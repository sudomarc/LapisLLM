#!/usr/bin/env python3
"""One-command Colab training orchestrator for LapisLLM.

The Colab UI uses Rich when available for a compact live dashboard while
keeping a plain-text fallback for non-interactive environments.
"""

from __future__ import annotations

# NOTE: The full pipeline implementation remains below; only the UI layer is
# changed here so training behavior and checkpoint semantics stay untouched.

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/sudomarc/LapisLLM.git"
REPO_ROOT = Path("/content/LapisLLM")
HISTORY_ROOT = REPO_ROOT / "training_history"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "colab-runs"
TRAINING_DATA_ROOT = REPO_ROOT / "training_data"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tiny.yaml"
DEFAULT_DATA = TRAINING_DATA_ROOT / "combined.txt"
REQUIRED_FILES = ("pyproject.toml", "configs/tiny.yaml", "scripts/train.py", "scripts/fetch_training_data.py")

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def print_banner(title: str) -> None:
    if RICH_AVAILABLE:
        Console().rule(f"[bold cyan]{title}[/]")
    else:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)


def run(cmd: list[str], *, check: bool = True, capture: bool = False, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO_ROOT, check=check, text=True, capture_output=capture, env=merged_env)


def prompt_int(label: str, default: int, minimum: int = 0) -> int:
    raw = input(f"{label} [{default}] : ").strip()
    if not raw:
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def bootstrap_repo() -> None:
    print_banner("LAPISLLM — INITIALISATION")
    if not REPO_ROOT.exists():
        subprocess.run(["git", "-c", "http.version=HTTP/1.1", "-c", "http.lowSpeedLimit=1000", "-c", "http.lowSpeedTime=30", "clone", "--depth", "1", "--single-branch", "--branch", "main", REPO_URL, str(REPO_ROOT)], check=True)
    elif not (REPO_ROOT / ".git").exists():
        raise RuntimeError(f"{REPO_ROOT} exists but is not a Git repository.")
    print(f"Répertoire : {REPO_ROOT}")


def status_path(line: str) -> str:
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.replace("\\", "/")


def sync_pull() -> None:
    print_banner("GIT — SYNCHRONISATION")
    for key, value in (("user.name", "LapisLLM Colab"), ("user.email", "lapisllm-colab@users.noreply.github.com")):
        if not run(["git", "config", key], check=False, capture=True).stdout.strip():
            run(["git", "config", key, value])
    status = run(["git", "status", "--porcelain", "--untracked-files=all"], capture=True).stdout.splitlines()
    protected = [line for line in status if not status_path(line).startswith("training_history/")]
    if protected:
        raise RuntimeError("Refus de synchroniser : des modifications locales hors training_history existent.\n" + "\n".join(protected))
    history = [line for line in status if status_path(line).startswith("training_history/")]
    if history:
        run(["git", "add", "training_history"])
        if run(["git", "diff", "--cached", "--name-only"], capture=True).stdout.strip():
            run(["git", "commit", "-m", "chore: save local training history"])
    run(["git", "fetch", "origin", "main"])
    run(["git", "pull", "--rebase", "origin", "main"])


def validate_project() -> None:
    print_banner("VALIDATION DU PROJET")
    for path in REQUIRED_FILES:
        if not (REPO_ROOT / path).exists():
            raise RuntimeError(f"Fichier requis absent : {path}")
        print(f"✅ {path}")


def install_project() -> None:
    print_banner("INSTALLATION")
    run([sys.executable, "-m", "pip", "install", "-e", ".[data]", "-q"])
    print("✅ Dépendances installées.")


def inspect_torch() -> str:
    print_banner("ENVIRONNEMENT TORCH")
    import torch
    cuda = bool(torch.cuda.is_available())
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA disponible : {cuda}")
    if cuda:
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    return "cuda" if cuda else "cpu"


def parse_max_steps(config: Path) -> int:
    match = re.search(r"(?m)^\s*max_steps:\s*(\d+)\s*$", config.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError(f"Impossible de trouver training.max_steps dans {config}")
    return int(match.group(1))


def build_corpus(max_wiki_articles: int = 40, max_doc_files: int = 20) -> None:
    print_banner("DONNÉES")
    TRAINING_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if DEFAULT_DATA.exists() and DEFAULT_DATA.stat().st_size >= 100_000:
        print(f"✅ Corpus existant : {DEFAULT_DATA.stat().st_size:,} octets")
        return
    run([sys.executable, "scripts/fetch_training_data.py", "--output-dir", "training_data", "--max-wiki-articles", str(max_wiki_articles), "--max-doc-files", str(max_doc_files)])
    if not DEFAULT_DATA.exists() or DEFAULT_DATA.stat().st_size < 100_000:
        raise RuntimeError("Le corpus n'a pas été généré correctement.")


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    seconds_i = int(seconds)
    minutes, seconds_i = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds_i:02d}s"


def parse_training_line(line: str) -> tuple[int, float] | None:
    match = re.search(r"step=(\d+)\s+loss=([0-9.eE+-]+)", line)
    return (int(match.group(1)), float(match.group(2))) if match else None


def dashboard_render(step: int, total: int, loss: float | None, speed: float, eta: str, device: str, gpu: str | None, vram: str | None) -> Panel:
    pct = (step / max(1, total)) * 100
    table = Table.grid(expand=True)
    table.add_column(justify="left", ratio=3)
    table.add_column(justify="right", ratio=2)
    table.add_row("Progress", f"{step:,} / {total:,}  ({pct:5.1f}%)")
    table.add_row("Loss", f"{loss:.5f}" if loss is not None else "--")
    table.add_row("Speed", f"{speed:.2f} step/s")
    table.add_row("ETA", eta)
    table.add_row("Device", device)
    if gpu:
        table.add_row("GPU", gpu)
    if vram:
        table.add_row("VRAM", vram)
    return Panel(table, title="[bold]LAPIS TRAINING[/]", border_style="cyan")


def train_one_run(*, run_number: int, total_runs: int, config: Path, data: Path, monitor_interval: int, device: str, seed: int) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"run-{timestamp}-{uuid.uuid4().hex[:6]}"
    local_dir = CHECKPOINT_ROOT / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = local_dir / "checkpoint.pt"
    monitor_log = local_dir / "learning_monitor.jsonl"
    target_steps = parse_max_steps(config)

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    vram = f"{torch.cuda.memory_reserved() / 1024**3:.1f} / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else None

    command = [sys.executable, "-u", "-m", "scripts.train", "--config", str(config.relative_to(REPO_ROOT)), "--device", device, "--data", str(data.relative_to(REPO_ROOT)), "--epochs", "1000", "--checkpoint", str(checkpoint.relative_to(REPO_ROOT)), "--monitor-interval", str(monitor_interval), "--monitor-sample-tokens", "64", "--monitor-log", str(monitor_log.relative_to(REPO_ROOT)), "--seed", str(seed)]
    started = time.monotonic()
    output_tail: list[str] = []
    last_step = 0
    last_loss: float | None = None
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    assert process.stdout is not None

    live = Live(dashboard_render(0, target_steps, None, 0.0, "--", device, gpu, vram), refresh_per_second=4) if RICH_AVAILABLE else None
    if live:
        live.start()
    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\n")
            if line.strip():
                output_tail.append(line)
                if len(output_tail) > 80:
                    output_tail.pop(0)
            parsed = parse_training_line(line)
            if parsed:
                last_step, last_loss = parsed
                elapsed = time.monotonic() - started
                speed = last_step / elapsed if elapsed > 0 else 0.0
                eta_seconds = (target_steps - last_step) / speed if speed > 0 else None
                if live:
                    if torch.cuda.is_available():
                        vram = f"{torch.cuda.memory_reserved() / 1024**3:.1f} / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
                    live.update(dashboard_render(last_step, target_steps, last_loss, speed, format_eta(eta_seconds), device, gpu, vram), refresh=True)
                else:
                    print(f"step={last_step:,}/{target_steps:,} loss={last_loss:.5f} speed={speed:.2f} step/s ETA={format_eta(eta_seconds)}", flush=True)
            elif not live and line.strip():
                print(line)
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise
    finally:
        if live:
            live.stop()

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Training failed (exit {return_code}) for {run_id}\n" + "\n".join(output_tail[-20:]))

    import torch
    if not checkpoint.exists() or checkpoint.stat().st_size < 1_000_000:
        raise RuntimeError(f"Checkpoint invalide : {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recorded_step = int(state.get("step", -1))
    if recorded_step != target_steps:
        raise RuntimeError(f"Checkpoint à l'étape {recorded_step}, attendu {target_steps}")

    metrics = []
    if monitor_log.exists():
        for line in monitor_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("type") == "metrics":
                    metrics.append(item)
    final = metrics[-1] if metrics else {}
    return {"run_id": run_id, "run_number": run_number, "total_runs": total_runs, "timestamp_utc": timestamp, "config": str(config.relative_to(REPO_ROOT)), "data": str(data.relative_to(REPO_ROOT)), "device": device, "seed": seed, "target_steps": target_steps, "recorded_step": recorded_step, "checkpoint_size_bytes": checkpoint.stat().st_size, "elapsed_seconds": round(time.monotonic() - started, 3), "final_loss": final.get("loss", last_loss), "final_perplexity": final.get("perplexity"), "tokens_seen": final.get("tokens_seen"), "samples": []}


def write_history(summary: dict) -> Path:
    run_dir = HISTORY_ROOT / summary["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return run_dir


def run_tests() -> None:
    print_banner("VALIDATION — TESTS")
    run([sys.executable, "-m", "pytest", "-q"])


def push_history() -> None:
    print_banner("GIT — PUSH DE L'HISTORIQUE")
    run(["git", "add", "training_history"])
    if not run(["git", "diff", "--cached", "--name-only"], capture=True).stdout.strip():
        print("✅ Aucun nouvel historique à pousser.")
        return
    run(["git", "commit", "-m", "chore: record automated training run history"])
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        run(["git", "-c", f"http.extraheader=AUTHORIZATION: bearer {token}", "push", "origin", "main"], check=False)
        return
    print("Aucun GITHUB_TOKEN/GH_TOKEN trouvé ; historique commité localement.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--monitor-interval", type=int, default=None)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--max-wiki-articles", type=int, default=40)
    parser.add_argument("--max-doc-files", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        bootstrap_repo(); sync_pull(); validate_project(); install_project(); detected_device = inspect_torch()
        config = REPO_ROOT / args.config
        if not config.exists(): raise RuntimeError(f"Configuration absente : {config}")
        if not args.skip_tests: run_tests()
        runs = args.runs if args.runs is not None else prompt_int("Combien de runs d'entraînement ?", 1, 1)
        monitor_interval = args.monitor_interval if args.monitor_interval is not None else prompt_int("Learning monitor interval", 500, 0)
        device = detected_device if args.device == "auto" else args.device
        if device == "cuda" and detected_device != "cuda": raise RuntimeError("CUDA demandé mais aucun GPU CUDA n'est disponible.")
        build_corpus(args.max_wiki_articles, args.max_doc_files)
        all_summaries = []
        for offset in range(runs):
            summary = train_one_run(run_number=offset + 1, total_runs=runs, config=config, data=DEFAULT_DATA, monitor_interval=monitor_interval, device=device, seed=args.seed + offset)
            write_history(summary); all_summaries.append(summary)
        if not args.no_push: push_history()
        return 0 if len(all_summaries) == runs else 1
    except KeyboardInterrupt:
        print("\n❌ Processus interrompu.")
        return 130
    except Exception as exc:
        print(f"\n❌ ERREUR : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())