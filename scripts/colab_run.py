#!/usr/bin/env python3
"""One-command Colab training orchestrator for LapisLLM.

Flow:
    clone/reuse checkout -> pull main -> install -> validate -> tests
    -> build corpus when missing -> run N verified training runs
    -> record history after every run -> push training history to main

Large generated artifacts (training_data/, checkpoints/) remain git-ignored.
"""

from __future__ import annotations

import argparse
import getpass
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
REQUIRED_FILES = (
    "pyproject.toml",
    "configs/tiny.yaml",
    "scripts/train.py",
    "scripts/fetch_training_data.py",
)


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("$", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=capture,
        env=merged_env,
    )


def prompt_int(label: str, default: int, minimum: int = 0) -> int:
    raw = input(f"{label} [{default}] : ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def bootstrap_repo() -> None:
    print_banner("LAPISLLM — INITIALISATION")
    if not REPO_ROOT.exists():
        print("Clonage du dépôt...")
        subprocess.run(
            [
                "git",
                "-c",
                "http.version=HTTP/1.1",
                "-c",
                "http.lowSpeedLimit=1000",
                "-c",
                "http.lowSpeedTime=30",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                "main",
                REPO_URL,
                str(REPO_ROOT),
            ],
            check=True,
        )
    elif not (REPO_ROOT / ".git").exists():
        raise RuntimeError(f"{REPO_ROOT} exists but is not a Git repository.")
    print(f"Répertoire : {REPO_ROOT}")


def path_from_status(status_line: str) -> str:
    path = status_line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.replace("\\", "/")


def sync_pull() -> None:
    print_banner("GIT — SYNCHRONISATION")
    for key, value in (
        ("user.name", "LapisLLM Colab"),
        ("user.email", "lapisllm-colab@users.noreply.github.com"),
    ):
        if not run(["git", "config", key], check=False, capture=True).stdout.strip():
            run(["git", "config", key, value])

    status = run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture=True,
    ).stdout.splitlines()
    protected = [
        line
        for line in status
        if not path_from_status(line).startswith("training_history/")
    ]
    if protected:
        raise RuntimeError(
            "Refus de synchroniser : des modifications locales hors "
            "training_history existent.\n" + "\n".join(protected)
        )

    history_changes = [
        line for line in status if path_from_status(line).startswith("training_history/")
    ]
    if history_changes:
        run(["git", "add", "training_history"])
        staged = run(
            ["git", "diff", "--cached", "--name-only"],
            capture=True,
        ).stdout.strip()
        if staged:
            run(["git", "commit", "-m", "chore: save local training history"])

    run(["git", "fetch", "origin", "main"])
    run(["git", "pull", "--rebase", "origin", "main"])
    print("✅ Git pull terminé.")


def validate_project() -> None:
    print_banner("VALIDATION DU PROJET")
    missing = [path for path in REQUIRED_FILES if not (REPO_ROOT / path).exists()]
    if missing:
        raise RuntimeError("Fichiers requis absents :\n" + "\n".join(missing))
    for path in REQUIRED_FILES:
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
    else:
        print("GPU : aucun")
    return "cuda" if cuda else "cpu"


def parse_max_steps(config: Path) -> int:
    text = config.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*max_steps:\s*(\d+)\s*$", text)
    if not match:
        raise RuntimeError(f"Impossible de trouver training.max_steps dans {config}")
    steps = int(match.group(1))
    if steps < 1:
        raise ValueError("training.max_steps doit être >= 1")
    print(f"✅ max_steps = {steps:,}")
    return steps


def build_corpus(max_wiki_articles: int = 40, max_doc_files: int = 20) -> None:
    print_banner("DONNÉES")
    TRAINING_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if DEFAULT_DATA.exists() and DEFAULT_DATA.stat().st_size >= 100_000:
        print(f"✅ Corpus existant : {DEFAULT_DATA.stat().st_size:,} octets")
        return

    print("📚 Corpus absent ou trop petit.")
    print("Construction du corpus...")
    run(
        [
            sys.executable,
            "scripts/fetch_training_data.py",
            "--output-dir",
            "training_data",
            "--max-wiki-articles",
            str(max_wiki_articles),
            "--max-doc-files",
            str(max_doc_files),
        ]
    )
    if not DEFAULT_DATA.exists() or DEFAULT_DATA.stat().st_size < 100_000:
        raise RuntimeError("Le corpus n'a pas été généré correctement.")
    print(f"✅ Corpus prêt : {DEFAULT_DATA.stat().st_size:,} octets")


def progress_bar(step: int, total: int, width: int = 34) -> str:
    ratio = min(1.0, max(0.0, step / max(1, total)))
    filled = int(width * ratio)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    seconds_i = int(seconds)
    minutes, seconds_i = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds_i:02d}s"


def parse_training_line(line: str) -> tuple[int, float] | None:
    match = re.search(r"step=(\d+)\s+loss=([0-9.eE+-]+)", line)
    if not match:
        return None
    return int(match.group(1)), float(match.group(2))


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def verify_run(checkpoint: Path, monitor_log: Path, expected_steps: int) -> dict:
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint absent : {checkpoint}")

    size = checkpoint.stat().st_size
    if size < 1_000_000:
        raise RuntimeError(f"Checkpoint suspectement petit : {size} octets")

    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recorded_step = int(state.get("step", -1))
    if recorded_step != expected_steps:
        raise RuntimeError(
            f"Checkpoint à l'étape {recorded_step}, attendu {expected_steps}"
        )

    records: list[dict] = []
    if monitor_log.exists():
        for line in monitor_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    metrics = [item for item in records if item.get("type") == "metrics"]
    samples = [item for item in records if item.get("type") == "samples"]

    return {
        "recorded_step": recorded_step,
        "checkpoint_size_bytes": size,
        "monitor_records": len(records),
        "final_metrics": metrics[-1] if metrics else {},
        "final_samples": samples[-1].get("samples", []) if samples else [],
    }


def write_history(summary: dict) -> Path:
    run_dir = HISTORY_ROOT / summary["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        f"# {summary['run_id']}",
        "",
        f"- Date (UTC): `{summary['timestamp_utc']}`",
        f"- Run: `{summary['run_number']}/{summary['total_runs']}`",
        f"- Steps: `{summary['recorded_step']:,}`",
        f"- Final loss: `{summary['final_loss']}`",
        f"- Final perplexity: `{summary['final_perplexity']}`",
        f"- Tokens seen: `{summary['tokens_seen']}`",
        f"- Duration: `{summary['elapsed_seconds']:.1f}s`",
        f"- Device: `{summary['device']}`",
        "",
        "## Final learning samples",
        "",
    ]
    for sample in summary.get("samples", []):
        lines += [
            f"### {sample.get('prompt', '<unknown>')}",
            "",
            sample.get("completion") or "<EOS>",
            "",
        ]
    (run_dir / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


def train_one_run(
    *,
    run_number: int,
    total_runs: int,
    config: Path,
    data: Path,
    monitor_interval: int,
    device: str,
    seed: int,
) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"run-{timestamp}-{uuid.uuid4().hex[:6]}"
    local_dir = CHECKPOINT_ROOT / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = local_dir / "checkpoint.pt"
    monitor_log = local_dir / "learning_monitor.jsonl"
    target_steps = parse_max_steps(config)

    print_banner(f"RUN {run_number}/{total_runs}")
    print(f"id          {run_id}")
    print(f"config      {config.relative_to(REPO_ROOT)}")
    print(f"target      {target_steps:,} optimizer steps")
    print(f"checkpoint  {checkpoint.relative_to(REPO_ROOT)}")
    print(f"seed        {seed}")
    print()

    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.train",
        "--config",
        str(config.relative_to(REPO_ROOT)),
        "--device",
        device,
        "--data",
        str(data.relative_to(REPO_ROOT)),
        "--epochs",
        "1000",
        "--checkpoint",
        str(checkpoint.relative_to(REPO_ROOT)),
        "--monitor-interval",
        str(monitor_interval),
        "--monitor-sample-tokens",
        "64",
        "--monitor-log",
        str(monitor_log.relative_to(REPO_ROOT)),
        "--seed",
        str(seed),
    ]

    started = time.monotonic()
    output_tail: list[str] = []
    last_step = 0
    last_loss: float | None = None
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    assert process.stdout is not None
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
                eta = (target_steps - last_step) / speed if speed > 0 else None
                print(
                    f"\r{progress_bar(last_step, target_steps)} "
                    f"{last_step:5,d}/{target_steps:,} "
                    f"{last_step / target_steps * 100:6.2f}% "
                    f"loss={last_loss:.4f} "
                    f"{speed:.2f} step/s ETA {format_eta(eta)}",
                    end="",
                    flush=True,
                )
            elif "LEARNING MONITOR" in line or "WHAT LAPIS IS LEARNING" in line:
                print("\n" + line)
            elif line.startswith("Prompt :") or line.startswith("Lapis  :"):
                print(line)
    except KeyboardInterrupt:
        print("\n⚠️ Interruption demandée : arrêt propre du processus...")
        terminate_process_group(process)
        raise KeyboardInterrupt from None

    print()
    return_code = process.wait()
    elapsed = time.monotonic() - started
    if return_code != 0:
        diagnostic = "\n".join(output_tail[-20:])
        raise RuntimeError(
            f"Training failed (exit {return_code}) for {run_id}\n{diagnostic}"
        )

    verification = verify_run(checkpoint, monitor_log, target_steps)
    metrics = verification["final_metrics"]
    return {
        "run_id": run_id,
        "run_number": run_number,
        "total_runs": total_runs,
        "timestamp_utc": timestamp,
        "config": str(config.relative_to(REPO_ROOT)),
        "data": str(data.relative_to(REPO_ROOT)),
        "device": device,
        "seed": seed,
        "target_steps": target_steps,
        "recorded_step": verification["recorded_step"],
        "checkpoint_size_bytes": verification["checkpoint_size_bytes"],
        "elapsed_seconds": round(elapsed, 3),
        "final_loss": metrics.get("loss", last_loss),
        "final_perplexity": metrics.get("perplexity"),
        "tokens_seen": metrics.get("tokens_seen"),
        "monitor_records": verification["monitor_records"],
        "samples": verification["final_samples"],
    }


def run_tests() -> None:
    print_banner("VALIDATION — TESTS")
    result = run([sys.executable, "-m", "pytest", "-q"], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"pytest a échoué avec le code {result.returncode}")
    print("✅ Tests réussis.")


def get_github_token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token.strip()

    try:
        from google.colab import userdata

        token = userdata.get("GITHUB_TOKEN")
        if token:
            return str(token).strip()
    except Exception:
        pass
    return None


def push_history() -> None:
    print_banner("GIT — PUSH DE L'HISTORIQUE")
    status = run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture=True,
    ).stdout.splitlines()
    protected = [
        line
        for line in status
        if not path_from_status(line).startswith("training_history/")
    ]
    if protected:
        raise RuntimeError(
            "Refus de push : modifications hors training_history détectées.\n"
            + "\n".join(protected)
        )

    run(["git", "add", "training_history"])
    staged = run(
        ["git", "diff", "--cached", "--name-only"],
        capture=True,
    ).stdout.strip()
    if not staged:
        print("✅ Aucun nouvel historique à pousser.")
        return

    run(["git", "commit", "-m", "chore: record automated training run history"])

    token = get_github_token()
    if token:
        result = run(
            [
                "git",
                "-c",
                f"http.extraheader=AUTHORIZATION: bearer {token}",
                "push",
                "origin",
                "main",
            ],
            check=False,
        )
    else:
        print("Aucun GITHUB_TOKEN/GH_TOKEN trouvé ; tentative avec les credentials Git existants.")
        result = run(["git", "push", "origin", "main"], check=False)
        if result.returncode != 0:
            token = getpass.getpass(
                "GitHub PAT pour pousser vers main (entrée masquée) : "
            ).strip()
            if not token:
                raise RuntimeError("Push impossible : aucun GitHub token fourni.")
            result = run(
                [
                    "git",
                    "-c",
                    f"http.extraheader=AUTHORIZATION: bearer {token}",
                    "push",
                    "origin",
                    "main",
                ],
                check=False,
            )

    if result.returncode != 0:
        raise RuntimeError(f"git push a échoué avec le code {result.returncode}")
    print("✅ Historique poussé sur origin/main.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatisation complète du training LapisLLM dans Google Colab"
    )
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
        bootstrap_repo()
        sync_pull()
        validate_project()
        install_project()
        detected_device = inspect_torch()

        config = REPO_ROOT / args.config
        if not config.exists():
            raise RuntimeError(f"Configuration absente : {config}")

        if not args.skip_tests:
            run_tests()

        runs = args.runs
        if runs is None:
            runs = prompt_int("Combien de runs d'entraînement ?", 1, 1)
        monitor_interval = args.monitor_interval
        if monitor_interval is None:
            monitor_interval = prompt_int("Learning monitor interval", 500, 0)

        device = detected_device if args.device == "auto" else args.device
        if device == "cuda" and detected_device != "cuda":
            raise RuntimeError("CUDA demandé mais aucun GPU CUDA n'est disponible.")

        target_steps = parse_max_steps(config)
        print_banner("PARAMÈTRES")
        print(f"Runs              : {runs}")
        print(f"Optimizer steps   : {target_steps:,} / run")
        print(f"Monitor interval  : {monitor_interval}")
        print(f"Device            : {device}")

        build_corpus(
            max_wiki_articles=args.max_wiki_articles,
            max_doc_files=args.max_doc_files,
        )

        all_summaries: list[dict] = []
        for offset in range(runs):
            seed = args.seed + offset
            try:
                summary = train_one_run(
                    run_number=offset + 1,
                    total_runs=runs,
                    config=config,
                    data=DEFAULT_DATA,
                    monitor_interval=monitor_interval,
                    device=device,
                    seed=seed,
                )
                history_dir = write_history(summary)
                all_summaries.append(summary)
                print_banner("✅ RUN VERIFIED")
                print(f"steps       : {summary['recorded_step']:,}")
                print(f"loss        : {summary['final_loss']}")
                print(f"perplexity  : {summary['final_perplexity']}")
                print(f"duration    : {summary['elapsed_seconds']:.1f}s")
                print(f"history     : {history_dir.relative_to(REPO_ROOT)}")
            except KeyboardInterrupt:
                print("\n❌ Training interrompu.")
                break
            except Exception as exc:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                failed = {
                    "run_id": f"failed-{timestamp}-{uuid.uuid4().hex[:6]}",
                    "run_number": offset + 1,
                    "total_runs": runs,
                    "timestamp_utc": timestamp,
                    "status": "failed",
                    "error": str(exc),
                    "device": device,
                    "seed": seed,
                }
                write_history(failed)
                print(f"\n❌ RUN {offset + 1}/{runs} échoué : {exc}")
                if not args.no_push:
                    push_history()
                return 1

        print_banner("PROCESSUS TERMINÉ")
        print(f"Runs vérifiés : {len(all_summaries)}/{runs}")
        if not args.no_push:
            push_history()
        return 0 if len(all_summaries) == runs else 130

    except KeyboardInterrupt:
        print("\n❌ Processus interrompu.")
        if not args.no_push:
            try:
                push_history()
            except Exception as exc:
                print(f"⚠️ Impossible de pousser l'historique : {exc}")
        return 130
    except Exception as exc:
        print(f"\n❌ ERREUR FATALE : {exc}", file=sys.stderr)
        if not args.no_push:
            try:
                push_history()
            except Exception as push_exc:
                print(f"⚠️ Push final impossible : {push_exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
