#!/usr/bin/env python3
"""Lapis Console: orchestrate training, history, evaluation helpers, and chat."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_ROOT = REPO_ROOT / "training_history"
LOCAL_RUN_ROOT = REPO_ROOT / "checkpoints" / "colab-runs"
TRAINING_DATA_ROOT = REPO_ROOT / "training_data"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tiny.yaml"
DEFAULT_DATA = TRAINING_DATA_ROOT / "combined.txt"

GENERATED_GIT_PREFIXES = ("checkpoints/", "training_data/")

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ACCENT = "\033[38;5;180m"
MUTED = "\033[38;5;245m"
USER = "\033[38;5;117m"
OK = "\033[38;5;114m"
WARN = "\033[38;5;221m"
ERR = "\033[38;5;203m"


def color(text: str, style: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("TERM") == "dumb":
        return text
    return f"{style}{text}{RESET}"


def banner() -> None:
    print()
    print(color("╭──────────────────────────────────────────────────────────╮", ACCENT))
    print(color("│                    L A P I S                           │", ACCENT + BOLD))
    print(color("│              experiment / training console             │", MUTED))
    print(color("╰──────────────────────────────────────────────────────────╯", ACCENT))
    print()


def run(
    cmd: list[str], *, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


def git_status() -> list[str]:
    result = run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def status_path(status_line: str) -> str:
    """Extract the path portion from a porcelain-v1 status line."""
    path = status_line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path


def is_generated_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(
        normalized == prefix[:-1] or normalized.startswith(prefix)
        for prefix in GENERATED_GIT_PREFIXES
    )


def clean_generated_artifacts_before_pull(status: list[str]) -> None:
    """Discard only known runtime-generated files before a fast-forward pull."""
    generated = [line for line in status if is_generated_path(status_path(line))]
    protected = [line for line in status if not is_generated_path(status_path(line))]

    if protected:
        raise RuntimeError(
            "Working tree has non-generated local changes. Refusing to pull:\n"
            + "\n".join(protected)
        )

    if not generated:
        return

    print(color("Git", BOLD) + "  cleaning generated Colab artifacts before pull...")

    tracked = run(
        ["git", "ls-files", "--", "checkpoints", "training_data"],
        capture=True,
    ).stdout.splitlines()
    tracked = [path for path in tracked if path.strip()]

    if tracked:
        run(["git", "restore", "--source=HEAD", "--worktree", "--", *tracked])
        print(color(f"✓ Restored {len(tracked)} tracked generated file(s)", OK))

    if TRAINING_DATA_ROOT.exists():
        untracked_data = [
            line for line in generated if status_path(line).replace("\\", "/").startswith("training_data/")
        ]
        if untracked_data:
            shutil.rmtree(TRAINING_DATA_ROOT)
            print(color("✓ Removed generated training_data/; it will be rebuilt", OK))


def git_sync_pull() -> None:
    status = git_status()
    clean_generated_artifacts_before_pull(status)

    status = git_status()
    if status:
        raise RuntimeError(
            "Working tree is not clean after generated-artifact cleanup:\n"
            + "\n".join(status)
        )

    print(color("Git", BOLD) + "  pulling origin/main...")
    run(["git", "pull", "--ff-only", "origin", "main"])
    print(color("✓ Git pull completed", OK))


def git_sync_push(message: str) -> None:
    status = git_status()
    if not status:
        print(color("✓ Nothing new to push", MUTED))
        return

    non_history = [
        line
        for line in status
        if not status_path(line).replace("\\", "/").startswith("training_history/")
    ]
    if non_history:
        raise RuntimeError(
            "Refusing to push because non-history local changes remain:\n"
            + "\n".join(non_history)
        )

    print(color("Git", BOLD) + "  staging training history...")
    run(["git", "add", "training_history"])
    staged = run(
        ["git", "diff", "--cached", "--name-only"],
        capture=True,
    ).stdout.strip()
    if not staged:
        print(color("✓ No pushable history changes", MUTED))
        return

    run(["git", "commit", "-m", message])
    print(color("Git", BOLD) + "  pushing origin/main...")
    run(["git", "push", "origin", "main"])
    print(color("✓ Changes pushed to origin/main", OK))


def progress_bar(step: int, total: int, width: int = 34) -> str:
    ratio = min(1.0, max(0.0, step / max(1, total)))
    filled = int(width * ratio)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    seconds_i = int(seconds)
    minutes, sec = divmod(seconds_i, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {sec:02d}s"


def parse_steps_config(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*max_steps:\s*(\d+)\s*$", text)
    if not match:
        raise RuntimeError(f"Cannot find training.max_steps in {path}")
    return int(match.group(1))


def parse_training_line(line: str) -> tuple[int, float | None] | None:
    match = re.search(r"step=(\d+)\s+loss=([0-9.]+)", line)
    if not match:
        return None
    return int(match.group(1)), float(match.group(2))


def train_one_run(
    *,
    run_number: int,
    total_runs: int,
    config: Path,
    corpus: Path,
    monitor_interval: int,
) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"run-{timestamp}-{uuid.uuid4().hex[:6]}"
    local_dir = LOCAL_RUN_ROOT / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = local_dir / "checkpoint.pt"
    monitor_log = local_dir / "learning_monitor.jsonl"
    target_steps = parse_steps_config(config)

    print()
    print(color(f"RUN {run_number}/{total_runs}", ACCENT + BOLD))
    print(f"id          {run_id}")
    print(f"config      {config.relative_to(REPO_ROOT)}")
    print(f"target      {target_steps:,} optimizer steps")
    print(f"checkpoint  {checkpoint.relative_to(REPO_ROOT)}")
    print()

    command = [
        sys.executable,
        "scripts/train.py",
        "--config",
        str(config.relative_to(REPO_ROOT)),
        "--device",
        "cuda",
        "--data",
        str(corpus.relative_to(REPO_ROOT)),
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
        str(41 + run_number),
    ]

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_step = 0
    last_loss: float | None = None

    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
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
            print()
            print(color(line, ACCENT + BOLD))
        elif line.startswith("step="):
            print("\n" + line)
        elif line.startswith("Prompt :") or line.startswith("Lapis  :"):
            print(line)

    print()
    return_code = process.wait()
    elapsed = time.monotonic() - started

    if return_code != 0:
        raise RuntimeError(f"Training process failed with exit code {return_code} for {run_id}")

    if not checkpoint.exists():
        raise RuntimeError(f"Training finished but checkpoint is missing: {checkpoint}")

    checkpoint_size = checkpoint.stat().st_size
    if checkpoint_size < 1_000_000:
        raise RuntimeError(f"Checkpoint is suspiciously small: {checkpoint_size} bytes")

    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    recorded_step = int(state.get("step", -1))
    if recorded_step != target_steps:
        raise RuntimeError(
            f"Training produced a checkpoint at step {recorded_step}, expected {target_steps}"
        )

    monitor_records = []
    if monitor_log.exists():
        for line in monitor_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                monitor_records.append(json.loads(line))
    if not monitor_records:
        raise RuntimeError("Learning monitor produced no records")

    metric_records = [r for r in monitor_records if r.get("type") == "metrics"]
    sample_records = [r for r in monitor_records if r.get("type") == "samples"]
    final_metrics = metric_records[-1] if metric_records else {}
    final_samples = sample_records[-1].get("samples", []) if sample_records else []

    summary = {
        "run_id": run_id,
        "run_number": run_number,
        "timestamp_utc": timestamp,
        "config": str(config.relative_to(REPO_ROOT)),
        "corpus": str(corpus.relative_to(REPO_ROOT)),
        "target_steps": target_steps,
        "recorded_step": recorded_step,
        "checkpoint_size_bytes": checkpoint_size,
        "elapsed_seconds": round(elapsed, 3),
        "final_loss": final_metrics.get("loss", last_loss),
        "final_perplexity": final_metrics.get("perplexity"),
        "tokens_seen": final_metrics.get("tokens_seen"),
        "monitor_records": len(monitor_records),
        "samples": final_samples,
    }
    return summary


def write_history(summary: dict) -> Path:
    run_dir = HISTORY_ROOT / summary["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {summary['run_id']}",
        "",
        f"- Date (UTC): `{summary['timestamp_utc']}`",
        f"- Steps: `{summary['recorded_step']:,}`",
        f"- Final loss: `{summary['final_loss']}`",
        f"- Final perplexity: `{summary['final_perplexity']}`",
        f"- Tokens seen: `{summary['tokens_seen']}`",
        f"- Duration: `{summary['elapsed_seconds']:.1f}s`",
        "",
        "## Final learning samples",
        "",
    ]
    for sample in summary.get("samples", []):
        lines.extend(
            [
                f"### {sample['prompt']}",
                "",
                sample["completion"] or "<EOS>",
                "",
            ]
        )
    (run_dir / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


def load_summaries() -> list[dict]:
    if not HISTORY_ROOT.exists():
        return []
    items: list[dict] = []
    for path in sorted(HISTORY_ROOT.glob("*/summary.json")):
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return items


def show_history() -> None:
    summaries = load_summaries()
    if not summaries:
        print("No training history yet.")
        return
    print(color("TRAINING HISTORY", ACCENT + BOLD))
    print("ID                         LOSS       PPL       STEPS     DURATION")
    print("─" * 70)
    for item in summaries:
        loss = item.get("final_loss")
        ppl = item.get("final_perplexity")
        duration = item.get("elapsed_seconds", 0)
        print(
            f"{item['run_id']:<26} "
            f"{str(loss):<10} "
            f"{str(ppl):<9} "
            f"{item.get('recorded_step', 0):>7,} "
            f"{duration:>8.1f}s"
        )


def choose_summary(label: str, summaries: list[dict]) -> dict:
    print(f"\n{label}")
    for idx, item in enumerate(summaries, 1):
        print(
            f"[{idx}] {item['run_id']}  "
            f"loss={item.get('final_loss')} ppl={item.get('final_perplexity')}"
        )
    raw = input("› ").strip()
    try:
        index = int(raw) - 1
        return summaries[index]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("Invalid history selection") from exc


def compare_runs() -> None:
    summaries = load_summaries()
    if len(summaries) < 2:
        print("At least two completed runs are required.")
        return
    left = choose_summary("Run A", summaries)
    right = choose_summary("Run B", summaries)
    print()
    print(color("RUN COMPARISON", ACCENT + BOLD))
    print("                         A                    B")
    print("─" * 65)
    for key, label in [
        ("final_loss", "Loss"),
        ("final_perplexity", "Perplexity"),
        ("tokens_seen", "Tokens"),
        ("elapsed_seconds", "Duration (s)"),
    ]:
        print(
            f"{label:<24} {str(left.get(key)):<20} {str(right.get(key)):<20}"
        )


def system_info() -> None:
    print(color("SYSTEM", ACCENT + BOLD))
    print(run(["git", "branch", "--show-current"], capture=True).stdout.strip())
    try:
        import torch

        print(f"PyTorch     : {torch.__version__}")
        print(f"CUDA        : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU         : {torch.cuda.get_device_name(0)}")
            print(f"VRAM        : {props.total_memory / (1024 ** 3):.2f} GB")
    except ImportError:
        print("PyTorch     : not installed")


def launch_chat() -> None:
    checkpoints = sorted(
        LOCAL_RUN_ROOT.glob("*/checkpoint.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not checkpoints:
        raise RuntimeError("No locally trained checkpoint found. Train a run first.")
    checkpoint = checkpoints[0]
    print(color(f"Launching latest trained checkpoint: {checkpoint}", OK))
    run(
        [
            sys.executable,
            "-m",
            "scripts.chat",
            "--checkpoint",
            str(checkpoint.relative_to(REPO_ROOT)),
            "--device",
            "cuda",
        ]
    )


def fetch_data() -> None:
    print(color("DATA", BOLD) + "  building corpus...")
    run(
        [
            sys.executable,
            "scripts/fetch_training_data.py",
            "--output-dir",
            "training_data",
            "--max-wiki-articles",
            "40",
            "--max-doc-files",
            "20",
        ]
    )
    if not DEFAULT_DATA.exists() or DEFAULT_DATA.stat().st_size < 100_000:
        raise RuntimeError("Training corpus is missing or suspiciously small")
    print(color("✓ Training corpus ready", OK))


def train_flow() -> None:
    raw = input("How many training runs? › ").strip()
    try:
        runs = int(raw)
    except ValueError as exc:
        raise RuntimeError("Please enter a positive integer") from exc
    if runs < 1:
        raise RuntimeError("Number of runs must be at least 1")

    monitor_raw = input("Learning monitor interval [500]? › ").strip()
    try:
        monitor_interval = int(monitor_raw) if monitor_raw else 500
    except ValueError as exc:
        raise RuntimeError("Monitor interval must be an integer") from exc
    if monitor_interval < 1:
        raise RuntimeError("Monitor interval must be at least 1")

    if not DEFAULT_CONFIG.exists():
        raise RuntimeError(f"Training config not found: {DEFAULT_CONFIG}")

    fetch_data()
    summaries = []
    for run_number in range(1, runs + 1):
        summary = train_one_run(
            run_number=run_number,
            total_runs=runs,
            config=DEFAULT_CONFIG,
            corpus=DEFAULT_DATA,
            monitor_interval=monitor_interval,
        )
        history_dir = write_history(summary)
        summaries.append(summary)
        print()
        print(color("✓ VERIFIED RUN", OK + BOLD))
        print(f"  steps       {summary['recorded_step']:,}")
        print(f"  loss        {summary['final_loss']}")
        print(f"  perplexity  {summary['final_perplexity']}")
        print(f"  history     {history_dir.relative_to(REPO_ROOT)}")

    message = f"train: {runs} verified Lapis run{'s' if runs != 1 else ''}"
    git_sync_push(message)

    print()
    print(color("All runs complete and history pushed.", OK + BOLD))
    launch_chat()


def interactive() -> None:
    banner()
    print("[1] Train")
    print("[2] Resume training")
    print("[3] Chat")
    print("[4] Evaluate")
    print("[5] Training history")
    print("[6] Compare runs")
    print("[7] System / GPU info")
    print()
    choice = input("› ").strip()
    if choice == "1":
        train_flow()
    elif choice == "2":
        print(
            "Resume mode is exposed through scripts/train.py --resume; "
            "use it after selecting a checkpoint."
        )
    elif choice == "3":
        launch_chat()
    elif choice == "4":
        run([sys.executable, "scripts/evaluate.py", "--help"])
    elif choice == "5":
        show_history()
    elif choice == "6":
        compare_runs()
    elif choice == "7":
        system_info()
    else:
        raise RuntimeError("Unknown menu selection")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lapis experiment console")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "train", help="pull, train, verify, push, then launch chat"
    )
    subparsers.add_parser(
        "chat", help="launch the latest locally trained checkpoint"
    )
    subparsers.add_parser("history", help="show training history")
    subparsers.add_parser("compare", help="compare two completed runs")
    subparsers.add_parser("system", help="show Git/GPU information")

    args = parser.parse_args()
    try:
        if args.command is None:
            git_sync_pull()
            interactive()
        elif args.command == "train":
            git_sync_pull()
            train_flow()
        elif args.command == "chat":
            launch_chat()
        elif args.command == "history":
            show_history()
        elif args.command == "compare":
            compare_runs()
        elif args.command == "system":
            system_info()
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        print(color(f"Lapis error: {exc}", ERR), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
