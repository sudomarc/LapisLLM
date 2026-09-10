#!/usr/bin/env python3
"""Stable, non-interactive Colab entry point for LapisLLM training."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import _colab_train_impl as _impl

HISTORY = _impl.HISTORY
METRIC_RE = _impl.METRIC_RE
TOKEN_RE = _impl.TOKEN_RE
format_duration = _impl.format_duration
get_github_token = _impl.get_github_token


def completed_run_numbers() -> set[int]:
    original = _impl.HISTORY
    _impl.HISTORY = HISTORY
    try:
        return _impl.completed_run_numbers()
    finally:
        _impl.HISTORY = original


def _stage_latest_checkpoint() -> None:
    subprocess.run(
        [sys.executable, "-m", "scripts.publish_checkpoint", "--no-push"],
        cwd=ROOT,
        check=True,
    )


def train_one_run(run_number: int, total_runs: int, args, device: str, config: Path) -> dict:
    """Run the real trainer directly instead of the broken Typer module invocation."""
    run_dir = _impl.CHECKPOINTS / f"run-{run_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"
    monitor_log = run_dir / "learning_monitor.jsonl"
    steps = _impl.target_steps(config)
    _impl.phase("RUN", f"{run_number} / {total_runs}")
    _impl.phase("RUN", f"target_steps={steps:,} | checkpoint={checkpoint}")
    _impl.phase("TOKENIZER", "STARTING | trainer will reuse a compatible on-disk tokenizer when available")
    _impl.phase("DATASET", "STARTING | corpus -> token IDs -> fixed-length samples")
    _impl.phase("MODEL", "STARTING | initialization and device placement occur inside scripts.train")
    _impl.phase("TRAINING", "START")

    command = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config", str(config.relative_to(ROOT)),
        "--data", str(_impl.CORPUS.relative_to(ROOT)),
        "--checkpoint", str(checkpoint.relative_to(ROOT)),
        "--device", device,
        "--epochs", "1000",
        "--monitor-interval", str(args.monitor_interval),
        "--monitor-sample-tokens", str(args.monitor_sample_tokens),
        "--monitor-log", str(monitor_log.relative_to(ROOT)),
    ]
    if args.monitor_prompts:
        command += ["--monitor-prompts", args.monitor_prompts]

    started = time.monotonic()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    try:
        metrics = _impl.monitor_child(
            process,
            run_number=run_number,
            total_runs=total_runs,
            target_steps=steps,
            started=started,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        metrics["device"] = device
        _impl.phase("TRAINING", f"COMPLETE | run={run_number}/{total_runs} | elapsed={format_duration(time.monotonic() - started)}")
        _impl.verify_checkpoint(checkpoint, steps)
        prompts = tuple(
            item.strip()
            for item in (
                args.monitor_prompts
                or "Explain a transformer.||Write a Python function to reverse a string.||Explique les réseaux de neurones."
            ).split("||")
            if item.strip()
        )
        samples = _impl.generate_preview(checkpoint, device, prompts)
        _impl.write_history(run_number, total_runs, checkpoint, metrics, samples, started, config)
        _stage_latest_checkpoint()
        return metrics
    except Exception as exc:
        _impl.write_failure_history(run_number, total_runs, exc, started)
        _impl.phase("RUN", f"FAILED | run={run_number}/{total_runs} | error={exc}")
        raise


def main() -> int:
    """Run Colab training with one non-interactive run by default."""
    if len(sys.argv) == 1:
        sys.argv.append("--runs")
        sys.argv.append("1")
    elif "--runs" not in sys.argv:
        sys.argv.extend(["--runs", "1"])
    _impl.train_one_run = train_one_run
    return _impl.main()


if __name__ == "__main__":
    raise SystemExit(main())
