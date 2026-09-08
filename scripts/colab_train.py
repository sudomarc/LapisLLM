#!/usr/bin/env python3
"""One-command Colab bootstrap, corpus build, and GPU training runner."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "colab.yaml"
CORPUS = ROOT / "training_data" / "colab_pretrain.txt"
MANIFEST = ROOT / "training_data" / "colab_pretrain_manifest.json"
CHECKPOINT = ROOT / "checkpoints" / "colab-pretrain.pt"


def banner(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}", flush=True)


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def ensure_dependencies() -> None:
    required = ("torch", "yaml", "datasets", "tokenizers", "typer")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        banner("INSTALLATION")
        run([sys.executable, "-m", "pip", "install", "-e", ".[data]"])


def check_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA unavailable. Select a GPU runtime in Google Colab first."
        )
    print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LapisLLM Colab GPU job")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-chars", type=int, default=200_000_000)
    parser.add_argument("--max-records-per-source", type=int, default=0)
    parser.add_argument("--monitor-interval", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def print_corpus_status(max_chars: int) -> None:
    size = CORPUS.stat().st_size if CORPUS.exists() else 0
    target = max_chars
    pct = min(100.0, (size / target) * 100.0) if target else 0.0
    print(
        f"[CORPUS] READY | {size / 1024 / 1024:.1f} MiB | "
        f"target={target / 1024 / 1024:.1f} MiB | {pct:.1f}%",
        flush=True,
    )
    print(f"[MANIFEST] {MANIFEST}", flush=True)


def run_training(args: argparse.Namespace) -> None:
    train = [
        sys.executable,
        "-m",
        "lapis.dev.cli",
        "train",
        "--config",
        str(CONFIG.relative_to(ROOT)),
        "--device",
        "cuda",
        "--data",
        str(CORPUS.relative_to(ROOT)),
        "--checkpoint",
        str(CHECKPOINT.relative_to(ROOT)),
        "--epochs",
        "1",
        "--monitor-interval",
        str(args.monitor_interval),
    ]
    if args.resume:
        train += ["--resume", str(CHECKPOINT.relative_to(ROOT))]

    banner("TRAINING")
    print("[1/4] Reading corpus + preparing tokenizer...", flush=True)
    print("[2/4] Building tokenized dataset + model...", flush=True)
    print("[3/4] Waiting for first optimizer step...", flush=True)
    print("[4/4] Live step/loss output follows below.", flush=True)
    started = time.monotonic()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        train,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                print(line, flush=True)
    finally:
        process.stdout.close()

    code = process.wait()
    elapsed = time.monotonic() - started
    if code != 0:
        raise SystemExit(f"Training failed with exit code {code} after {elapsed:.1f}s")

    print(f"[TRAINING] DONE | elapsed={elapsed / 60:.1f} min", flush=True)


def main() -> int:
    args = parse_args()
    ensure_dependencies()
    check_gpu()

    ROOT.joinpath("training_data").mkdir(parents=True, exist_ok=True)
    ROOT.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)

    banner("LAPISLLM COLAB")
    max_chars = 5_000_000 if args.smoke_test else args.max_chars
    print(f"[CONFIG] {CONFIG}", flush=True)
    print(f"[TARGET] {max_chars / 1024 / 1024:.1f} MiB corpus", flush=True)

    build = [
        sys.executable,
        "scripts/build_colab_corpus.py",
        "--output",
        str(CORPUS.relative_to(ROOT)),
        "--manifest",
        str(MANIFEST.relative_to(ROOT)),
        "--max-chars",
        str(max_chars),
    ]
    if args.max_records_per_source > 0:
        build += [
            "--max-records-per-source",
            str(args.max_records_per_source),
        ]

    banner("CORPUS")
    run(build)
    print_corpus_status(max_chars)
    print("[NEXT] Corpus complete. Entering training phase...", flush=True)

    run_training(args)

    banner("OUTPUTS")
    print(f"[CHECKPOINT] {CHECKPOINT}", flush=True)
    print(f"[MANIFEST]   {MANIFEST}", flush=True)
    if CHECKPOINT.exists():
        print(
            f"[CHECKPOINT] size={CHECKPOINT.stat().st_size / 1024 / 1024:.1f} MiB",
            flush=True,
        )
    print("[STATUS] Training pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
