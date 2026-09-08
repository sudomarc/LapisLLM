#!/usr/bin/env python3
"""One-command Colab bootstrap, corpus build, and GPU training runner."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "colab.yaml"
CORPUS = ROOT / "training_data" / "colab_pretrain.txt"
MANIFEST = ROOT / "training_data" / "colab_pretrain_manifest.json"
CHECKPOINT = ROOT / "checkpoints" / "colab-pretrain.pt"


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_dependencies() -> None:
    required = ("torch", "yaml", "datasets", "tokenizers", "typer")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        run([sys.executable, "-m", "pip", "install", "-e", ".[data]"])


def check_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA unavailable. Select a GPU runtime in Google Colab first."
        )
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first LapisLLM Colab GPU job")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-chars", type=int, default=200_000_000)
    parser.add_argument("--max-records-per-source", type=int, default=0)
    parser.add_argument("--monitor-interval", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dependencies()
    check_gpu()

    ROOT.joinpath("training_data").mkdir(parents=True, exist_ok=True)
    ROOT.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)

    max_chars = 5_000_000 if args.smoke_test else args.max_chars
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
    run(build)

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
    run(train)

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Manifest:   {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
