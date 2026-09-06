#!/usr/bin/env python3
"""Run the complete Lapis Colab training pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "fineweb", "fineweb_edu", "c4", "wikipedia", "s2orc_arxiv",
    "cosmopedia", "gutenberg", "openr1_math", "oasst1", "math", "code",
]
PROMPTS = [
    "The future of artificial intelligence is",
    "A language model learns by",
    "Explain why the sky is blue:",
]
CHAT_PROMPTS = [
    "Hello Lapis, introduce yourself in one short sentence.",
    "What is 2 + 2?",
]


def run(command: list[str], *, input_text: str | None = None) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, input=input_text, text=True, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command Lapis Colab pipeline")
    parser.add_argument("--max-chars", type=int, default=50_000_000)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="outputs/lapis-checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--push-manifest", action="store_true", help="Explicitly commit and push the training manifest")
    parser.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    args = parser.parse_args()
    if args.resume and args.fresh:
        parser.error("--resume and --fresh cannot be used together")
    return args


def print_system_info() -> None:
    import torch

    print("=" * 72)
    print("LAPIS COLAB RUN")
    print("=" * 72)
    print(f"Repository : {ROOT}")
    print(f"Python     : {sys.version.split()[0]}")
    print(f"PyTorch    : {torch.__version__}")
    print(f"CUDA       : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 72, flush=True)


def collect_data(args: argparse.Namespace, corpus: Path) -> None:
    if args.skip_data and corpus.exists():
        print(f"\n[DATA] Reusing existing corpus: {corpus}")
        return
    run([
        sys.executable, "scripts/fetch_open_corpus.py", "--sources", *args.sources,
        "--max-chars", str(args.max_chars), "--max-examples", str(args.max_examples),
    ])


def train(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    should_resume = args.resume and checkpoint.exists()
    if args.resume and not checkpoint.exists():
        raise SystemExit(f"Resume requested but checkpoint does not exist: {checkpoint}")
    if args.fresh and checkpoint.exists():
        checkpoint.unlink()

    command = [
        sys.executable, "-u", "-m", "scripts.train", "--config", args.config,
        "--data", "training_data/open/combined.txt", "--device", args.device,
        "--checkpoint", str(checkpoint),
    ]
    if should_resume:
        command.extend(["--resume", str(checkpoint)])
    run(command)


def verify_generation(args: argparse.Namespace) -> None:
    for prompt in PROMPTS:
        run([
            sys.executable, "scripts/generate.py", "--checkpoint", args.checkpoint,
            "--prompt", prompt, "--max-new-tokens", "16", "--temperature", "0.8",
            "--top-k", "40", "--top-p", "0.95", "--device", "auto",
        ])


def verify_chat(args: argparse.Namespace) -> None:
    if args.skip_chat:
        return
    scripted_input = "\n".join(CHAT_PROMPTS + ["quit", ""])
    run([
        sys.executable, "scripts/chat.py", "--checkpoint", args.checkpoint,
        "--device", "auto", "--max-new-tokens", "8", "--temperature", "0.8",
        "--top-k", "40", "--top-p", "0.95",
    ], input_text=scripted_input)


def verify(args: argparse.Namespace, corpus: Path) -> None:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint missing: {checkpoint}")
    if not corpus.exists():
        raise SystemExit(f"Training corpus missing: {corpus}")
    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"])
    verify_generation(args)
    verify_chat(args)


def push_manifest() -> None:
    run(["git", "add", "training_data/open/manifest.json"])
    status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if status.returncode == 0:
        print("[GIT] No manifest changes to commit.")
        return
    run(["git", "commit", "-m", "data: update training manifest"])
    run(["git", "push", "origin", "main"])


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    corpus = ROOT / "training_data" / "open" / "combined.txt"
    print_system_info()
    collect_data(args, corpus)
    train(args)
    verify(args, corpus)
    if args.push_manifest:
        push_manifest()
    print("\nLAPIS Colab pipeline completed successfully.")
    print(f"Checkpoint: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
