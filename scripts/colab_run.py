#!/usr/bin/env python3
"""Run the complete Lapis Colab training pipeline.

This is the single entry point for a Colab run:
1. collect the bounded open corpus
2. rebuild the tokenizer
3. train Lapis on CUDA when available
4. run automated post-training verification and sample generations
5. optionally commit the checkpoint/manifest and push to GitHub

Usage from the repository root:
    python scripts/colab_run.py

For a fresh run, the default command trains from scratch.
Use --resume to continue from an existing checkpoint.

GitHub authentication is requested interactively at push time and is never
written into the repository URL or notebook source.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "latest.pt"
CORPUS = ROOT / "training_data" / "open" / "combined.txt"
MANIFEST = ROOT / "training_data" / "open" / "manifest.json"

SOURCES = [
    "fineweb",
    "fineweb_edu",
    "c4",
    "wikipedia",
    "s2orc_arxiv",
    "cosmopedia",
    "gutenberg",
    "openr1_math",
    "oasst1",
    "math",
    "code",
]

PROMPTS = [
    "The future of artificial intelligence is",
    "A language model learns by",
    "Explain why the sky is blue:",
]


def run(command: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    """Run a command while keeping its output live in Colab."""
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command Lapis Colab pipeline")
    parser.add_argument("--max-chars", type=int, default=50_000_000)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    parser.add_argument("--skip-data", action="store_true", help="Reuse the existing corpus")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest verification")
    parser.add_argument("--skip-push", action="store_true", help="Do not push the checkpoint to GitHub")
    parser.add_argument("--sources", nargs="+", default=SOURCES, choices=SOURCES)
    return parser.parse_args()


def print_system_info() -> None:
    print("=" * 72)
    print("LAPIS COLAB RUN")
    print("=" * 72)
    print(f"Repository : {ROOT}")
    print(f"Python     : {sys.version.split()[0]}")
    try:
        import torch

        print(f"PyTorch    : {torch.__version__}")
        print(f"CUDA       : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU        : {torch.cuda.get_device_name(0)}")
            print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception as exc:
        print(f"Torch info unavailable: {exc}")
    print("=" * 72, flush=True)


def collect_data(args: argparse.Namespace) -> None:
    if args.skip_data and CORPUS.exists():
        print(f"\n[DATA] Reusing existing corpus: {CORPUS}")
        return

    command = [
        sys.executable,
        "scripts/fetch_open_corpus.py",
        "--sources",
        *args.sources,
        "--max-chars",
        str(args.max_chars),
        "--max-examples",
        str(args.max_examples),
    ]
    run(command)


def train(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.train",
        "--config",
        args.config,
        "--data",
        str(CORPUS),
        "--device",
        args.device,
        "--checkpoint",
        args.checkpoint,
    ]
    if args.resume:
        if not CHECKPOINT.exists():
            raise SystemExit("--resume was requested but checkpoints/latest.pt does not exist.")
        command.extend(["--resume", args.checkpoint])
    else:
        tokenizer = ROOT / "artifacts" / "tokenizer"
        if tokenizer.exists():
            print(f"\n[TRAIN] Removing old tokenizer: {tokenizer}")
            shutil.rmtree(tokenizer)

    run(command)


def verify(args: argparse.Namespace) -> None:
    print("\n" + "=" * 72)
    print("POST-TRAINING VERIFICATION")
    print("=" * 72)

    if not CHECKPOINT.exists():
        raise SystemExit(f"Checkpoint missing: {CHECKPOINT}")
    if not (CHECKPOINT.parent / "tokenizer" / "tokenizer.json").exists():
        raise SystemExit("Checkpoint tokenizer missing.")
    if not CORPUS.exists():
        raise SystemExit("Training corpus missing.")

    print(f"Checkpoint : {CHECKPOINT} ({CHECKPOINT.stat().st_size / 1024**2:.1f} MB)")
    print(f"Corpus     : {CORPUS} ({CORPUS.stat().st_size / 1024**2:.1f} MB)")
    if MANIFEST.exists():
        print(f"Manifest   : {MANIFEST}")

    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"])

    for prompt in PROMPTS:
        print("\n" + "-" * 72)
        print(f"PROMPT: {prompt}")
        run(
            [
                sys.executable,
                "scripts/generate.py",
                "--checkpoint",
                args.checkpoint,
                "--prompt",
                prompt,
                "--max-new-tokens",
                "48",
                "--temperature",
                "0.8",
                "--top-k",
                "40",
                "--top-p",
                "0.95",
                "--device",
                "auto",
            ]
        )


def configure_git() -> None:
    run(["git", "config", "user.name", "sudomarc"])
    run(["git", "config", "user.email", "sudomarc@users.noreply.github.com"])
    run(["git", "add", "-f", str(CHECKPOINT.relative_to(ROOT))])
    tokenizer = CHECKPOINT.parent / "tokenizer"
    if tokenizer.exists():
        run(["git", "add", "-f", str(tokenizer.relative_to(ROOT))])
    if MANIFEST.exists():
        run(["git", "add", str(MANIFEST.relative_to(ROOT))])

    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT)
    if staged.returncode == 0:
        print("\n[GIT] No new checkpoint changes to commit.")
        return
    run(["git", "commit", "-m", "train: update Lapis checkpoint"])


def push_to_github() -> None:
    token = getpass("GitHub PAT (input hidden): ").strip()
    if not token:
        raise SystemExit("No GitHub token supplied; push cancelled.")

    askpass = Path(tempfile.mktemp(prefix="lapis-git-askpass-", suffix=".sh"))
    askpass.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' 'sudomarc' ;;\n"
        "  *Password*) printf '%s\\n' \"$LAPIS_GITHUB_TOKEN\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(stat.S_IRWXU)

    env = os.environ.copy()
    env["LAPIS_GITHUB_TOKEN"] = token
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_TERMINAL_PROMPT"] = "0"

    try:
        run(["git", "remote", "set-url", "origin", "https://github.com/sudomarc/LapisLLM.git"])
        run(["git", "push", "origin", "main"], env=env)
    finally:
        token = ""
        env.pop("LAPIS_GITHUB_TOKEN", None)
        try:
            askpass.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)

    print_system_info()
    print(f"\n[1/5] DATA — {len(args.sources)} sources / {args.max_chars:,} chars")
    collect_data(args)

    print("\n[2/5] TRAINING")
    train(args)

    print("\n[3/5] VERIFICATION")
    verify(args)

    if not args.skip_push:
        print("\n[4/5] GITHUB")
        configure_git()
        push_to_github()
    else:
        print("\n[4/5] GITHUB — skipped")

    print("\n[5/5] COMPLETE")
    print("Lapis training pipeline finished successfully.")
    print(f"Checkpoint: {CHECKPOINT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
