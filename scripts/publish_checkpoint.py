#!/usr/bin/env python3
"""Publish a verified Lapis checkpoint to the repository for CHAD."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "checkpoints"
LATEST_CHECKPOINT = CHECKPOINT_ROOT / "latest.pt"
LATEST_TOKENIZER = CHECKPOINT_ROOT / "tokenizer"


def run(*args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=capture,
        check=True,
    )


def status() -> list[str]:
    result = run("git", "status", "--porcelain", "--untracked-files=all", capture=True)
    return [line for line in result.stdout.splitlines() if line.strip()]


def verify_checkpoint(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Checkpoint missing or empty: {path}")
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in state or "config" not in state:
        raise RuntimeError("Checkpoint is missing model/config metadata")
    return state


def find_source(path: Path | None) -> Path:
    if path:
        candidate = path if path.is_absolute() else ROOT / path
        if candidate.is_file():
            return candidate
        raise RuntimeError(f"Checkpoint not found: {candidate}")

    candidates = sorted(
        (CHECKPOINT_ROOT / "colab-runs").glob("run-*/checkpoint.pt"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ) if (CHECKPOINT_ROOT / "colab-runs").is_dir() else []
    if not candidates:
        raise RuntimeError("No checkpoint was found under checkpoints/colab-runs")
    return candidates[0]


def tokenizer_source(checkpoint: Path) -> Path:
    tokenizer = checkpoint.parent / "tokenizer"
    if not tokenizer.is_dir() or not (tokenizer / "tokenizer.json").is_file():
        raise RuntimeError(f"Checkpoint tokenizer not found: {tokenizer}")
    return tokenizer


def publish(source: Path) -> None:
    verify_checkpoint(source)
    tokenizer = tokenizer_source(source)

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, LATEST_CHECKPOINT)
    if LATEST_TOKENIZER.exists():
        shutil.rmtree(LATEST_TOKENIZER)
    shutil.copytree(tokenizer, LATEST_TOKENIZER)


def push(commit_message: str) -> bool:
    current = status()
    protected = [
        line
        for line in current
        if not line[3:].replace("\\", "/").startswith("checkpoints/")
    ]
    if protected:
        raise RuntimeError(
            "Refusing checkpoint publication because unrelated local changes exist:\n"
            + "\n".join(protected)
        )

    run("git", "add", "checkpoints")
    staged = run("git", "diff", "--cached", "--name-only", capture=True).stdout.strip()
    if not staged:
        print("Checkpoint already synchronized.")
        return False

    run("git", "commit", "-m", commit_message)
    run("git", "push", "origin", "main")
    print("Checkpoint published to origin/main.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a verified Lapis checkpoint")
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    source = find_source(args.checkpoint)
    publish(source)
    print(f"Published checkpoint: {LATEST_CHECKPOINT}")
    if not args.no_push:
        push(f"chore: publish Lapis checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
