#!/usr/bin/env python3
"""Publish a verified inference checkpoint for external runtime consumers."""

from __future__ import annotations

import argparse
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
        timeout=120,
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
    if not isinstance(state["model_state_dict"], dict) or not state["model_state_dict"]:
        raise RuntimeError("Checkpoint model_state_dict is empty or invalid")
    return state


def find_source(path: Path | None) -> Path:
    if path:
        candidate = path if path.is_absolute() else ROOT / path
        if candidate.is_file():
            return candidate
        raise RuntimeError(f"Checkpoint not found: {candidate}")

    run_root = CHECKPOINT_ROOT / "colab-runs"
    candidates = (
        sorted(
            run_root.glob("run-*/checkpoint.pt"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if run_root.is_dir()
        else []
    )
    if not candidates:
        raise RuntimeError("No checkpoint was found under checkpoints/colab-runs")
    return candidates[0]


def tokenizer_source(checkpoint: Path) -> Path:
    tokenizer = checkpoint.parent / "tokenizer"
    if not tokenizer.is_dir() or not (tokenizer / "tokenizer.json").is_file():
        raise RuntimeError(f"Checkpoint tokenizer not found: {tokenizer}")
    return tokenizer


def publish(source: Path) -> None:
    """Create the smaller inference artifact consumed by LapisRuntime/CHAD."""
    state = verify_checkpoint(source)
    tokenizer = tokenizer_source(source)

    inference_state = {
        "model_state_dict": state["model_state_dict"],
        "config": state["config"],
        "tokenizer_version": state.get("tokenizer_version"),
        "checkpoint_format": "lapis-inference-v1",
    }

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = LATEST_CHECKPOINT.with_suffix(".tmp")
    import torch

    try:
        torch.save(inference_state, temporary)
        if LATEST_CHECKPOINT.exists():
            LATEST_CHECKPOINT.unlink()
        temporary.replace(LATEST_CHECKPOINT)
    finally:
        temporary.unlink(missing_ok=True)

    if LATEST_TOKENIZER.exists():
        shutil.rmtree(LATEST_TOKENIZER)
    shutil.copytree(tokenizer, LATEST_TOKENIZER)

    published = verify_checkpoint(LATEST_CHECKPOINT)
    if published.get("checkpoint_format") != "lapis-inference-v1":
        raise RuntimeError("Published checkpoint format marker is missing")
    print(
        f"Inference checkpoint published locally: {LATEST_CHECKPOINT} "
        f"({LATEST_CHECKPOINT.stat().st_size / 1024 / 1024:.1f} MiB)"
    )


def push(commit_message: str) -> bool:
    current = status()
    protected = [
        line
        for line in current
        if not line[3:].lstrip().replace("\\", "/").startswith("checkpoints/")
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
    print("Inference checkpoint published to origin/main.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a verified Lapis inference checkpoint")
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    source = find_source(args.checkpoint)
    publish(source)
    print(f"Published checkpoint: {LATEST_CHECKPOINT}")
    if not args.no_push:
        push("chore: publish Lapis inference checkpoint")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
