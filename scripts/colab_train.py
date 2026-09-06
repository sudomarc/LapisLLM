#!/usr/bin/env python3
"""One-command Colab pipeline for collecting data, training LAPIS, and syncing artifacts.

Usage in Google Colab:
    !python scripts/colab_train.py

GitHub authentication is intentionally read from the environment (GITHUB_TOKEN)
and is never stored in this file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    # Install project dependencies.
    run([sys.executable, "-m", "pip", "install", "-e", "."])
    run([sys.executable, "-m", "pip", "install", "datasets"])

    # Build the open training corpus used by the current LAPIS pipeline.
    run([
        sys.executable,
        "scripts/fetch_open_corpus.py",
        "--sources",
        "fineweb",
        "fineweb_edu",
        "c4",
        "wikipedia",
        "cosmopedia",
        "math",
        "code",
        "--max-chars",
        "50000000",
    ])

    # The tokenizer must match the new model configuration; never reuse an
    # incompatible tokenizer from an older checkpoint.
    tokenizer_dir = ROOT / "artifacts" / "tokenizer"
    if tokenizer_dir.exists():
        import shutil
        shutil.rmtree(tokenizer_dir)

    run([
        sys.executable,
        "-u",
        "-m",
        "scripts.train",
        "--config",
        "configs/tiny.yaml",
        "--data",
        "training_data/open/combined.txt",
        "--device",
        "cuda",
        "--checkpoint",
        "checkpoints/latest.pt",
    ])

    # Optionally sync the trained checkpoint/metadata to GitHub.
    # Authentication must be supplied as GITHUB_TOKEN by the Colab runtime.
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("\nTraining complete. GITHUB_TOKEN not set; skipping GitHub push.", flush=True)
        return

    remote = f"https://x-access-token:{token}@github.com/sudomarc/LapisLLM.git"
    safe_env = os.environ.copy()
    run(["git", "config", "user.email", "actions@github.com"])
    run(["git", "config", "user.name", "Lapis Colab Trainer"])
    run(["git", "remote", "set-url", "origin", remote], env=safe_env)

    # Keep source code/config changes, but explicitly include the ignored
    # checkpoint and tokenizer produced by this training run.
    run(["git", "add", "training_data/manifest.json"])
    run(["git", "add", "-f", "checkpoints/latest.pt", "checkpoints/tokenizer"])

    status = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if status.returncode == 0:
        print("\nNothing new to commit.", flush=True)
        return

    run(["git", "commit", "-m", "train: update Lapis checkpoint"])
    run(["git", "push", "origin", "main"], env=safe_env)
    print("\nLAPIS Colab pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
