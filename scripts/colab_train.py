#!/usr/bin/env python3
"""One-command Colab pipeline for collecting data, training LAPIS, and syncing optional metadata."""

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
    run([sys.executable, "-m", "pip", "install", "-e", "."])
    run([sys.executable, "-m", "pip", "install", "datasets"])

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

    tokenizer_dir = ROOT / "artifacts" / "tokenizer"
    if tokenizer_dir.exists():
        import shutil
        shutil.rmtree(tokenizer_dir)

    checkpoint_path = ROOT / "outputs" / "latest.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
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
        str(checkpoint_path),
    ])

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("\nTraining complete. GITHUB_TOKEN not set; skipping GitHub push.", flush=True)
        return

    safe_env = os.environ.copy()
    safe_env["GIT_ASKPASS"] = "true"
    safe_env.pop("GITHUB_TOKEN", None)

    push_checkpoint = os.environ.get("LAPIS_PUSH_CHECKPOINT", "0").lower() in {"1", "true", "yes"}
    if not push_checkpoint:
        print(
            "\nLAPIS_PUSH_CHECKPOINT is not enabled; generated checkpoints remain local.",
            flush=True,
        )
        return

    remote = f"https://x-access-token:{token}@github.com/sudomarc/LapisLLM.git"
    safe_env["GIT_ASKPASS"] = "echo"
    run(["git", "config", "user.email", "actions@github.com"])
    run(["git", "config", "user.name", "Lapis Colab Trainer"])
    run(["git", "remote", "set-url", "origin", remote], env=safe_env)
    run(["git", "add", "training_data/open/manifest.json"])

    status = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if status.returncode == 0:
        print("\nNo metadata changes to commit.", flush=True)
        return

    run(["git", "commit", "-m", "data: update training manifest"])
    run(["git", "push", "origin", "main"], env=safe_env)
    print("\nLAPIS Colab metadata sync complete.", flush=True)


if __name__ == "__main__":
    main()
