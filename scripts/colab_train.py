#!/usr/bin/env python3
"""One-command Colab pipeline for collecting data, training LAPIS, and syncing optional metadata."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)


def run(command: list[str], *, env: dict[str, str] | None = None, display_command: list[str] | None = None) -> None:
    shown = display_command or command
    print("\n$", " ".join(shown), flush=True)
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

    push_manifest = os.environ.get("LAPIS_PUSH_MANIFEST", "0").lower() in {"1", "true", "yes"}
    if not push_manifest:
        print("\nLAPIS_PUSH_MANIFEST is not enabled; generated artifacts remain local.", flush=True)
        return

    askpass = Path(tempfile.mktemp(prefix="lapis-askpass-", suffix=".sh"))
    askpass.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  *Username*) printf '%s\\n' 'x-access-token' ;;\n"
        "  *Password*) printf '%s\\n' \"$LAPIS_GITHUB_TOKEN\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    askpass.chmod(stat.S_IRWXU)
    env = os.environ.copy()
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["LAPIS_GITHUB_TOKEN"] = token

    try:
        run(["git", "config", "user.email", "actions@github.com"])
        run(["git", "config", "user.name", "Lapis Colab Trainer"])
        run(["git", "add", "training_data/open/manifest.json"])
        status = subprocess.run(["git", "diff", "--cached", "--quiet"])
        if status.returncode == 0:
            print("\nNo metadata changes to commit.", flush=True)
            return
        run(["git", "commit", "-m", "data: update training manifest"])
        run(
            ["git", "-c", "credential.helper=", "push", "https://github.com/sudomarc/LapisLLM.git", "HEAD:main"],
            env=env,
            display_command=["git", "-c", "credential.helper=", "push", "https://github.com/sudomarc/LapisLLM.git", "HEAD:main"],
        )
    finally:
        env.pop("LAPIS_GITHUB_TOKEN", None)
        try:
            askpass.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
