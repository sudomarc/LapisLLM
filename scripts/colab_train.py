#!/usr/bin/env python3
"""Canonical non-interactive Colab entry point for LapisLLM training.

This module preserves the stable public helper surface while delegating the
workflow implementation to ``scripts._colab_train_impl``.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path

from scripts import _colab_train_impl as _impl

CONFIG = _impl.CONFIG
SMOKE_CONFIG = _impl.SMOKE_CONFIG
CORPUS = _impl.CORPUS
MANIFEST = _impl.MANIFEST
HISTORY = _impl.HISTORY
CHECKPOINTS = _impl.CHECKPOINTS
METRIC_RE = _impl.METRIC_RE
TOKEN_RE = _impl.TOKEN_RE
format_duration = _impl.format_duration
corpus_valid = _impl.corpus_valid
prepare_corpus = _impl.prepare_corpus
parse_args = _impl.parse_args
banner = _impl.banner
phase = _impl.phase
ensure_dependencies = _impl.ensure_dependencies
check_gpu = _impl.check_gpu
train_one_run = _impl.train_one_run


def get_github_token() -> str | None:
    return _impl.get_github_token()


def completed_run_numbers() -> set[int]:
    """Return completed runs, preserving status-less legacy summaries."""
    numbers: set[int] = set()
    if not HISTORY.is_dir():
        return numbers
    for summary in HISTORY.glob("run-*/summary.json"):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            number = int(data["run_number"])
            if data.get("status", "completed") == "completed":
                numbers.add(number)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
            continue
    return numbers


def git_push(token: str | None, smoke: bool) -> bool:
    """Publish only training history and canonical inference checkpoint outputs."""
    if smoke:
        phase("GITHUB", "SKIPPED | smoke test")
        return True
    if not token:
        phase("GITHUB", "AUTHENTICATION MISSING | failing before modifying Git state")
        raise RuntimeError("No GitHub credentials available for non-interactive Colab push.")

    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=_impl.ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    status = result.stdout.splitlines()
    allowed_prefixes = (
        "training_history/",
        "checkpoints/latest.pt",
        "checkpoints/tokenizer/",
    )
    protected = []
    pushable = []
    for line in status:
        path = line[3:].lstrip().replace("\\", "/") if len(line) >= 4 else ""
        if not path:
            continue
        if path.startswith(allowed_prefixes):
            pushable.append(path)
        else:
            protected.append(line)
    if protected:
        raise RuntimeError(
            "Refusing GitHub push because unrelated local changes exist:\n"
            + "\n".join(protected)
        )

    if pushable:
        phase("GITHUB", f"COMMIT | files={len(pushable)}")
        subprocess.run(
            [
                "git",
                "add",
                "training_history",
                "checkpoints/latest.pt",
                "checkpoints/tokenizer",
            ],
            cwd=_impl.ROOT,
            check=True,
            timeout=30,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=_impl.ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        if staged:
            subprocess.run(
                ["git", "commit", "-m", "chore: record Colab training outputs"],
                cwd=_impl.ROOT,
                check=True,
                timeout=60,
            )

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "PYTHONUNBUFFERED": "1"}
    with tempfile.TemporaryDirectory(prefix="lapis-git-auth-") as tmp:
        askpass = Path(tmp) / "askpass.sh"
        askpass.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *[Uu]sername*) printf '%s\\n' 'x-access-token' ;;\n"
            "  *) printf '%s\\n' \"$LAPIS_GIT_TOKEN\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        env.update({"GIT_ASKPASS": str(askpass), "LAPIS_GIT_TOKEN": token})
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=_impl.ROOT,
            env=env,
            check=True,
            timeout=120,
        )

    phase("GITHUB", "PUSH COMPLETE")
    return True


# Keep the implementation module's main() aligned with this stable entry point.
_impl.completed_run_numbers = completed_run_numbers
_impl.git_push = git_push
main = _impl.main

__all__ = [
    "CHECKPOINTS",
    "CONFIG",
    "CORPUS",
    "HISTORY",
    "MANIFEST",
    "METRIC_RE",
    "SMOKE_CONFIG",
    "TOKEN_RE",
    "banner",
    "check_gpu",
    "completed_run_numbers",
    "corpus_valid",
    "ensure_dependencies",
    "format_duration",
    "get_github_token",
    "git_push",
    "main",
    "parse_args",
    "phase",
    "prepare_corpus",
    "train_one_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
