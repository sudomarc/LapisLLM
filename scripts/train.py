#!/usr/bin/env python3
"""Stable training entry point with absolute artifact-path enforcement."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts import _train_impl as _impl
from scripts._train_impl import *  # noqa: F401,F403


def _absolute_arg(flag: str, default: str) -> Path:
    argv = sys.argv
    try:
        index = argv.index(flag)
    except ValueError:
        return Path(default).resolve()
    if index + 1 >= len(argv):
        raise SystemExit(f"{flag} requires a value")
    return Path(argv[index + 1]).resolve()


def main() -> None:
    _impl.main()
    checkpoint = _absolute_arg("--checkpoint", "checkpoints/latest.pt")
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise RuntimeError(
            "Training reported success but the checkpoint artifact is missing or empty: "
            f"{checkpoint}"
        )
    tokenizer = checkpoint.parent / "tokenizer" / "tokenizer.json"
    if not tokenizer.is_file() or tokenizer.stat().st_size <= 0:
        raise RuntimeError(
            "Training reported success but the tokenizer artifact is missing or empty: "
            f"{tokenizer}"
        )
    print(f"Artifact verification: checkpoint={checkpoint} | size={checkpoint.stat().st_size} bytes")
    print(f"Artifact verification: tokenizer={tokenizer} | size={tokenizer.stat().st_size} bytes")


if __name__ == "__main__":
    main()
