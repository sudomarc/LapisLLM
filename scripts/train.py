#!/usr/bin/env python3
"""Stable training entry point with absolute artifact-path enforcement."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts import _train_impl as _impl
from scripts._train_impl import *  # noqa: F401,F403
from scripts.streaming_train import StreamingTextDataset

_PATH_FLAGS = {"--config", "--data", "--checkpoint", "--monitor-log", "--resume", "--tokenizer"}


def _normalize_path_args() -> None:
    argv = sys.argv
    for index, value in enumerate(argv[:-1]):
        if value in _PATH_FLAGS:
            argv[index + 1] = str(Path(argv[index + 1]).resolve())


def _absolute_checkpoint() -> Path:
    argv = sys.argv
    try:
        index = argv.index("--checkpoint")
    except ValueError:
        return Path("checkpoints/latest.pt").resolve()
    if index + 1 >= len(argv):
        raise SystemExit("--checkpoint requires a value")
    return Path(argv[index + 1]).resolve()


def _has_data_file() -> bool:
    return "--data" in sys.argv


def main() -> None:
    _normalize_path_args()
    if _has_data_file():
        from scripts import streaming_train

        streaming_train.main()
    else:
        _impl.main()

    checkpoint = _absolute_checkpoint()
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
