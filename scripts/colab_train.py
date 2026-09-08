#!/usr/bin/env python3
"""Observable Colab entry point with absolute child artifact paths."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import _colab_train_impl as _impl
from scripts._colab_train_impl import *  # noqa: F401,F403

_ORIGINAL_TRAIN_ONE_RUN = _impl.train_one_run
_ORIGINAL_POPEN = subprocess.Popen
_PATH_FLAGS = {"--config", "--data", "--checkpoint", "--monitor-log", "--resume", "--tokenizer"}


def _absolute_child_paths(command: list[str]) -> list[str]:
    normalized = list(command)
    for index, value in enumerate(normalized[:-1]):
        if value in _PATH_FLAGS:
            normalized[index + 1] = str(Path(normalized[index + 1]).resolve())
    return normalized


def train_one_run(run_number: int, total_runs: int, args, device: str, config: Path) -> dict:
    """Run the canonical trainer while forcing absolute child artifact paths."""

    def popen(command, *popen_args, **popen_kwargs):
        return _ORIGINAL_POPEN(
            _absolute_child_paths(list(command)), *popen_args, **popen_kwargs
        )

    _impl.subprocess.Popen = popen
    try:
        return _ORIGINAL_TRAIN_ONE_RUN(run_number, total_runs, args, device, config)
    finally:
        _impl.subprocess.Popen = _ORIGINAL_POPEN


_impl.train_one_run = train_one_run
main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
