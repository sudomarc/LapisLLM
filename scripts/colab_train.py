#!/usr/bin/env python3
"""Observable Colab entry point with absolute child artifact paths."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Make the repository root importable when this file is executed directly
# (e.g. ``python scripts/colab_train.py`` in Google Colab).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import _colab_train_impl as _impl

_ORIGINAL_TRAIN_ONE_RUN = _impl.train_one_run
_ORIGINAL_POPEN = subprocess.Popen
_PATH_FLAGS = {"--config", "--data", "--checkpoint", "--monitor-log", "--resume", "--tokenizer"}
HISTORY = _impl.HISTORY
# Preserve the public API used by the legacy ``colab_run`` entry point and
# observability tests without reintroducing the wildcard import that previously
# exposed unused/repeated names.
METRIC_RE = _impl.METRIC_RE
TOKEN_RE = _impl.TOKEN_RE
format_duration = _impl.format_duration
get_github_token = _impl.get_github_token
main = _impl.main


def _absolute_child_paths(command: list[str]) -> list[str]:
    normalized = list(command)
    for index, value in enumerate(normalized[:-1]):
        if value in _PATH_FLAGS:
            normalized[index + 1] = str(Path(normalized[index + 1]).resolve())
    return normalized


def completed_run_numbers() -> set[int]:
    """Return completed runs while preserving legacy summaries without status."""
    # Keep the wrapper's historical monkeypatch surface while delegating the
    # actual classification policy to the canonical implementation.
    original_history = _impl.HISTORY
    _impl.HISTORY = HISTORY
    try:
        return _impl.completed_run_numbers()
    finally:
        _impl.HISTORY = original_history


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


if __name__ == "__main__":
    raise SystemExit(main())
