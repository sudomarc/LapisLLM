#!/usr/bin/env python3
"""Backward-compatible data-preparation entry point.

The canonical collector lives in :mod:`scripts.fetch_training_data`. This
wrapper keeps the historical ``prepare_data`` command and documentation valid
without maintaining a second data pipeline.
"""

from __future__ import annotations

from scripts.fetch_training_data import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
