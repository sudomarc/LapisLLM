#!/usr/bin/env python3
"""Backward-compatible entry point for the observable Colab orchestrator.

The canonical workflow now lives in ``scripts.colab_train`` so both notebook
entry points share the same run management, progress reporting, resume logic,
checkpoint verification, history, and GitHub behavior.
"""

from __future__ import annotations

from scripts.colab_train import main


if __name__ == "__main__":
    raise SystemExit(main())
