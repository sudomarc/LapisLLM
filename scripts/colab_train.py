#!/usr/bin/env python3
"""Compatibility entry point for the automated Colab training pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("[COLAB] Delegating to the automated training pipeline.")
    result = subprocess.run(
        [sys.executable, "scripts/colab_run.py", *sys.argv[1:]],
        cwd=ROOT,
        check=False,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
