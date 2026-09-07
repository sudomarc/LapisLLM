#!/usr/bin/env python3
"""Legacy Colab entry point delegating to the canonical Lapis console.

The previous implementation trained and then force-added large checkpoints to
Git, which conflicted with the repository's artifact policy and duplicated the
main training orchestration. Keep this wrapper for notebook compatibility.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("[DEPRECATED] scripts/colab_run.py now delegates to the canonical Lapis console.")
    print("Use: python -m scripts.lapis train")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.lapis", "train"],
        cwd=ROOT,
        check=False,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
