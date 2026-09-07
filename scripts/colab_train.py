#!/usr/bin/env python3
"""Legacy Colab entry point delegating to the canonical Lapis console.

Use ``lapis train`` as the supported training entry point. This wrapper is
kept so existing notebooks do not silently use the old artifact-pushing flow.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("[DEPRECATED] scripts/colab_train.py now delegates to the canonical Lapis console.")
    print("Use: python -m scripts.lapis train")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.lapis", "train"],
        cwd=ROOT,
        check=False,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
