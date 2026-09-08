#!/usr/bin/env python3
"""Backward-compatible entry point for the observable Colab orchestrator.

The canonical workflow now lives in ``scripts.colab_train`` so both notebook
entry points share the same run management, progress reporting, resume logic,
checkpoint verification, history, and GitHub behavior.
"""

from __future__ import annotations

import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from scripts.colab_train import get_github_token, main


@contextmanager
def github_auth_env(token: str):
    """Provide ephemeral Git askpass credentials without mutating os.environ."""
    if not token:
        yield {}
        return

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
        yield {
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "LAPIS_GIT_TOKEN": token,
        }


__all__ = ["get_github_token", "github_auth_env", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
