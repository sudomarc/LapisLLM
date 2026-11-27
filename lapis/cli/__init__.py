"""LapisLLM command line interface.

Usage::

    python -m lapis doctor
    lapis doctor
    lapis doctor --benchmark --iterations 20
"""

from __future__ import annotations

from .doctor import build_parser, doctor_command

__all__ = ["build_parser", "doctor_command", "main"]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``lapis`` console script and ``python -m lapis``."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        code: object = exc.code
        if isinstance(code, int):
            return code
        return 1
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return_code: int = args.func(args)
    return return_code
