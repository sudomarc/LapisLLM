"""Allow ``python -m lapis`` as the CLI entry point."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
