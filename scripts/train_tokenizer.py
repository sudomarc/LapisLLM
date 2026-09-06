#!/usr/bin/env python3
"""Build and save the current LAPIS tokenizer from a text corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from lapis.tokenizer.tokenizer import Tokenizer

DEFAULT_CORPUS = "LAPIS is an independent language model project.\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/export the LAPIS tokenizer")
    parser.add_argument("--data", help="UTF-8 text file used as tokenizer corpus")
    parser.add_argument("--output", default="artifacts/tokenizer")
    args = parser.parse_args()

    corpus = Path(args.data).read_text(encoding="utf-8") if args.data else DEFAULT_CORPUS
    tokenizer = Tokenizer()
    tokenizer.save(args.output)
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size}")
    print(f"Corpus tokens: {len(tokenizer.encode(corpus))}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
