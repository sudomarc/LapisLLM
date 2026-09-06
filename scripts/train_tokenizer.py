#!/usr/bin/env python3
"""Train and save the LAPIS BPE tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from lapis.tokenizer.tokenizer import Tokenizer

DEFAULT_CORPUS = """
LAPIS is an independent language model research project.
We build the tokenizer, transformer, training loop, evaluation pipeline and inference stack ourselves.
A language model learns to predict the next token from context.
This small development corpus exists only to exercise the tokenizer pipeline.
""".strip()


def _read_text_files(directory: str) -> list[str]:
    root = Path(directory)
    return [str(path) for path in sorted(root.rglob("*.txt")) if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LAPIS BPE tokenizer")
    parser.add_argument("--data", help="UTF-8 text file used as tokenizer corpus")
    parser.add_argument("--data-dir", help="Directory containing UTF-8 .txt corpus files")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--output", default="artifacts/tokenizer")
    args = parser.parse_args()

    if args.data and args.data_dir:
        parser.error("Use either --data or --data-dir, not both")

    if args.data:
        corpus = Path(args.data).read_text(encoding="utf-8")
        if not corpus.strip():
            raise ValueError("Tokenizer corpus is empty")
        tokenizer = Tokenizer.train_from_iterator(
            [corpus], vocab_size=args.vocab_size, min_frequency=args.min_frequency
        )
        token_count = len(tokenizer.encode(corpus))
    elif args.data_dir:
        files = _read_text_files(args.data_dir)
        if not files:
            raise ValueError(f"No .txt files found in {args.data_dir}")
        tokenizer = Tokenizer.train_from_files(
            files, vocab_size=args.vocab_size, min_frequency=args.min_frequency
        )
        token_count = sum(
            len(tokenizer.encode(Path(path).read_text(encoding="utf-8")))
            for path in files
        )
    else:
        tokenizer = Tokenizer.train_from_iterator(
            [DEFAULT_CORPUS],
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        token_count = len(tokenizer.encode(DEFAULT_CORPUS))

    tokenizer.save(args.output)
    print(f"Tokenizer: {tokenizer}")
    print(f"Vocabulary: {tokenizer.vocab_size}")
    print(f"Corpus tokens: {token_count}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
