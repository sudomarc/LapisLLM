#!/usr/bin/env python3
"""Prepare a deterministic UTF-8 text corpus from a directory of text files."""

from __future__ import annotations

import argparse
from pathlib import Path


def prepare_data(input_dir: Path, output_file: Path) -> int:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_file.open("w", encoding="utf-8", newline="\n") as output:
        for path in files:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                continue
            if written:
                output.write("\n\n")
            output.write(text.rstrip())
            written += 1

    if written == 0:
        output_file.unlink(missing_ok=True)
        raise ValueError(f"All .txt files in {input_dir} are empty")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deterministic Lapis text corpus")
    parser.add_argument("--input-dir", required=True, help="Directory containing UTF-8 .txt files")
    parser.add_argument("--output", required=True, help="Output UTF-8 text file")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_file = Path(args.output).expanduser().resolve()
    count = prepare_data(input_dir, output_file)
    print(f"Prepared {count} text file(s): {output_file}")


if __name__ == "__main__":
    main()
