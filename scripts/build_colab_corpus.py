#!/usr/bin/env python3
"""Build a bounded, reproducible Colab pretraining corpus from the Lapis registry.

The builder streams only the explicitly selected, registry-approved pretraining
sources and writes one UTF-8 text file plus a provenance manifest. It is designed
for the first Colab run, not for downloading an entire web-scale corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset

DEFAULT_SOURCES = (
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    ("wikipedia", "wikimedia/wikipedia", "20231101.fr", "train", "text"),
    ("openwebmath", "open-web-math/open-web-math", None, "train", "text"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bounded LapisLLM Colab corpus")
    parser.add_argument("--output", default="training_data/colab_pretrain.txt")
    parser.add_argument("--manifest", default="training_data/colab_pretrain_manifest.json")
    parser.add_argument("--max-chars", type=int, default=200_000_000)
    parser.add_argument("--max-records-per-source", type=int, default=0)
    parser.add_argument("--source", action="append", choices=[item[0] for item in DEFAULT_SOURCES])
    return parser.parse_args()


def clean_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def selected_sources(args: argparse.Namespace):
    requested = set(args.source or [item[0] for item in DEFAULT_SOURCES])
    return [item for item in DEFAULT_SOURCES if item[0] in requested]


def load_kwargs(dataset_id: str, config: str | None, split: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"path": dataset_id, "split": split, "streaming": True}
    if config:
        kwargs["name"] = config
    return kwargs


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    source_stats = []
    seen_sources = []

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for source_id, dataset_id, config, split, field in selected_sources(args):
            if total_chars >= args.max_chars:
                break
            dataset = load_dataset(**load_kwargs(dataset_id, config, split))
            records = 0
            chars = 0
            for row in dataset:
                value = row.get(field) if isinstance(row, dict) else None
                if not isinstance(value, str):
                    continue
                text = clean_text(value)
                if not text:
                    continue
                remaining = args.max_chars - total_chars
                if remaining <= 0:
                    break
                block = f"\n\n===== {source_id} =====\n\n{text}\n"
                if len(block) > remaining:
                    block = block[:remaining]
                handle.write(block)
                written = len(block)
                total_chars += written
                chars += written
                records += 1
                if args.max_records_per_source and records >= args.max_records_per_source:
                    break
            source_stats.append({
                "id": source_id,
                "dataset": dataset_id,
                "config": config,
                "split": split,
                "field": field,
                "records_written": records,
                "chars_written": chars,
            })
            seen_sources.append(source_id)

    manifest = {
        "project": "LapisLLM",
        "builder": "scripts/build_colab_corpus.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_chars": args.max_chars,
        "actual_chars": total_chars,
        "sources": source_stats,
        "source_order": seen_sources,
        "output": str(output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Corpus: {output} ({total_chars:,} chars)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
