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

PROGRESS_EVERY_RECORDS = 250
PROGRESS_EVERY_SECONDS = 5.0


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


def format_size(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{value:,} B"


def print_progress(
    *,
    source_id: str,
    source_records: int,
    source_chars: int,
    total_chars: int,
    max_chars: int,
    started: float,
) -> None:
    elapsed = max(0.001, time.monotonic() - started)
    percent = min(100.0, (total_chars / max_chars) * 100) if max_chars else 100.0
    chars_per_sec = total_chars / elapsed
    remaining_chars = max(0, max_chars - total_chars)
    eta = remaining_chars / chars_per_sec if chars_per_sec > 0 else 0.0
    print(
        f"[CORPUS] {percent:6.2f}% | {format_size(total_chars)} / "
        f"{format_size(max_chars)} | source={source_id} | "
        f"records={source_records:,} | source_chars={format_size(source_chars)} | "
        f"speed={format_size(int(chars_per_sec))}/s | ETA={eta / 60:.1f} min",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    manifest_path = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    source_stats = []
    seen_sources = []
    started = time.monotonic()

    print(
        f"[CORPUS] Starting | target={format_size(args.max_chars)} | "
        f"sources={', '.join(item[0] for item in selected_sources(args))}",
        flush=True,
    )

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for source_id, dataset_id, config, split, field in selected_sources(args):
            if total_chars >= args.max_chars:
                break

            source_started = time.monotonic()
            print(
                f"[CORPUS] Loading source: {source_id} "
                f"({dataset_id}, config={config or 'default'}, split={split})",
                flush=True,
            )
            dataset = load_dataset(**load_kwargs(dataset_id, config, split))
            records = 0
            chars = 0
            last_progress = source_started

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

                now = time.monotonic()
                if (
                    records % PROGRESS_EVERY_RECORDS == 0
                    or now - last_progress >= PROGRESS_EVERY_SECONDS
                ):
                    print_progress(
                        source_id=source_id,
                        source_records=records,
                        source_chars=chars,
                        total_chars=total_chars,
                        max_chars=args.max_chars,
                        started=started,
                    )
                    last_progress = now

                if args.max_records_per_source and records >= args.max_records_per_source:
                    break

            source_stats.append(
                {
                    "id": source_id,
                    "dataset": dataset_id,
                    "config": config,
                    "split": split,
                    "field": field,
                    "records_written": records,
                    "chars_written": chars,
                }
            )
            seen_sources.append(source_id)
            print_progress(
                source_id=source_id,
                source_records=records,
                source_chars=chars,
                total_chars=total_chars,
                max_chars=args.max_chars,
                started=started,
            )
            print(f"[CORPUS] Completed source: {source_id}", flush=True)

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
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    elapsed = time.monotonic() - started
    print(
        f"[CORPUS] DONE | {format_size(total_chars)} | "
        f"elapsed={elapsed / 60:.1f} min | output={output}",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
