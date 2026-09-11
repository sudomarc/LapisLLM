#!/usr/bin/env python3
"""Build a bounded, reproducible Colab pretraining corpus from the Lapis registry.

The builder streams only the explicitly selected, registry-approved pretraining
sources and writes one UTF-8 text file plus a provenance manifest. It is designed
for the first Colab run, not for downloading an entire web-scale corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

# FineWeb-Edu currently uses Hugging Face Hub/Xet-backed Parquet shards. Keep the
# Colab corpus path on the standard Hub HTTP transport unless the caller already
# selected another setting. These timeouts also prevent a broken network path
# from appearing idle indefinitely.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from datasets import load_dataset

DEFAULT_SOURCES = (
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    ("wikipedia", "wikimedia/wikipedia", "20231101.fr", "train", "text"),
    ("openwebmath", "open-web-math/open-web-math", None, "train", "text"),
)

CORPUS_CONTRACT_VERSION = "lapis-colab-corpus-v2"
PROGRESS_EVERY_RECORDS = 250
PROGRESS_EVERY_SECONDS = 5.0
SOURCE_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


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
    return re.sub(r"\n{3,}", "\n\n", value).strip()


def selected_sources(args: argparse.Namespace):
    requested = set(args.source or [item[0] for item in DEFAULT_SOURCES])
    return [item for item in DEFAULT_SOURCES if item[0] in requested]


def allocate_source_budget(remaining_global: int, remaining_sources: int) -> int:
    """Give the current source a fair share of the budget still available."""
    if remaining_global < 0 or remaining_sources < 1:
        raise ValueError("remaining_global must be >= 0 and remaining_sources must be >= 1")
    return max(1, (remaining_global + remaining_sources - 1) // remaining_sources)


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


def load_source(source_id: str, dataset_id: str, config: str | None, split: str):
    """Open one streaming source with a bounded retry budget."""
    last_error: Exception | None = None
    for attempt in range(1, SOURCE_RETRIES + 1):
        try:
            if attempt > 1:
                delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 2))
                print(
                    f"[CORPUS] RETRY | source={source_id} | attempt={attempt}/{SOURCE_RETRIES} | "
                    f"sleep={delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
            return load_dataset(**load_kwargs(dataset_id, config, split))
        except Exception as exc:
            last_error = exc
            print(
                f"[CORPUS] SOURCE OPEN FAILED | source={source_id} | "
                f"attempt={attempt}/{SOURCE_RETRIES} | error={exc}",
                flush=True,
            )
    assert last_error is not None
    raise last_error


def write_source(
    handle,
    *,
    source_id: str,
    dataset_id: str,
    config: str | None,
    split: str,
    field: str,
    total_chars: int,
    max_chars: int,
    source_char_limit: int,
    max_records_per_source: int,
    started: float,
) -> tuple[int, int, int]:
    """Write one source within its current share of the global budget."""
    source_start = handle.tell()

    for attempt in range(1, SOURCE_RETRIES + 1):
        handle.seek(source_start)
        handle.truncate()
        dataset = load_source(source_id, dataset_id, config, split)
        records = 0
        chars = 0
        last_progress = time.monotonic()
        try:
            for row in dataset:
                value = row.get(field) if isinstance(row, dict) else None
                if not isinstance(value, str):
                    continue
                text = clean_text(value)
                if not text:
                    continue
                remaining_global = max_chars - total_chars
                remaining_source = source_char_limit - chars
                remaining = min(remaining_global, remaining_source)
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
                        max_chars=max_chars,
                        started=started,
                    )
                    last_progress = now

                if max_records_per_source and records >= max_records_per_source:
                    break

            return records, chars, total_chars
        except Exception as exc:
            print(
                f"[CORPUS] SOURCE FAILED | source={source_id} | "
                f"attempt={attempt}/{SOURCE_RETRIES} | error={exc}",
                flush=True,
            )
            handle.seek(source_start)
            handle.truncate()
            total_chars -= chars
            if attempt < SOURCE_RETRIES:
                delay = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(
                    f"[CORPUS] RETRYING SOURCE | source={source_id} | sleep={delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
            else:
                raise

    raise RuntimeError(f"Source exhausted retry budget: {source_id}")


def main() -> int:
    args = parse_args()
    if args.max_chars < 1:
        raise SystemExit("--max-chars must be >= 1")
    if args.max_records_per_source < 0:
        raise SystemExit("--max-records-per-source must be >= 0")

    output = Path(args.output)
    manifest_path = Path(args.manifest)
    sources = selected_sources(args)
    if not sources:
        raise SystemExit("At least one corpus source must be selected")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    source_stats = []
    seen_sources = []
    source_errors = []
    started = time.monotonic()

    print(
        f"[CORPUS] Starting | target={format_size(args.max_chars)} | "
        f"sources={', '.join(item[0] for item in sources)} | "
        f"contract={CORPUS_CONTRACT_VERSION} | "
        f"xet={'disabled' if os.environ.get('HF_HUB_DISABLE_XET', '').lower() in {'1', 'true', 'yes', 'on'} else 'enabled'}",
        flush=True,
    )

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (source_id, dataset_id, config, split, field) in enumerate(sources):
            if total_chars >= args.max_chars:
                break
            remaining_sources = len(sources) - index
            remaining_global = args.max_chars - total_chars
            source_budget = allocate_source_budget(remaining_global, remaining_sources)

            source_started = time.monotonic()
            print(
                f"[CORPUS] Loading source: {source_id} "
                f"({dataset_id}, config={config or 'default'}, split={split}) | "
                f"budget={format_size(source_budget)}",
                flush=True,
            )
            try:
                records, chars, total_chars = write_source(
                    handle,
                    source_id=source_id,
                    dataset_id=dataset_id,
                    config=config,
                    split=split,
                    field=field,
                    total_chars=total_chars,
                    max_chars=args.max_chars,
                    source_char_limit=source_budget,
                    max_records_per_source=args.max_records_per_source,
                    started=started,
                )
            except Exception as exc:
                source_errors.append({"id": source_id, "dataset": dataset_id, "error": str(exc)})
                print(
                    f"[CORPUS] SOURCE SKIPPED | source={source_id} | error={exc}",
                    flush=True,
                )
                continue

            source_stats.append(
                {
                    "id": source_id,
                    "dataset": dataset_id,
                    "config": config,
                    "split": split,
                    "field": field,
                    "budget_chars": source_budget,
                    "records_written": records,
                    "chars_written": chars,
                    "elapsed_seconds": time.monotonic() - source_started,
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
        "contract_version": CORPUS_CONTRACT_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_chars": args.max_chars,
        "actual_chars": total_chars,
        "sources": source_stats,
        "source_order": seen_sources,
        "source_errors": source_errors,
        "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET", ""),
        "hf_hub_etag_timeout": os.environ.get("HF_HUB_ETAG_TIMEOUT", ""),
        "hf_hub_download_timeout": os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", ""),
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
    if source_errors:
        print(f"[CORPUS] WARNINGS | skipped_sources={len(source_errors)}", flush=True)
    if total_chars <= 0:
        raise RuntimeError("Corpus builder produced no training data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
