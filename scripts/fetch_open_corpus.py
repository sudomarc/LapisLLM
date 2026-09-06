#!/usr/bin/env python3
"""Collect a bounded, diverse training corpus from public/open datasets.

The collector is intentionally bounded so Colab runs do not accidentally
consume huge datasets. Hugging Face streaming is used throughout and every
source is recorded in manifest.json for provenance.

Install once:
    pip install datasets

Example:
    python scripts/fetch_open_corpus.py --max-chars 50000000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("lapis.fetch_open_corpus")

# Broad mixture: general web, education, books, knowledge, science,
# mathematics, code, and assistant conversations/reasoning. All are streamed
# and bounded. These are open/public datasets; they are not Claude's private
# training data, which is not publicly disclosed by Anthropic.
SOURCES: dict[str, dict[str, Any]] = {
    "fineweb": {"dataset": "HuggingFaceFW/fineweb", "config": "sample-10BT", "split": "train", "field": "text"},
    "fineweb_edu": {"dataset": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "split": "train", "field": "text"},
    "c4": {"dataset": "allenai/c4", "config": "en", "split": "train", "field": "text"},
    "wikipedia": {"dataset": "wikimedia/wikipedia", "config": "20231101.en", "split": "train", "field": "text"},
    "s2orc_arxiv": {"dataset": "AlgorithmicResearchGroup/s2orc_arxiv", "config": None, "split": "train", "field": "text"},
    "cosmopedia": {"dataset": "HuggingFaceTB/cosmopedia", "config": "web_samples_v1", "split": "train", "field": "text"},
    "gutenberg": {"dataset": "common-pile/project_gutenberg_filtered", "config": "default", "split": "train", "field": "text"},
    "openr1_math": {"dataset": "open-r1/OpenR1-Math-220k", "config": "default", "split": "train", "field": "messages"},
    "math": {"dataset": "open-web-math/open-web-math", "config": "default", "split": "train", "field": "text"},
    "oasst1": {"dataset": "OpenAssistant/oasst1", "config": None, "split": "train", "field": "text"},
    "code": {"dataset": "bigcode/the-stack-smol", "config": "data", "split": "train", "field": "content"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a bounded open Lapis training corpus")
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCES), default=[
        "fineweb", "fineweb_edu", "c4", "wikipedia", "s2orc_arxiv",
        "cosmopedia", "gutenberg", "openr1_math", "oasst1", "math", "code",
    ])
    parser.add_argument("--output-dir", default="training_data/open")
    parser.add_argument("--max-chars", type=int, default=50_000_000, help="Global character budget")
    parser.add_argument("--max-chars-per-source", type=int, default=4_500_000)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument("--max-examples-per-source", type=int, default=25_000)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--language", default=None, help="Optional dataset language filter when supported")
    parser.add_argument("--no-code", action="store_true", help="Skip code even if selected")
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_streaming(spec: dict[str, Any]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: install it with `pip install datasets`.") from exc

    kwargs: dict[str, Any] = {
        "path": spec["dataset"],
        "split": spec.get("split", "train"),
        "streaming": True,
    }
    if spec.get("config") not in {None, "default"}:
        kwargs["name"] = spec["config"]
    elif spec.get("config") == "default":
        kwargs["name"] = "default"
    return load_dataset(**kwargs)


def extract_text(example: dict[str, Any], field: str) -> str | None:
    value = example.get(field)
    if isinstance(value, str):
        return value

    # OpenR1-style message arrays and other conversational structures.
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content") or item.get("text")
                if isinstance(content, str):
                    prefix = f"{role}: " if isinstance(role, str) else ""
                    parts.append(prefix + content)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n\n".join(parts)

    # Common instruction/reasoning dataset shapes.
    parts: list[str] = []
    for key in ("problem", "question", "prompt", "solution", "answer", "completion", "response", "content", "body", "text"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n\n".join(parts) if parts else None


def write_source(path: Path, source_name: str, records: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for text in records:
            handle.write(f"===== {source_name} =====\n\n{text}\n\n")
    return path.stat().st_size


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    total_examples = 0
    manifest: dict[str, Any] = {
        "project": "LapisLLM",
        "collector": "scripts/fetch_open_corpus.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budgets": vars(args),
        "sources": {},
    }
    combined_path = output / "combined.txt"
    combined_path.write_text("", encoding="utf-8")

    for name in args.sources:
        if args.no_code and name == "code":
            continue
        if total_chars >= args.max_chars or total_examples >= args.max_examples:
            break
        spec = SOURCES[name]
        LOGGER.info("Loading %s (%s)", name, spec["dataset"])
        stream = None
        try:
            stream = load_streaming(spec)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", name, exc)
            manifest["sources"][name] = {"status": "load_failed", "error": str(exc)}
            continue

        records: list[str] = []
        chars = 0
        examples = 0
        try:
            for example in stream:
                if examples >= args.max_examples_per_source:
                    break
                if total_examples >= args.max_examples or total_chars >= args.max_chars:
                    break
                raw = extract_text(example, spec["field"])
                if not raw:
                    continue
                text = clean_text(raw)
                if len(text) < args.min_chars:
                    continue
                remaining_global = args.max_chars - total_chars
                remaining_source = args.max_chars_per_source - chars
                budget = min(remaining_global, remaining_source)
                if budget <= 0:
                    break
                if len(text) > budget:
                    text = text[:budget]
                records.append(text)
                size = len(text)
                chars += size
                total_chars += size
                examples += 1
                total_examples += 1
        except Exception as exc:
            LOGGER.warning("Stream interrupted for %s after %d examples: %s", name, examples, exc)
        finally:
            del stream
            gc.collect()

        source_path = output / f"{name}.txt"
        size = write_source(source_path, name, records)
        with combined_path.open("a", encoding="utf-8", newline="\n") as combined:
            for text in records:
                combined.write(f"===== {name} =====\n\n{text}\n\n")
        manifest["sources"][name] = {
            "dataset": spec["dataset"],
            "config": spec.get("config"),
            "split": spec.get("split", "train"),
            "field": spec["field"],
            "examples": examples,
            "characters": chars,
            "file_bytes": size,
            "status": "ok" if examples else "empty",
        }
        LOGGER.info("%s: %d examples, %d chars", name, examples, chars)

    manifest["totals"] = {"examples": total_examples, "characters": total_chars}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Corpus complete: %d examples / %d characters", total_examples, total_chars)
    LOGGER.info("Combined corpus: %s", combined_path)
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code == 0:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    raise SystemExit(exit_code)
