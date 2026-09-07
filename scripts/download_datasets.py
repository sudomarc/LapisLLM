#!/usr/bin/env python3
"""Download and normalize the LapisLLM open-data registry.

This script never commits raw datasets to Git. It writes them under the local
``data/raw/`` tree and emits per-source provenance/manifests. Dataset selection
is declarative in ``configs/data/datasets.yaml``.

Examples:
  python scripts/download_datasets.py --list
  python scripts/download_datasets.py --profile development
  python scripts/download_datasets.py --profile recommended --max-records 10000
  python scripts/download_datasets.py --source fineweb --config sample-10BT --max-records 100000
  python scripts/download_datasets.py --source numinamath_cot
  python scripts/download_datasets.py --profile all --allow-source-specific-license

The downloader intentionally refuses sources whose registry entry declares
``source_specific`` licensing unless the caller explicitly opts in. This keeps
LAPIS aligned with AGENTS.md: unknown/source-specific licensing must not become
implicit training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing PyYAML. Install with: pip install pyyaml") from exc

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing datasets. Install with: pip install 'lapis[data]'") from exc

LOGGER = logging.getLogger("lapis.download_datasets")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = REPO_ROOT / "configs" / "data" / "datasets.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LapisLLM dataset registry sources")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", choices=("all", "recommended", "development"), default="recommended")
    parser.add_argument("--source", action="append", default=[], help="Specific registry id; repeatable")
    parser.add_argument("--list", action="store_true", help="List registry sources and exit")
    parser.add_argument("--stage", choices=("pretrain", "sft"), action="append", help="Filter by training stage")
    parser.add_argument("--config", help="Override the HF dataset config for all --source selections")
    parser.add_argument("--split", help="Override the HF split for all selected sources")
    parser.add_argument("--max-records", type=int, default=0, help="Limit records per source; 0 means no explicit limit")
    parser.add_argument("--streaming", action="store_true", help="Stream datasets instead of caching the full dataset")
    parser.add_argument("--allow-source-specific-license", action="store_true", help="Permit entries marked source_specific")
    parser.add_argument("--no-write", action="store_true", help="Resolve selections and show actions without writing data")
    parser.add_argument("--timeout", type=float, default=30.0, help="Reserved for future HTTP-backed sources")
    return parser.parse_args()


def load_catalog(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ValueError(f"Invalid dataset catalog: {path}")
    return data


def source_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in catalog["sources"]:
        if not isinstance(raw, dict) or "id" not in raw or "dataset" not in raw:
            raise ValueError("Every dataset source needs at least id and dataset")
        source_id = str(raw["id"])
        if source_id in result:
            raise ValueError(f"Duplicate dataset source id: {source_id}")
        result[source_id] = raw
    return result


def selected_sources(catalog: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    sources = source_map(catalog)
    if args.source:
        unknown = [source_id for source_id in args.source if source_id not in sources]
        if unknown:
            raise SystemExit(f"Unknown source id(s): {', '.join(unknown)}")
        selected = [sources[source_id] for source_id in args.source]
    else:
        profile = catalog.get("profiles", {}).get(args.profile, {})
        include = profile.get("include", [])
        selected = [sources[source_id] for source_id in include if source_id in sources]

    if args.stage:
        stages = set(args.stage)
        selected = [item for item in selected if item.get("stage") in stages]
    return selected


def print_sources(catalog: dict[str, Any]) -> None:
    for source in catalog["sources"]:
        print(
            f"{source['id']:<28} {source.get('stage', '?'):<8} "
            f"{source['dataset']:<48} license={source.get('license', '?')}"
        )


def safe_filename(value: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    cleaned = "".join(ch if ch in keep else "_" for ch in value)
    return cleaned.strip("._")[:180] or "dataset"


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iter_rows(dataset: Any, max_records: int) -> Iterable[dict[str, Any]]:
    if hasattr(dataset, "to_iterable_dataset"):
        dataset = dataset.to_iterable_dataset()
    count = 0
    for row in dataset:
        if not isinstance(row, dict):
            row = {"value": row}
        yield row
        count += 1
        if max_records > 0 and count >= max_records:
            break


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    bytes_written = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            bytes_written += len(line.encode("utf-8"))
            records += 1
    return records, bytes_written


def resolve_loader_kwargs(source: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "path": source["dataset"],
        "split": args.split or source.get("split", "train"),
        "streaming": bool(args.streaming),
    }
    config = args.config if args.config is not None else source.get("config")
    if config not in (None, "", "null"):
        kwargs["name"] = config
    return kwargs


def download_one(source: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source_id = str(source["id"])
    license_name = str(source.get("license", "unknown"))
    if license_name == "source_specific" and not args.allow_source_specific_license:
        return {
            "id": source_id,
            "status": "blocked_license_review",
            "reason": "Registry marks this dataset as source_specific; pass --allow-source-specific-license after reviewing upstream licensing.",
        }

    kwargs = resolve_loader_kwargs(source, args)
    destination = args.output_dir / safe_filename(source_id)
    manifest_path = destination / "manifest.json"
    config_label = kwargs.get("name") or "default"
    LOGGER.info("%s -> %s/%s split=%s streaming=%s", source_id, source["dataset"], config_label, kwargs["split"], kwargs["streaming"])

    if args.no_write:
        return {
            "id": source_id,
            "status": "planned",
            "loader": kwargs,
            "output": str(destination),
            "license": license_name,
        }

    started = time.time()
    dataset = load_dataset(**kwargs)
    raw_path = destination / "records.jsonl"
    records, bytes_written = write_jsonl(raw_path, iter_rows(dataset, args.max_records))

    manifest = {
        "project": "LapisLLM",
        "source_id": source_id,
        "dataset": source["dataset"],
        "stage": source.get("stage"),
        "config": config_label,
        "split": kwargs["split"],
        "streaming": bool(args.streaming),
        "records_written": records,
        "bytes_written": bytes_written,
        "license": license_name,
        "license_url": source.get("license_url"),
        "source_url": source.get("source_url"),
        "text_fields": source.get("text_fields", []),
        "catalog_source_hash": stable_json_hash(source),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"id": source_id, "status": "downloaded", **manifest, "output": str(raw_path)}


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    catalog = load_catalog(args.catalog)
    if args.list:
        print_sources(catalog)
        return 0

    selected = selected_sources(catalog, args)
    if not selected:
        raise SystemExit("No dataset sources selected")

    if args.profile == "all" and not args.allow_source_specific_license:
        LOGGER.info("Profile=all: source-specific-license entries will be listed but blocked until explicitly approved.")

    results = []
    for source in selected:
        try:
            results.append(download_one(source, args))
        except Exception as exc:  # isolate a source so one unavailable dataset does not hide others
            LOGGER.exception("Dataset %s failed", source.get("id"))
            results.append({"id": source.get("id"), "status": "failed", "error": str(exc)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "project": "LapisLLM",
        "catalog": str(args.catalog),
        "profile": args.profile,
        "sources_requested": [source.get("id") for source in selected],
        "results": results,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version,
            "pid": os.getpid(),
        },
    }
    run_manifest_path = args.output_dir / "download-manifest.json"
    if not args.no_write:
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for result in results:
        print(f"{result['id']}: {result['status']}")
    failures = [result for result in results if result["status"] == "failed"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
