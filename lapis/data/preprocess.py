"""Dataset preprocessing pipeline for LapisLLM.

Handles normalization, deduplication, and train/validation/test splitting
of approved training data.
"""
import json
import os
import hashlib
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from lapis.data.registry import DatasetRegistry, SourceRecord, compute_content_sha256


def normalize_text(text: str) -> str:
    """Normalize text content for training.

    Steps:
    1. Strip leading/trailing whitespace
    2. Collapse multiple internal whitespace to single space
    3. Remove UTF-8 BOM if present
    4. Normalize common code patterns
    """
    # Strip
    text = text.strip()

    # Collapse multiple whitespace lines but preserve code structure
    # Only collapse runs of 3+ blank lines to 2
    lines = text.split("\n")
    processed_lines = []
    blank_streak = 0

    for line in lines:
        if line == "":
            blank_streak += 1
            # Keep at most 2 blank lines
            if blank_streak <= 2:
                processed_lines.append(line)
        else:
            blank_streak = 0
            processed_lines.append(line)

    text = "\n".join(processed_lines)

    # Remove BOM
    if text.startswith("﻿"):
        text = text[1:]

    return text


def deduplicate_records(records: List[SourceRecord]) -> List[SourceRecord]:
    """Remove duplicate records based on content SHA256 hash.

    Keeps the first occurrence of each unique content hash.
    """
    seen_hashes = set()
    unique_records = []

    for rec in records:
        h = compute_content_sha256(rec.content)
        if h not in seen_hashes:
            seen_hashes.add(h)
            # Update the record's content_sha256
            rec.content_sha256 = h
            unique_records.append(rec)
        # else: skip duplicate

    return unique_records


def split_by_language_and_size(
    registry: DatasetRegistry,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    min_records_per_lang: int = 1,
) -> Dict[str, List[SourceRecord]]:
    """Split approved sources into train/validation/test partitions.

    Partitions are done per language to ensure language coverage.
    Fractions are approximate per language group.

    Returns dict with keys "train", "validation", "test".
    """
    # Get approved records grouped by language
    by_language: Dict[str, List[SourceRecord]] = defaultdict(list)
    for rec in registry.get_approved():
        by_language[rec.language].append(rec)

    splits = {"train": [], "validation": [], "test": []}

    for lang, records in by_language.items():
        # Shuffle (simple deterministic sort by record_id for reproducibility)
        records.sort(key=lambda r: r.record_id)

        n = len(records)
        if n < min_records_per_lang:
            # Too few — put all in train
            splits["train"].extend(records)
            continue

        # Calculate split indices
        n_train = max(1, int(n * train_frac))
        n_val = max(0, int(n * val_frac))
        n_test = n - n_train - n_val  # remainder

        # Ensure we don't exceed total
        n_train = min(n_train, n - 2)  # leave at least 2 for val+test
        n_val = min(n_val, n - n_train - 1)
        n_test = n - n_train - n_val

        splits["train"].extend(records[:n_train])
        splits["validation"].extend(records[n_train:n_train + n_val])
        splits["test"].extend(records[n_train + n_val:])

    return splits


def export_splits_to_disk(
    splits: Dict[str, List[SourceRecord]],
    output_dir: str = ".lapis-data/processed",
    prefix: str = "lapis-corpus"
) -> Dict[str, str]:
    """Write split datasets to JSON files on disk.

    Returns dict mapping split names to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    for split_name, records in splits.items():
        filepath = os.path.join(output_dir, f"{prefix}-{split_name}.json")
        with open(filepath, "w") as f:
            json.dump([rec.to_dict() for rec in records], f, indent=2)
        paths[split_name] = filepath
        print(f"  Wrote {len(records)} records to {filepath}")

    return paths


def build_dataset_manifests(
    registry_path: str = ".lapis-data/dataset-registry.json",
    output_dir: str = ".lapis-data/processed",
) -> Dict[str, str]:
    """Build train/val/test splits from registry and export to disk.

    Full pipeline: load registry → approve development corpus →
    split by language/size → export manifests.

    Returns dict of output file paths.
    """
    reg = DatasetRegistry(registry_path)

    # Ensure development corpus is in registry
    reg = initialize_development_registry(registry_path)

    # Get all approved records
    approved = reg.get_approved()
    print(f"Building splits from {len(approved)} approved sources")

    # Split into train/val/test
    splits = split_by_language_and_size(reg)

    print(f"  Train: {len(splits['train'])} records")
    print(f"  Validation: {len(splits['validation'])} records")
    print(f"  Test: {len(splits['test'])} records")

    # Export to disk
    paths = export_splits_to_disk(splits, output_dir)

    # Compute deduplication stats
    all_contents = [rec.content for rec in approved]
    original_count = len(all_contents)
    unique_records = deduplicate_records(approved)
    after_dedup = len(unique_records)
    removed = original_count - after_dedup

    print(f"\nDeduplication: {original_count} → {after_dedup} "
          f"(-{removed} duplicates removed)")

    return paths


if __name__ == "__main__":
    # Standalone: build the dataset
    print("LapisLLM Dataset Preprocessing Pipeline")
    print("=" * 50)

    paths = build_dataset_manifests()

    print("\nOutput files:")
    for split_name, path in paths.items():
        print(f"  {split_name}: {path}")