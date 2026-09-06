import json
import os
import hashlib
from datetime import datetime


def compute_hash(filepath):
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def save_manifest(output_dir, manifest_data):
    """Save a dataset manifest as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    
    return manifest_path


def load_manifest(manifest_path):
    """Load a dataset manifest from JSON."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_manifest(dataset_name, version, documents, tokens, 
                    language_distribution, sources, filters_used,
                    deduplication, tokenizer_name, data_dir="data"):
    """Create a dataset manifest.
    
    Returns manifest dict following the LAPIS format.
    """
    # Compute hash of all data files
    data_hash = ""
    if os.path.exists(data_dir):
        for entry in os.listdir(data_dir):
            entry_path = os.path.join(data_dir, entry)
            if os.path.isfile(entry_path):
                data_hash = compute_hash(entry_path)
                break
    
    manifest = {
        "dataset_name": dataset_name,
        "version": version,
        "documents": documents,
        "tokens": tokens,
        "language_distribution": language_distribution,
        "sources": sources,
        "filters": filters_used,
        "deduplication": deduplication,
        "tokenizer": tokenizer_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "hash": data_hash
    }
    
    return manifest