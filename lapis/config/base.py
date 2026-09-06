import os
from pathlib import Path

import yaml


def resolve_path(path):
    """Resolve a config path from the working tree or installed package."""
    candidate = Path(path)
    if candidate.exists():
        return candidate

    package_root = Path(__file__).resolve().parents[2]
    packaged = package_root / candidate
    if packaged.exists():
        return packaged

    return candidate


def load_config(path):
    """Load configuration from a YAML file path."""
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["_base_path"] = os.path.dirname(os.path.abspath(path))
    return config
