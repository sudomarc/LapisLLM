import os
from pathlib import Path

import torch
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


def resolve_device(requested: str) -> torch.device:
    """Resolve a device and reject unavailable explicit CUDA indices."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        if requested == "cuda":
            return torch.device(requested)
        try:
            index = int(requested.split(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid CUDA device: {requested}") from exc
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise ValueError(
                f"CUDA device index {index} is out of range; {count} device(s) available."
            )
    return torch.device(requested)


def load_config(path):
    """Load configuration from a YAML file path."""
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["_base_path"] = os.path.dirname(os.path.abspath(path))
    return config
