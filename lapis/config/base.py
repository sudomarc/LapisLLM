from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a user path without depending on the current working directory."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML configuration file."""
    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    config["_base_path"] = str(config_path.parent)
    return config
