"""Centralized, typed configuration for LapisLLM.

Configuration sources, in ascending priority:

1. Built-in defaults (:data:`DEFAULTS`).
2. A TOML config file (default ``lapis.toml`` in the current directory, or the
   path given by ``LAPIS_CONFIG``).
3. Environment variables prefixed with ``LAPIS_`` (e.g. ``LAPIS_LOG_LEVEL=DEBUG``).

All paths are :class:`pathlib.Path` objects. Relative paths are resolved against
the directory of the config file that defines them, falling back to the current
working directory. Nothing is written to disk unless explicitly requested.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .exceptions import ConfigError

__all__ = [
    "DEFAULTS",
    "LapisConfig",
    "ConfigSource",
    "load_config",
    "save_config",
]

ENV_PREFIX = "LAPIS_"
ENV_CONFIG_KEY = "LAPIS_CONFIG"
DEFAULT_CONFIG_NAME = "lapis.toml"

DEFAULTS: dict[str, Any] = {
    "project_name": "lapisllm",
    "config_version": 1,
    "data_root": ".lapis-data",
    "runs_dir": ".lapis-data/runs",
    "log_level": "INFO",
    "log_file": None,
    "default_device": "auto",
    "default_precision": "fp32",
    "seed": 42,
    "max_cpu_threads": None,
}

_VALID_DEVICES = {"cpu", "cuda", "mps", "rocm", "auto"}
_VALID_PRECISIONS = {"fp32", "fp16", "bf16"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class LapisConfig(BaseModel):
    """Project-wide typed configuration.

    Attributes marked with ``--`` are derived from the environment or the config
    file and validated on construction.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    project_name: str = Field(default=DEFAULTS["project_name"], min_length=1)
    config_version: int = Field(default=DEFAULTS["config_version"], ge=1)
    data_root: Path = Field(default=Path(DEFAULTS["data_root"]))
    runs_dir: Path = Field(default=Path(DEFAULTS["runs_dir"]))
    log_level: str = DEFAULTS["log_level"]
    log_file: Path | None = None
    default_device: str = DEFAULTS["default_device"]
    default_precision: str = DEFAULTS["default_precision"]
    seed: int = DEFAULTS["seed"]
    max_cpu_threads: int | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, value: Any) -> str:
        if value is None:
            return str(DEFAULTS["log_level"])
        text = str(value).upper()
        if text not in _VALID_LOG_LEVELS:
            raise ConfigError(
                f"Invalid log_level {value!r}; expected one of {sorted(_VALID_LOG_LEVELS)}"
            )
        return text

    @field_validator("default_device", mode="before")
    @classmethod
    def _validate_device(cls, value: Any) -> str:
        text = str(value).lower()
        if text not in _VALID_DEVICES:
            raise ConfigError(
                f"Invalid default_device {value!r}; expected one of {sorted(_VALID_DEVICES)}"
            )
        return text

    @field_validator("default_precision", mode="before")
    @classmethod
    def _validate_precision(cls, value: Any) -> str:
        text = str(value).lower()
        if text not in _VALID_PRECISIONS:
            raise ConfigError(
                f"Invalid default_precision {value!r}; expected one of {sorted(_VALID_PRECISIONS)}"
            )
        return text

    @field_validator("data_root", "runs_dir", "log_file", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Any:
        if value is None or isinstance(value, Path):
            return value
        return Path(str(value))

    @model_validator(mode="after")
    def _resolve_paths(self) -> LapisConfig:
        # Relative paths stay relative so callers can re-root the whole tree;
        # absolute paths are kept as-is. Nothing is created on disk here.
        return self

    def model_dump_paths(self) -> dict[str, str]:
        """Serialize the config with paths as POSIX strings.

        Only non-empty values are emitted: an absent key reloads the built-in
        default, which keeps round-trips stable.
        """
        items: list[tuple[str, str]] = [
            ("project_name", self.project_name),
            ("config_version", str(self.config_version)),
            ("data_root", str(self.data_root)),
            ("runs_dir", str(self.runs_dir)),
            ("log_level", self.log_level),
            ("default_device", self.default_device),
            ("default_precision", self.default_precision),
            ("seed", str(self.seed)),
        ]
        if self.log_file is not None:
            items.append(("log_file", str(self.log_file)))
        if self.max_cpu_threads is not None:
            items.append(("max_cpu_threads", str(self.max_cpu_threads)))
        return {key: value for key, value in items if value}


class ConfigSource:
    """Resolution order and raw values for a single configuration load."""

    def __init__(self, defaults: dict[str, Any], file_path: Path | None) -> None:
        self.defaults = defaults
        self.file_path = file_path
        self.file_values: dict[str, Any] = {}
        self.env_values: dict[str, Any] = {}

    def merged(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        merged.update(self.defaults)
        merged.update(self.file_values)
        merged.update(self.env_values)
        return merged


def _env_overrides() -> dict[str, Any]:
    """Collect ``LAPIS_*`` environment variables as typed overrides."""
    overrides: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        name = key[len(ENV_PREFIX) :].lower()
        if name not in DEFAULTS:
            continue
        default = DEFAULTS[name]
        if isinstance(default, bool):
            overrides[name] = value.lower() in {"1", "true", "yes", "on"}
        elif isinstance(default, int):
            try:
                overrides[name] = int(value)
            except ValueError as exc:
                raise ConfigError(
                    f"Environment variable {key}={value!r} is not a valid integer"
                ) from exc
        elif isinstance(default, Path):
            overrides[name] = Path(value)
        else:
            overrides[name] = value
    return overrides


def load_config(config_path: str | Path | None = None) -> tuple[LapisConfig, ConfigSource]:
    """Load configuration from defaults, an optional TOML file, and the environment.

    Parameters
    ----------
    config_path:
        Explicit path to a ``lapis.toml`` file. When omitted, looks for
        ``LAPIS_CONFIG`` then ``lapis.toml`` in the current directory. A missing
        file is not an error when no explicit path was given.
    """
    resolved: Path | None = None
    if config_path is not None:
        resolved = Path(config_path)
        if not resolved.is_file():
            raise ConfigError(f"Config file not found: {resolved}")
    else:
        env_path = os.environ.get(ENV_CONFIG_KEY)
        if env_path:
            resolved = Path(env_path)
            if not resolved.is_file():
                raise ConfigError(f"LAPIS_CONFIG file not found: {resolved}")
        else:
            candidate = Path.cwd() / DEFAULT_CONFIG_NAME
            if candidate.is_file():
                resolved = candidate

    source = ConfigSource(dict(DEFAULTS), resolved)

    file_dir = resolved.parent if resolved is not None else Path.cwd()
    if resolved is not None:
        try:
            with resolved.open("rb") as handle:
                raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"Invalid TOML in {resolved}: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"Cannot read config file {resolved}: {exc}") from exc

        project_block = raw.get("project", {}) if isinstance(raw, dict) else {}
        if not isinstance(project_block, dict):
            raise ConfigError(f"[project] table in {resolved} must be a table")

        for key, value in project_block.items():
            if key not in DEFAULTS:
                raise ConfigError(f"Unknown config key {key!r} in {resolved}")
            if isinstance(value, str):
                candidate_path = Path(value)
                if (
                    key in {"data_root", "runs_dir", "log_file"}
                    and not candidate_path.is_absolute()
                ):
                    value = (file_dir / candidate_path).as_posix()
            source.file_values[key] = value

    source.env_values = _env_overrides()
    try:
        config = LapisConfig(**source.merged())
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigError(f"Invalid configuration: {exc}") from exc
    return config, source


def save_config(config: LapisConfig, path: str | Path) -> None:
    """Persist configuration to a TOML file without secrets (there are none)."""
    output = Path(path)
    try:
        data = tomli_w.dumps({"project": config.model_dump_paths()})
    except Exception as exc:  # pragma: no cover - tomli_w failure
        raise ConfigError(f"Cannot serialize configuration: {exc}") from exc
    try:
        output.write_text(data, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot write config file {output}: {exc}") from exc
