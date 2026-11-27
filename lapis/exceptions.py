"""LapisLLM error hierarchy.

Every error raised by the project derives from :class:`LapisError`. Errors carry
plain, actionable messages and never embed credentials, secrets or personal
data.
"""

from __future__ import annotations


class LapisError(Exception):
    """Base exception for all LapisLLM failures."""


class ConfigError(LapisError):
    """Invalid, missing or incompatible configuration."""


class HardwareError(LapisError):
    """The requested hardware capability is unavailable or misreported."""


class DeviceUnavailableError(HardwareError):
    """A requested accelerator device cannot be used."""


class PrecisionUnavailableError(HardwareError):
    """A requested numeric precision is not supported on the selected device."""


class CheckpointError(LapisError):
    """Checkpoint save, validation or restore failed."""


class HardwareAuditError(HardwareError):
    """Hardware diagnostics could not be collected for a subsystem."""


class TokenizerError(LapisError):
    """Tokenizer training, encoding or compatibility failure."""


class DataError(LapisError):
    """Dataset ingestion, validation or provenance failure."""


class TrainingError(LapisError):
    """Training engine failure."""


class InferenceError(LapisError):
    """Local inference failure."""


__all__ = [
    "CheckpointError",
    "ConfigError",
    "DataError",
    "DeviceUnavailableError",
    "HardwareAuditError",
    "HardwareError",
    "InferenceError",
    "LapisError",
    "PrecisionUnavailableError",
    "TokenizerError",
    "TrainingError",
]
