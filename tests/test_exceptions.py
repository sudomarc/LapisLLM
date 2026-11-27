"""Behavior tests for the LapisLLM error hierarchy."""

from __future__ import annotations

from lapis.exceptions import (
    CheckpointError,
    ConfigError,
    DataError,
    DeviceUnavailableError,
    HardwareAuditError,
    HardwareError,
    InferenceError,
    LapisError,
    PrecisionUnavailableError,
    TokenizerError,
    TrainingError,
)


def test_all_errors_derive_from_lapis_error() -> None:
    for exc_class in (
        CheckpointError,
        ConfigError,
        DataError,
        DeviceUnavailableError,
        HardwareAuditError,
        HardwareError,
        InferenceError,
        PrecisionUnavailableError,
        TokenizerError,
        TrainingError,
    ):
        assert issubclass(exc_class, LapisError)
        instance = exc_class("something went wrong")
        assert str(instance) == "something went wrong"
        assert isinstance(instance, Exception)


def test_device_and_precision_errors_are_hardware_errors() -> None:
    assert issubclass(DeviceUnavailableError, HardwareError)
    assert issubclass(PrecisionUnavailableError, HardwareError)
    assert issubclass(HardwareAuditError, HardwareError)
