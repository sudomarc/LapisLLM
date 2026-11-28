"""Hardware detection tests for LapisLLM Stage 1."""
import pytest
from lapis.config.hardware import detect_hardware, get_device


def test_hardware_detection_returns_dict():
    """Hardware detection should return a dict with expected keys."""
    report = detect_hardware()
    assert isinstance(report, dict)
    assert "os" in report
    assert "cpu" in report
    assert "memory_gb" in report
    assert "pytorch" in report
    assert "device" in report


def test_hardware_has_cpu_info():
    """CPU info should be present in the report."""
    report = detect_hardware()
    assert report["cpu"]["cores"] is not None
    assert report["cpu"]["cores"] > 0


def test_device_is_cpu_when_no_cuda():
    """When no CUDA, device should be CPU."""
    device = get_device()
    assert device == "cpu"


def test_pytorch_version_is_string():
    """PyTorch version should be a string if available."""
    report = detect_hardware()
    if report["pytorch"]["version"]:
        assert isinstance(report["pytorch"]["version"], str)