"""Behavior tests for the local hardware audit (issue #74)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from lapis.exceptions import HardwareAuditError
from lapis.hardware import (  # noqa: E402
    HardwareReport,
    audit_hardware,
    collect_cpu_info,
    collect_disk_info,
    collect_memory_info,
    collect_os_info,
    collect_torch_info,
    detect_accelerators,
    precision_support,
    render_report,
)

torch = pytest.importorskip("torch")


def test_collect_os_info_identifies_platform() -> None:
    info = collect_os_info()
    assert info.system in {"Windows", "Linux", "Darwin"}
    assert info.release != "unavailable"
    assert info.machine != "unavailable"


def test_collect_cpu_info_reports_cores() -> None:
    info = collect_cpu_info()
    assert info.logical_cores is not None and info.logical_cores >= 1
    # Physical cores are best-effort: None is acceptable, but when reported it
    # must be sane.
    if info.physical_cores is not None:
        assert 1 <= info.physical_cores <= info.logical_cores or info.logical_cores == 1


def test_collect_memory_info_reports_ram() -> None:
    mem = collect_memory_info()
    assert mem.total_bytes is not None and mem.total_bytes > 0
    if mem.available_bytes is not None:
        assert 0 < mem.available_bytes <= mem.total_bytes


def test_collect_torch_info_matches_installed_torch() -> None:
    info = collect_torch_info()
    assert info.version == torch.__version__
    assert isinstance(info.cuda_available, bool)
    assert isinstance(info.mps_available, bool)
    assert isinstance(info.mps_built, bool)
    assert isinstance(info.rocm_available, bool)
    if info.cuda_available:
        assert info.cuda_version is not None
    if info.rocm_available:
        assert info.rocm_version is not None


def test_detect_accelerators_returns_device_list() -> None:
    devices = detect_accelerators()
    assert isinstance(devices, list)
    for device in devices:
        assert device.index >= -1
        assert device.backend in {"cuda", "mps", "rocm"}
        assert device.name != "unavailable" or device.backend == "cuda"


def test_precision_support_cpu_is_verifiable() -> None:
    support = precision_support("cpu")
    assert support.device == "cpu"
    assert support.fp32 is True  # float32 must work on CPU


def test_precision_support_unknown_device_is_false() -> None:
    support = precision_support("cuda:99")
    assert support.fp32 is False


def test_audit_hardware_returns_complete_report() -> None:
    report = audit_hardware()
    assert isinstance(report, HardwareReport)
    datetime.fromisoformat(report.timestamp_utc)  # valid UTC timestamp
    assert report.python_version
    assert report.os.system != "unavailable"
    assert report.cpu.logical_cores is not None
    assert report.memory.total_bytes is not None
    assert report.torch.version == torch.__version__
    assert isinstance(report.accelerators, list)
    assert "cpu" in report.precision
    assert report.precision["cpu"].fp32 is True
    assert isinstance(report.notes, list)


def test_audit_hardware_disk_paths(tmp_path: Path) -> None:
    report = audit_hardware(disk_paths=[tmp_path])
    assert len(report.disk) == 1
    assert report.disk[0].path == str(tmp_path)
    assert report.disk[0].total_bytes is not None and report.disk[0].total_bytes > 0
    assert report.disk[0].free_bytes is not None and report.disk[0].free_bytes >= 0


def test_collect_disk_info() -> None:
    disk = collect_disk_info(".")
    assert disk.total_bytes is not None and disk.total_bytes > 0
    assert disk.free_bytes is not None and disk.free_bytes >= 0


def test_render_report_contains_key_facts() -> None:
    report = audit_hardware()
    text = render_report(report)
    assert report.python_version.split()[0] in text
    assert report.os.system in text
    if report.accelerators:
        assert report.accelerators[0].name in text
    else:
        assert "CPU-only" in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_precision_support_cuda() -> None:
    support = precision_support("cuda:0")
    assert support.fp32 is True


def test_hardware_audit_never_invents_cuda_when_torch_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lapis.hardware as hardware_module

    def boom() -> object:
        raise HardwareAuditError("torch missing")

    monkeypatch.setattr(hardware_module, "_import_torch", boom)
    assert detect_accelerators() == []
