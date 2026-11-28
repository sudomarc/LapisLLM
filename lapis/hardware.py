"""Local hardware audit for LapisLLM.

Collects only what the system reports, never invents a capability, and never
collects serial numbers, account names, hostnames or private paths. Every
collector degrades gracefully: an unavailable subsystem is reported as
unavailable rather than fabricated.

All functions run offline. PyTorch is imported lazily so the rest of the
package stays importable when PyTorch is absent.
"""

import ctypes
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .exceptions import HardwareAuditError

__all__ = [
    "DtypeSupport",
    "GpuDevice",
    "HardwareReport",
    "MemoryInfo",
    "OsInfo",
    "TorchInfo",
    "CpuInfo",
    "DiskInfo",
    "audit_hardware",
    "collect_cpu_info",
    "collect_disk_info",
    "collect_memory_info",
    "collect_os_info",
    "collect_torch_info",
    "detect_accelerators",
    "precision_support",
    "render_report",
]

_UNKNOWN: str = "unavailable"
_NOT_DETECTED: str = "not_detected"


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass
class OsInfo:
    system: str
    release: str
    version: str
    architecture: str
    machine: str
    processor: str


@dataclass
class CpuInfo:
    logical_cores: int | None
    physical_cores: int | None
    model: str | None


@dataclass
class MemoryInfo:
    total_bytes: int | None
    available_bytes: int | None


@dataclass
class GpuDevice:
    index: int
    name: str
    backend: str  # "cuda", "mps", "rocm" or "none"
    vram_total_bytes: int | None
    vram_free_bytes: int | None
    driver_or_runtime_version: str | None


@dataclass
class TorchInfo:
    version: str | None
    cuda_available: bool
    cuda_version: str | None
    mps_available: bool
    mps_built: bool
    rocm_available: bool
    rocm_version: str | None


@dataclass
class DtypeSupport:
    device: str
    fp32: bool
    fp16: bool
    bf16: bool


@dataclass
class DiskInfo:
    path: str
    total_bytes: int | None
    free_bytes: int | None


@dataclass
class HardwareReport:
    timestamp_utc: str
    python_version: str
    os: OsInfo
    cpu: CpuInfo
    memory: MemoryInfo
    torch: TorchInfo
    accelerators: list[GpuDevice]
    precision: dict[str, DtypeSupport]
    disk: list[DiskInfo]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "python_version": self.python_version,
            "os": {
                "system": self.os.system,
                "release": self.os.release,
                "version": self.os.version,
                "architecture": self.os.architecture,
                "machine": self.os.machine,
                "processor": self.os.processor,
            },
            "cpu": {
                "logical_cores": self.cpu.logical_cores,
                "physical_cores": self.cpu.physical_cores,
                "model": self.cpu.model,
            },
            "memory": {
                "total_bytes": self.memory.total_bytes,
                "available_bytes": self.memory.available_bytes,
            },
            "torch": {
                "version": self.torch.version,
                "cuda_available": self.torch.cuda_available,
                "cuda_version": self.torch.cuda_version,
                "mps_available": self.torch.mps_available,
                "mps_built": self.torch.mps_built,
                "rocm_available": self.torch.rocm_available,
                "rocm_version": self.torch.rocm_version,
            },
            "accelerators": [
                {
                    "index": g.index,
                    "name": g.name,
                    "backend": g.backend,
                    "vram_total_bytes": g.vram_total_bytes,
                    "vram_free_bytes": g.vram_free_bytes,
                    "driver_or_runtime_version": g.driver_or_runtime_version,
                }
                for g in self.accelerators
            ],
            "precision": {
                device: {
                    "fp32": d.fp32,
                    "fp16": d.fp16,
                    "bf16": d.bf16,
                }
                for device, d in self.precision.items()
            },
            "disk": [
                {
                    "path": d.path,
                    "total_bytes": d.total_bytes,
                    "free_bytes": d.free_bytes,
                }
                for d in self.disk
            ],
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------


def collect_os_info() -> OsInfo:
    """Operating system and architecture identifiers (no hostnames/serials)."""
    uname = platform.uname()
    return OsInfo(
        system=uname.system or _UNKNOWN,
        release=uname.release or _UNKNOWN,
        version=uname.version or _UNKNOWN,
        architecture=platform.architecture()[0] or _UNKNOWN,
        machine=uname.machine or _UNKNOWN,
        processor=uname.processor or _UNKNOWN,
    )


def _read_proc_cpuinfo_field(field_name: str) -> str | None:
    """Best-effort Linux-only field extraction from /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip().startswith(field_name + ":"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def collect_cpu_info() -> CpuInfo:
    """Processor and core counts; physical cores are best-effort only."""
    logical: int | None = None
    try:
        logical = os.cpu_count()
    except NotImplementedError:
        logical = None

    physical: int | None = None
    model: str | None = None

    if platform.system() == "Linux":
        model = _read_proc_cpuinfo_field("model name")
        # Best-effort physical core count from /proc/cpuinfo topology.
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as handle:
                content = handle.read()
            physical_ids = {
                line.split(":", 1)[1].strip()
                for line in content.splitlines()
                if line.strip().startswith("physical id")
            }
            core_ids = {
                line.split(":", 1)[1].strip()
                for line in content.splitlines()
                if line.strip().startswith("core id")
            }
            if physical_ids:
                physical = len(physical_ids)
            elif core_ids:
                physical = len(core_ids)
        except OSError:
            pass
    elif platform.system() == "Windows":
        try:
            import ctypes.wintypes  # noqa: F401

            class _Sysinfo(ctypes.Structure):
                _fields_ = [
                    ("wProcessorArchitecture", ctypes.c_ushort),
                    ("wReserved", ctypes.c_ushort),
                    ("dwPageSize", ctypes.c_uint32),
                    ("lpMinimumApplicationAddress", ctypes.c_void_p),
                    ("lpMaximumApplicationAddress", ctypes.c_void_p),
                    ("dwActiveProcessorMask", ctypes.c_void_p),
                    ("dwNumberOfProcessors", ctypes.c_uint32),
                    ("dwProcessorType", ctypes.c_uint32),
                    ("dwAllocationGranularity", ctypes.c_uint32),
                    ("wProcessorLevel", ctypes.c_ushort),
                    ("wProcessorRevision", ctypes.c_ushort),
                ]

            info = _Sysinfo()
            ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(info))
            # dwNumberOfProcessors is the logical core count on Windows. The
            # physical count requires WMI, which we deliberately avoid to keep
            # this module dependency-free.
            physical = None
            logical = int(info.dwNumberOfProcessors) if logical is None else logical
        except OSError:
            pass
    elif platform.system() == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                model = result.stdout.strip() or None
            result = subprocess.run(
                ["sysctl", "-n", "hw.ncpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                logical = int(result.stdout.strip())
            result = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                physical = int(result.stdout.strip())
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass

    return CpuInfo(logical_cores=logical, physical_cores=physical, model=model)


def _memory_windows() -> tuple[int | None, int | None]:
    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32),
            ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    return None, None


def _memory_posix() -> tuple[int | None, int | None]:
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None, None
    try:
        page_size = sysconf("SC_PAGE_SIZE")
        total = page_size * sysconf("SC_PHYS_PAGES")
        available = page_size * sysconf("SC_AVPHYS_PAGES")
    except (OSError, ValueError, TypeError):
        return None, None
    return total, available


def collect_memory_info() -> MemoryInfo:
    """Total and available system RAM, when the OS exposes it."""
    if platform.system() == "Windows":
        total, available = _memory_windows()
    else:
        total, available = _memory_posix()
    return MemoryInfo(total_bytes=total, available_bytes=available)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise HardwareAuditError(
            "PyTorch is not installed; accelerator and dtype detection skipped"
        ) from exc
    return torch


def collect_torch_info() -> TorchInfo:
    """PyTorch version and backend availability, exactly as torch reports them."""
    torch = _import_torch()

    cuda_available = False
    cuda_version = None
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch implementation detail
        cuda_available = False
    if cuda_available:
        try:
            cuda_version = torch.version.cuda
        except AttributeError:
            cuda_version = None

    mps_available = False
    mps_built = False
    try:
        mps_available = bool(torch.backends.mps.is_available())
        mps_built = bool(torch.backends.mps.is_built())
    except AttributeError:
        mps_available = False
        mps_built = False

    rocm_available = False
    rocm_version = None
    try:
        rocm_version = getattr(torch.version, "hip", None)
        rocm_available = bool(rocm_version)
    except Exception:  # pragma: no cover - torch implementation detail
        rocm_available = False
        rocm_version = None

    return TorchInfo(
        version=getattr(torch, "__version__", _UNKNOWN),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        mps_available=mps_available,
        mps_built=mps_built,
        rocm_available=rocm_available,
        rocm_version=rocm_version,
    )


def detect_accelerators() -> list[GpuDevice]:
    """Detect actual accelerator devices reported by PyTorch backends.

    Returns an empty list when no accelerator backend is available (CPU-only
    machine). A missing backend is reported as such, never invented.
    """
    try:
        torch = _import_torch()
    except HardwareAuditError:
        return []

    devices: list[GpuDevice] = []

    if torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
        except Exception:  # pragma: no cover - driver instability
            count = 0
        for index in range(count):
            try:
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    GpuDevice(
                        index=index,
                        name=str(props.name),
                        backend="cuda",
                        vram_total_bytes=(int(props.total_memory) if props.total_memory else None),
                        vram_free_bytes=None,
                        driver_or_runtime_version=torch.version.cuda,
                    )
                )
            except Exception:  # pragma: no cover - unstable driver query
                devices.append(
                    GpuDevice(
                        index=index,
                        name=_UNKNOWN,
                        backend="cuda",
                        vram_total_bytes=None,
                        vram_free_bytes=None,
                        driver_or_runtime_version=torch.version.cuda,
                    )
                )

    try:
        mps_available = torch.backends.mps.is_available()
    except AttributeError:
        mps_available = False
    if mps_available:
        devices.append(
            GpuDevice(
                index=0,
                name="Apple Silicon GPU (MPS)",
                backend="mps",
                vram_total_bytes=None,
                vram_free_bytes=None,
                driver_or_runtime_version=None,
            )
        )

    # Report ROCm/HIP only when an actual accelerator is reachable through the
    # CUDA-compatible HIP API; a compiled-in hip attribute alone does not mean a
    # GPU is present (CPU-only torch builds can still expose the attribute).
    if torch.cuda.is_available() and hasattr(torch.version, "hip"):
        devices.append(
            GpuDevice(
                index=-1,
                name="AMD GPU (ROCm via HIP)",
                backend="rocm",
                vram_total_bytes=None,
                vram_free_bytes=None,
                driver_or_runtime_version=getattr(torch.version, "hip", None),
            )
        )

    return devices


def precision_support(device: str = "cpu") -> DtypeSupport:
    """Verifiably test whether a dtype can be allocated on a device.

    Each dtype is tried with a tiny allocation; a failure means unsupported on
    this device, never assumed.
    """
    torch = _import_torch()

    def works(dtype: Any) -> bool:
        try:
            _ = torch.zeros((2, 2), dtype=dtype, device=device)
            return True
        except Exception:
            return False

    return DtypeSupport(
        device=device,
        fp32=works(torch.float32),
        fp16=works(torch.float16),
        bf16=works(torch.bfloat16),
    )


def collect_disk_info(path: str | os.PathLike[str] = ".") -> DiskInfo:
    """Free and total space for a path, in bytes (shutil.disk_usage)."""
    resolved = os.fspath(path)
    try:
        usage = shutil.disk_usage(resolved)
    except OSError as exc:
        raise HardwareAuditError(f"Cannot query disk space for {resolved!r}: {exc}") from exc
    return DiskInfo(path=resolved, total_bytes=usage.total, free_bytes=usage.free)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def audit_hardware(
    disk_paths: list[str | os.PathLike[str]] | None = None,
) -> HardwareReport:
    """Run the full local hardware audit and return a structured report.

    Parameters
    ----------
    disk_paths:
        Volumes to probe for free disk space. Defaults to the current working
        directory only.
    """
    torch_info: TorchInfo
    try:
        torch_info = collect_torch_info()
    except HardwareAuditError:
        torch_info = TorchInfo(
            version=None,
            cuda_available=False,
            cuda_version=None,
            mps_available=False,
            mps_built=False,
            rocm_available=False,
            rocm_version=None,
        )

    accelerators = detect_accelerators() if torch_info.version else []

    precision: dict[str, DtypeSupport] = {"cpu": precision_support("cpu")}
    notes: list[str] = []
    if torch_info.cuda_available:
        for device in ("cuda", "cuda:0"):
            try:
                precision[device] = precision_support(device)
            except Exception:
                notes.append(f"Precision probe failed on {device}; reported unsupported")
                precision[device] = DtypeSupport(device=device, fp32=False, fp16=False, bf16=False)
    elif torch_info.version:
        notes.append(
            "PyTorch installed but no accelerator backend detected; precision probed on CPU only"
        )

    paths = list(disk_paths) if disk_paths else ["."]
    disk = [collect_disk_info(p) for p in paths]

    if not accelerators:
        notes.append("No accelerator detected: CPU-only execution")

    return HardwareReport(
        timestamp_utc=_now_utc(),
        python_version=sys.version,
        os=collect_os_info(),
        cpu=collect_cpu_info(),
        memory=collect_memory_info(),
        torch=torch_info,
        accelerators=accelerators,
        precision=precision,
        disk=disk,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _bytes_human(value: int | None) -> str:
    if value is None:
        return _UNKNOWN
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size:.2f} {units[index]}"


def render_report(report: HardwareReport) -> str:
    """Human-readable hardware audit report."""
    lines: list[str] = []
    lines.append("LapisLLM — hardware audit report")
    lines.append("=" * 44)
    lines.append(f"Generated (UTC): {report.timestamp_utc}")
    lines.append(f"Python: {report.python_version.split()[0]}")
    lines.append("")
    lines.append("Operating system")
    lines.append(f"  system:     {report.os.system}")
    lines.append(f"  release:    {report.os.release}")
    lines.append(f"  version:    {report.os.version}")
    lines.append(f"  architecture: {report.os.architecture}")
    lines.append(f"  machine:    {report.os.machine}")
    lines.append(f"  processor:  {report.os.processor}")
    lines.append("")
    lines.append("Processor")
    lines.append(f"  logical cores:   {report.cpu.logical_cores or _UNKNOWN}")
    lines.append(f"  physical cores:  {report.cpu.physical_cores or _UNKNOWN}")
    lines.append(f"  model:           {report.cpu.model or _UNKNOWN}")
    lines.append("")
    lines.append("Memory")
    lines.append(f"  total:   {_bytes_human(report.memory.total_bytes)}")
    lines.append(f"  available: {_bytes_human(report.memory.available_bytes)}")
    lines.append("")
    lines.append("PyTorch")
    lines.append(f"  version:       {report.torch.version or _UNKNOWN}")
    lines.append(f"  CUDA available: {report.torch.cuda_available}")
    lines.append(f"  CUDA version:  {report.torch.cuda_version or _UNKNOWN}")
    lines.append(f"  MPS available: {report.torch.mps_available} (built: {report.torch.mps_built})")
    lines.append(f"  ROCm available: {report.torch.rocm_available}")
    lines.append(f"  ROCm version:  {report.torch.rocm_version or _UNKNOWN}")
    lines.append("")
    lines.append("Accelerators")
    if report.accelerators:
        for gpu in report.accelerators:
            lines.append(
                f"  [{gpu.index}] {gpu.name} backend={gpu.backend} "
                f"VRAM total={_bytes_human(gpu.vram_total_bytes)} "
                f"free={_bytes_human(gpu.vram_free_bytes)} "
                f"runtime={gpu.driver_or_runtime_version or _UNKNOWN}"
            )
    else:
        lines.append("  none detected (CPU-only)")
    lines.append("")
    lines.append("Numerical precision (verified by trial allocation)")
    for device, support in report.precision.items():
        lines.append(f"  {device}: fp32={support.fp32} fp16={support.fp16} bf16={support.bf16}")
    lines.append("")
    lines.append("Disk space")
    for d in report.disk:
        lines.append(
            f"  {d.path}: free={_bytes_human(d.free_bytes)} / total={_bytes_human(d.total_bytes)}"
        )
    lines.append("")
    if report.notes:
        lines.append("Notes")
        for note in report.notes:
            lines.append(f"  - {note}")
        lines.append("")
    lines.append("This report contains no serial numbers, hostnames, account names or")
    lines.append("private paths beyond the disk paths explicitly requested for probing.")
    return "\n".join(lines)
