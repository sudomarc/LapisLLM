# Hardware detection for LapisLLM — works with or without PyTorch
import os
import sys
from typing import Literal, Optional


def detect_hardware() -> dict:
    """Detect available hardware and return a report. Works without PyTorch."""
    report = {
        "os": {
            "name": sys.platform,
            "uname": os.uname().sysname if hasattr(os, 'uname') else "unknown",
        },
        "cpu": {
            "model": os.uname().machine if hasattr(os, 'uname') else "unknown",
            "cores": os.cpu_count() or 1,
            "flags": [],
        },
        "memory_gb": 0,
        "gpu": None,
        "pytorch": {
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
        },
    }

    # Read total memory (MemTotal / 1024 / 1024 for GB)
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                    report["memory_gb"] = round(total_kb / 1024 / 1024, 1)
                    break
    except (FileNotFoundError, ValueError):
        pass

    # Check PyTorch and CUDA (optional)
    try:
        import torch
        _HAS_TORCH = True
        report["pytorch"]["version"] = torch.__version__
        report["pytorch"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["pytorch"]["cuda_version"] = torch.version.cuda
            report["pytorch"]["device_count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                gpu_name = torch.cuda.get_device_name(0)
                report["pytorch"]["gpu_name"] = gpu_name
                report["pytorch"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
                report["gpu"] = {
                    "name": gpu_name,
                    "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024,
                }
    except ImportError:
        _HAS_TORCH = False
        # PyTorch not available — report stays with defaults (CPU-only)
        pass

    # Set device based on availability
    if _HAS_TORCH and report["pytorch"]["cuda_available"] and report["pytorch"]["device_count"] > 0:
        report["device"] = "cuda"
        report["device_index"] = 0
    else:
        report["device"] = "cpu"

    return report


def get_device() -> Literal["cpu", "cuda"]:
    """Return the recommended compute device."""
    report = detect_hardware()
    return report["device"]