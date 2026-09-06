from __future__ import annotations

import os
from copy import deepcopy

import torch


CPU_FAST_OVERRIDES = {
    "model": {
        "hidden_size": 128,
        "intermediate_size": 512,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
    },
    "training": {
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_steps": 500,
    },
    "data": {"max_seq_length": 127},
}


def _positive_int(value, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def configure_cpu_runtime(config: dict, device: torch.device) -> dict:
    """Configure safe CPU runtime settings without changing model semantics."""
    runtime = config.setdefault("runtime", {})
    cpu = runtime.setdefault("cpu", {})

    threads = int(cpu.get("threads", 0) or 0)
    if threads < 0:
        raise ValueError("runtime.cpu.threads must be >= 0")
    if threads:
        torch.set_num_threads(threads)

    interop_threads = int(cpu.get("interop_threads", 0) or 0)
    if interop_threads < 0:
        raise ValueError("runtime.cpu.interop_threads must be >= 0")
    if interop_threads:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as exc:
            # PyTorch only allows this setting before parallel work has started.
            if "after parallel work" not in str(exc).lower():
                raise

    precision = str(runtime.get("matmul_precision", "high"))
    if precision in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(precision)

    if device.type == "cpu":
        cpu["effective_threads"] = torch.get_num_threads()
        cpu.setdefault("effective_interop_threads", torch.get_num_interop_threads())
        cpu.setdefault("recommended", os.cpu_count() or 1)

    return config


def apply_cpu_fast_profile(config: dict) -> dict:
    """Return a copy of config tuned for quick CPU iteration.

    This profile intentionally changes model size and sequence length. It must not
    be used to resume a checkpoint produced by another architecture.
    """
    tuned = deepcopy(config)
    for section, values in CPU_FAST_OVERRIDES.items():
        tuned.setdefault(section, {}).update(values)
    tuned.setdefault("runtime", {})["device"] = "cpu"
    tuned.setdefault("runtime", {}).setdefault("dtype", "float32")
    tuned.setdefault("runtime", {}).setdefault("matmul_precision", "high")
    tuned.setdefault("runtime", {}).setdefault("cpu", {})
    tuned["runtime"]["cpu"].setdefault("threads", 0)
    tuned["runtime"]["cpu"].setdefault("interop_threads", 0)
    tuned["runtime"]["cpu"].setdefault("dataloader_workers", 0)
    tuned["runtime"]["cpu"].setdefault("pin_memory", False)
    return tuned


def get_dataloader_options(config: dict, device: torch.device) -> dict:
    """Build DataLoader options that are safe for the selected device."""
    runtime = config.get("runtime", {})
    cpu = runtime.get("cpu", {})
    workers = int(cpu.get("dataloader_workers", 0) or 0) if device.type == "cpu" else 0
    if workers < 0:
        raise ValueError("runtime.cpu.dataloader_workers must be >= 0")

    options = {
        "num_workers": workers,
        "pin_memory": bool(cpu.get("pin_memory", False)) if device.type == "cpu" else device.type == "cuda",
    }
    if workers > 0:
        options["persistent_workers"] = bool(cpu.get("persistent_workers", True))
    return options


def validate_fast_profile(config: dict) -> None:
    """Validate the CPU-fast shape before model construction."""
    model = config.get("model", {})
    hidden = _positive_int(model.get("hidden_size", 0), "hidden_size")
    heads = _positive_int(model.get("num_attention_heads", 0), "num_attention_heads")
    kv_heads = _positive_int(model.get("num_key_value_heads", 0), "num_key_value_heads")
    if hidden % heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if heads % kv_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
