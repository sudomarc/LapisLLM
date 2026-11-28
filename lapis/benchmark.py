"""Bounded performance benchmark for LapisLLM.

Measures wall-clock time, tokens per second and peak memory for a small,
fixed number of iterations. This is a harness, not a model: callers pass a
model factory and an input factory. It is intentionally safe to run in CI and
on the maintainer's PC — no long runs, no gradient checkpointing, no data
downloads.

Peak memory is measured per-backend: CUDA allocated memory when available,
otherwise the tracemalloc delta of the Python process.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from .exceptions import HardwareError

__all__ = ["BenchmarkResult", "benchmark_forward_backward"]


@dataclass
class BenchmarkResult:
    device: str
    dtype: str
    batch_size: int
    sequence_length: int
    vocab_size: int
    iterations: int
    wall_time_seconds: float
    tokens_per_second: float
    peak_memory_bytes: int | None
    peak_memory_human: str
    notes: list[str]


def _bytes_human(value: int | None) -> str:
    if value is None:
        return "unavailable"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size:.2f} {units[index]}"


def benchmark_forward_backward(
    model_factory: Callable[[], object],
    input_factory: Callable[[int], dict[str, object]],
    *,
    device: str = "cpu",
    dtype: str = "fp32",
    batch_size: int = 2,
    sequence_length: int = 64,
    vocab_size: int = 512,
    iterations: int = 10,
    warmup: int = 2,
) -> BenchmarkResult:
    """Run a bounded forward/backward benchmark.

    Parameters
    ----------
    model_factory:
        Callable returning a model in train mode. The model must accept the
        mapping returned by ``input_factory`` and return an object with either
        a ``loss`` tensor attribute (first positional return) or a ``loss``
        attribute.
    input_factory:
        Callable(batch_size) returning model inputs for one step.
    iterations:
        Number of measured steps (bounded, small by default).
    warmup:
        Number of unmeasured steps before measuring.
    """
    import torch

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if dtype not in dtype_map:
        raise HardwareError(f"Unsupported dtype {dtype!r}; expected one of {sorted(dtype_map)}")
    torch_dtype = dtype_map[dtype]

    # Validate the device by actually allocating on it: torch.device()
    # never raises for invalid device names.
    try:
        _ = torch.zeros((1,), device=device)
    except Exception as exc:
        raise HardwareError(f"Invalid device {device!r}: {exc}") from exc

    model = cast(Any, model_factory()).to(device=device, dtype=torch_dtype)
    model.train()

    def _cast_inputs(inputs: dict[str, object]) -> dict[str, object]:
        """Move inputs to the device, but only cast floating-point tensors.

        Integer tensors (token ids, labels) must keep their integral dtype.
        """
        cast: dict[str, object] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    cast[key] = value.to(device=device, dtype=torch_dtype)
                else:
                    cast[key] = value.to(device=device)
            else:
                cast[key] = value
        return cast

    # Warm-up runs (not measured).
    for _ in range(max(warmup, 0)):
        inputs = _cast_inputs(input_factory(batch_size))
        output = model(**inputs)
        loss = output.loss if hasattr(output, "loss") else output[0]
        loss.backward()
        model.zero_grad(set_to_none=True)

    # Measured runs.
    total_tokens = 0
    peak_cpu = None

    tracemalloc.start()
    try:
        # CUDA event timing is unavailable on CPU-only builds.
        use_cuda_events = device == "cuda"
        start_events: torch.cuda.Event | None = (
            torch.cuda.Event(enable_timing=True) if use_cuda_events else None
        )
        end_events: torch.cuda.Event | None = (
            torch.cuda.Event(enable_timing=True) if use_cuda_events else None
        )

        if start_events is not None:
            start_events.record()

        wall_start = time.perf_counter()
        for _step in range(iterations):
            inputs = _cast_inputs(input_factory(batch_size))
            output = model(**inputs)
            loss = output.loss if hasattr(output, "loss") else output[0]
            loss.backward()
            model.zero_grad(set_to_none=True)
            total_tokens += batch_size * sequence_length

        wall_end = time.perf_counter()
        if end_events is not None:
            end_events.record()
            torch.cuda.synchronize(device)
        _, peak_cpu = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    if start_events is not None:
        cuda_peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        peak_memory_bytes = cuda_peak
    else:
        peak_memory_bytes = peak_cpu

    wall_time = wall_end - wall_start
    tokens_per_second = total_tokens / wall_time if wall_time > 0 else 0.0

    notes: list[str] = []
    if device == "cuda" and not torch.cuda.is_available():
        notes.append("CUDA requested but unavailable; measurements reflect CPU")

    return BenchmarkResult(
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        iterations=iterations,
        wall_time_seconds=wall_time,
        tokens_per_second=tokens_per_second,
        peak_memory_bytes=peak_memory_bytes,
        peak_memory_human=_bytes_human(peak_memory_bytes),
        notes=notes,
    )
