"""Behavior tests for the bounded benchmark harness (issue #74)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lapis.benchmark import BenchmarkResult, benchmark_forward_backward  # noqa: E402
from lapis.exceptions import HardwareError  # noqa: E402


def _tiny_model_factory(vocab_size: int = 64, hidden: int = 16):
    import torch.nn as nn

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, hidden)
            self.head = nn.Linear(hidden, vocab_size)

        def forward(self, input_ids, labels=None):  # type: ignore[no-untyped-def]
            logits = self.head(self.emb(input_ids))
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )
            return type("Out", (), {"logits": logits, "loss": loss})()

    return TinyModel


def _input_factory(vocab_size: int = 64, sequence_length: int = 16):
    def factory(batch_size: int) -> dict:
        import torch

        ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
        return {"input_ids": ids, "labels": ids.clone()}

    return factory


def test_benchmark_runs_cpu_and_reports_throughput() -> None:
    result = benchmark_forward_backward(
        model_factory=_tiny_model_factory(),
        input_factory=_input_factory(),
        device="cpu",
        dtype="fp32",
        batch_size=2,
        sequence_length=16,
        vocab_size=64,
        iterations=4,
        warmup=1,
    )
    assert isinstance(result, BenchmarkResult)
    assert result.device == "cpu"
    assert result.iterations == 4
    assert result.wall_time_seconds > 0
    assert result.tokens_per_second > 0
    assert result.peak_memory_bytes is not None and result.peak_memory_bytes > 0


def test_benchmark_invalid_dtype_raises() -> None:
    model = _tiny_model_factory()()
    with pytest.raises(HardwareError, match="Unsupported dtype"):
        benchmark_forward_backward(
            model_factory=lambda: model,
            input_factory=_input_factory(),
            dtype="tf32",
        )


def test_benchmark_invalid_device_raises() -> None:
    model = _tiny_model_factory()()
    with pytest.raises(HardwareError, match="Invalid device"):
        benchmark_forward_backward(
            model_factory=lambda: model,
            input_factory=_input_factory(),
            device="cuda:99",
        )


def test_benchmark_counts_expected_tokens() -> None:
    result = benchmark_forward_backward(
        model_factory=_tiny_model_factory(),
        input_factory=_input_factory(),
        batch_size=3,
        sequence_length=10,
        iterations=5,
    )
    assert result.iterations == 5
