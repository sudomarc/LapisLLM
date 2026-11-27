"""``lapis doctor`` — local hardware audit and bounded benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from types import SimpleNamespace

import torch

from ..benchmark import BenchmarkResult, benchmark_forward_backward
from ..exceptions import HardwareError
from ..hardware import HardwareReport, audit_hardware, render_report
from ..logging import get_logger, setup_logging

logger = get_logger("cli.doctor")


class _TinyLanguageModel(torch.nn.Module):
    """Smallest causal-next-token model useful for benchmarking the harness.

    This is a throwaway smoke model; the real Transformer arrives in stage 5
    (issue #61). It is only here so the benchmark harness can be executed and
    tested before the model exists.
    """

    def __init__(self, vocab_size: int, hidden: int = 64) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden)
        self.proj = torch.nn.Linear(hidden, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> SimpleNamespace:
        hidden_states = self.proj(self.embedding(input_ids))
        logits = self.head(hidden_states)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        return SimpleNamespace(logits=logits, loss=loss)


def _benchmark_model_factory(
    vocab_size: int,
) -> Callable[[], _TinyLanguageModel]:
    def factory() -> _TinyLanguageModel:
        return _TinyLanguageModel(vocab_size=vocab_size)

    return factory


def _benchmark_input_factory(
    vocab_size: int, sequence_length: int
) -> Callable[[int], dict[str, object]]:
    def factory(batch_size: int) -> dict[str, object]:
        input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

    return factory


def _print_benchmark(result: BenchmarkResult) -> None:
    lines = [
        "Bounded benchmark (synthetic smoke model, not a trained LapisLLM)",
        "-" * 56,
        f"device:        {result.device}",
        f"dtype:         {result.dtype}",
        f"batch size:    {result.batch_size}",
        f"sequence:      {result.sequence_length}",
        f"vocab size:    {result.vocab_size}",
        f"iterations:    {result.iterations}",
        f"wall time:     {result.wall_time_seconds:.4f}s",
        f"tokens/sec:    {result.tokens_per_second:.2f}",
        f"peak memory:   {result.peak_memory_human}",
    ]
    if result.notes:
        lines.append("notes:")
        for note in result.notes:
            lines.append(f"  - {note}")
    logger.info("\n".join(lines))


def run_audit(as_json: bool = False) -> HardwareReport:
    """Collect and return the hardware audit report."""
    report = audit_hardware()
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render_report(report))
    return report


def run_benchmark(
    *,
    device: str = "cpu",
    dtype: str = "fp32",
    batch_size: int = 2,
    sequence_length: int = 64,
    vocab_size: int = 512,
    iterations: int = 10,
    warmup: int = 2,
) -> BenchmarkResult:
    result = benchmark_forward_backward(
        model_factory=_benchmark_model_factory(vocab_size),
        input_factory=_benchmark_input_factory(vocab_size, sequence_length),
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        iterations=iterations,
        warmup=warmup,
    )
    _print_benchmark(result)
    return result


def doctor_command(args: argparse.Namespace) -> int:
    setup_logging(level=args.log_level, log_file=args.log_file)
    try:
        if args.bench:
            run_benchmark(
                device=args.device,
                dtype=args.dtype,
                batch_size=args.batch,
                sequence_length=args.seq,
                vocab_size=args.vocab,
                iterations=args.iterations,
                warmup=args.warmup,
            )
        else:
            run_audit(as_json=args.json)
    except HardwareError as exc:
        logger.error(str(exc))
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lapis",
        description="LapisLLM — local development tools for the from-scratch coding model.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console log level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path (default: console only).",
    )

    sub = parser.add_subparsers(required=True)

    doctor = sub.add_parser("doctor", help="Run the local hardware audit.")
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit the hardware report as JSON instead of plain text.",
    )
    doctor.add_argument(
        "--benchmark",
        dest="bench",
        action="store_true",
        help="Also run a bounded synthetic benchmark (CPU, seconds-long).",
    )
    doctor.add_argument("--device", default="cpu", help="Benchmark device (default: cpu).")
    doctor.add_argument(
        "--dtype", default="fp32", choices=["fp32", "fp16", "bf16"], help="Benchmark dtype."
    )
    doctor.add_argument("--batch", type=int, default=2, help="Benchmark batch size.")
    doctor.add_argument("--seq", type=int, default=64, help="Benchmark sequence length.")
    doctor.add_argument("--vocab", type=int, default=512, help="Benchmark vocabulary size.")
    doctor.add_argument("--iterations", type=int, default=10, help="Benchmark iterations.")
    doctor.add_argument("--warmup", type=int, default=2, help="Benchmark warmup steps.")
    doctor.set_defaults(func=doctor_command)

    return parser
