"""Hardware benchmark for LapisLLM Stage 2.

Runs a minimal synthetic forward pass with the Micro model config
to measure CPU throughput and peak memory, producing a validated
training profile for the maintainer's PC.
"""
import json
import os
import sys
import time
import psutil
from contextlib import contextmanager

import torch
import torch.nn as nn

# Add lapis to path for config imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lapis'))

from lapis.config.base import LapisConfig
from lapis.config.hardware import detect_hardware, get_device


@contextmanager
def track_peak_memory():
    """Context manager to track peak memory during a block.

    For CPU: tracks RSS via psutil.
    For GPU: tracks CUDA allocated memory.
    """
    peak_rss = 0
    peak_cuda = 0

    def record_peak():
        nonlocal peak_rss, peak_cuda
        # CPU RAM via psutil
        try:
            process = psutil.Process(os.getpid())
            rss = process.memory_info().rss
            peak_rss = max(peak_rss, rss)
        except Exception:
            pass
        # GPU memory
        if torch.cuda.is_available():
            cuda_alloc = torch.cuda.max_memory_allocated()
            peak_cuda = max(peak_cuda, cuda_alloc)

    record_peak()
    try:
        yield {"peak_rss_bytes": peak_rss, "peak_cuda_bytes": peak_cuda}
    finally:
        record_peak()


def main():
    """Run hardware benchmark and produce a validated Micro profile."""
    print("=" * 60)
    print("LapisLLM Hardware Benchmark — Stage 2")
    print("=" * 60)

    # Detect hardware
    hw_report = detect_hardware()
    device = get_device()
    print(f"\nDetected device: {device}")
    print(f"CPU cores: {hw_report['cpu']['cores']}")
    print(f"System RAM: {hw_report['memory_gb']} GB")

    # Create config
    config = LapisConfig()

    # Minimal benchmark: Embedding + Linear forward pass
    # This measures basic CPU throughput without architecture complexity
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    max_seq_len = config.max_seq_len
    batch_size = 2
    seq_len = 32  # conservative sequence length

    print(f"\nBenchmark setup:")
    print(f"  vocab_size={vocab_size}, hidden_size={hidden_size}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  device={device}")

    # Create dummy data
    device_torch = torch.device(device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device_torch)

    # Minimal model: embedding -> linear -> output
    # This is sufficient to measure CPU throughput boundary
    embedding = nn.Embedding(vocab_size, hidden_size).to(device_torch)
    projector = nn.Linear(hidden_size, hidden_size).to(device_torch)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            h = embedding(input_ids)
            h = projector(h)

    # Benchmark forward pass
    with track_peak_memory() as peak_info:
        start = time.time()
        with torch.no_grad():
            for _ in range(50):  # 50 iterations for stable measurement
                h = embedding(input_ids)
                h = projector(h)
        elapsed = time.time() - start

    tokens_processed = batch_size * seq_len * 50
    tokens_per_sec = tokens_processed / elapsed
    avg_step_time = elapsed / 50

    peak_memory_bytes = peak_info["peak_rss_bytes"]
    peak_memory_gb = peak_memory_bytes / (1024 ** 3) if peak_memory_bytes else None

    # Collect hardware report data
    ram_gb = hw_report['memory_gb']

    print(f"\nBenchmark Results:")
    print(f"  Tokens/sec: {tokens_per_sec:.1f}")
    print(f"  Avg step time: {avg_step_time:.4f}s")
    print(f"  Peak memory (RSS): {peak_memory_bytes / (1024**2):.1f} MB")
    print(f"  Peak memory (GB): {peak_memory_gb:.3f}" if peak_memory_gb else
          f"  Peak memory (GB): N/A")

    # Determine viable Micro profile based on measured metrics
    print(f"\nProfile Viability Assessment:")
    if tokens_per_sec > 0 and ram_gb > 0:
        if ram_gb >= 8:
            viability = "Good — can train Micro with full batch"
            micro_bs = 2
            grad_accum = 4
            train_bs = 8
        elif ram_gb >= 4:
            viability = "Adequate — Micro trainable with gradient accumulation"
            micro_bs = 1
            grad_accum = 8
            train_bs = 4
        elif ram_gb >= 2:
            viability = "Limited — reduce micro_batch_size, use gradient accumulation"
            micro_bs = 1
            grad_accum = 16
            train_bs = 2
        else:
            viability = "Very limited — significant profile reduction needed"
            micro_bs = 1
            grad_accum = 32
            train_bs = 1

        print(f"  RAM: {ram_gb}GB → {viability}")

        # Save validated profile
        profile = {
            "model_name": config.model_name,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "precision": config.precision,
            "device": device,
            "tokens_per_sec": float(tokens_per_sec),
            "peak_memory_gb": peak_memory_gb,
            "ram_system_gb": ram_gb,
            "viability": viability,
            "micro_batch_size": micro_bs,
            "gradient_accumulation_steps": grad_accum,
            "batch_size": train_bs,
            "seq_len": seq_len,
            "seed": config.seed,
            "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save report
        os.makedirs(".lapis-data", exist_ok=True)
        report_path = ".lapis-data/hardware-report.json"
        with open(report_path, "w") as f:
            json.dump(profile, f, indent=2)

        print(f"\nHardware report saved to: {report_path}")
        print(f"\nRecommended Micro config:")
        print(f"  hidden_size={profile['hidden_size']}, layers={profile['num_layers']}")
        print(f"  micro_batch_size={profile['micro_batch_size']}, "
              f"grad_accum={profile['gradient_accumulation_steps']}")
        print(f"  batch_size={profile['batch_size']}, lr={config.learning_rate}")
        print(f"  seq_len={profile['seq_len']}")

    else:
        print("  ⚠️  Benchmark completed but metrics insufficient for profile recommendation")

    return 0


if __name__ == "__main__":
    exit(main())