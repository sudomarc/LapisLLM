"""Developer command namespace for LapisLLM."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.table import Table

from lapis.inference.runtime import LapisRuntime, SamplingConfig

app = typer.Typer(name="dev", help="Developer and research commands.")
console = Console()
REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(module: str, *args: str) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run([sys.executable, "-u", "-m", module, *args], cwd=REPO_ROOT, env=env)
    raise typer.Exit(result.returncode)


@app.command("train")
def train(
    config: Path = typer.Option(Path("configs/local-dev.yaml"), "--config", "-c"),
    device: str | None = typer.Option(None, "--device"),
    data: Path | None = typer.Option(None, "--data"),
    checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"),
    epochs: int = typer.Option(1000, "--epochs", min=1),
    monitor_interval: int = typer.Option(500, "--monitor-interval", min=0),
    monitor_sample_tokens: int = typer.Option(48, "--monitor-sample-tokens", min=1),
    monitor_prompts: str | None = typer.Option(None, "--monitor-prompts"),
    monitor_log: Path | None = typer.Option(None, "--monitor-log"),
    resume: Path | None = typer.Option(None, "--resume"),
) -> None:
    args = ["--config", str(config), "--epochs", str(epochs), "--checkpoint", str(checkpoint), "--monitor-interval", str(monitor_interval), "--monitor-sample-tokens", str(monitor_sample_tokens)]
    if device:
        args += ["--device", device]
    if data:
        args += ["--data", str(data)]
    if monitor_prompts:
        args += ["--monitor-prompts", monitor_prompts]
    if monitor_log:
        args += ["--monitor-log", str(monitor_log)]
    if resume:
        args += ["--resume", str(resume)]
    _run("scripts.train", *args)


@app.command("evaluate")
def evaluate(checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"), device: str = typer.Option("auto", "--device")) -> None:
    _run("scripts.evaluate", "--checkpoint", str(checkpoint), "--device", device)


@app.command("generate")
def generate(
    prompt: str = typer.Argument(...),
    checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens", min=1),
    temperature: float = typer.Option(0.8, "--temperature", min=0.01),
    top_k: int = typer.Option(40, "--top-k", min=0),
    top_p: float = typer.Option(0.95, "--top-p", min=0.01, max=1.0),
    seed: int | None = typer.Option(None, "--seed"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    args = [
        "--checkpoint",
        str(checkpoint),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--top-p",
        str(top_p),
        "--device",
        device,
    ]
    if seed is not None:
        args += ["--seed", str(seed)]
    _run("scripts.generate", *args)


@app.command("chat")
def chat(
    checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"),
    device: str = typer.Option("auto", "--device"),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens", min=1),
    temperature: float = typer.Option(0.8, "--temperature", min=0.01),
    top_k: int = typer.Option(40, "--top-k", min=0),
    top_p: float = typer.Option(0.95, "--top-p", min=0.01),
    no_color: bool = typer.Option(False, "--no-color"),
) -> None:
    """Start the developer inference/testing console."""
    args = ["--checkpoint", str(checkpoint), "--device", device, "--max-new-tokens", str(max_new_tokens), "--temperature", str(temperature), "--top-k", str(top_k), "--top-p", str(top_p)]
    if no_color:
        args.append("--no-color")
    _run("scripts.chat", *args)


@app.command("inspect")
def inspect(checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint")) -> None:
    """Inspect checkpoint metadata without modifying it or starting training."""
    if not checkpoint.is_file():
        raise typer.BadParameter(f"Checkpoint not found: {checkpoint}")
    try:
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise typer.BadParameter(f"Unable to read checkpoint: {exc}") from exc
    config = data.get("config", {})
    model = config.get("model", {})
    table = Table(title=f"Checkpoint: {checkpoint}")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in (("step", data.get("step", "—")), ("epoch", data.get("epoch", "—")), ("tokenizer_version", data.get("tokenizer_version", "—")), ("vocab_size", model.get("vocab_size", "—")), ("layers", model.get("num_layers", "—")), ("hidden_size", model.get("hidden_size", "—"))):
        table.add_row(key, str(value))
    console.print(table)


@app.command("benchmark")
def benchmark(
    checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"),
    prompt: str = typer.Option("The future of computing is", "--prompt"),
    tokens: int = typer.Option(32, "--tokens", min=1),
    device: str = typer.Option("auto", "--device"),
    warmup: int = typer.Option(1, "--warmup", min=0),
    runs: int = typer.Option(5, "--runs", min=1),
    seed: int | None = typer.Option(None, "--seed"),
    quantize: bool = typer.Option(False, "--quantize"),
) -> None:
    """Measure inference latency and throughput without changing the checkpoint."""
    runtime = LapisRuntime.from_checkpoint(checkpoint, device, quantize=quantize)
    sampling = SamplingConfig(max_new_tokens=tokens, seed=seed)

    for _ in range(warmup):
        runtime.generate(prompt, sampling)

    latencies_ms: list[float] = []
    ttfts_ms: list[float] = []
    rates: list[float] = []
    sample_text = ""

    for _ in range(runs):
        meta = runtime.generate_with_metadata(prompt, sampling)
        sample_text = meta["text"]
        latencies_ms.append(meta["total_time_ms"])
        ttfts_ms.append(meta["time_to_first_token_ms"])
        rates.append(meta["tokens_per_second"])

    latencies_ms.sort()
    ttfts_ms.sort()
    rates.sort()

    p50_latency = latencies_ms[len(latencies_ms) // 2]
    p95_index = min(int(len(latencies_ms) * 0.95), len(latencies_ms) - 1)
    p95_latency = latencies_ms[p95_index]
    avg_rate = sum(rates) / len(rates)
    avg_ttft = sum(ttfts_ms) / len(ttfts_ms)

    memory_info = ""
    if torch.cuda.is_available() and "cuda" in str(runtime.device):
        max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        memory_info = f"peak_gpu_mem={max_mem_mb:.1f}MB"
    else:
        try:
            import psutil

            process = psutil.Process()
            rss_mb = process.memory_info().rss / (1024 * 1024)
            memory_info = f"rss_mem={rss_mb:.1f}MB"
        except ImportError:
            pass

    table = Table(title=f"Inference Benchmark ({runs} runs, warmup={warmup})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Device", str(runtime.device))
    table.add_row("Quantized", str(runtime.quantized))
    table.add_row("Generated Tokens / run", str(tokens))
    table.add_row("Avg Throughput (tok/s)", f"{avg_rate:.2f}")
    table.add_row("Avg TTFT (ms)", f"{avg_ttft:.2f}")
    table.add_row("p50 Latency (ms)", f"{p50_latency:.2f}")
    table.add_row("p95 Latency (ms)", f"{p95_latency:.2f}")
    if memory_info:
        table.add_row("Memory Footprint", memory_info)

    console.print(table)
    console.print(f"[dim]Sample Output:[/dim] {sample_text}")


@app.command("publish")
def publish() -> None:
    """Publish the newest verified checkpoint to origin/main for CHAD."""
    _run("scripts.publish_checkpoint")


checkpoint_app = typer.Typer(name="checkpoint", help="Checkpoint inspection and validation tools.")
app.add_typer(checkpoint_app, name="checkpoint")


@checkpoint_app.command("inspect")
def checkpoint_inspect(checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint")) -> None:
    inspect(checkpoint)
