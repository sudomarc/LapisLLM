"""Developer command namespace for LapisLLM."""

from __future__ import annotations

import os
import subprocess
import sys
import time
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
    device: str = typer.Option("auto", "--device"),
) -> None:
    _run("scripts.generate", "--checkpoint", str(checkpoint), "--prompt", prompt, "--max-new-tokens", str(max_new_tokens), "--temperature", str(temperature), "--top-k", str(top_k), "--top-p", str(top_p), "--device", device)


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
) -> None:
    """Measure inference throughput without changing the checkpoint."""
    runtime = LapisRuntime.from_checkpoint(checkpoint, device)
    started = time.perf_counter()
    text = runtime.generate(prompt, SamplingConfig(max_new_tokens=tokens))
    elapsed = time.perf_counter() - started
    rate = tokens / elapsed if elapsed > 0 else 0.0
    console.print(f"device={runtime.device} tokens={tokens} elapsed={elapsed:.3f}s tok/s={rate:.1f}")
    console.print(text)


@app.command("publish")
def publish() -> None:
    """Publish the newest verified checkpoint to origin/main for CHAD."""
    _run("scripts.publish_checkpoint")


checkpoint_app = typer.Typer(name="checkpoint", help="Checkpoint inspection and validation tools.")
app.add_typer(checkpoint_app, name="checkpoint")


@checkpoint_app.command("inspect")
def checkpoint_inspect(checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint")) -> None:
    inspect(checkpoint)
