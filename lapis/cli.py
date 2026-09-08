"""Professional Lapis command-line interface.

The CLI keeps the existing scripts as the execution layer while providing a
single, consistent terminal experience with Rich progress, status panels and
system metrics.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import psutil
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

app = typer.Typer(
    name="lapis",
    help="Lapis language-model training and inference CLI.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    """Run an existing Lapis script and preserve its exit status."""
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode:
        raise typer.Exit(result.returncode)


def _max_steps(config: Path) -> int:
    text = config.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*max_steps:\s*(\d+)\s*$", text)
    if not match:
        return 0
    return int(match.group(1))


def _training_command(
    config: Path,
    device: str | None,
    data: Path | None,
    checkpoint: Path,
    epochs: int,
    monitor_interval: int,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(config),
        "--epochs",
        str(epochs),
        "--checkpoint",
        str(checkpoint),
        "--monitor-interval",
        str(monitor_interval),
    ]
    if device:
        command.extend(["--device", device])
    if data:
        command.extend(["--data", str(data)])
    return command


@app.command("train")
def train(
    config: Path = typer.Option(Path("configs/local-dev.yaml"), "--config", "-c"),
    device: str | None = typer.Option(None, "--device"),
    data: Path | None = typer.Option(None, "--data"),
    checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"),
    epochs: int = typer.Option(1000, "--epochs", min=1),
    monitor_interval: int = typer.Option(500, "--monitor-interval", min=0),
) -> None:
    """Train Lapis with a live Rich progress display."""
    total = _max_steps(REPO_ROOT / config)
    command = _training_command(config, device, data, checkpoint, epochs, monitor_interval)

    console.print(Panel.fit(
        f"[bold]Lapis Training[/bold]\n"
        f"Config: {config}\n"
        f"Device: {device or 'auto'}\n"
        f"Checkpoint: {checkpoint}",
        border_style="bright_blue",
    ))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("loss {task.fields[loss]}"),
        TextColumn("step {task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
    )
    task_total = total if total > 0 else None
    started = time.monotonic()
    last_lines: list[str] = []

    with progress:
        task = progress.add_task("training", total=task_total, loss="--")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            last_lines.append(line)
            del last_lines[:-3]
            match = re.search(r"step=(\d+)\s+loss=([0-9.eE+-]+)", line)
            if match:
                step = int(match.group(1))
                progress.update(task, completed=step, loss=match.group(2))
            elif "Checkpoint saved:" in line:
                progress.update(task, description="saving checkpoint")

        return_code = process.wait()

    elapsed = time.monotonic() - started
    if return_code:
        console.print(Panel("\n".join(last_lines) or "Training failed.", title="Training failed", border_style="red"))
        raise typer.Exit(return_code)

    console.print(Panel.fit(
        f"[bold green]Training complete[/bold green]\n"
        f"Elapsed: {elapsed:.1f}s\n"
        f"Checkpoint: {checkpoint}",
        border_style="green",
    ))


@app.command("generate")
def generate(
    prompt: str = typer.Argument(..., help="Prompt to generate from."),
    checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens", min=1),
    temperature: float = typer.Option(0.8, "--temperature", min=0.01),
    top_k: int = typer.Option(40, "--top-k", min=0),
    top_p: float = typer.Option(0.95, "--top-p", min=0.01, max=1.0),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Generate text from a checkpoint."""
    _run([
        sys.executable,
        "-m",
        "scripts.generate",
        "--checkpoint", str(checkpoint),
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--top-k", str(top_k),
        "--top-p", str(top_p),
        "--device", device,
    ])


@app.command("evaluate")
def evaluate(
    checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Evaluate a trained checkpoint."""
    _run([sys.executable, "-m", "scripts.evaluate", "--checkpoint", str(checkpoint), "--device", device])


@app.command("chat")
def chat() -> None:
    """Start the Lapis chat interface."""
    _run([sys.executable, "-m", "scripts.chat"])


@app.command("serve")
def serve() -> None:
    """Start the Lapis API server."""
    _run([sys.executable, "-m", "scripts.serve"])


@app.command("console")
def legacy_console() -> None:
    """Open the original Lapis experiment/training console."""
    _run([sys.executable, "-m", "scripts.lapis"])


@app.command("system")
def system() -> None:
    """Display CPU, memory and disk information for the current machine."""
    table = Table(title="Lapis System")
    table.add_column("Resource", style="bold")
    table.add_column("Usage")
    table.add_row("CPU", f"{psutil.cpu_percent(interval=0.2):.0f}%")
    memory = psutil.virtual_memory()
    table.add_row("RAM", f"{memory.percent:.0f}% ({memory.used / 2**30:.1f} / {memory.total / 2**30:.1f} GiB)")
    disk = psutil.disk_usage(str(REPO_ROOT))
    table.add_row("Disk", f"{disk.percent:.0f}% ({disk.used / 2**30:.1f} / {disk.total / 2**30:.1f} GiB)")
    console.print(table)


if __name__ == "__main__":
    app()
