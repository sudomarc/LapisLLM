"""Professional Lapis command-line interface."""

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
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from lapis.dev.cli import app as dev_app
from lapis.ui.training_console import can_use_tui, run_training_console

app = typer.Typer(name="lapis", help="Lapis language-model platform.", no_args_is_help=False, rich_markup_mode="rich")
api_app = typer.Typer(name="api", help="Run and inspect the Lapis inference API.", no_args_is_help=True, rich_markup_mode="rich")
console = Console()
REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode:
        raise typer.Exit(result.returncode)


def _max_steps(config: Path) -> int:
    text = config.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*max_steps:\s*(\d+)\s*$", text)
    return int(match.group(1)) if match else 0


def _training_command(config: Path, device: str | None, data: Path | None, checkpoint: Path, epochs: int, monitor_interval: int) -> list[str]:
    command = [sys.executable, "-m", "scripts.train", "--config", str(config), "--epochs", str(epochs), "--checkpoint", str(checkpoint), "--monitor-interval", str(monitor_interval)]
    if device:
        command.extend(["--device", device])
    if data:
        command.extend(["--data", str(data)])
    return command


def _serve_api(checkpoint: Path, host: str, port: int, device: str) -> None:
    _run([
        sys.executable,
        "-m",
        "scripts.serve",
        "--checkpoint",
        str(checkpoint),
        "--host",
        host,
        "--port",
        str(port),
        "--device",
        device,
    ])


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """Launch the end-user experience when no command is supplied."""
    if ctx.invoked_subcommand is None and not ctx.resilient_parsing:
        chat()


app.add_typer(dev_app, name="dev")
app.add_typer(api_app, name="api")


@app.command("train", hidden=True)
def train(config: Path = typer.Option(Path("configs/local-dev.yaml"), "--config", "-c"), device: str | None = typer.Option(None, "--device"), data: Path | None = typer.Option(None, "--data"), checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"), epochs: int = typer.Option(1000, "--epochs", min=1), monitor_interval: int = typer.Option(500, "--monitor-interval", min=0), no_tui: bool = typer.Option(False, "--no-tui", help="Force the legacy non-interactive renderer.")) -> None:
    """Legacy training command. Prefer ``lapis dev train`` for new usage."""
    total = _max_steps(REPO_ROOT / config)
    command = _training_command(config, device, data, checkpoint, epochs, monitor_interval)
    if not no_tui and can_use_tui():
        raise typer.Exit(run_training_console(command, config=str(config), checkpoint=str(checkpoint)))
    console.print(Panel.fit(f"[bold]Lapis Training[/bold]\nConfig: {config}\nDevice: {device or 'auto'}\nCheckpoint: {checkpoint}", border_style="bright_blue"))
    progress = Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(), TaskProgressColumn(), TextColumn("loss {task.fields[loss]}"), TextColumn("step {task.completed}/{task.total}"), TimeRemainingColumn(), console=console)
    started = time.monotonic()
    last_lines: list[str] = []
    with progress:
        task = progress.add_task("training", total=total if total > 0 else None, loss="--")
        process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            last_lines.append(line)
            del last_lines[:-3]
            match = re.search(r"step=(\d+)\s+loss=([0-9.eE+-]+)", line)
            if match:
                progress.update(task, completed=int(match.group(1)), loss=match.group(2))
            elif "Checkpoint saved:" in line:
                progress.update(task, description="saving checkpoint")
        return_code = process.wait()
    if return_code:
        console.print(Panel("\n".join(last_lines) or "Training failed.", title="Training failed", border_style="red"))
        raise typer.Exit(return_code)
    console.print(Panel.fit(f"[bold green]Training complete[/bold green]\nElapsed: {time.monotonic() - started:.1f}s\nCheckpoint: {checkpoint}", border_style="green"))


@app.command("generate", hidden=True)
def generate(prompt: str = typer.Argument(...), checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"), max_new_tokens: int = typer.Option(64, "--max-new-tokens", min=1), temperature: float = typer.Option(0.8, "--temperature", min=0.01), top_k: int = typer.Option(40, "--top-k", min=0), top_p: float = typer.Option(0.95, "--top-p", min=0.01, max=1.0), device: str = typer.Option("auto", "--device")) -> None:
    """Legacy direct generation command."""
    _run([sys.executable, "-m", "scripts.generate", "--checkpoint", str(checkpoint), "--prompt", prompt, "--max-new-tokens", str(max_new_tokens), "--temperature", str(temperature), "--top-k", str(top_k), "--top-p", str(top_p), "--device", device])


@app.command("evaluate", hidden=True)
def evaluate(checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"), device: str = typer.Option("auto", "--device")) -> None:
    """Legacy evaluation command. Prefer ``lapis dev evaluate``."""
    _run([sys.executable, "-m", "scripts.evaluate", "--checkpoint", str(checkpoint), "--device", device])


@app.command("chat")
def chat() -> None:
    """Start the user-facing Lapis chat interface."""
    _run([sys.executable, "-m", "scripts.chat"])


@api_app.command("serve")
def api_serve(
    checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Start the Lapis inference API server."""
    _serve_api(checkpoint, host, port, device)


@api_app.command("start", hidden=True)
def api_start(
    checkpoint: Path = typer.Option(Path("checkpoints/latest.pt"), "--checkpoint"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port", min=1, max=65535),
    device: str = typer.Option("auto", "--device"),
) -> None:
    """Alias for ``lapis api serve``."""
    _serve_api(checkpoint, host, port, device)


@app.command("serve", hidden=True)
def legacy_serve() -> None:
    """Legacy alias for ``lapis api serve``."""
    _serve_api(Path("checkpoints/latest.pt"), "127.0.0.1", 8000, "auto")


@app.command("console", hidden=True)
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
