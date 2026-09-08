"""Full-screen training observability console.

The console deliberately stays outside the ML training loop. It consumes the
existing ``scripts.train`` stdout and JSONL learning-monitor stream, so the UI
cannot change model math or become a hot-path dependency of optimization.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


STATUS_STATES = {
    "starting",
    "preparing data",
    "training",
    "paused",
    "resuming",
    "checkpointing",
    "completed",
    "failed",
    "interrupted",
}

_METRIC_RE = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<loss>[0-9.eE+\-]+)"
    r"(?:\s+ppl=(?P<ppl>[0-9.eE+\-]+|inf))?"
    r"(?:\s+lr=(?P<lr>[0-9.eE+\-]+))?"
)
_CHECKPOINT_RE = re.compile(r"Checkpoint saved:\s*(?P<path>.+?)\s*$")
_TOTAL_STEPS_RE = re.compile(r"(?:/|of)\s*(?P<total>\d+)\b")


def sanitize_output(value: object, *, max_width: int = 96, max_lines: int = 8) -> str:
    """Render model/log text safely without hiding model-generated content."""
    text = str(value).replace("\x1b", "")
    text = "".join(char for char in text if char in "\n\t" or ord(char) >= 32)
    lines = [line.rstrip() for line in text.splitlines()]
    clipped = [line[:max_width] for line in lines[:max_lines]]
    if len(lines) > max_lines:
        clipped.append("…")
    return "\n".join(clipped)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {sec:02d}s"


def _trend(values: Iterable[float]) -> str:
    data = list(values)
    if len(data) < 4:
        return "—"
    split = len(data) // 2
    before = sum(data[:split]) / split
    after = sum(data[split:]) / max(1, len(data) - split)
    scale = max(abs(before), 1e-12)
    delta = (after - before) / scale
    if delta < -0.02:
        return "↓ improving"
    if delta > 0.02:
        return "↑ worsening"
    return "→ stable"


def _status_symbol(status: str) -> str:
    normalized = status.lower()
    if normalized in {"training", "resuming", "starting", "preparing data", "checkpointing"}:
        return "●"
    if normalized == "completed":
        return "✓"
    if normalized in {"failed", "interrupted"}:
        return "✗"
    return "!"


@dataclass
class Sample:
    prompt: str
    completion: str
    step: int
    generation_time: float | None = None


@dataclass
class CheckpointInfo:
    path: str | None = None
    step: int | None = None
    loss: float | None = None
    timestamp: float | None = None
    available: bool = False


@dataclass
class TrainingState:
    run_id: str
    config: str
    status: str = "STARTING"
    current_step: int = 0
    total_steps: int = 0
    loss: float | None = None
    best_loss: float | None = None
    learning_rate: float | None = None
    perplexity: float | None = None
    tokens_seen: int | None = None
    step_rate: float | None = None
    tokens_per_second: float | None = None
    elapsed: float = 0.0
    eta: float | None = None
    gradient_norm: float | None = None
    gpu: str | None = None
    gpu_utilization: float | None = None
    vram_used: float | None = None
    vram_total: float | None = None
    gpu_temperature: float | None = None
    cpu_percent: float | None = None
    ram_used: float | None = None
    ram_total: float | None = None
    cuda: str | None = None
    dataloader: str | None = None
    checkpoint: CheckpointInfo = field(default_factory=CheckpointInfo)
    best_checkpoint: CheckpointInfo = field(default_factory=CheckpointInfo)
    samples: list[Sample] = field(default_factory=list)
    events: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=100))
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    lr_history: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    throughput_history: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    anomaly: str | None = None
    failure_reason: str | None = None
    started_at: float = field(default_factory=time.monotonic)

    def snapshot(self) -> "TrainingState":
        copy = TrainingState(run_id=self.run_id, config=self.config)
        for key, value in self.__dict__.items():
            if key == "events":
                copy.events = deque(value, maxlen=value.maxlen)
            elif key == "logs":
                copy.logs = deque(value, maxlen=value.maxlen)
            elif key == "loss_history":
                copy.loss_history = deque(value, maxlen=value.maxlen)
            elif key == "lr_history":
                copy.lr_history = deque(value, maxlen=value.maxlen)
            elif key == "throughput_history":
                copy.throughput_history = deque(value, maxlen=value.maxlen)
            elif key == "samples":
                copy.samples = list(value)
            elif key == "checkpoint":
                copy.checkpoint = CheckpointInfo(**vars(value))
            elif key == "best_checkpoint":
                copy.best_checkpoint = CheckpointInfo(**vars(value))
            elif key != "started_at":
                setattr(copy, key, value)
        copy.started_at = self.started_at
        return copy


class TrainingTelemetry:
    """Thread-safe state store fed by the training subprocess adapter."""

    def __init__(self, run_id: str, config: str, total_steps: int = 0) -> None:
        self._lock = threading.RLock()
        self.state = TrainingState(run_id=run_id, config=config, total_steps=total_steps)
        self._step_times: deque[tuple[int, float]] = deque(maxlen=40)
        self._token_times: deque[tuple[int, float]] = deque(maxlen=40)

    def update_metric(
        self,
        *,
        step: int,
        loss: float,
        learning_rate: float | None,
        perplexity: float | None = None,
        tokens_seen: int | None = None,
    ) -> None:
        now = time.monotonic()
        with self._lock:
            s = self.state
            s.current_step = step
            s.loss = loss
            s.perplexity = perplexity
            if learning_rate is not None:
                s.learning_rate = learning_rate
                s.lr_history.append(learning_rate)
            s.loss_history.append(loss)
            self._step_times.append((step, now))
            if tokens_seen is not None:
                s.tokens_seen = tokens_seen
                self._token_times.append((tokens_seen, now))

            if len(self._step_times) >= 2:
                first_step, first_time = self._step_times[0]
                last_step, last_time = self._step_times[-1]
                if last_time > first_time and last_step >= first_step:
                    s.step_rate = (last_step - first_step) / (last_time - first_time)
            if len(self._token_times) >= 2:
                first_tokens, first_time = self._token_times[0]
                last_tokens, last_time = self._token_times[-1]
                if last_time > first_time and last_tokens >= first_tokens:
                    s.tokens_per_second = (last_tokens - first_tokens) / (last_time - first_time)
            s.throughput_history.append(s.step_rate or 0.0)
            s.elapsed = now - s.started_at
            if s.step_rate and s.total_steps:
                remaining = max(0, s.total_steps - s.current_step)
                s.eta = remaining / s.step_rate

            if s.best_loss is None or loss < s.best_loss:
                s.best_loss = loss
            if not math.isfinite(loss):
                s.anomaly = f"LOSS ANOMALY · non-finite loss at step {step}"
            elif len(s.loss_history) >= 12:
                recent = list(s.loss_history)[-12:]
                if max(recent) > min(recent) * 3 and recent[-1] > recent[0]:
                    s.anomaly = f"LOSS ANOMALY · possible explosion around step {step}"
                elif max(recent) - min(recent) <= max(abs(recent[0]), 1e-12) * 0.002:
                    s.anomaly = "LOSS PLATEAU · no significant recent improvement"
                else:
                    s.anomaly = None

    def add_event(self, label: str, detail: str) -> None:
        with self._lock:
            self.state.events.append((label, detail))

    def add_log(self, line: str) -> None:
        clean = sanitize_output(line, max_width=240, max_lines=4)
        if not clean:
            return
        with self._lock:
            self.state.logs.append(clean)

    def add_sample(self, sample: Sample) -> None:
        with self._lock:
            self.state.samples.append(sample)
            self.state.samples = self.state.samples[-8:]

    def set_status(self, status: str, detail: str | None = None) -> None:
        normalized = status.upper()
        if normalized.lower() not in STATUS_STATES:
            normalized = status
        with self._lock:
            self.state.status = normalized
            if detail:
                self.state.events.append((normalized, detail))

    def set_checkpoint(self, path: str, step: int | None = None, loss: float | None = None) -> None:
        with self._lock:
            checkpoint = CheckpointInfo(
                path=path,
                step=step,
                loss=loss,
                timestamp=time.time(),
                available=True,
            )
            self.state.checkpoint = checkpoint
            if loss is not None and (self.state.best_checkpoint.loss is None or loss < self.state.best_checkpoint.loss):
                self.state.best_checkpoint = CheckpointInfo(**vars(checkpoint))
            self.state.events.append(("CHECKPOINT", f"saved · {path}"))

    def set_system(self, **values: object) -> None:
        with self._lock:
            for key, value in values.items():
                if hasattr(self.state, key) and value is not None:
                    setattr(self.state, key, value)

    def fail(self, reason: str) -> None:
        with self._lock:
            self.state.status = "FAILED"
            self.state.failure_reason = sanitize_output(reason, max_width=180, max_lines=3)
            self.state.events.append(("FAILED", self.state.failure_reason))

    def snapshot(self) -> TrainingState:
        with self._lock:
            return self.state.snapshot()


def _spark(values: Iterable[float], *, width: int = 42, chars: str = "▁▂▃▄▅▆▇█") -> str:
    data = list(values)
    if not data:
        return "·" * min(width, 42)
    data = data[-width:]
    lo = min(data)
    hi = max(data)
    if hi - lo <= 1e-12:
        return chars[0] * len(data)
    return "".join(chars[int((value - lo) / (hi - lo) * (len(chars) - 1))] for value in data)


def _metric_card(label: str, value: str, context: str = "") -> str:
    context_line = f"\n{context}" if context else ""
    return f"{label}\n{value}{context_line}"


class TrainingConsoleRunner:
    """Launch the training process and connect it to the full-screen UI."""

    def __init__(self, command: list[str], *, config: str, checkpoint: str) -> None:
        self.command = command
        self.config = config
        self.checkpoint = Path(checkpoint)
        self.monitor_path = self.checkpoint.parent / "learning_monitor.jsonl"
        self.run_id = time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid():x}"
        self.telemetry = TrainingTelemetry(self.run_id, config)
        self.process: subprocess.Popen[str] | None = None
        self._stop = threading.Event()
        self._pause_requested = False
        self._monitor_position = 0

    def pause_resume(self) -> bool:
        """Use a real POSIX process pause, when the platform supports it."""
        if self.process is None or self.process.poll() is not None or os.name != "posix":
            return False
        try:
            if self._pause_requested:
                os.kill(self.process.pid, signal.SIGCONT)
                self._pause_requested = False
                self.telemetry.set_status("RESUMING", "SIGCONT sent to training process")
            else:
                os.kill(self.process.pid, signal.SIGSTOP)
                self._pause_requested = True
                self.telemetry.set_status("PAUSED", "training process suspended")
            return True
        except OSError:
            return False

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)

    def _read_monitor_file(self) -> None:
        if not self.monitor_path.exists():
            return
        try:
            with self.monitor_path.open("r", encoding="utf-8") as handle:
                handle.seek(self._monitor_position)
                for line in handle:
                    self._monitor_position = handle.tell()
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("type") == "metrics":
                        self.telemetry.update_metric(
                            step=int(record.get("step", 0)),
                            loss=float(record.get("loss", 0.0)),
                            learning_rate=float(record["learning_rate"]) if record.get("learning_rate") is not None else None,
                            perplexity=float(record["perplexity"]) if record.get("perplexity") not in {None, "inf"} else None,
                            tokens_seen=int(record["tokens_seen"]) if record.get("tokens_seen") is not None else None,
                        )
                    elif record.get("type") == "samples":
                        step = int(record.get("step", 0))
                        samples = record.get("samples", [])
                        for item in samples:
                            self.telemetry.add_sample(
                                Sample(
                                    prompt=sanitize_output(item.get("prompt", ""), max_width=120, max_lines=2),
                                    completion=sanitize_output(item.get("completion", ""), max_width=120, max_lines=5),
                                    step=step,
                                )
                            )
        except OSError:
            return

    def poll_process(self) -> None:
        self._read_monitor_file()
        if self.process is None:
            return
        return_code = self.process.poll()
        if return_code is None:
            return
        self._read_monitor_file()
        if return_code == 0:
            self.telemetry.set_status("COMPLETED", "training process exited successfully")
        elif return_code in {-signal.SIGINT, 130}:
            self.telemetry.set_status("INTERRUPTED", f"process exited with code {return_code}")
        else:
            self.telemetry.fail(f"training process exited with code {return_code}")

    def run_process(self) -> int:
        self.telemetry.set_status("PREPARING DATA")
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        self.telemetry.set_status("TRAINING")
        self.telemetry.add_event("RUN", self.run_id)

        for raw_line in self.process.stdout:
            line = raw_line.rstrip("\n")
            self.telemetry.add_log(line)
            lower = line.lower()
            if "learning monitor:" in lower:
                self.telemetry.add_event("MONITOR", line.split(":", 1)[-1].strip())
            if "model:" in lower:
                self.telemetry.add_event("MODEL", line.strip())
            if "tokenizer:" in lower:
                self.telemetry.add_event("TOKENIZER", line.strip())
            if "parameters:" in lower:
                self.telemetry.add_event("MODEL", line.strip())
            if "device:" in lower:
                self.telemetry.add_event("SYSTEM", line.strip())
            if "checkpoint saved:" in lower:
                match = _CHECKPOINT_RE.search(line)
                if match:
                    self.telemetry.set_checkpoint(
                        match.group("path"),
                        step=self.telemetry.snapshot().current_step,
                        loss=self.telemetry.snapshot().loss,
                    )
                self.telemetry.set_status("CHECKPOINTING")
            match = _METRIC_RE.search(line)
            if match:
                step = int(match.group("step"))
                loss = float(match.group("loss"))
                lr = float(match.group("lr")) if match.group("lr") else None
                ppl = None
                if match.group("ppl") and match.group("ppl") != "inf":
                    ppl = float(match.group("ppl"))
                self.telemetry.update_metric(step=step, loss=loss, learning_rate=lr, perplexity=ppl)
                if self.telemetry.state.total_steps == 0:
                    candidate = _TOTAL_STEPS_RE.search(line)
                    if candidate:
                        self.telemetry.state.total_steps = int(candidate.group("total"))
                self.telemetry.set_status("TRAINING")
            self._read_monitor_file()

        result = self.process.wait()
        self.poll_process()
        return result


class TrainingAppController:
    """Small bridge object kept separate from Textual-specific rendering."""

    def __init__(self, runner: TrainingConsoleRunner) -> None:
        self.runner = runner
        self.telemetry = runner.telemetry
        self._app = None

    def toggle_pause(self) -> None:
        if not self.runner.pause_resume():
            self.telemetry.add_event("PAUSE", "pause unavailable on this platform/process")

    def stop(self) -> None:
        self.runner.stop()

    def launch(self) -> None:
        try:
            from textual.app import App, ComposeResult
            from textual.containers import Container, Horizontal, Vertical
            from textual.widgets import Footer, Header, Static
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise RuntimeError("Textual is required for the interactive training console") from exc

        controller = self

        class TrainingApp(App[None]):
            TITLE = "LapisLLM Training Console"
            CSS = """
            Screen { background: #0B0F14; color: #F3F6F9; }
            Header { background: #111720; color: #F3F6F9; height: 3; }
            Footer { background: #111720; color: #8B98A8; height: 1; }
            #shell { height: 1fr; }
            #sidebar { width: 23; background: #111720; border-right: solid #202936; padding: 1 1; }
            #workspace { width: 1fr; padding: 1; }
            .panel { background: #111720; border: solid #202936; padding: 1; margin: 0 0 1 0; height: auto; }
            #metric-grid { height: auto; }
            .metric { width: 1fr; min-width: 18; margin: 0 1 0 0; }
            #main { height: 1fr; }
            #view { height: 1fr; }
            .muted { color: #8B98A8; }
            .accent { color: #B8E986; }
            .warning { color: #F0C56B; }
            .error { color: #F07C7C; }
            @media (max-width: 100) { #sidebar { width: 20; } .metric { min-width: 14; } }
            @media (max-width: 85) { #sidebar { display: none; } #shell { height: 1fr; } }
            @media (min-width: 160) { #workspace { padding: 2; } .panel { margin-right: 1; } }
            """
            BINDINGS = [
                ("1", "show_view('overview')", "Overview"),
                ("2", "show_view('metrics')", "Metrics"),
                ("3", "show_view('learning')", "Learning"),
                ("4", "show_view('system')", "System"),
                ("5", "show_view('checkpoints')", "Checkpoints"),
                ("6", "show_view('timeline')", "Timeline"),
                ("7", "show_view('logs')", "Logs"),
                ("p", "pause_resume", "Pause/resume"),
                ("r", "refresh_now", "Refresh"),
                ("q", "quit_console", "Quit"),
            ]

            def __init__(self) -> None:
                super().__init__()
                self.current_view = "overview"
                self.sample_index = 0
                self._worker: threading.Thread | None = None

            def compose(self) -> ComposeResult:
                yield Header(show_clock=False)
                with Horizontal(id="shell"):
                    yield Static(id="sidebar")
                    with Vertical(id="workspace"):
                        yield Static(id="main")
                yield Footer()

            def on_mount(self) -> None:
                self._worker = threading.Thread(target=controller.runner.run_process, daemon=True)
                self._worker.start()
                self.set_interval(1.0, self.refresh_dashboard)
                self.refresh_dashboard()

            def action_show_view(self, view: str) -> None:
                self.current_view = view
                self.refresh_dashboard()

            def action_pause_resume(self) -> None:
                controller.toggle_pause()
                self.refresh_dashboard()

            def action_refresh_now(self) -> None:
                controller.runner._read_monitor_file()
                self.refresh_dashboard()

            def action_quit_console(self) -> None:
                controller.stop()
                self.exit()

            def on_key(self, event) -> None:
                if event.key in {"j", "down"}:
                    self.sample_index += 1
                    self.refresh_dashboard()
                elif event.key in {"k", "up"}:
                    self.sample_index = max(0, self.sample_index - 1)
                    self.refresh_dashboard()

            def refresh_dashboard(self) -> None:
                controller.runner.poll_process()
                state = controller.telemetry.snapshot()
                self.query_one("#sidebar", Static).update(self.render_sidebar(state))
                self.query_one("#main", Static).update(self.render_view(state))
                status = state.status
                self.title = f"LapisLLM · {status} · {state.run_id}"

            def render_sidebar(self, state: TrainingState) -> Text:
                text = Text()
                text.append("LAPISLLM\n", style="bold")
                text.append("TRAINING\n\n", style="dim")
                entries = [
                    ("1", "Overview", "overview"),
                    ("2", "Metrics", "metrics"),
                    ("3", "Learning", "learning"),
                    ("4", "System", "system"),
                    ("5", "Checkpoints", "checkpoints"),
                    ("6", "Timeline", "timeline"),
                    ("7", "Logs", "logs"),
                ]
                for key, label, view in entries:
                    marker = "●" if view == self.current_view else "○"
                    style = "bold #B8E986" if view == self.current_view else "#8B98A8"
                    text.append(f"{marker} {key} {label}\n", style=style)
                text.append("\nRUN\n", style="dim")
                text.append(f"{state.run_id}\n", style="bold")
                text.append(f"{_status_symbol(state.status)} {state.status}\n", style="bold")
                if state.config:
                    text.append(f"\nCONFIG\n{state.config}", style="dim")
                return text

            def render_view(self, state: TrainingState):
                if self.current_view == "overview":
                    return self.render_overview(state)
                if self.current_view == "metrics":
                    return self.render_metrics(state)
                if self.current_view == "learning":
                    return self.render_learning(state)
                if self.current_view == "system":
                    return self.render_system(state)
                if self.current_view == "checkpoints":
                    return self.render_checkpoints(state)
                if self.current_view == "timeline":
                    return self.render_timeline(state)
                return self.render_logs(state)

            def render_metric_grid(self, state: TrainingState):
                table = Table.grid(expand=True, padding=(0, 1))
                table.add_column(ratio=1)
                table.add_column(ratio=1)
                table.add_column(ratio=1)
                table.add_column(ratio=1)
                progress = f"{state.current_step:,} / {state.total_steps:,}" if state.total_steps else f"{state.current_step:,}"
                pct = (state.current_step / state.total_steps * 100) if state.total_steps else 0
                table.add_row(
                    _metric_card("STEP", progress, f"{pct:.1f}%" if state.total_steps else ""),
                    _metric_card("LOSS", f"{state.loss:.4g}" if state.loss is not None else "—", f"best {state.best_loss:.4g}" if state.best_loss is not None else ""),
                    _metric_card("THROUGHPUT", f"{state.step_rate:.2f} step/s" if state.step_rate else "—", f"{state.tokens_per_second:,.0f} tok/s" if state.tokens_per_second else ""),
                    _metric_card("ETA", f"~{format_duration(state.eta)}" if state.eta is not None else "—", "stable estimate"),
                )
                return table

            def render_overview(self, state: TrainingState):
                outer = Table.grid(expand=True)
                outer.add_row(self.render_metric_grid(state))
                pct = (state.current_step / state.total_steps) if state.total_steps else 0
                bars = max(0, min(50, int(pct * 50)))
                progress_text = f"{state.current_step:,} / {state.total_steps:,}  {pct * 100:.1f}%  [#B8E986]{'█' * bars}[/#B8E986]{'░' * (50 - bars)}"
                outer.add_row(f"PROGRESS\n{progress_text}\nElapsed {format_duration(state.elapsed)}")
                loss_table = Table.grid(expand=True)
                loss_table.add_column(ratio=3)
                loss_table.add_column(ratio=1)
                loss_table.add_row(
                    f"LOSS · LAST {len(state.loss_history):,} STEPS\n[#B8E986]{_spark(state.loss_history)}[/#B8E986]\ncurrent {state.loss:.4g} · {_trend(state.loss_history)}" if state.loss is not None else "LOSS\nno samples yet",
                    f"GPU {state.gpu_utilization:.0f}%" if state.gpu_utilization is not None else "GPU —",
                )
                outer.add_row(loss_table)
                sample = state.samples[min(self.sample_index, len(state.samples) - 1)] if state.samples else None
                sample_text = "LIVE SAMPLE\nNo monitor sample yet."
                if sample:
                    sample_text = f"LIVE SAMPLE · step {sample.step}\nPROMPT\n{sanitize_output(sample.prompt, max_width=120, max_lines=2)}\n\nCOMPLETION\n{sanitize_output(sample.completion, max_width=120, max_lines=5)}"
                outer.add_row(sample_text)
                system_line = []
                if state.gpu and state.gpu_utilization is not None:
                    system_line.append(f"GPU {state.gpu} · {state.gpu_utilization:.0f}%")
                if state.vram_used is not None and state.vram_total is not None:
                    system_line.append(f"VRAM {state.vram_used:.1f}/{state.vram_total:.1f}GB")
                if state.cuda:
                    system_line.append(f"CUDA {state.cuda}")
                outer.add_row(" · ".join(system_line) if system_line else "SYSTEM · metrics unavailable")
                if state.anomaly:
                    outer.add_row(f"[bold #F0C56B]! {state.anomaly}[/#F0C56B]")
                return outer

            def render_metrics(self, state: TrainingState):
                table = Table.grid(expand=True)
                table.add_column(ratio=1)
                table.add_column(ratio=1)
                table.add_row(
                    _metric_card("LOSS", f"{state.loss:.6g}" if state.loss is not None else "—", _trend(state.loss_history)),
                    _metric_card("LEARNING RATE", f"{state.learning_rate:.3g}" if state.learning_rate is not None else "—", _spark(state.lr_history, width=32)),
                )
                table.add_row(
                    _metric_card("STEP RATE", f"{state.step_rate:.3f} step/s" if state.step_rate else "—", _spark(state.throughput_history, width=32)),
                    _metric_card("TOKENS/S", f"{state.tokens_per_second:,.0f}" if state.tokens_per_second else "—", f"seen {state.tokens_seen:,}" if state.tokens_seen is not None else ""),
                )
                table.add_row(
                    _metric_card("ELAPSED", format_duration(state.elapsed)),
                    _metric_card("ETA", f"~{format_duration(state.eta)}" if state.eta is not None else "—"),
                )
                table.add_row(
                    _metric_card("OPTIMIZER", "reported by trainer only"),
                    _metric_card("GRADIENT NORM", f"{state.gradient_norm:.4g}" if state.gradient_norm is not None else "—"),
                )
                if state.anomaly:
                    table.add_row(f"[bold #F0C56B]! {state.anomaly}[/#F0C56B]", "")
                return table

            def render_learning(self, state: TrainingState):
                if not state.samples:
                    return "LEARNING\n\nWaiting for the first learning-monitor sample…"
                outer = Table.grid(expand=True)
                outer.add_column()
                for index, sample in enumerate(state.samples):
                    marker = "→" if index == min(self.sample_index, len(state.samples) - 1) else " "
                    outer.add_row(
                        f"{marker} SAMPLE {index + 1:02d} · STEP {sample.step}\n"
                        f"PROMPT\n{sanitize_output(sample.prompt, max_width=120, max_lines=2)}\n"
                        f"\nCOMPLETION\n{sanitize_output(sample.completion, max_width=120, max_lines=6)}"
                    )
                return outer

            def render_system(self, state: TrainingState):
                table = Table.grid(expand=True)
                table.add_column()
                table.add_column()
                rows = []
                if state.gpu:
                    rows.extend([("GPU", state.gpu), ("GPU UTILIZATION", f"{state.gpu_utilization:.0f}%" if state.gpu_utilization is not None else "—")])
                if state.vram_total is not None:
                    rows.append(("VRAM", f"{state.vram_used:.1f} / {state.vram_total:.1f} GB"))
                if state.gpu_temperature is not None:
                    rows.append(("TEMPERATURE", f"{state.gpu_temperature:.0f}°C"))
                if state.cuda:
                    rows.append(("CUDA", state.cuda))
                if state.cpu_percent is not None:
                    rows.append(("CPU", f"{state.cpu_percent:.0f}%"))
                if state.ram_total is not None:
                    rows.append(("RAM", f"{state.ram_used:.1f} / {state.ram_total:.1f} GB"))
                if state.dataloader:
                    rows.append(("DATALOADER", state.dataloader))
                if not rows:
                    return "SYSTEM\n\nNo system metrics available."
                for left, right in rows:
                    table.add_row(left, right)
                return table

            def render_checkpoints(self, state: TrainingState):
                table = Table.grid(expand=True)
                table.add_column()
                table.add_column()
                table.add_row("LATEST", state.checkpoint.path or "not saved yet")
                table.add_row("STATUS", "✓ available" if state.checkpoint.available else "— unavailable")
                if state.checkpoint.step is not None:
                    table.add_row("STEP", str(state.checkpoint.step))
                if state.checkpoint.loss is not None:
                    table.add_row("LOSS", f"{state.checkpoint.loss:.6g}")
                table.add_row("BEST", state.best_checkpoint.path or "not known")
                if state.best_checkpoint.loss is not None:
                    table.add_row("BEST LOSS", f"{state.best_checkpoint.loss:.6g}")
                return table

            def render_timeline(self, state: TrainingState):
                if not state.events:
                    return "TIMELINE\n\nNo events yet."
                table = Table.grid(expand=True)
                for label, detail in reversed(list(state.events)[-30:]):
                    table.add_row(f"{label:<12} {sanitize_output(detail, max_width=140, max_lines=2)}")
                return table

            def render_logs(self, state: TrainingState):
                lines = list(state.logs)[-45:]
                if not lines:
                    return "LOGS\n\nNo logs yet."
                return "LOGS · live\n\n" + "\n".join(lines)

        self._app = TrainingApp()
        self._app.run()


def can_use_tui() -> bool:
    """Return whether full-screen mode is appropriate for the current process."""
    if os.environ.get("NO_COLOR") is not None or os.environ.get("LAPIS_NO_TUI"):
        return False
    return bool(getattr(__import__("sys"), "stdout").isatty())


def run_training_console(command: list[str], *, config: str, checkpoint: str) -> int:
    """Run training inside the full-screen console when possible."""
    runner = TrainingConsoleRunner(command, config=config, checkpoint=checkpoint)
    TrainingAppController(runner).launch()
    if runner.process is None:
        return 1
    return runner.process.returncode if runner.process.returncode is not None else 1
