"""Full-screen Textual training observability console for Lapis."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

STATUS_STATES = {
    "STARTING", "PREPARING DATA", "TRAINING", "PAUSED", "RESUMING",
    "CHECKPOINTING", "COMPLETED", "FAILED", "INTERRUPTED",
}
_MAX_STEPS_RE = __import__("re").compile(r"(?m)^\s*max_steps:\s*(\d+)\s*$")


def sanitize_output(value: object, *, max_width: int = 96, max_lines: int = 8) -> str:
    text = str(value).replace("\x1b", "")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    lines = [line.rstrip() for line in text.splitlines()]
    result = [line[:max_width] for line in lines[:max_lines]]
    if len(lines) > max_lines:
        result.append("…")
    return "\n".join(result)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {sec:02d}s"


def _trend(values: Iterable[float]) -> str:
    values = list(values)
    if len(values) < 4:
        return "—"
    split = len(values) // 2
    before = sum(values[:split]) / split
    after = sum(values[split:]) / max(1, len(values) - split)
    delta = (after - before) / max(abs(before), 1e-12)
    return "↓ improving" if delta < -0.02 else "↑ worsening" if delta > 0.02 else "→ stable"


def _spark(values: Iterable[float], width: int = 48) -> str:
    chars = "▁▂▃▄▅▆▇█"
    data = list(values)[-width:]
    if not data:
        return "·" * min(width, 48)
    lo, hi = min(data), max(data)
    if hi - lo <= 1e-12:
        return chars[0] * len(data)
    return "".join(chars[int((v - lo) / (hi - lo) * (len(chars) - 1))] for v in data)


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
    events: deque[tuple[str, str, float]] = field(default_factory=lambda: deque(maxlen=100))
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    lr_history: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    throughput_history: deque[float] = field(default_factory=lambda: deque(maxlen=2000))
    anomaly: str | None = None
    failure_reason: str | None = None
    started_at: float = field(default_factory=time.monotonic)

    def snapshot(self) -> "TrainingState":
        result = TrainingState(self.run_id, self.config)
        for key, value in self.__dict__.items():
            if key in {"events", "logs"} or key.endswith("_history"):
                setattr(result, key, deque(value, maxlen=value.maxlen))
            elif key == "samples":
                setattr(result, key, list(value))
            elif key in {"checkpoint", "best_checkpoint"}:
                setattr(result, key, CheckpointInfo(**vars(value)))
            elif key != "started_at":
                setattr(result, key, value)
        result.started_at = self.started_at
        return result


class TrainingTelemetry:
    def __init__(self, run_id: str, config: str, total_steps: int = 0) -> None:
        self._lock = threading.RLock()
        self.state = TrainingState(run_id=run_id, config=config, total_steps=total_steps)
        self._step_times: deque[tuple[int, float]] = deque(maxlen=40)
        self._token_times: deque[tuple[int, float]] = deque(maxlen=40)

    def update_metric(self, *, step: int, loss: float, learning_rate: float | None,
                      perplexity: float | None = None, tokens_seen: int | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            s = self.state
            s.current_step, s.loss, s.perplexity = step, loss, perplexity
            if learning_rate is not None:
                s.learning_rate = learning_rate
                s.lr_history.append(learning_rate)
            s.loss_history.append(loss)
            self._step_times.append((step, now))
            if tokens_seen is not None:
                s.tokens_seen = tokens_seen
                self._token_times.append((tokens_seen, now))
            if len(self._step_times) >= 2:
                a, ta = self._step_times[0]
                b, tb = self._step_times[-1]
                if b >= a and tb > ta:
                    s.step_rate = (b - a) / (tb - ta)
            if len(self._token_times) >= 2:
                a, ta = self._token_times[0]
                b, tb = self._token_times[-1]
                if b >= a and tb > ta:
                    s.tokens_per_second = (b - a) / (tb - ta)
            s.throughput_history.append(s.step_rate or 0.0)
            s.elapsed = now - s.started_at
            if s.step_rate and s.total_steps:
                s.eta = max(0, s.total_steps - step) / s.step_rate
            if s.best_loss is None or loss < s.best_loss:
                s.best_loss = loss
            if not math.isfinite(loss):
                s.anomaly = f"! LOSS ANOMALY · non-finite loss at step {step}"
            elif len(s.loss_history) >= 12:
                recent = list(s.loss_history)[-12:]
                if max(recent) > max(min(recent), 1e-12) * 3 and recent[-1] > recent[0]:
                    s.anomaly = f"! LOSS ANOMALY · possible explosion around step {step}"
                elif max(recent) - min(recent) <= max(abs(recent[0]), 1e-12) * 0.002:
                    s.anomaly = "! LOSS PLATEAU · no significant recent improvement"
                else:
                    s.anomaly = None

    def add_event(self, label: str, detail: str) -> None:
        with self._lock:
            self.state.events.append((label, sanitize_output(detail, max_width=180, max_lines=2), time.time()))

    def add_log(self, line: str) -> None:
        line = sanitize_output(line, max_width=240, max_lines=4)
        if line:
            with self._lock:
                self.state.logs.append(line)

    def add_sample(self, sample: Sample) -> None:
        with self._lock:
            self.state.samples = (self.state.samples + [sample])[-8:]

    def set_status(self, status: str, detail: str | None = None) -> None:
        status = status.upper()
        if status not in STATUS_STATES:
            return
        with self._lock:
            self.state.status = status
            if detail:
                self.add_event(status, detail)

    def set_checkpoint(self, path: str, step: int | None, loss: float | None) -> None:
        with self._lock:
            info = CheckpointInfo(path, step, loss, time.time(), True)
            self.state.checkpoint = info
            if loss is not None and (self.state.best_checkpoint.loss is None or loss < self.state.best_checkpoint.loss):
                self.state.best_checkpoint = CheckpointInfo(**vars(info))
            self.state.events.append(("CHECKPOINT", f"saved · {path}", time.time()))

    def fail(self, reason: str) -> None:
        with self._lock:
            self.state.status = "FAILED"
            self.state.failure_reason = sanitize_output(reason, max_width=180, max_lines=3)
            self.state.events.append(("FAILED", self.state.failure_reason, time.time()))

    def set_system(self, **values: object) -> None:
        with self._lock:
            for key, value in values.items():
                if hasattr(self.state, key) and value is not None:
                    setattr(self.state, key, value)

    def snapshot(self) -> TrainingState:
        with self._lock:
            return self.state.snapshot()


def _gpu_stats() -> dict[str, object]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.0, check=True,
        )
        row = result.stdout.strip().splitlines()[0].split(", ")
        if len(row) != 5:
            return {}
        return {"gpu": row[0], "gpu_utilization": float(row[1]), "vram_used": float(row[2]) / 1024,
                "vram_total": float(row[3]) / 1024, "gpu_temperature": float(row[4])}
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return {}


class TrainingConsoleRunner:
    def __init__(self, command: list[str], *, config: str, checkpoint: str) -> None:
        self.command = command
        self.config = config
        self.checkpoint = Path(checkpoint)
        self.monitor_path = self.checkpoint.parent / "learning_monitor.jsonl"
        try:
            match = _MAX_STEPS_RE.search(Path(config).read_text(encoding="utf-8"))
            total = int(match.group(1)) if match else 0
        except OSError:
            total = 0
        self.telemetry = TrainingTelemetry(time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid():x}", config, total)
        self.process: subprocess.Popen[str] | None = None
        self._paused = False
        self._monitor_position = 0
        self._system_last = 0.0

    def pause_resume(self) -> bool:
        if self.process is None or self.process.poll() is not None or os.name != "posix":
            return False
        try:
            os.kill(self.process.pid, signal.SIGCONT if self._paused else signal.SIGSTOP)
            self._paused = not self._paused
            self.telemetry.set_status("RESUMING" if not self._paused else "PAUSED")
            return True
        except OSError:
            return False

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            if self._paused:
                try:
                    os.kill(self.process.pid, signal.SIGCONT)
                except OSError:
                    pass
                self._paused = False
            self.process.send_signal(signal.SIGINT)

    def _read_monitor_file(self) -> None:
        if not self.monitor_path.exists():
            return
        try:
            with self.monitor_path.open("r", encoding="utf-8") as handle:
                handle.seek(self._monitor_position)
                while True:
                    line = handle.readline()
                    if not line:
                        break
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
                        for item in record.get("samples", []):
                            self.telemetry.add_sample(Sample(
                                prompt=sanitize_output(item.get("prompt", ""), max_width=120, max_lines=2),
                                completion=sanitize_output(item.get("completion", ""), max_width=120, max_lines=6),
                                step=int(record.get("step", 0)),
                            ))
        except OSError:
            return

    def refresh_system(self) -> None:
        now = time.monotonic()
        if now - self._system_last < 2.0:
            return
        self._system_last = now
        try:
            import psutil
        except ImportError:
            return
        memory = psutil.virtual_memory()
        values: dict[str, object] = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used": memory.used / 2**30,
            "ram_total": memory.total / 2**30,
        }
        values.update(_gpu_stats())
        self.telemetry.set_system(**values)

    def poll(self) -> None:
        self._read_monitor_file()
        self.refresh_system()
        if self.process is None:
            return
        code = self.process.poll()
        if code is None:
            return
        self._read_monitor_file()
        if code == 0:
            self.telemetry.set_status("COMPLETED", "training process exited successfully")
        elif code in {-signal.SIGINT, 130}:
            self.telemetry.set_status("INTERRUPTED", f"process exited with code {code}")
        else:
            self.telemetry.fail(f"training process exited with code {code}")

    def run_process(self) -> int:
        self.telemetry.set_status("TRAINING")
        self.process = subprocess.Popen(self.command, cwd=Path(self.config).resolve().parents[1] if Path(self.config).is_absolute() else Path.cwd(),
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        assert self.process.stdout is not None
        for raw_line in self.process.stdout:
            line = raw_line.rstrip()
            if line:
                self.telemetry.add_log(line)
            match = __import__("re").search(r"step=(\d+)\s+loss=([0-9.eE+-]+)", line)
            if match:
                self.telemetry.update_metric(step=int(match.group(1)), loss=float(match.group(2)), learning_rate=None)
            checkpoint_match = __import__("re").search(r"Checkpoint saved:\s*(.+?)\s*$", line)
            if checkpoint_match:
                self.telemetry.set_checkpoint(checkpoint_match.group(1), self.telemetry.state.current_step, self.telemetry.state.loss)
        code = self.process.wait()
        self.poll()
        return code


try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, VerticalScroll
    from textual.widgets import Footer, Header, Static
except ImportError:
    App = None


def can_use_tui() -> bool:
    if os.environ.get("LAPIS_NO_TUI") or "NO_COLOR" in os.environ or not sys.stdout.isatty():
        return False
    return App is not None


class TrainingAppController:
    def __init__(self, runner: TrainingConsoleRunner) -> None:
        self.runner = runner

    def launch(self) -> None:
        if App is None:
            raise RuntimeError("Textual is required for the interactive training console")
        runner = self.runner

        class TrainingApp(App[None]):
            TITLE = "LapisLLM Training Console"
            CSS = """
            Screen { background: #0B0F14; color: #F3F6F9; }
            Header { background: #111720; color: #F3F6F9; height: 3; }
            Footer { background: #111720; color: #8B98A8; height: 1; }
            #shell { height: 1fr; }
            #sidebar { width: 23; background: #111720; border-right: solid #202936; padding: 1; }
            #workspace { width: 1fr; padding: 1; }
            #view { height: 1fr; }
            """
            BINDINGS = [("r", "refresh", "Refresh"), ("p", "pause_resume", "Pause/resume"), ("q", "quit_app", "Quit")]

            def compose(self) -> ComposeResult:
                yield Header(show_clock=False)
                with Horizontal(id="shell"):
                    yield Static(id="sidebar")
                    with VerticalScroll(id="workspace"):
                        yield Static(id="view")
                yield Footer()

            def on_mount(self) -> None:
                self.worker = threading.Thread(target=runner.run_process, daemon=True)
                self.worker.start()
                self.set_interval(1.0, self.refresh_dashboard)
                self.refresh_dashboard()

            def action_refresh(self) -> None:
                runner.poll()
                self.refresh_dashboard()

            def action_pause_resume(self) -> None:
                runner.pause_resume()
                self.refresh_dashboard()

            def action_quit_app(self) -> None:
                runner.stop()
                self.exit()

            def refresh_dashboard(self) -> None:
                runner.poll()
                state = runner.telemetry.snapshot()
                self.query_one("#sidebar", Static).update(self.render_sidebar(state))
                self.query_one("#view", Static).update(self.render_view(state))
                self.title = f"LapisLLM · {state.status} · {state.run_id}"

            def render_sidebar(self, state: TrainingState) -> str:
                return "\n".join(["LAPISLLM", "TRAINING", "", f"STATUS  {state.status}", f"RUN     {state.run_id}", "", "CONTROLS", "r  refresh", "p  pause/resume", "q  quit"])

            def render_view(self, state: TrainingState) -> str:
                pct = (100.0 * state.current_step / state.total_steps) if state.total_steps else 0.0
                loss = f"{state.loss:.6g}" if state.loss is not None else "—"
                best = f"{state.best_loss:.6g}" if state.best_loss is not None else "—"
                speed = f"{state.step_rate:.2f} step/s" if state.step_rate else "—"
                eta = format_duration(state.eta) if state.eta is not None else "—"
                lines = ["LAPISLLM TRAINING", "", f"STEP             {state.current_step:,} / {state.total_steps:,}   ({pct:.1f}%)", f"LOSS             {loss}", f"BEST LOSS        {best}", f"TREND            {_trend(state.loss_history)}", f"THROUGHPUT       {speed}", f"TOKENS/SECOND    {state.tokens_per_second:,.0f}" if state.tokens_per_second else "TOKENS/SECOND    —", f"ETA              {eta}", f"ELAPSED          {format_duration(state.elapsed)}", "", f"LOSS HISTORY\n{_spark(state.loss_history)}"]
                if state.gpu:
                    lines += ["", f"GPU              {state.gpu}", f"GPU UTILIZATION  {state.gpu_utilization:.0f}%" if state.gpu_utilization is not None else "GPU UTILIZATION  —", f"VRAM             {state.vram_used:.1f} / {state.vram_total:.1f} GB" if state.vram_total is not None else "VRAM             —"]
                if state.samples:
                    sample = state.samples[-1]
                    lines += ["", f"LEARNING SAMPLE · STEP {sample.step}", f"PROMPT\n{sample.prompt}", f"COMPLETION\n{sample.completion}"]
                if state.anomaly:
                    lines += ["", state.anomaly]
                if state.failure_reason:
                    lines += ["", f"FAILURE\n{state.failure_reason}"]
                if state.status == "COMPLETED":
                    lines += ["", "STATUS           ✓ COMPLETED"]
                return "\n".join(lines)

        TrainingApp().run()


def run_training_console(command: list[str], *, config: str, checkpoint: str) -> int:
    runner = TrainingConsoleRunner(command, config=config, checkpoint=checkpoint)
    TrainingAppController(runner).launch()
    return runner.process.returncode if runner.process and runner.process.returncode is not None else 0
