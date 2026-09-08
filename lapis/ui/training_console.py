"""Full-screen Lapis training observability console."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import psutil

STATUS_STATES = {
    "STARTING", "PREPARING DATA", "TRAINING", "PAUSED", "RESUMING",
    "CHECKPOINTING", "COMPLETED", "FAILED", "INTERRUPTED",
}
_METRIC_RE = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<loss>[0-9.eE+\-]+)"
    r"(?:\s+ppl=(?P<ppl>[0-9.eE+\-]+|inf))?"
    r"(?:\s+lr=(?P<lr>[0-9.eE+\-]+))?"
)
_CHECKPOINT_RE = re.compile(r"Checkpoint saved:\s*(?P<path>.+?)\s*$")
_MAX_STEPS_RE = re.compile(r"(?m)^\s*max_steps:\s*(\d+)\s*$")


def sanitize_output(value: object, *, max_width: int = 96, max_lines: int = 8) -> str:
    """Remove terminal controls and bound display size without changing content semantics."""
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
            if key == "events":
                setattr(result, key, deque(value, maxlen=value.maxlen))
            elif key == "logs":
                setattr(result, key, deque(value, maxlen=value.maxlen))
            elif key.endswith("_history"):
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
    """Thread-safe observable state; the training engine never imports this class."""

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
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.0, check=True,
        )
        row = result.stdout.strip().splitlines()[0].split(", ")
        if len(row) != 5:
            return {}
        return {"gpu": row[0], "gpu_utilization": float(row[1]), "vram_used": float(row[2]) / 1024,
                "vram_total": float(row[3]) / 1024, "gpu_temperature": float(row[4])}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


class TrainingConsoleRunner:
    """Subprocess adapter; it keeps UI I/O off the training hot path."""

    def __init__(self, command: list[str], *, config: str, checkpoint: str) -> None:
        self.command = command
        self.config = config
        self.checkpoint = Path(checkpoint)
        self.monitor_path = self.checkpoint.parent / "learning_monitor.jsonl"
        total = 0
        try:
            match = _MAX_STEPS_RE.search(Path(config).read_text(encoding="utf-8"))
            total = int(match.group(1)) if match else 0
        except OSError:
            pass
        self.telemetry = TrainingTelemetry(
            run_id=time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid():x}",
            config=config,
            total_steps=total,
        )
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
        self.telemetry.set_status("PREPARING DATA")
        self.process = subprocess.Popen(
            self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert self.process.stdout is not None
        self.telemetry.set_status("TRAINING")
        self.telemetry.add_event("RUN", self.telemetry.state.run_id)
        for raw_line in self.process.stdout:
            line = raw_line.rstrip("\n")
            self.telemetry.add_log(line)
            lower = line.lower()
            metric = _METRIC_RE.search(line)
            if metric:
                self.telemetry.update_metric(
                    step=int(metric.group("step")),
                    loss=float(metric.group("loss")),
                    learning_rate=float(metric.group("lr")) if metric.group("lr") else None,
                    perplexity=float(metric.group("ppl")) if metric.group("ppl") not in {None, "inf"} else None,
                )
                self.telemetry.set_status("TRAINING")
            if "checkpoint saved:" in lower:
                match = _CHECKPOINT_RE.search(line)
                if match:
                    state = self.telemetry.snapshot()
                    self.telemetry.set_checkpoint(match.group("path"), state.current_step, state.loss)
                self.telemetry.set_status("CHECKPOINTING")
            for marker, label in (("model:", "MODEL"), ("tokenizer:", "TOKENIZER"), ("learning monitor:", "MONITOR"), ("dataset", "DATA")):
                if marker in lower:
                    self.telemetry.add_event(label, line)
                    break
            self._read_monitor_file()
        code = self.process.wait()
        self.poll()
        return code


class TrainingAppController:
    def __init__(self, runner: TrainingConsoleRunner) -> None:
        self.runner = runner

    def launch(self) -> None:
        try:
            from textual.app import App, ComposeResult
            from textual.containers import Horizontal, VerticalScroll
            from textual.widgets import Footer, Header, Static
        except ImportError as exc:
            raise RuntimeError("Textual is required for the interactive training console") from exc

        runner = self.runner

        class TrainingApp(App[None]):
            TITLE = "LapisLLM Training Console"
            CSS = """
            Screen { background: #0B0F14; color: #F3F6F9; }
            Header { background: #111720; color: #F3F6F9; height: 3; }
            Footer { background: #111720; color: #8B98A8; height: 1; }
            #shell { height: 1fr; }
            #sidebar { width: 23; background: #111720; border-right: solid #202936; padding: 1 1; }
            #workspace { width: 1fr; padding: 1; }
            #view { height: 1fr; }
            @media (max-width: 100) { #sidebar { width: 20; } }
            @media (max-width: 85) { #sidebar { display: none; } }
            @media (min-width: 160) { #workspace { padding: 2; } }
            """
            BINDINGS = [
                ("1", "view('overview')", "Overview"), ("2", "view('metrics')", "Metrics"),
                ("3", "view('learning')", "Learning"), ("4", "view('system')", "System"),
                ("5", "view('checkpoints')", "Checkpoints"), ("6", "view('timeline')", "Timeline"),
                ("7", "view('logs')", "Logs"), ("p", "pause_resume", "Pause/resume"),
                ("r", "refresh", "Refresh"), ("q", "quit_app", "Quit"),
            ]

            def __init__(self) -> None:
                super().__init__()
                self.current_view = "overview"
                self.sample_index = 0
                self.worker: threading.Thread | None = None

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

            def action_view(self, name: str) -> None:
                self.current_view = name
                self.refresh_dashboard()

            def action_pause_resume(self) -> None:
                runner.pause_resume()
                self.refresh_dashboard()

            def action_refresh(self) -> None:
                runner.poll()
                self.refresh_dashboard()

            def action_quit_app(self) -> None:
                runner.stop()
                self.exit()

            def on_key(self, event) -> None:
                if event.key in {"down", "j"}:
                    self.sample_index += 1
                    self.refresh_dashboard()
                elif event.key in {"up", "k"}:
                    self.sample_index = max(0, self.sample_index - 1)
                    self.refresh_dashboard()

            def refresh_dashboard(self) -> None:
                runner.poll()
                state = runner.telemetry.snapshot()
                self.query_one("#sidebar", Static).update(self.render_sidebar(state))
                self.query_one("#view", Static).update(self.render_view(state))
                self.title = f"LapisLLM · {state.status} · {state.run_id}"

            def render_sidebar(self, state: TrainingState) -> str:
                rows = ["LAPISLLM", "TRAINING", ""]
                for number, label, name in ((1, "Overview", "overview"), (2, "Metrics", "metrics"), (3, "Learning", "learning"),
                                             (4, "System", "system"), (5, "Checkpoints", "checkpoints"), (6, "Timeline", "timeline"), (7, "Logs", "logs")):
                    rows.append(f"{'●' if name == self.current_view else '○'} {number} {label}")
                rows += ["", "RUN", state.run_id, f"{state.status}"]
                return "\n".join(rows)

            def render_view(self, state: TrainingState) -> str:
                if state.status == "FAILED":
                    return self.render_failure(state)
                if state.status == "COMPLETED":
                    return self.render_completion(state)
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

            def render_overview(self, s: TrainingState) -> str:
                pct = (s.current_step / s.total_steps * 100) if s.total_steps else 0.0
                filled = max(0, min(50, int(pct / 2)))
                loss = f"{s.loss:.5f}" if s.loss is not None else "—"
                best = f"best {s.best_loss:.5f}" if s.best_loss is not None else ""
                speed = f"{s.step_rate:.2f} step/s" if s.step_rate else "—"
                eta = f"~{format_duration(s.eta)}" if s.eta is not None else "—"
                sample = s.samples[min(self.sample_index, len(s.samples) - 1)] if s.samples else None
                lines = [
                    f"STEP\n{s.current_step:,} / {s.total_steps:,}    {pct:.1f}%",
                    f"LOSS\n{loss}    {best}    {_trend(s.loss_history)}",
                    f"THROUGHPUT\n{speed}    {s.tokens_per_second:,.0f} tok/s" if s.tokens_per_second else f"THROUGHPUT\n{speed}",
                    f"ETA\n{eta}    elapsed {format_duration(s.elapsed)}",
                    "",
                    f"PROGRESS\n{'█' * filled}{'░' * (50 - filled)}",
                    "",
                    f"LOSS · LAST {len(s.loss_history):,} POINTS\n{_spark(s.loss_history)}",
                ]
                if sample:
                    lines += ["", f"LIVE SAMPLE · STEP {sample.step}", f"PROMPT\n{sample.prompt}", f"COMPLETION\n{sample.completion}"]
                if s.gpu:
                    lines += ["", f"GPU {s.gpu} · {s.gpu_utilization:.0f}% · VRAM {s.vram_used:.1f}/{s.vram_total:.1f}GB" if s.vram_total else f"GPU {s.gpu}"]
                if s.anomaly:
                    lines += ["", s.anomaly]
                return "\n".join(lines)

            def render_metrics(self, s: TrainingState) -> str:
                return "\n".join([
                    "METRICS",
                    f"loss              {s.loss:.6g}" if s.loss is not None else "loss              —",
                    f"learning rate     {s.learning_rate:.4g}" if s.learning_rate is not None else "learning rate     —",
                    f"gradient norm     {s.gradient_norm:.4g}" if s.gradient_norm is not None else "gradient norm     —",
                    f"step rate         {s.step_rate:.3f} step/s" if s.step_rate else "step rate         —",
                    f"tokens/s          {s.tokens_per_second:,.0f}" if s.tokens_per_second else "tokens/s          —",
                    f"tokens seen       {s.tokens_seen:,}" if s.tokens_seen is not None else "tokens seen       —",
                    f"elapsed           {format_duration(s.elapsed)}",
                    f"ETA               ~{format_duration(s.eta)}" if s.eta is not None else "ETA               —",
                    "",
                    f"LOSS\n{_spark(s.loss_history)}",
                    f"LEARNING RATE\n{_spark(s.lr_history)}",
                    f"THROUGHPUT\n{_spark(s.throughput_history)}",
                ])

            def render_learning(self, s: TrainingState) -> str:
                if not s.samples:
                    return "LEARNING\n\nWaiting for the learning monitor…"
                blocks = []
                for index, sample in enumerate(s.samples):
                    blocks.append(f"{'→' if index == min(self.sample_index, len(s.samples) - 1) else ' '} SAMPLE {index + 1:02d} · STEP {sample.step}\nPROMPT\n{sample.prompt}\n\nCOMPLETION\n{sample.completion}")
                return "\n\n".join(blocks)

            def render_system(self, s: TrainingState) -> str:
                rows = ["SYSTEM"]
                if s.gpu:
                    rows += [f"GPU              {s.gpu}", f"GPU utilization   {s.gpu_utilization:.0f}%" if s.gpu_utilization is not None else "GPU utilization   —"]
                if s.vram_total is not None:
                    rows.append(f"VRAM              {s.vram_used:.1f} / {s.vram_total:.1f} GB")
                if s.gpu_temperature is not None:
                    rows.append(f"Temperature       {s.gpu_temperature:.0f}°C")
                if s.cpu_percent is not None:
                    rows.append(f"CPU               {s.cpu_percent:.0f}%")
                if s.ram_total is not None:
                    rows.append(f"RAM               {s.ram_used:.1f} / {s.ram_total:.1f} GB")
                if not s.gpu and s.cpu_percent is None:
                    rows.append("No system metrics available.")
                return "\n".join(rows)

            def render_checkpoints(self, s: TrainingState) -> str:
                latest = s.checkpoint
                best = s.best_checkpoint
                return "\n".join([
                    "CHECKPOINTS",
                    f"LATEST            {latest.path or 'not saved yet'}",
                    f"STATUS            {'✓ available' if latest.available else '— unavailable'}",
                    f"STEP              {latest.step}" if latest.step is not None else "STEP              —",
                    f"LOSS              {latest.loss:.6g}" if latest.loss is not None else "LOSS              —",
                    "",
                    f"BEST              {best.path or 'not known'}",
                    f"BEST LOSS         {best.loss:.6g}" if best.loss is not None else "BEST LOSS         —",
                ])

            def render_timeline(self, s: TrainingState) -> str:
                lines = ["TIMELINE"]
                for label, detail, stamp in reversed(list(s.events)[-30:]):
                    lines.append(f"{time.strftime('%H:%M:%S', time.localtime(stamp))}  {label:<12} {detail}")
                return "\n".join(lines)

            def render_logs(self, s: TrainingState) -> str:
                return "LOGS · live\n\n" + "\n".join(list(s.logs)[-60:])

            def render_failure(self, s: TrainingState) -> str:
                return "\n".join([
                    "TRAINING FAILED",
                    "",
                    f"Step              {s.current_step:,} / {s.total_steps:,}" if s.total_steps else f"Step              {s.current_step:,}",
                    f"Loss              {s.loss:.6g}" if s.loss is not None else "Loss              —",
                    f"Reason            {s.failure_reason or 'unknown failure'}",
                    "",
                    f"LAST CHECKPOINT   {'✓ available' if s.checkpoint.available else '— unavailable'}",
                    f"Resume            {'available' if s.checkpoint.available else 'not available'}",
                    "",
                    "Original stderr/stdout remains available in Logs (7).",
                ])

            def render_completion(self, s: TrainingState) -> str:
                return "\n".join([
                    "LAPISLLM",
                    "TRAINING COMPLETE",
                    "",
                    f"{s.current_step:,} / {s.total_steps:,}" if s.total_steps else f"{s.current_step:,} steps",
                    f"Final loss        {s.loss:.6g}" if s.loss is not None else "Final loss        —",
                    f"Best loss         {s.best_loss:.6g}" if s.best_loss is not None else "Best loss         —",
                    f"Duration          {format_duration(s.elapsed)}",
                    f"Average throughput {s.step_rate:.2f} step/s" if s.step_rate else "Average throughput —",
                    f"Checkpoint        {s.checkpoint.path or '—'}",
                    "",
                    "STATUS            ✓ COMPLETED",
                    "",
                    "Commands are available only when implemented by the CLI; q exits.",
                ])

        TrainingApp().run()


def can_use_tui() -> bool:
    """Only use full-screen mode for an actual interactive TTY."""
    return not os.environ.get("LAPIS_NO_TUI") and "NO_COLOR" not in os.environ and sys.stdout.isatty()


def run_training_console(command: list[str], *, config: str, checkpoint: str) -> int:
    runner = TrainingConsoleRunner(command, config=config, checkpoint=checkpoint)
    TrainingAppController(runner).launch()
    return runner.process.returncode if runner.process and runner.process.returncode is not None else 1
