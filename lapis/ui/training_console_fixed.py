"""Compatibility TUI for Lapis training without unsupported TCSS media rules."""

from __future__ import annotations

import os
import sys
import threading
import time

from .training_console_legacy import TrainingConsoleRunner, TrainingState, format_duration, _spark, _trend


def can_use_tui() -> bool:
    """Return True only when an interactive TTY and Textual are available."""
    if os.environ.get("LAPIS_NO_TUI") or "NO_COLOR" in os.environ or not sys.stdout.isatty():
        return False
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


class TrainingAppController:
    """Small, Textual-compatible full-screen training dashboard."""

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
            Screen {
                background: #0B0F14;
                color: #F3F6F9;
            }
            Header {
                background: #111720;
                color: #F3F6F9;
                height: 3;
            }
            Footer {
                background: #111720;
                color: #8B98A8;
                height: 1;
            }
            #shell {
                height: 1fr;
            }
            #sidebar {
                width: 23;
                background: #111720;
                border-right: solid #202936;
                padding: 1 1;
            }
            #workspace {
                width: 1fr;
                padding: 1;
            }
            #view {
                height: 1fr;
            }
            """
            BINDINGS = [
                ("r", "refresh", "Refresh"),
                ("p", "pause_resume", "Pause/resume"),
                ("q", "quit_app", "Quit"),
            ]

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
                return "\n".join(
                    [
                        "LAPISLLM",
                        "TRAINING",
                        "",
                        f"STATUS  {state.status}",
                        f"RUN     {state.run_id}",
                        "",
                        "CONTROLS",
                        "r  refresh",
                        "p  pause/resume",
                        "q  quit",
                    ]
                )

            def render_view(self, state: TrainingState) -> str:
                pct = (100.0 * state.current_step / state.total_steps) if state.total_steps else 0.0
                loss = f"{state.loss:.6g}" if state.loss is not None else "—"
                best = f"{state.best_loss:.6g}" if state.best_loss is not None else "—"
                speed = f"{state.step_rate:.2f} step/s" if state.step_rate else "—"
                eta = format_duration(state.eta) if state.eta is not None else "—"
                lines = [
                    "LAPISLLM TRAINING",
                    "",
                    f"STEP             {state.current_step:,} / {state.total_steps:,}   ({pct:.1f}%)",
                    f"LOSS             {loss}",
                    f"BEST LOSS        {best}",
                    f"TREND            {_trend(state.loss_history)}",
                    f"THROUGHPUT       {speed}",
                    f"TOKENS/SECOND    {state.tokens_per_second:,.0f}" if state.tokens_per_second else "TOKENS/SECOND    —",
                    f"ETA              {eta}",
                    f"ELAPSED          {format_duration(state.elapsed)}",
                    "",
                    f"LOSS HISTORY\n{_spark(state.loss_history)}",
                ]
                if state.gpu:
                    lines += [
                        "",
                        f"GPU              {state.gpu}",
                        f"GPU UTILIZATION  {state.gpu_utilization:.0f}%" if state.gpu_utilization is not None else "GPU UTILIZATION  —",
                        f"VRAM             {state.vram_used:.1f} / {state.vram_total:.1f} GB" if state.vram_total is not None else "VRAM             —",
                        f"TEMPERATURE      {state.gpu_temperature:.0f}°C" if state.gpu_temperature is not None else "TEMPERATURE      —",
                    ]
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
    if runner.process and runner.process.returncode is not None:
        return runner.process.returncode
    return 0
