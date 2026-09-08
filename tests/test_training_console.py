from __future__ import annotations

from lapis.ui.training_console import (
    TrainingTelemetry,
    can_use_tui,
    format_duration,
    sanitize_output,
)


def test_sanitize_output_removes_controls_and_clips() -> None:
    value = "hello\x1b[31m\nworld\x00\nthird"
    result = sanitize_output(value, max_width=5, max_lines=2)
    assert "hello" in result
    assert "\x00" not in result
    assert "third" not in result
    assert result.endswith("…")


def test_format_duration_is_compact() -> None:
    assert format_duration(0) == "0m 00s"
    assert format_duration(125) == "2m 05s"
    assert format_duration(3661) == "1h 01m"
    assert format_duration(None) == "—"


def test_telemetry_tracks_history_best_loss_and_eta() -> None:
    telemetry = TrainingTelemetry("run-test", "tiny.yaml", total_steps=100)
    telemetry.update_metric(step=10, loss=1.0, learning_rate=3e-4)
    telemetry.update_metric(step=20, loss=0.8, learning_rate=2e-4)
    state = telemetry.snapshot()
    assert state.current_step == 20
    assert state.best_loss == 0.8
    assert len(state.loss_history) == 2
    assert state.learning_rate == 2e-4
    assert state.step_rate is not None


def test_can_use_tui_is_safe_in_non_tty_tests(monkeypatch) -> None:
    monkeypatch.setenv("LAPIS_NO_TUI", "1")
    assert can_use_tui() is False
