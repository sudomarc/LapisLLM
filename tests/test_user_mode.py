from pathlib import Path

from typer.testing import CliRunner

from lapis.cli import app
from lapis.inference.runtime import CheckpointLoadError, LapisRuntime
from lapis.user.runtime import UserRuntime


runner = CliRunner()


def test_default_mode_is_user(monkeypatch) -> None:
    called = []
    monkeypatch.setattr("lapis.cli.chat", lambda: called.append("chat"))
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert called == ["chat"]


def test_dev_mode_help() -> None:
    result = runner.invoke(app, ["dev", "--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "benchmark" in result.stdout
    assert "inspect" in result.stdout


def test_user_config_contains_only_inference_settings() -> None:
    config_path = Path("configs/user/default.yaml")
    text = config_path.read_text(encoding="utf-8")
    assert "temperature:" in text
    assert "top_k:" in text
    assert "top_p:" in text
    assert "max_new_tokens:" in text
    assert "learning_rate:" not in text
    assert "optimizer" not in text
    assert "gradient_accumulation" not in text


def test_user_runtime_exposes_generation_only() -> None:
    assert hasattr(UserRuntime, "generate")
    assert not hasattr(UserRuntime, "train")
    assert not hasattr(UserRuntime, "evaluate")
    assert not hasattr(UserRuntime, "save_checkpoint")


def test_invalid_checkpoint_is_cleanly_rejected(tmp_path) -> None:
    missing = tmp_path / "missing.pt"
    try:
        LapisRuntime.from_checkpoint(missing)
    except CheckpointLoadError as exc:
        assert "Checkpoint not found" in str(exc)
    else:
        raise AssertionError("missing checkpoint should fail")
