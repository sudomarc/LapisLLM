from typer.testing import CliRunner

from lapis.cli import app


def test_cli_help_is_developer_only() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "language-model engine" in result.stdout
    assert "developer" in result.stdout.lower()
    assert "dev" in result.stdout
    assert "system" in result.stdout
    assert "chat" not in result.stdout
    assert "simple user inference" not in result.stdout


def test_cli_without_command_shows_help() -> None:
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 2
    assert "Usage:" in result.stdout
    assert "dev" in result.stdout


def test_dev_mode_help_includes_developer_inference() -> None:
    result = CliRunner().invoke(app, ["dev", "--help"])
    assert result.exit_code == 0
    for command in ("chat", "train", "evaluate", "generate", "benchmark", "inspect"):
        assert command in result.stdout


def test_system_command() -> None:
    result = CliRunner().invoke(app, ["system"])
    assert result.exit_code == 0
    assert "Lapis System" in result.stdout
    assert "CPU" in result.stdout
    assert "RAM" in result.stdout
