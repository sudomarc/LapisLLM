from typer.testing import CliRunner

from lapis.cli import app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "simple user inference" in result.stdout
    assert "chat" in result.stdout
    assert "dev" in result.stdout
    assert "system" in result.stdout
    assert "train" not in result.stdout


def test_system_command() -> None:
    result = CliRunner().invoke(app, ["system"])
    assert result.exit_code == 0
    assert "Lapis System" in result.stdout
    assert "CPU" in result.stdout
    assert "RAM" in result.stdout
