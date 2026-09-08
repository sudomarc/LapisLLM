from typer.testing import CliRunner

from lapis.cli import app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Lapis language-model training and inference CLI" in result.stdout
    assert "train" in result.stdout
    assert "generate" in result.stdout
    assert "system" in result.stdout


def test_system_command() -> None:
    result = CliRunner().invoke(app, ["system"])
    assert result.exit_code == 0
    assert "Lapis System" in result.stdout
    assert "CPU" in result.stdout
    assert "RAM" in result.stdout
