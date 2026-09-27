from typer.testing import CliRunner

from lapis.cli import app


runner = CliRunner()


def test_cli_help_is_developer_only() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "language-model engine" in result.stdout
    assert "developer" in result.stdout.lower()
    assert "dev" in result.stdout
    assert "api" in result.stdout
    assert "system" in result.stdout
    assert "chat" not in result.stdout
    assert "simple user inference" not in result.stdout


def test_cli_without_command_shows_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout
    assert "dev" in result.stdout
    assert "api" in result.stdout
    assert "chat" not in result.stdout


def test_top_level_chat_is_not_exposed() -> None:
    result = runner.invoke(app, ["chat"])
    assert result.exit_code != 0
    assert "No such command" in result.output


def test_dev_mode_help_includes_developer_inference() -> None:
    result = runner.invoke(app, ["dev", "--help"])
    assert result.exit_code == 0
    for command in ("chat", "train", "evaluate", "generate", "benchmark", "inspect"):
        assert command in result.stdout


def test_api_mode_help_includes_serve() -> None:
    result = runner.invoke(app, ["api", "--help"])
    assert result.exit_code == 0
    assert "serve" in result.stdout


def test_system_command() -> None:
    result = runner.invoke(app, ["system"])
    assert result.exit_code == 0
    assert "Lapis System" in result.stdout
    assert "CPU" in result.stdout
    assert "RAM" in result.stdout


def test_dev_benchmark_command(monkeypatch) -> None:
    class MockRuntime:
        device = "cpu"
        dtype = "float32"
        quantized = False

        def generate(self, prompt, sampling):
            return "output"

        def generate_with_metadata(self, prompt, sampling):
            return {
                "text": "benchmark text",
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
                "time_to_first_token_ms": 12.5,
                "total_time_ms": 50.0,
                "tokens_per_second": 200.0,
            }

    monkeypatch.setattr(
        "lapis.dev.cli.LapisRuntime.from_checkpoint",
        lambda checkpoint, device, quantize=False, dtype="float32": MockRuntime(),
    )
    result = runner.invoke(app, ["dev", "benchmark", "--runs", "2", "--warmup", "1"])
    assert result.exit_code == 0
    assert "Inference Benchmark" in result.stdout
    assert "Avg Throughput" in result.stdout
    assert "p50 Latency" in result.stdout
    assert "p95 Latency" in result.stdout
