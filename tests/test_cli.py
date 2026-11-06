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


def test_dev_benchmark_save_and_compare_baseline(tmp_path, monkeypatch) -> None:
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
                "time_to_first_token_ms": 10.0,
                "total_time_ms": 40.0,
                "tokens_per_second": 250.0,
            }

        def generate_batch(self, prompts, sampling):
            return ["output"] * len(prompts)

        def generate_batch_with_metadata(self, prompts, sampling):
            return {
                "results": [
                    {
                        "text": "benchmark text",
                        "prompt_tokens": 5,
                        "completion_tokens": 10,
                        "total_tokens": 15,
                        "time_to_first_token_ms": 10.0,
                        "total_time_ms": 40.0,
                        "tokens_per_second": 250.0,
                    }
                    for _ in prompts
                ],
                "batch_size": len(prompts),
                "total_prompt_tokens": 5 * len(prompts),
                "total_completion_tokens": 10 * len(prompts),
                "total_tokens": 15 * len(prompts),
                "total_time_ms": 40.0,
                "tokens_per_second": 250.0 * len(prompts),
            }

    monkeypatch.setattr(
        "lapis.dev.cli.LapisRuntime.from_checkpoint",
        lambda checkpoint, device, quantize=False, dtype="float32": MockRuntime(),
    )

    baseline_file = tmp_path / "baseline.json"
    # 1. Run and save baseline with batch-size=2
    save_result = runner.invoke(
        app,
        ["dev", "benchmark", "--runs", "2", "--warmup", "1", "--batch-size", "2", "--save-baseline", str(baseline_file)],
    )
    assert save_result.exit_code == 0
    assert baseline_file.is_file()

    # 2. Compare against baseline with strict max-regression-pct (should pass as metrics match)
    cmp_result = runner.invoke(
        app,
        ["dev", "benchmark", "--runs", "2", "--warmup", "1", "--batch-size", "2", "--compare-baseline", str(baseline_file), "--max-regression-pct", "10.0"],
    )
    assert cmp_result.exit_code == 0
    assert "Baseline Comparison" in cmp_result.stdout
    assert "Delta (%)" in cmp_result.stdout

    # 3. Simulate regression threshold failure by setting threshold to 0.0 with a mocked regressed run
    bad_baseline = tmp_path / "fast_baseline.json"
    import json
    with open(bad_baseline, "w", encoding="utf-8") as f:
        json.dump(
            {
                "p50_latency_ms": 1.0,  # current is 40.0ms, huge regression
                "avg_throughput_tokens_per_s": 5000.0,
            },
            f,
        )

    fail_result = runner.invoke(
        app,
        ["dev", "benchmark", "--runs", "2", "--warmup", "0", "--compare-baseline", str(bad_baseline), "--max-regression-pct", "10.0"],
    )
    assert fail_result.exit_code == 1
    assert "REGRESSION ERROR" in fail_result.stdout
