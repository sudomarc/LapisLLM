"""Behavior tests for the ``lapis doctor`` command (issue #74)."""

from __future__ import annotations

import json

from lapis.cli import build_parser, main
from lapis.cli.doctor import doctor_command, run_audit, run_benchmark


def test_main_without_subcommand_returns_usage_error() -> None:
    assert main([]) == 2


def test_build_parser_has_doctor_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["doctor", "--json"])
    assert args.func is doctor_command
    assert args.json is True


def test_run_audit_returns_hardware_report() -> None:
    report = run_audit()
    assert report.python_version
    assert isinstance(report.accelerators, list)
    assert report.precision["cpu"].fp32 is True


def test_run_audit_as_json_emits_valid_json(capsys) -> None:
    run_audit(as_json=True)
    output = capsys.readouterr().out
    data = json.loads(output)
    assert "timestamp_utc" in data
    assert "cpu" in data


def test_doctor_command_runs(capsys) -> None:
    class Args:
        log_level = "WARNING"
        log_file = None
        bench = False
        json = False

    assert doctor_command(Args()) == 0
    output = capsys.readouterr().out
    assert "LapisLLM — hardware audit report" in output


def test_run_benchmark_cpu_is_bounded_and_measurable() -> None:
    result = run_benchmark(
        device="cpu",
        dtype="fp32",
        batch_size=2,
        sequence_length=32,
        vocab_size=256,
        iterations=5,
        warmup=1,
    )
    assert result.tokens_per_second > 0
    assert result.wall_time_seconds > 0
    assert result.iterations == 5


def test_main_benchmark_returns_zero(capsys) -> None:
    assert main(["doctor", "--benchmark", "--iterations", "3"]) == 0
    output = capsys.readouterr().out
    assert "tokens/sec" in output
