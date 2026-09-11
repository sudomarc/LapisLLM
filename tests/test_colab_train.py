from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "colab_train.py"


def _module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("lapis_colab_train", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_colab_runner_has_no_input_call() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    input_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "input"
    ]
    assert not input_calls


def test_colab_runner_defaults_and_documented_flags() -> None:
    module = _module()
    original = sys.argv[:]
    try:
        sys.argv = [str(SCRIPT)]
        args = module.parse_args()
    finally:
        sys.argv = original

    assert args.runs == 1
    assert args.monitor_interval == 0
    assert args.device == "auto"
    assert args.max_chars == 200_000_000
    assert args.resume is False
    assert args.smoke_test is False


def test_documented_flags_parse() -> None:
    module = _module()
    original = sys.argv[:]
    try:
        sys.argv = [str(SCRIPT), "--resume", "--smoke-test"]
        args = module.parse_args()
    finally:
        sys.argv = original

    assert args.resume is True
    assert args.smoke_test is True


def test_corpus_cache_requires_current_builder_marker(tmp_path: Path) -> None:
    module = _module()
    module.CORPUS = tmp_path / "corpus.txt"
    module.MANIFEST = tmp_path / "manifest.json"
    module.CORPUS.write_text("data", encoding="utf-8")

    module.MANIFEST.write_text(
        '{"actual_chars": 4, "max_chars": 4, "output": "corpus.txt"}\n',
        encoding="utf-8",
    )
    assert not module.corpus_valid(4)

    module.MANIFEST.write_text(
        '{"actual_chars": 4, "max_chars": 4, "output": "corpus.txt", '
        '"hf_hub_disable_xet": "1"}\n',
        encoding="utf-8",
    )
    assert module.corpus_valid(4)


def test_legacy_observability_helpers_remain_available(tmp_path: Path) -> None:
    module = _module()
    history = tmp_path / "training_history"
    (history / "run-001").mkdir(parents=True)
    (history / "run-002").mkdir(parents=True)
    (history / "run-003").mkdir(parents=True)
    (history / "run-001" / "summary.json").write_text(
        json.dumps({"run_number": 1, "status": "completed"})
    )
    (history / "run-002" / "summary.json").write_text(
        json.dumps({"run_number": 2, "status": "failed"})
    )
    (history / "run-003" / "summary.json").write_text(
        json.dumps({"run_number": 3})
    )
    module.HISTORY = history

    assert module.completed_run_numbers() == {1, 3}
    assert module.completed_runs() == {1, 3}
    assert module.format_duration(0) == "0m 00s"
    assert module.format_duration(65) == "1m 05s"
    assert module.format_duration(3661) == "1h 01m 01s"


def test_write_history_uses_json_null_for_non_finite_metrics(tmp_path: Path) -> None:
    module = _module()
    module.HISTORY = tmp_path / "history"
    module.MANIFEST = tmp_path / "manifest.json"
    module.MANIFEST.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    module.write_history(
        run_number=1,
        runs=1,
        checkpoint=checkpoint,
        metrics={"step": 10.0, "loss": float("nan"), "ppl": float("inf"), "lr": 1e-4, "tokens": 100.0},
        samples=[],
        started=0.0,
        device="cpu",
    )

    summary = json.loads((module.HISTORY / "run-001" / "summary.json").read_text())
    assert summary["loss"] is None
    assert summary["perplexity"] is None
    assert summary["learning_rate"] == 1e-4
    assert json.dumps(summary, allow_nan=False)


def test_preview_continues_after_one_prompt_failure(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    prompts = iter(module.DEFAULT_PROMPTS)
    calls = 0

    def fake_run(command, *, capture=False, env=None):
        nonlocal calls
        calls += 1
        next(prompts)
        if calls == 1:
            raise subprocess.CalledProcessError(1, command, stderr="generation failed")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(module, "run", fake_run)
    samples = module.preview(checkpoint, "cpu")

    assert calls == len(module.DEFAULT_PROMPTS)
    assert samples[0]["completion"] == ""
    assert samples[1]["completion"] == "ok"
    assert samples[2]["completion"] == "ok"
