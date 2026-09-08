from __future__ import annotations

import json
from pathlib import Path

from scripts import colab_train


def test_training_metric_regex_extracts_live_metrics() -> None:
    line = "step=0125 loss=4.1821 ppl=65.58 lr=2.81e-4 tokens=1,280,000"
    match = colab_train.METRIC_RE.search(line)
    assert match is not None
    assert int(match.group("step")) == 125
    assert float(match.group("loss")) == 4.1821
    assert float(match.group("ppl")) == 65.58
    assert float(match.group("lr")) == 2.81e-4
    token_match = colab_train.TOKEN_RE.search(line)
    assert token_match is not None
    assert int(token_match.group("tokens").replace(",", "")) == 1_280_000


def test_completed_run_numbers_ignore_failed_runs_and_keep_legacy_history(
    tmp_path: Path, monkeypatch
) -> None:
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
    monkeypatch.setattr(colab_train, "HISTORY", history)
    assert colab_train.completed_run_numbers() == {1, 3}


def test_format_duration_is_stable() -> None:
    assert colab_train.format_duration(0) == "0m 00s"
    assert colab_train.format_duration(65) == "1m 05s"
    assert colab_train.format_duration(3661) == "1h 01m 01s"
