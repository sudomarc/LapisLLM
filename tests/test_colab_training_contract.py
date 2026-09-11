from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts import _colab_train_impl as colab
from scripts import build_colab_corpus


ROOT = Path(__file__).resolve().parents[1]


def test_colab_runner_contains_no_input_calls():
    for relative in ("scripts/colab_train.py", "scripts/_colab_train_impl.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "input"
            for node in ast.walk(tree)
        )


def test_colab_defaults_to_one_run_without_in_training_generation(monkeypatch):
    monkeypatch.setattr(colab, "ensure_dependencies", lambda: None)
    monkeypatch.setattr(colab, "check_gpu", lambda requested: "cuda")
    monkeypatch.setattr(colab, "prepare_corpus", lambda args: None)
    monkeypatch.setattr(colab, "completed_run_numbers", lambda: set())
    calls = []
    monkeypatch.setattr(
        colab,
        "train_one_run",
        lambda run_number, total_runs, args, device, config: calls.append(
            (run_number, total_runs, args.monitor_interval, device, config)
        ),
    )
    monkeypatch.setattr(colab, "banner", lambda _title: None)
    monkeypatch.setattr(colab, "phase", lambda _name, _message: None)
    monkeypatch.setattr(colab, "get_github_token", lambda: None)
    monkeypatch.setattr(colab, "git_push", lambda _token, _smoke: True)
    monkeypatch.setattr("sys.argv", ["scripts.colab_train"])

    assert colab.main() == 0
    assert calls == [(1, 1, 0, "cuda", colab.CONFIG)]


def test_source_budgets_cover_global_budget_without_first_source_starvation():
    remaining = 200_000_000
    budgets = []
    for remaining_sources in (3, 2, 1):
        budget = build_colab_corpus.allocate_source_budget(remaining, remaining_sources)
        budgets.append(budget)
        remaining -= budget

    assert budgets == [66_666_667, 66_666_667, 66_666_666]
    assert sum(budgets) == 200_000_000


def test_corpus_cache_requires_current_builder_contract(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    manifest = tmp_path / "manifest.json"
    corpus.write_text("data", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "contract_version": "old",
                "max_chars": 200_000_000,
                "actual_chars": 4,
                "sources": [{"id": "old"}],
                "output": str(corpus),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(colab, "CORPUS", corpus)
    monkeypatch.setattr(colab, "MANIFEST", manifest)

    assert not colab.corpus_valid(200_000_000)


def test_training_monitor_default_is_disabled(monkeypatch):
    monkeypatch.setattr("sys.argv", ["scripts.colab_train"])
    args = colab.parse_args()
    assert args.monitor_interval == 0
    assert args.runs == 1
