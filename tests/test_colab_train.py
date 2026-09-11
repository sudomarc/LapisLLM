from __future__ import annotations

import ast
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


def test_colab_runner_defaults_to_one_run_and_disables_sampling() -> None:
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
        '"HF_HUB_DISABLE_XET": "1"}\n',
        encoding="utf-8",
    )
    assert module.corpus_valid(4)
