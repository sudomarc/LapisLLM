import json
from pathlib import Path

import pytest

from lapis.data.cleaner import clean_dataset


def test_clean_dataset_fails_on_missing_input(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Input directory not found"):
        clean_dataset(tmp_path / "missing", tmp_path / "out")


def test_clean_dataset_fails_on_empty_input(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="No supported input files"):
        clean_dataset(source, tmp_path / "out")


def test_clean_dataset_preserves_json_arrays(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.json").write_text(json.dumps([" café  ", "hello"]), encoding="utf-8")

    output = tmp_path / "out"
    clean_dataset(source, output)
    cleaned = json.loads((output / "data.json").read_text(encoding="utf-8"))
    assert cleaned == ["cafe", "hello"]


def test_clean_dataset_rejects_invalid_json(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON input"):
        clean_dataset(source, tmp_path / "out")
