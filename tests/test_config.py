"""Behavior tests for centralized configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from lapis.config import DEFAULTS, LapisConfig, load_config, save_config
from lapis.exceptions import ConfigError


def test_load_defaults_only() -> None:
    config, source = load_config()
    assert config.project_name == DEFAULTS["project_name"]
    assert config.config_version == DEFAULTS["config_version"]
    assert config.data_root == Path(DEFAULTS["data_root"])
    assert config.runs_dir == Path(DEFAULTS["runs_dir"])
    assert config.log_level == "INFO"
    assert config.default_device == "auto"
    assert config.default_precision == "fp32"
    assert config.seed == DEFAULTS["seed"]
    assert config.max_cpu_threads is None
    assert source.file_path is None
    assert source.file_values == {}


def test_load_toml_file(tmp_path: Path) -> None:
    config_file = tmp_path / "lapis.toml"
    config_file.write_text(
        '[project]\nlog_level = "DEBUG"\ndata_root = "local-store"\nseed = 7\n',
        encoding="utf-8",
    )
    config, source = load_config(config_file)
    assert config.log_level == "DEBUG"
    assert config.data_root == tmp_path / "local-store"
    assert config.seed == 7
    assert source.file_path == config_file
    assert source.file_values["log_level"] == "DEBUG"


def test_relative_paths_resolve_from_config_directory(tmp_path: Path) -> None:
    sub = tmp_path / "configs"
    sub.mkdir()
    config_file = sub / "lapis.toml"
    config_file.write_text('[project]\nruns_dir = "runs"\n', encoding="utf-8")
    config, _ = load_config(config_file)
    assert config.runs_dir == sub / "runs"


def test_absolute_paths_kept_as_is(tmp_path: Path) -> None:
    config_file = tmp_path / "lapis.toml"
    absolute = tmp_path / "absolute-data"
    config_file.write_text(f'[project]\ndata_root = "{absolute.as_posix()}"\n', encoding="utf-8")
    config, _ = load_config(config_file)
    assert config.data_root == tmp_path / "absolute-data"


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAPIS_SEED", "4242")
    monkeypatch.setenv("LAPIS_LOG_LEVEL", "WARNING")
    config, source = load_config()
    assert config.seed == 4242
    assert config.log_level == "WARNING"
    assert source.env_values["seed"] == 4242


def test_env_var_invalid_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAPIS_SEED", "not-a-number")
    with pytest.raises(ConfigError, match="not a valid integer"):
        load_config()


def test_unknown_config_key_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "lapis.toml"
    config_file.write_text('[project]\nunknown_key = "value"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown config key"):
        load_config(config_file)


def test_invalid_log_level_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "lapis.toml"
    config_file.write_text('[project]\nlog_level = "VERBOSE"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid log_level"):
        load_config(config_file)


def test_invalid_device_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "lapis.toml"
    config_file.write_text('[project]\ndefault_device = "cuda:42"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid default_device"):
        load_config(config_file)


def test_invalid_precision_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "lapis.toml"
    config_file.write_text('[project]\ndefault_precision = "tf32"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid default_precision"):
        load_config(config_file)


def test_explicit_missing_config_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does-not-exist.toml")


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    config = LapisConfig(log_level="ERROR", seed=99)
    target = tmp_path / "saved.toml"
    save_config(config, target)
    loaded, _ = load_config(target)
    assert loaded.log_level == "ERROR"
    assert loaded.seed == 99
    # Relative paths are resolved against the config file's directory.
    assert loaded.data_root == tmp_path / ".lapis-data"


def test_env_priority_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "lapis.toml"
    config_file.write_text("[project]\nseed = 1\n", encoding="utf-8")
    monkeypatch.setenv("LAPIS_SEED", "2")
    config, _ = load_config(config_file)
    assert config.seed == 2
