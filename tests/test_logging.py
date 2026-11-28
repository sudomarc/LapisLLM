"""Behavior tests for application logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lapis.logging import LoggingError, get_logger, is_configured, setup_logging


def _drop_file_handlers() -> None:
    logger = logging.getLogger("lapisllm")
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)


def test_setup_logging_configures_once(tmp_path: Path) -> None:
    setup_logging(level="INFO", log_file=tmp_path / "run.log")
    assert is_configured()
    logger = logging.getLogger("lapisllm")
    handlers = logger.handlers
    assert len(handlers) >= 1  # stream + file

    # Calling again resets handlers instead of duplicating them.
    setup_logging(level="DEBUG")
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG
    _drop_file_handlers()


def test_log_file_is_created_and_written(tmp_path: Path) -> None:
    log_file = tmp_path / "sub" / "run.log"
    setup_logging(level="INFO", log_file=log_file)
    logger = get_logger("app.tests")
    logger.info("hello logging test")
    assert log_file.is_file()
    content = log_file.read_text(encoding="utf-8")
    assert "hello logging test" in content
    assert "[lapisllm.app.tests]" in content
    _drop_file_handlers()


def test_log_file_invalid_path_raises(tmp_path: Path) -> None:
    # Point the log file inside an existing regular file: parent creation fails.
    blocking_file = tmp_path / "blocking.txt"
    blocking_file.write_text("x", encoding="utf-8")
    try:
        with pytest.raises(LoggingError):
            setup_logging(level="INFO", log_file=blocking_file / "nested.log")
    finally:
        _drop_file_handlers()


def test_get_logger_returns_child_logger() -> None:
    setup_logging(level="WARNING")
    first = get_logger("alpha")
    second = get_logger("alpha.beta")
    assert second.name == "lapisllm.alpha.beta"
    assert first.name == "lapisllm.alpha"
    # Children propagate to the configured "lapisllm" logger.
    assert first.parent is not None
