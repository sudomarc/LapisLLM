"""Application logging for LapisLLM.

Configuration is intentionally small: a level, optional file destination and a
single message format. Log records never carry credentials or personal data;
callers are responsible for scrubbing secrets before passing objects to log
calls.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

from .exceptions import LapisError

DEFAULT_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_LOGGER_NAME = "lapisllm"
_CONFIGURED = False


class LoggingError(LapisError):
    """Logging setup failed."""


def _redact(value: Any) -> Any:
    """Best-effort redaction of obvious secret-looking strings.

    This is a courtesy, not a security boundary: never log credentials.
    """
    if isinstance(value, str):
        lowered = value.lower()
        for token in ("token", "key", "secret", "password", "passwd", "credential"):
            if token in lowered:
                return "<redacted>"
    return value


def setup_logging(
    level: str | int = "INFO",
    log_file: str | Path | None = None,
    *,
    reset: bool = True,
) -> None:
    """Configure the ``lapisllm`` logger.

    Parameters
    ----------
    level:
        Logging level name or integer. Falls back to WARNING when unrecognizable.
    log_file:
        Optional file destination. The parent directory is created if missing.
        A failure to open the file raises :class:`LoggingError`.
    reset:
        Remove previously attached handlers before configuring (avoids duplicate
        output when called multiple times).
    """
    global _CONFIGURED

    logger = logging.getLogger(_LOGGER_NAME)
    if reset:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    logger.propagate = True

    try:
        resolved_level = (
            logging.getLevelName(str(level).upper()) if isinstance(level, str) else level
        )
        if not isinstance(resolved_level, int):
            resolved_level = logging.WARNING
        logger.setLevel(resolved_level)
    except (AttributeError, TypeError):
        logger.setLevel(logging.WARNING)

    formatter = logging.Formatter(DEFAULT_FORMAT, datefmt=DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(_SecretScrubbingFilter())
    logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.addFilter(_SecretScrubbingFilter())
            logger.addHandler(file_handler)
        except OSError as exc:
            raise LoggingError(f"Cannot open log file {path}: {exc}") from exc

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``lapisllm`` namespace."""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


class _SecretScrubbingFilter(logging.Filter):
    """Best-effort scrubbing of secret-looking kwargs in extra payloads."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key in ("password", "passwd", "secret", "token", "api_key", "credential"):
            if hasattr(record, key):
                setattr(record, key, _redact(getattr(record, key)))
        return True


def is_configured() -> bool:
    return _CONFIGURED
