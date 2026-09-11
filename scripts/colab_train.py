#!/usr/bin/env python3
"""Canonical non-interactive Colab entry point for LapisLLM training.

This module preserves the stable public helper surface while delegating the
workflow implementation to ``scripts._colab_train_impl``.
"""

from __future__ import annotations

from scripts import _colab_train_impl as _impl

CONFIG = _impl.CONFIG
SMOKE_CONFIG = _impl.SMOKE_CONFIG
CORPUS = _impl.CORPUS
MANIFEST = _impl.MANIFEST
HISTORY = _impl.HISTORY
CHECKPOINTS = _impl.CHECKPOINTS
METRIC_RE = _impl.METRIC_RE
TOKEN_RE = _impl.TOKEN_RE
format_duration = _impl.format_duration
corpus_valid = _impl.corpus_valid
prepare_corpus = _impl.prepare_corpus
banner = _impl.banner
phase = _impl.phase
ensure_dependencies = _impl.ensure_dependencies
check_gpu = _impl.check_gpu
completed_run_numbers = _impl.completed_run_numbers
train_one_run = _impl.train_one_run
git_push = _impl.git_push
get_github_token = _impl.get_github_token
parse_args = _impl.parse_args
main = _impl.main

__all__ = [
    "CHECKPOINTS",
    "CONFIG",
    "CORPUS",
    "HISTORY",
    "MANIFEST",
    "METRIC_RE",
    "SMOKE_CONFIG",
    "TOKEN_RE",
    "banner",
    "check_gpu",
    "completed_run_numbers",
    "corpus_valid",
    "ensure_dependencies",
    "format_duration",
    "get_github_token",
    "git_push",
    "main",
    "parse_args",
    "phase",
    "prepare_corpus",
    "train_one_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
