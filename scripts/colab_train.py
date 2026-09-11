#!/usr/bin/env python3
"""Canonical non-interactive Colab entry point for LapisLLM training."""

from __future__ import annotations

from scripts._colab_train_impl import get_github_token, main

__all__ = ["get_github_token", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
