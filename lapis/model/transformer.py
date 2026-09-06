"""Backward-compatible Transformer entry point.

The canonical implementation lives in :mod:`lapis.model.lapis_model`.
"""

from lapis.model.lapis_model import LapisModel


class LapisTransformer(LapisModel):
    """Compatibility wrapper around the canonical LAPIS Transformer."""

    def __init__(self, config):
        super().__init__(**config)
