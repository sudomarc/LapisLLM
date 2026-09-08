"""Small user-facing facade over the shared inference runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lapis.config.base import load_config
from lapis.inference.runtime import LapisRuntime, SamplingConfig


class UserRuntime:
    """Public inference surface for end users.

    It exposes generation only; training, dataset and checkpoint-management APIs
    intentionally remain outside this facade.
    """

    def __init__(self, runtime: LapisRuntime, sampling: SamplingConfig) -> None:
        self._runtime = runtime
        self.sampling = sampling

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/user/default.yaml") -> "UserRuntime":
        config: dict[str, Any] = load_config(config_path)
        checkpoint = str(config.get("model", "latest"))
        if checkpoint == "latest":
            checkpoint = "checkpoints/latest.pt"
        runtime = LapisRuntime.from_checkpoint(checkpoint, str(config.get("device", "auto")))
        sampling = SamplingConfig(
            max_new_tokens=int(config.get("max_new_tokens", 128)),
            temperature=float(config.get("temperature", 0.8)),
            top_k=int(config.get("top_k", 40)),
            top_p=float(config.get("top_p", 0.95)),
        )
        sampling.validate()
        return cls(runtime, sampling)

    @property
    def runtime(self) -> LapisRuntime:
        """Read-only access to inference metadata for UI integrations."""
        return self._runtime

    def generate(self, prompt: str) -> str:
        return self._runtime.generate(prompt, self.sampling)
