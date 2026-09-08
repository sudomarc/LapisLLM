"""Checkpoint-backed inference runtime.

This module intentionally contains inference-only primitives. Training, optimizer,
dataset, and checkpoint-writing operations are not exposed by this API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lapis.config.base import resolve_device
from lapis.config.model_config import model_config_kwargs
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer


class CheckpointLoadError(RuntimeError):
    """Raised when an inference checkpoint cannot be loaded safely."""


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95

    def validate(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the range (0, 1]")


def _sample_next_token(logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    config.validate()
    logits = logits / config.temperature

    if config.top_k > 0:
        values, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities > config.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Sampling produced non-finite probabilities")
    return torch.multinomial(probabilities, num_samples=1)


def _validate_checkpoint_tokenizer(checkpoint: dict[str, Any], tokenizer: Tokenizer) -> None:
    checkpoint_version = checkpoint.get("tokenizer_version")
    if checkpoint_version is not None and checkpoint_version != tokenizer.VERSION:
        raise CheckpointLoadError(
            "Checkpoint tokenizer version does not match the installed tokenizer."
        )

    checkpoint_vocab = checkpoint.get("config", {}).get("model", {}).get("vocab_size")
    if checkpoint_vocab is not None and int(checkpoint_vocab) != tokenizer.vocab_size:
        raise CheckpointLoadError("Checkpoint vocabulary size does not match the tokenizer.")


class LapisRuntime:
    """Inference-only runtime backed by an approved/read-only checkpoint."""

    def __init__(self, model: LapisModel, tokenizer: Tokenizer, device: torch.device, checkpoint: Path) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint = checkpoint

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: str = "auto") -> "LapisRuntime":
        path = Path(checkpoint)
        if not path.exists() or not path.is_file():
            raise CheckpointLoadError(f"Checkpoint not found: {path}")

        try:
            target = resolve_device(device)
            data = torch.load(path, map_location=target, weights_only=True)
            tokenizer_path = path.parent / "tokenizer"
            if not tokenizer_path.is_dir():
                raise CheckpointLoadError(f"Tokenizer directory not found: {tokenizer_path}")
            tokenizer = Tokenizer.load(str(tokenizer_path))
            _validate_checkpoint_tokenizer(data, tokenizer)
            model = LapisModel(**model_config_kwargs(data["config"])).to(target)
            model.load_state_dict(data["model_state_dict"])
            model.eval()
            return cls(model, tokenizer, target, path)
        except CheckpointLoadError:
            raise
        except (KeyError, RuntimeError, ValueError, OSError, TypeError) as exc:
            raise CheckpointLoadError(f"Unable to load checkpoint {path}: {exc}") from exc

    def generate(self, prompt: str, sampling: SamplingConfig | None = None) -> str:
        config = sampling or SamplingConfig()
        config.validate()
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = [self.tokenizer.bos_id]
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.inference_mode():
            for _ in range(config.max_new_tokens):
                context = ids[:, -self.model.max_position_embeddings :]
                logits, _ = self.model(context)
                next_id = _sample_next_token(logits[:, -1, :], config)
                ids = torch.cat([ids, next_id], dim=1)
                if next_id.item() == self.tokenizer.eos_id:
                    break

        return self.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)
