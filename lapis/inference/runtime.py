"""Product-agnostic checkpoint-backed inference runtime."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

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
    seed: int | None = None

    def validate(self) -> None:
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool):
            raise ValueError("max_new_tokens must be an integer")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and greater than 0")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
            raise ValueError("top_k must be an integer")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if (
            not isinstance(self.top_p, (int, float))
            or isinstance(self.top_p, bool)
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be finite and in the range (0, 1]")
        if self.seed is not None and (
            not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer or None")


def _sample_next_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    config.validate()
    if not torch.isfinite(logits).all():
        raise RuntimeError("Model produced non-finite logits")
    logits = logits / config.temperature

    if config.top_k > 0:
        values, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Sampling produced non-finite probabilities")
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities > config.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Sampling produced non-finite probabilities")
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


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
    """Inference-only runtime backed by a read-only model checkpoint."""

    def __init__(
        self,
        model: LapisModel | torch.nn.Module,
        tokenizer: Tokenizer,
        device: torch.device | str,
        checkpoint: Path,
        quantized: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint = checkpoint
        self.quantized = quantized

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        device: str = "auto",
        quantize: bool = False,
    ) -> "LapisRuntime":
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

            quantized = False
            if quantize:
                if target.type != "cpu":
                    raise CheckpointLoadError(
                        "Dynamic quantization is currently supported on CPU target device."
                    )
                model = torch.ao.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                quantized = True

            return cls(model, tokenizer, target, path, quantized=quantized)
        except CheckpointLoadError:
            raise
        except (KeyError, RuntimeError, ValueError, OSError, TypeError) as exc:
            raise CheckpointLoadError(f"Unable to load checkpoint {path}: {exc}") from exc

    def tokenize(self, text: str) -> list[int]:
        """Encode text using the checkpoint-compatible tokenizer."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self.tokenizer.encode(text, add_special_tokens=False)

    def get_model_info(self) -> dict[str, Any]:
        """Return stable runtime metadata without exposing model internals."""
        return {
            "checkpoint": str(self.checkpoint),
            "device": str(self.device),
            "vocab_size": self.tokenizer.vocab_size,
            "context_length": getattr(self.model, "max_position_embeddings", 512),
            "parameter_count": sum(parameter.numel() for parameter in self.model.parameters()),
            "tokenizer_version": self.tokenizer.VERSION,
            "quantized": getattr(self, "quantized", False),
            "capabilities": self.get_capabilities(),
        }

    def get_capabilities(self) -> dict[str, Any]:
        """Return capabilities metadata exposed to consumer applications like CHAD."""
        return {
            "context_length": getattr(self.model, "max_position_embeddings", 512),
            "tokenizer_version": self.tokenizer.VERSION,
            "vocab_size": self.tokenizer.vocab_size,
            "streaming": True,
            "quantized": getattr(self, "quantized", False),
            "sampling_parameters": ["max_new_tokens", "temperature", "top_k", "top_p", "seed"],
        }

    def _prepare_ids(self, prompt: str) -> torch.Tensor:
        token_ids = self.tokenize(prompt)
        if not token_ids:
            token_ids = [self.tokenizer.bos_id]
        context_limit = getattr(self.model, "max_position_embeddings", 512)
        if len(token_ids) > context_limit:
            token_ids = token_ids[-context_limit:]
        device_arg = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        return torch.tensor([token_ids], dtype=torch.long, device=device_arg)

    def _iter_generated_token_ids(self, prompt: str, config: SamplingConfig) -> Iterator[int]:
        ids = self._prepare_ids(prompt)
        generator = None
        if config.seed is not None:
            dev_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
            generator = torch.Generator(device=dev_type)
            generator.manual_seed(config.seed)

        context_limit = getattr(self.model, "max_position_embeddings", 512)

        with torch.inference_mode():
            logits, _, kv_cache = self.model(ids, use_cache=True, start_pos=0)
            next_id = _sample_next_token(logits[:, -1, :], config, generator=generator)
            start_pos = ids.size(1)

            for _ in range(config.max_new_tokens):
                token_id = int(next_id.item())
                if token_id == self.tokenizer.eos_id:
                    break
                yield token_id
                if start_pos >= context_limit:
                    break
                logits, _, kv_cache = self.model(
                    next_id, kv_cache_list=kv_cache, start_pos=start_pos, use_cache=True
                )
                start_pos += 1
                next_id = _sample_next_token(logits[:, -1, :], config, generator=generator)

    def generate(self, prompt: str, sampling: SamplingConfig | None = None) -> str:
        config = sampling or SamplingConfig()
        config.validate()
        generated = list(self._iter_generated_token_ids(prompt, config))
        prompt_ids = self.tokenize(prompt)
        context_limit = getattr(self.model, "max_position_embeddings", 512)
        if len(prompt_ids) > context_limit:
            prompt_ids = prompt_ids[-context_limit:]
        return self.tokenizer.decode(prompt_ids + generated, skip_special_tokens=True)

    def generate_with_metadata(
        self, prompt: str, sampling: SamplingConfig | None = None
    ) -> dict[str, Any]:
        """Generate text and return response metadata (tokens, TTFT, total time, throughput)."""
        config = sampling or SamplingConfig()
        config.validate()

        start_time = time.perf_counter()
        ttft_s: float | None = None
        generated: list[int] = []

        for token_id in self._iter_generated_token_ids(prompt, config):
            if ttft_s is None:
                ttft_s = time.perf_counter() - start_time
            generated.append(token_id)

        total_time_s = time.perf_counter() - start_time
        if ttft_s is None:
            ttft_s = total_time_s

        prompt_ids = self.tokenize(prompt)
        context_limit = getattr(self.model, "max_position_embeddings", 512)
        if len(prompt_ids) > context_limit:
            prompt_ids = prompt_ids[-context_limit:]

        text = self.tokenizer.decode(prompt_ids + generated, skip_special_tokens=True)
        completion_tokens = len(generated)
        prompt_tokens = len(prompt_ids)
        total_tokens = prompt_tokens + completion_tokens

        tokens_per_second = (
            completion_tokens / total_time_s if total_time_s > 0 else 0.0
        )

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "time_to_first_token_ms": round(ttft_s * 1000, 2),
            "total_time_ms": round(total_time_s * 1000, 2),
            "tokens_per_second": round(tokens_per_second, 2),
        }

    def stream_generate(self, prompt: str, sampling: SamplingConfig | None = None) -> Iterator[str]:
        """Yield text chunks only after the decoded prefix is stable."""
        config = sampling or SamplingConfig()
        config.validate()
        generated: list[int] = []
        emitted = ""
        previous = ""

        for token_id in self._iter_generated_token_ids(prompt, config):
            generated.append(token_id)
            current = self.tokenizer.decode(generated, skip_special_tokens=True)
            stable_length = 0
            limit = min(len(previous), len(current))
            while stable_length < limit and previous[stable_length] == current[stable_length]:
                stable_length += 1
            if stable_length > len(emitted):
                yield current[len(emitted) : stable_length]
                emitted = current[:stable_length]
            previous = current

        if len(previous) > len(emitted):
            yield previous[len(emitted) :]
