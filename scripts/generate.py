#!/usr/bin/env python3
"""Generate text from a trained Lapis checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from lapis.checkpoint import CheckpointError, load_checkpoint
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested, but CUDA is unavailable")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested CUDA device {device.index}, but only {torch.cuda.device_count()} device(s) are available"
        )
    return device


def load_model(checkpoint_path: str, device: torch.device) -> tuple[LapisModel, dict]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise CheckpointError("Checkpoint does not contain a valid model configuration")
    try:
        model = LapisModel(**config["model"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError(
            "Checkpoint model state is incompatible with its model configuration"
        ) from exc
    model.eval()
    return model, checkpoint


def load_tokenizer(checkpoint_path: str, checkpoint: dict) -> Tokenizer:
    embedded = checkpoint.get("tokenizer_json")
    if embedded:
        tokenizer = Tokenizer.from_json(embedded)
    else:
        tokenizer = Tokenizer.load(str(Path(checkpoint_path).parent / "tokenizer"))
    expected = checkpoint.get("tokenizer_vocab_size")
    if expected is not None and int(expected) != tokenizer.vocab_size:
        raise CheckpointError("Checkpoint tokenizer metadata does not match the tokenizer")
    return tokenizer


def sample_next_token(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> torch.Tensor:
    if not torch.isfinite(logits).all():
        raise RuntimeError("Model produced non-finite generation logits")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be finite, greater than 0, and at most 1")
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scaled = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
        cutoff = values[..., -1, None]
        scaled = torch.where(
            scaled < cutoff,
            torch.full_like(scaled, float("-inf")),
            scaled,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf"))
        scaled.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probabilities = torch.softmax(scaled, dim=-1)
    if not torch.isfinite(probabilities).all() or torch.any(probabilities.sum(dim=-1) <= 0):
        raise RuntimeError("Generation parameters produced an invalid probability distribution")
    return torch.multinomial(probabilities, num_samples=1)


def generate(
    model: LapisModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not math.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be finite, greater than 0, and at most 1")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")

    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = ids[:, -model.max_position_embeddings :]
            logits, _ = model(context)
            next_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_id:
                break

    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with Lapis")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and non-negative")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if not math.isfinite(args.top_p) or not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be finite, greater than 0, and at most 1")

    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.checkpoint, checkpoint)
    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
        )
    )


if __name__ == "__main__":
    main()
