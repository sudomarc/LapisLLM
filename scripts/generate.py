#!/usr/bin/env python3
"""Generate text from a trained Lapis checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from lapis.config.base import resolve_device
from lapis.config.model_config import model_config_kwargs
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer


def validate_checkpoint_tokenizer(checkpoint: dict, tokenizer: Tokenizer) -> None:
    """Validate tokenizer metadata embedded in a checkpoint."""
    checkpoint_version = checkpoint.get("tokenizer_version")
    if checkpoint_version is not None and checkpoint_version != tokenizer.VERSION:
        raise ValueError(
            "Checkpoint tokenizer_version does not match the active tokenizer: "
            f"{checkpoint_version!r} != {tokenizer.VERSION!r}"
        )

    checkpoint_vocab = checkpoint.get("config", {}).get("model", {}).get("vocab_size")
    if checkpoint_vocab is not None and int(checkpoint_vocab) != tokenizer.vocab_size:
        raise ValueError(
            "Checkpoint vocabulary size does not match the tokenizer: "
            f"{checkpoint_vocab} != {tokenizer.vocab_size}"
        )


def validate_sampling_args(
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int | None = None,
) -> None:
    """Validate generation controls before sampling."""
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise ValueError("max_new_tokens must be an integer")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than 0")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in the range (0, 1]")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("seed must be a non-negative integer or None")


def sample_next_token(logits, temperature, top_k, top_p, generator=None):
    """Apply validated temperature, top-k, and top-p sampling to model logits."""
    validate_sampling_args(1, temperature, top_k, top_p)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Model produced non-finite logits")
    logits = logits / temperature

    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = values[..., -1, None]
        logits = torch.where(
            logits < cutoff,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Sampling produced non-finite probabilities")
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Sampling produced non-finite probabilities")
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, seed=None):
    """Generate a decoded continuation from an already-loaded model and tokenizer."""
    validate_sampling_args(max_new_tokens, temperature, top_k, top_p, seed=seed)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    context_limit = model.max_position_embeddings
    if len(token_ids) > context_limit:
        token_ids = token_ids[-context_limit:]

    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated = []

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device.type if hasattr(device, "type") else str(device))
        generator.manual_seed(seed)

    with torch.inference_mode():
        logits, _, kv_cache = model(ids, use_cache=True, start_pos=0)
        next_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, generator=generator)
        start_pos = ids.size(1)

        for _ in range(max_new_tokens):
            token_val = int(next_id.item())
            if token_val == tokenizer.eos_id:
                break
            generated.append(token_val)
            if start_pos >= context_limit:
                break
            logits, _, kv_cache = model(next_id, kv_cache_list=kv_cache, start_pos=start_pos, use_cache=True)
            start_pos += 1
            next_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p, generator=generator)

    return tokenizer.decode(token_ids + generated, skip_special_tokens=True)


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def main() -> None:
    """Run the command-line generation utility."""
    parser = argparse.ArgumentParser(description="Generate text with Lapis")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer directory; defaults to <checkpoint-dir>/tokenizer.",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    try:
        validate_sampling_args(
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        parser.error(f"Checkpoint not found: {checkpoint}")

    device = resolve_device(args.device)
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)

    tokenizer_dir = (
        Path(args.tokenizer)
        if args.tokenizer
        else checkpoint.parent / "tokenizer"
    )
    if not tokenizer_dir.exists():
        parser.error(f"Tokenizer directory not found: {tokenizer_dir}")
    tokenizer = Tokenizer.load(str(tokenizer_dir))
    validate_checkpoint_tokenizer(checkpoint_data, tokenizer)

    dtype_str = args.dtype.lower()
    if dtype_str not in _DTYPE_MAP:
        parser.error(f"Unsupported dtype: {args.dtype}")
    target_dtype = _DTYPE_MAP[dtype_str]

    model = LapisModel(**model_config_kwargs(checkpoint_data["config"])).to(device=device, dtype=target_dtype)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    print(
        generate(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
