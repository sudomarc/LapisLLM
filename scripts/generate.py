#!/usr/bin/env python3
"""Generate text from a trained Lapis checkpoint."""

from __future__ import annotations

import argparse

import torch

from lapis.config.base import resolve_path
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.train import load_checkpoint


def load_model(checkpoint_path: str, device: torch.device) -> tuple[LapisModel, Tokenizer]:
    path = resolve_path(checkpoint_path)
    checkpoint = load_checkpoint(path, device)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint is missing model configuration metadata")

    model_cfg = config["model"]
    model = LapisModel(**model_cfg).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise ValueError(f"Checkpoint model weights are incompatible: {exc}") from exc

    tokenizer_dir = path.parent / "tokenizer"
    tokenizer = Tokenizer.load(str(tokenizer_dir))
    checkpoint_vocab = checkpoint.get("tokenizer_vocab_size", model.vocab_size)
    if int(checkpoint_vocab) != tokenizer.vocab_size:
        raise ValueError(
            f"Checkpoint tokenizer vocabulary size {checkpoint_vocab} does not match "
            f"loaded tokenizer vocabulary size {tokenizer.vocab_size}"
        )
    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary size {tokenizer.vocab_size} does not match "
            f"model vocabulary size {model.vocab_size}"
        )
    if checkpoint.get("tokenizer_version") not in (None, tokenizer.VERSION):
        raise ValueError("Checkpoint tokenizer version is incompatible")
    model.eval()
    return model, tokenizer


def validate_generation_parameters(
    max_new_tokens: int, temperature: float, top_k: int, top_p: float
) -> None:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must satisfy 0 < top_p <= 1")


def sample_next_token(logits, temperature, top_k, top_p):
    """Sample one token after applying temperature, top-k and nucleus filtering."""
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        cutoff = values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities >= top_p
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise RuntimeError("Generation produced invalid token probabilities")
    return torch.multinomial(probabilities, num_samples=1)


def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p):
    validate_generation_parameters(max_new_tokens, temperature, top_k, top_p)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    if len(token_ids) > model.max_position_embeddings:
        raise ValueError(
            f"Prompt contains {len(token_ids)} tokens, exceeding the model context "
            f"limit of {model.max_position_embeddings}"
        )

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

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(args.device)
        except RuntimeError as exc:
            raise ValueError(f"Invalid device: {args.device}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")

    model, tokenizer = load_model(args.checkpoint, device)
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
