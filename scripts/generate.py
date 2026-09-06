#!/usr/bin/env python3
"""Generate text from a trained Lapis checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lapis.config.base import resolve_device
from lapis.config.model_config import model_config_kwargs
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_checkpoint_tokenizer(checkpoint: dict, tokenizer: Tokenizer) -> None:
    checkpoint_version = checkpoint.get("tokenizer_version")
    if checkpoint_version is not None and checkpoint_version != tokenizer.VERSION:
        raise ValueError(
            "Checkpoint tokenizer_version does not match the active tokenizer: "
            f"{checkpoint_version!r} != {tokenizer.VERSION!r}"
        )


def load_model(checkpoint_path: str, device: torch.device) -> LapisModel:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = LapisModel(**model_config_kwargs(config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def sample_next_token(logits, temperature, top_k, top_p):
    logits = logits / max(temperature, 1e-6)

    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probabilities = torch.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p):
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
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

    device = resolve_device(args.device)
    checkpoint = Path(args.checkpoint)
    checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
    tokenizer_dir = checkpoint.parent / "tokenizer"
    tokenizer = Tokenizer.load(str(tokenizer_dir))
    validate_checkpoint_tokenizer(checkpoint_data, tokenizer)
    model = LapisModel(**model_config_kwargs(checkpoint_data["config"])).to(device)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    print(generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
    ))


if __name__ == "__main__":
    main()
