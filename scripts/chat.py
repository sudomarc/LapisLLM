#!/usr/bin/env python3
"""Interactive chat with a trained Lapis checkpoint."""

from __future__ import annotations

import argparse

import torch

from lapis.config.base import resolve_path
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.generate import load_model, sample_next_token, validate_generation_parameters


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device: {requested}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return device


def load_chat_model(checkpoint_path: Path, device: torch.device) -> tuple[LapisModel, Tokenizer]:
    return load_model(str(checkpoint_path), device)


def stream_response(
    model: LapisModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> None:
    validate_generation_parameters(max_new_tokens, temperature, top_k, top_p)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]
    if len(token_ids) > model.max_position_embeddings:
        raise ValueError(
            f"Prompt contains {len(token_ids)} tokens, exceeding the model context limit of "
            f"{model.max_position_embeddings}"
        )

    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    displayed = ""

    print("Lapis: ", end="", flush=True)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = ids[:, -model.max_position_embeddings :]
            logits, _ = model(context)
            next_id = sample_next_token(logits[:, -1, :], temperature, top_k, top_p)
            ids = torch.cat([ids, next_id], dim=1)
            token_id = int(next_id.item())
            if token_id == tokenizer.eos_id:
                break
            generated.append(token_id)
            text = tokenizer.decode(generated, skip_special_tokens=True)
            delta = text[len(displayed) :] if text.startswith(displayed) else text
            if delta:
                print(delta, end="", flush=True)
                displayed = text
    print(flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with LAPIS")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    device = resolve_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint)
    model, tokenizer = load_chat_model(checkpoint_path, device)

    print(f"LAPIS Chat — device: {device} — type 'quit' to exit", flush=True)
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"quit", "exit"}:
            break
        if not prompt:
            continue
        stream_response(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
        )


if __name__ == "__main__":
    main()
