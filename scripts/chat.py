#!/usr/bin/env python3
"""Interactive chat with a trained Lapis checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lapis.checkpoint import CheckpointError, load_checkpoint
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.generate import resolve_device, sample_next_token


def load_chat_model(checkpoint_path: Path, device: torch.device) -> tuple[LapisModel, Tokenizer]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    try:
        model = LapisModel(**checkpoint["config"]["model"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except (RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise CheckpointError("Checkpoint model state is incompatible with its configuration") from exc

    embedded = checkpoint.get("tokenizer_json")
    tokenizer = (
        Tokenizer.from_json(embedded)
        if embedded
        else Tokenizer.load(str(checkpoint_path.parent / "tokenizer"))
    )
    expected = checkpoint.get("tokenizer_vocab_size")
    if expected is not None and int(expected) != tokenizer.vocab_size:
        raise CheckpointError("Checkpoint tokenizer metadata does not match tokenizer")
    model.eval()
    return model, tokenizer


def stream_response(
    model: LapisModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> None:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be greater than 0 and at most 1")

    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

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
    model, tokenizer = load_chat_model(Path(args.checkpoint), device)

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
