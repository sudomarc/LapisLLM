#!/usr/bin/env python3
"""Interactive terminal chat for a trained Lapis checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lapis.config.base import resolve_device
from lapis.config.model_config import model_config_kwargs
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.generate import sample_next_token, validate_checkpoint_tokenizer


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ACCENT = "\033[38;5;180m"
MUTED = "\033[38;5;245m"
USER = "\033[38;5;117m"
ASSISTANT = "\033[38;5;183m"
ERROR = "\033[38;5;203m"


def paint(value: str, style: str, enabled: bool) -> str:
    return f"{style}{value}{RESET}" if enabled else value


def ui_enabled(no_color: bool) -> bool:
    return not no_color and sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def load_chat_model(checkpoint_path: Path, device: torch.device) -> tuple[LapisModel, Tokenizer]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train a model or pass --checkpoint."
        )

    tokenizer_path = checkpoint_path.parent / "tokenizer"
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. A checkpoint must ship with its tokenizer."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = Tokenizer.load(str(tokenizer_path))
    validate_checkpoint_tokenizer(checkpoint, tokenizer)
    model = LapisModel(**model_config_kwargs(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
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
    color: bool,
) -> None:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    displayed = ""

    print(paint("Lapis", ASSISTANT, color) + paint(" › ", DIM, color), end="", flush=True)
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


def print_header(device: torch.device, checkpoint: Path, color: bool) -> None:
    print()
    print(paint("╭─ L A P I S", ACCENT + BOLD, color))
    print(paint("│  local model · terminal chat", MUTED, color))
    print(
        paint("╰─ ", ACCENT, color)
        + paint(f"{device}", BOLD, color)
        + paint(" · ", DIM, color)
        + paint(str(checkpoint), MUTED, color)
    )
    print()
    print(paint("/help", USER, color) + paint(" commands · /clear reset · /exit quit", MUTED, color))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Lapis in the terminal")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    if not 0.0 < args.temperature:
        parser.error("--temperature must be greater than 0")

    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in the range (0, 1]")

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    color = ui_enabled(args.no_color)

    try:
        model, tokenizer = load_chat_model(checkpoint_path, device)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(paint(f"Lapis startup error: {exc}", ERROR, color), file=sys.stderr)
        raise SystemExit(1) from exc

    print_header(device, checkpoint_path, color)

    while True:
        try:
            prompt = input(paint("› ", USER + BOLD, color)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue

        command = prompt.lower()
        if command in {"/exit", "/quit", "exit", "quit"}:
            break
        if command == "/help":
            print(paint("/help", USER, color) + " show commands")
            print(paint("/clear", USER, color) + " clear the terminal")
            print(paint("/exit", USER, color) + " leave the chat")
            continue
        if command == "/clear":
            print("\033[2J\033[H" if color else "\n" * 3, end="")
            print_header(device, checkpoint_path, color)
            continue

        print()
        stream_response(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            color,
        )
        print()


if __name__ == "__main__":
    main()
