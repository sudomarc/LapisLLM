#!/usr/bin/env python3
"""Interactive terminal chat for a trained Lapis checkpoint.

The interface intentionally stays dependency-free: ANSI escape sequences provide
layout and color while the model/inference code remains unchanged.
"""

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
ACCENT = "\033[38;5;141m"
CYAN = "\033[38;5;81m"
GREEN = "\033[38;5;114m"
YELLOW = "\033[38;5;221m"
RED = "\033[38;5;203m"


def use_color(plain: bool) -> bool:
    return not plain and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def style(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{RESET}" if enabled else text


def clear_screen(enabled: bool) -> None:
    if enabled:
        print("\033[2J\033[H", end="")


def render_header(device: torch.device, checkpoint: Path, colored: bool) -> None:
    print(style("╭─ L A P I S ─────────────────────────────────────────────╮", ACCENT, colored))
    print(style("│", ACCENT, colored) + style(" Lapis Chat", BOLD, colored) + " — local model session", end="")
    print(" " * max(0, 58 - len(" Lapis Chat — local model session")) + style("│", ACCENT, colored))
    print(style("├───────────────────────────────────────────────────────────┤", ACCENT, colored))
    print(style("│", ACCENT, colored) + f" device   {device}" + " " * max(0, 47 - len(str(device))) + style("│", ACCENT, colored))
    print(style("│", ACCENT, colored) + f" checkpoint {checkpoint}" + " " * max(0, 42 - len(str(checkpoint))) + style("│", ACCENT, colored))
    print(style("╰───────────────────────────────────────────────────────────╯", ACCENT, colored))
    print(style("Type /help for commands. Ctrl+C stops generation. /exit quits.", DIM, colored))
    print()


def render_user(prompt: str, colored: bool) -> None:
    print(style("You", CYAN, colored) + "  " + prompt)


def render_assistant_start(colored: bool) -> None:
    print(style("Lapis", ACCENT, colored) + "  " + style("thinking…", DIM, colored), end="\r", flush=True)
    print(" " * 40 + "\r", end="")
    print(style("Lapis", ACCENT, colored) + "  ", end="", flush=True)


def render_error(message: str, colored: bool) -> None:
    print(style(f"Error: {message}", RED, colored))


def load_chat_model(checkpoint_path: Path, device: torch.device) -> tuple[LapisModel, Tokenizer]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = Tokenizer.load(str(checkpoint_path.parent / "tokenizer"))
    validate_checkpoint_tokenizer(checkpoint, tokenizer)
    model = LapisModel(**model_config_kwargs(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def build_prompt(history: list[tuple[str, str]]) -> str:
    """Build a compact conversational prompt for the base Lapis model."""
    return "\n".join(f"{role}: {content}" for role, content in history) + "\nLapis:"


def stream_response(
    model: LapisModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    colored: bool = False,
) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    displayed = ""

    render_assistant_start(colored)
    try:
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
    except KeyboardInterrupt:
        print("\n" + style("Generation stopped.", YELLOW, colored), end="")

    print(flush=True)
    return displayed.strip()


def print_help(colored: bool) -> None:
    print(style("Commands", BOLD, colored))
    print("  /help      Show this help")
    print("  /clear     Clear the terminal and start a fresh session")
    print("  /stats     Show current runtime settings")
    print("  /exit      Exit Lapis Chat")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Lapis")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--plain", action="store_true", help="Disable ANSI styling")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the range (0, 1]")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train a model first or pass --checkpoint PATH."
        )

    device = resolve_device(args.device)
    model, tokenizer = load_chat_model(checkpoint_path, device)
    colored = use_color(args.plain)
    history: list[tuple[str, str]] = []

    clear_screen(colored)
    render_header(device, checkpoint_path, colored)

    while True:
        try:
            prompt = input(style("You › ", BOLD, colored)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit", "quit", "exit"}:
            print(style("Session closed.", DIM, colored))
            break
        if prompt.lower() == "/help":
            print_help(colored)
            continue
        if prompt.lower() == "/clear":
            history.clear()
            clear_screen(colored)
            render_header(device, checkpoint_path, colored)
            continue
        if prompt.lower() == "/stats":
            print(
                f"device={device} | max_new_tokens={args.max_new_tokens} | "
                f"temperature={args.temperature} | top_k={args.top_k} | top_p={args.top_p}"
            )
            print()
            continue

        render_user(prompt, colored)
        history.append(("You", prompt))
        response = stream_response(
            model,
            tokenizer,
            build_prompt(history),
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.top_p,
            colored,
        )
        if response:
            history.append(("Lapis", response))
        else:
            render_error("The model returned an empty response.", colored)
        print()


if __name__ == "__main__":
    main()
