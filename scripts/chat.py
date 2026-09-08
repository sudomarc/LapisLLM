#!/usr/bin/env python3
"""Interactive local Lapis chat with a terminal-native developer-console UI."""

from __future__ import annotations

import argparse
import os
import sys
import time
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
SUCCESS = "\033[38;5;114m"


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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    tokenizer = Tokenizer.load(str(tokenizer_path))
    validate_checkpoint_tokenizer(checkpoint, tokenizer)
    model = LapisModel(**model_config_kwargs(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def render_header(
    *,
    model: LapisModel,
    device: torch.device,
    checkpoint: Path,
    color: bool,
    messages: list[tuple[str, str]],
    generated_tokens: int,
    elapsed: float,
) -> None:
    width = 66
    print()
    print(paint("╭" + "─" * width + "╮", ACCENT, color))
    print(paint("│", ACCENT, color) + paint("  L A P I S", BOLD + ACCENT, color) + " " * (width - 13) + paint("│", ACCENT, color))
    status = f"  local inference · {device} · context {model.max_position_embeddings - 1}"
    print(paint("│", ACCENT, color) + paint(status, MUTED, color) + " " * max(0, width - len(status)) + paint("│", ACCENT, color))
    info = f"  {checkpoint}"
    print(paint("│", ACCENT, color) + paint(info, DIM, color) + " " * max(0, width - len(info)) + paint("│", ACCENT, color))
    print(paint("╰" + "─" * width + "╯", ACCENT, color))
    print()
    print(
        paint("/help", USER, color)
        + paint(" commands  ", MUTED, color)
        + paint("/stats", USER, color)
        + paint(" session  ", MUTED, color)
        + paint("/context", USER, color)
        + paint(" context  ", MUTED, color)
        + paint("/exit", USER, color)
        + paint(" quit", MUTED, color)
    )
    print()


def build_prompt(messages: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for role, content in messages:
        prefix = "User" if role == "user" else "Lapis"
        lines.append(f"{prefix}: {content}")
    lines.append("Lapis:")
    return "\n".join(lines)


def count_context_tokens(tokenizer: Tokenizer, messages: list[tuple[str, str]]) -> int:
    return len(tokenizer.encode(build_prompt(messages), add_special_tokens=False))


def stream_response(
    model: LapisModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    color: bool,
) -> str:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    device = next(model.parameters()).device
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated: list[int] = []
    displayed = ""

    print(paint("Lapis", ASSISTANT, color) + paint(" › ", DIM, color), end="", flush=True)
    started = time.monotonic()
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

    duration = time.monotonic() - started
    tokens_per_second = len(generated) / duration if duration > 0 else 0.0
    print()
    print(
        paint(
            f"  {len(generated)} tokens · {tokens_per_second:.1f} tok/s",
            DIM,
            color,
        )
    )
    return displayed


def save_conversation(path: Path, messages: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for role, content in messages:
        lines.append(f"## {'User' if role == 'user' else 'Lapis'}\n\n{content}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Lapis in the terminal")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.temperature <= 0:
        parser.error("--temperature must be greater than 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in the range (0, 1]")

    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    color = ui_enabled(args.no_color)

    try:
        model, tokenizer = load_chat_model(checkpoint_path, device)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(paint(f"Lapis startup error: {exc}", ERROR, color), file=sys.stderr)
        raise SystemExit(1) from exc

    messages: list[tuple[str, str]] = []
    generated_tokens = 0
    session_started = time.monotonic()
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens

    if color:
        print("\033[2J\033[H", end="")
    render_header(
        model=model,
        device=device,
        checkpoint=checkpoint_path,
        color=color,
        messages=messages,
        generated_tokens=generated_tokens,
        elapsed=0.0,
    )

    while True:
        try:
            user_input = input(paint("› ", USER + BOLD, color)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        command, _, argument = user_input.partition(" ")
        command = command.lower()
        argument = argument.strip()

        if command in {"/exit", "/quit", "exit", "quit"}:
            break
        if command == "/help":
            print(paint("/help", USER, color) + "  show commands")
            print(paint("/clear", USER, color) + "  clear terminal and redraw")
            print(paint("/reset", USER, color) + "  clear conversation context")
            print(paint("/stats", USER, color) + "  show session and model stats")
            print(paint("/context", USER, color) + "  show current context usage")
            print(paint("/model", USER, color) + "  show checkpoint/device")
            print(paint("/temperature <n>", USER, color) + "  change sampling temperature")
            print(paint("/tokens <n>", USER, color) + "  change maximum output tokens")
            print(paint("/save [file]", USER, color) + "  save this conversation")
            print(paint("/exit", USER, color) + "  leave the chat")
            print()
            continue

        if command == "/clear":
            print("\033[2J\033[H" if color else "\n" * 3, end="")
            render_header(
                model=model,
                device=device,
                checkpoint=checkpoint_path,
                color=color,
                messages=messages,
                generated_tokens=generated_tokens,
                elapsed=time.monotonic() - session_started,
            )
            continue

        if command == "/reset":
            messages.clear()
            print(paint("✓ Conversation context reset", SUCCESS, color))
            continue

        if command == "/stats":
            context_tokens = count_context_tokens(tokenizer, messages)
            elapsed = time.monotonic() - session_started
            print(paint("SESSION", ACCENT + BOLD, color))
            print(f"  model           {checkpoint_path.name}")
            print(f"  device          {device}")
            print(f"  parameters      {sum(p.numel() for p in model.parameters()):,}")
            print(f"  messages        {len(messages)}")
            print(f"  context tokens  {context_tokens} / {model.max_position_embeddings - 1}")
            print(f"  generated       {generated_tokens} tokens")
            print(f"  session         {elapsed:.1f}s")
            print(f"  temperature     {temperature:.2f}")
            print(f"  max tokens      {max_new_tokens}")
            print()
            continue

        if command == "/context":
            context_tokens = count_context_tokens(tokenizer, messages)
            limit = model.max_position_embeddings - 1
            ratio = context_tokens / max(1, limit)
            width = 32
            filled = min(width, int(width * ratio))
            print(
                f"Context [{('█' * filled) + ('░' * (width - filled))}] "
                f"{context_tokens}/{limit} tokens"
            )
            continue

        if command == "/model":
            print(f"Model     : {checkpoint_path}")
            print(f"Device    : {device}")
            print(f"Tokenizer : {tokenizer.vocab_size:,} vocab")
            print(f"Context   : {model.max_position_embeddings - 1} tokens")
            print()
            continue

        if command == "/temperature":
            if not argument:
                print(f"temperature = {temperature:.2f}")
                continue
            try:
                candidate = float(argument)
            except ValueError:
                print(paint("Temperature must be a number.", ERROR, color))
                continue
            if candidate <= 0:
                print(paint("Temperature must be greater than 0.", ERROR, color))
                continue
            temperature = candidate
            print(paint(f"✓ temperature = {temperature:.2f}", SUCCESS, color))
            continue

        if command == "/tokens":
            if not argument:
                print(f"max-new-tokens = {max_new_tokens}")
                continue
            try:
                candidate = int(argument)
            except ValueError:
                print(paint("Token count must be an integer.", ERROR, color))
                continue
            if candidate < 1:
                print(paint("Token count must be at least 1.", ERROR, color))
                continue
            max_new_tokens = candidate
            print(paint(f"✓ max-new-tokens = {max_new_tokens}", SUCCESS, color))
            continue

        if command == "/save":
            target = Path(argument) if argument else Path("outputs/chat-session.md")
            save_conversation(target, messages)
            print(paint(f"✓ Conversation saved to {target}", SUCCESS, color))
            continue

        # Keep the prompt inside the model context window. Prefer recent turns.
        messages.append(("user", user_input))
        prompt = build_prompt(messages)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        context_limit = model.max_position_embeddings - 1
        while len(prompt_ids) > context_limit and len(messages) > 2:
            del messages[0:2]
            prompt = build_prompt(messages)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        print()
        response = stream_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            temperature,
            args.top_k,
            args.top_p,
            color,
        )
        generated_tokens += len(tokenizer.encode(response, add_special_tokens=False))
        messages.append(("assistant", response))
        print()


if __name__ == "__main__":
    main()
