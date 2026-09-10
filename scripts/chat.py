#!/usr/bin/env python3
"""Interactive developer inference/testing console for LapisLLM."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lapis.inference.runtime import LapisRuntime, SamplingConfig

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ACCENT = "\033[38;5;180m"
MUTED = "\033[38;5;245m"
INPUT = "\033[38;5;117m"
ASSISTANT = "\033[38;5;183m"
ERROR = "\033[38;5;203m"
SUCCESS = "\033[38;5;114m"


def paint(value: str, style: str, enabled: bool) -> str:
    return f"{style}{value}{RESET}" if enabled else value


def ui_enabled(no_color: bool) -> bool:
    return not no_color and sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def load_chat_model(checkpoint_path: Path, device: str) -> LapisRuntime:
    """Load a checkpoint through the shared inference runtime."""
    return LapisRuntime.from_checkpoint(checkpoint_path, device)


def render_header(*, runtime: LapisRuntime, color: bool) -> None:
    width = 66
    print()
    print(paint("╭" + "─" * width + "╮", ACCENT, color))
    title = "│  L A P I S"
    print(paint(title + " " * (width + 2 - len(title)) + "│", BOLD + ACCENT, color))
    status = f"  developer inference · {runtime.device} · context {runtime.model.max_position_embeddings - 1}"
    print(paint("│" + status + " " * max(0, width - len(status)) + "│", MUTED, color))
    print(paint("╰" + "─" * width + "╯", ACCENT, color))
    print()
    print(paint("/help", INPUT, color) + " commands  " + paint("/stats", INPUT, color) + " runtime  " + paint("/context", INPUT, color) + " context  " + paint("/exit", INPUT, color) + " quit")
    print()


def build_prompt(messages: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for role, content in messages:
        prefix = "Input" if role == "input" else "Lapis"
        lines.append(f"{prefix}: {content}")
    lines.append("Lapis:")
    return "\n".join(lines)


def count_context_tokens(runtime: LapisRuntime, messages: list[tuple[str, str]]) -> int:
    return len(runtime.tokenizer.encode(build_prompt(messages), add_special_tokens=False))


def sample_next_token(logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    """Compatibility helper for the streaming developer console."""
    config.validate()
    logits = logits / config.temperature
    if config.top_k > 0:
        values, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)
    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative - probabilities > config.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Sampling produced non-finite probabilities")
    return torch.multinomial(probabilities, num_samples=1)


def stream_response(runtime: LapisRuntime, prompt: str, config: SamplingConfig, color: bool) -> str:
    token_ids = runtime.tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [runtime.tokenizer.bos_id]
    ids = torch.tensor([token_ids], dtype=torch.long, device=runtime.device)
    generated: list[int] = []
    displayed = ""
    print(paint("Lapis", ASSISTANT, color) + paint(" › ", DIM, color), end="", flush=True)
    started = time.monotonic()
    with torch.inference_mode():
        for _ in range(config.max_new_tokens):
            context = ids[:, -runtime.model.max_position_embeddings :]
            logits, _ = runtime.model(context)
            next_id = sample_next_token(logits[:, -1, :], config)
            ids = torch.cat([ids, next_id], dim=1)
            token_id = int(next_id.item())
            if token_id == runtime.tokenizer.eos_id:
                break
            generated.append(token_id)
            text = runtime.tokenizer.decode(generated, skip_special_tokens=True)
            delta = text[len(displayed) :] if text.startswith(displayed) else text
            if delta:
                print(delta, end="", flush=True)
                displayed = text
    duration = time.monotonic() - started
    tokens_per_second = len(generated) / duration if duration > 0 else 0.0
    print()
    print(paint(f"  {len(generated)} tokens · {tokens_per_second:.1f} tok/s", DIM, color))
    return displayed


def save_session(path: Path, messages: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"## {'Input' if role == 'input' else 'Lapis'}\n\n{content}\n" for role, content in messages]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Developer inference and checkpoint-testing console")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    try:
        sampling = SamplingConfig(args.max_new_tokens, args.temperature, args.top_k, args.top_p)
        sampling.validate()
        runtime = load_chat_model(Path(args.checkpoint), args.device)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, OSError, TypeError) as exc:
        print(paint(f"Unable to load checkpoint: {exc}", ERROR, not args.no_color), file=sys.stderr)
        raise SystemExit(1) from exc

    color = ui_enabled(args.no_color)
    messages: list[tuple[str, str]] = []
    generated_tokens = 0
    session_started = time.monotonic()
    if color:
        print("\033[2J\033[H", end="")
    render_header(runtime=runtime, color=color)

    while True:
        try:
            user_input = input(paint("› ", INPUT + BOLD, color)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        command, _, argument = user_input.partition(" ")
        command, argument = command.lower(), argument.strip()
        if command in {"/exit", "/quit", "exit", "quit"}:
            break
        if command == "/help":
            print("/help  /clear  /reset  /stats  /context  /model  /temperature <n>  /tokens <n>  /save [file]  /exit\n")
            continue
        if command == "/clear":
            print("\033[2J\033[H" if color else "\n" * 3, end="")
            render_header(runtime=runtime, color=color)
            continue
        if command == "/reset":
            messages.clear()
            print(paint("✓ Session context reset", SUCCESS, color))
            continue
        if command == "/stats":
            context_tokens = count_context_tokens(runtime, messages)
            print(f"checkpoint {runtime.checkpoint}\ndevice {runtime.device}\nparameters {sum(p.numel() for p in runtime.model.parameters()):,}\nmessages {len(messages)}\ncontext {context_tokens}/{runtime.model.max_position_embeddings - 1}\ngenerated {generated_tokens} tokens\nsession {time.monotonic() - session_started:.1f}s\ntemperature {sampling.temperature:.2f}\nmax tokens {sampling.max_new_tokens}\n")
            continue
        if command == "/context":
            context_tokens = count_context_tokens(runtime, messages)
            limit = runtime.model.max_position_embeddings - 1
            print(f"Context: {context_tokens}/{limit} tokens")
            continue
        if command == "/model":
            print(f"Checkpoint: {runtime.checkpoint}\nDevice: {runtime.device}\nTokenizer: {runtime.tokenizer.vocab_size:,} vocab\nContext: {runtime.model.max_position_embeddings - 1} tokens\n")
            continue
        if command == "/temperature":
            if not argument:
                print(f"temperature = {sampling.temperature:.2f}")
                continue
            try:
                sampling = SamplingConfig(sampling.max_new_tokens, float(argument), sampling.top_k, sampling.top_p)
                sampling.validate()
                print(paint(f"✓ temperature = {sampling.temperature:.2f}", SUCCESS, color))
            except ValueError as exc:
                print(paint(str(exc), ERROR, color))
            continue
        if command == "/tokens":
            if not argument:
                print(f"max-new-tokens = {sampling.max_new_tokens}")
                continue
            try:
                sampling = SamplingConfig(int(argument), sampling.temperature, sampling.top_k, sampling.top_p)
                sampling.validate()
                print(paint(f"✓ max-new-tokens = {sampling.max_new_tokens}", SUCCESS, color))
            except ValueError as exc:
                print(paint(str(exc), ERROR, color))
            continue
        if command == "/save":
            target = Path(argument) if argument else Path("outputs/developer-session.md")
            save_session(target, messages)
            print(paint(f"✓ Session saved to {target}", SUCCESS, color))
            continue

        messages.append(("input", user_input))
        prompt = build_prompt(messages)
        prompt_ids = runtime.tokenizer.encode(prompt, add_special_tokens=False)
        context_limit = runtime.model.max_position_embeddings - 1
        while len(prompt_ids) > context_limit and len(messages) > 2:
            del messages[0:2]
            prompt = build_prompt(messages)
            prompt_ids = runtime.tokenizer.encode(prompt, add_special_tokens=False)
        print()
        response = stream_response(runtime, prompt, sampling, color)
        generated_tokens += len(runtime.tokenizer.encode(response, add_special_tokens=False))
        messages.append(("assistant", response))
        print()


if __name__ == "__main__":
    main()
