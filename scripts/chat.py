#!/usr/bin/env python3
"""Interactive developer inference/testing console for LapisLLM."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

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


def render_header(*, runtime: LapisRuntime, color: bool) -> None:
    width = 66
    title = "│  L A P I S"
    status = f"  developer inference · {runtime.device} · context {runtime.model.max_position_embeddings - 1}"
    print()
    print(paint("╭" + "─" * width + "╮", ACCENT, color))
    print(paint(title + " " * (width + 2 - len(title)) + "│", BOLD + ACCENT, color))
    print(paint("│" + status + " " * max(0, width - len(status)) + "│", MUTED, color))
    print(paint("╰" + "─" * width + "╯", ACCENT, color))
    print()
    print(paint("/help", INPUT, color) + " commands  " + paint("/stats", INPUT, color) + " runtime  " + paint("/context", INPUT, color) + " context  " + paint("/exit", INPUT, color) + " quit")
    print()


def build_prompt(messages: list[tuple[str, str]]) -> str:
    lines = [f"{'Input' if role == 'input' else 'Lapis'}: {content}" for role, content in messages]
    lines.append("Lapis:")
    return "\n".join(lines)


def count_context_tokens(runtime: LapisRuntime, messages: list[tuple[str, str]]) -> int:
    return len(runtime.tokenize(build_prompt(messages)))


def stream_response(runtime: LapisRuntime, prompt: str, config: SamplingConfig, color: bool) -> str:
    print(paint("Lapis", ASSISTANT, color) + paint(" › ", DIM, color), end="", flush=True)
    started = time.monotonic()
    chunks: list[str] = []
    for chunk in runtime.stream_generate(prompt, config):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    duration = time.monotonic() - started
    text = "".join(chunks)
    token_count = len(runtime.tokenize(text))
    rate = token_count / duration if duration > 0 else 0.0
    print()
    print(paint(f"  {token_count} tokens · {rate:.1f} tok/s", DIM, color))
    return text


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
        runtime = LapisRuntime.from_checkpoint(Path(args.checkpoint), args.device)
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
            print(f"Context: {count_context_tokens(runtime, messages)}/{runtime.model.max_position_embeddings - 1} tokens")
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
        prompt_ids = runtime.tokenize(prompt)
        context_limit = runtime.model.max_position_embeddings - 1
        while len(prompt_ids) > context_limit and len(messages) > 2:
            del messages[0:2]
            prompt = build_prompt(messages)
            prompt_ids = runtime.tokenize(prompt)
        print()
        response = stream_response(runtime, prompt, sampling, color)
        generated_tokens += len(runtime.tokenize(response))
        messages.append(("assistant", response))
        print()


if __name__ == "__main__":
    main()
