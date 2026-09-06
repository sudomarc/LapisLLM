#!/usr/bin/env python3
"""Serve a Lapis checkpoint through a small OpenAI-style HTTP API."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from lapis.checkpoint import CheckpointError, load_checkpoint
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.generate import generate, resolve_device


class Message(BaseModel):
    role: str = Field(min_length=1, max_length=32)
    content: str = Field(max_length=100_000)


class ChatRequest(BaseModel):
    model: str = Field(default="lapis-tiny", min_length=1, max_length=64)
    messages: list[Message] = Field(min_length=1, max_length=128)
    temperature: float = Field(default=0.8, ge=0.0)
    max_tokens: int = Field(default=64, ge=1, le=4096)


def load_runtime(checkpoint_path: str, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    try:
        model = LapisModel(**checkpoint["config"]["model"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError("Checkpoint model state is incompatible with its configuration") from exc

    embedded = checkpoint.get("tokenizer_json")
    tokenizer = (
        Tokenizer.from_json(embedded)
        if embedded
        else Tokenizer.load(str(Path(checkpoint_path).parent / "tokenizer"))
    )
    expected = checkpoint.get("tokenizer_vocab_size")
    if expected is not None and int(expected) != tokenizer.vocab_size:
        raise CheckpointError("Checkpoint tokenizer metadata does not match tokenizer")
    model.eval()
    return model, tokenizer


def create_app(checkpoint_path: str, device: torch.device) -> FastAPI:
    model, tokenizer = load_runtime(checkpoint_path, device)
    app = FastAPI(title="LAPIS API", version="0.1.1")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "lapis-tiny", "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
        text = generate(
            model,
            tokenizer,
            prompt,
            request.max_tokens,
            request.temperature,
            40,
            0.95,
        )
        return {
            "id": "lapis-completion",
            "object": "chat.completion",
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve LAPIS")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    device = resolve_device(args.device)
    uvicorn.run(create_app(args.checkpoint, device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
