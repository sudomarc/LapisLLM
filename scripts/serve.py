#!/usr/bin/env python3
"""Serve a Lapis checkpoint through a small OpenAI-style HTTP API."""

from __future__ import annotations

import argparse

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lapis.config.base import resolve_path
from scripts.generate import generate, load_model


class Message(BaseModel):
    role: str = Field(min_length=1)
    content: str


class ChatRequest(BaseModel):
    model: str = "lapis-tiny"
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.8, ge=0.0)
    max_tokens: int = Field(default=64, ge=1, le=4096)


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


def load_runtime(checkpoint_path: str, device: torch.device):
    return load_model(str(resolve_path(checkpoint_path)), device)


def create_app(checkpoint_path: str, device: torch.device) -> FastAPI:
    model, tokenizer = load_runtime(checkpoint_path, device)
    app = FastAPI(title="LAPIS API", version="0.1.2")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "lapis-tiny", "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        if request.model != "lapis-tiny":
            raise HTTPException(status_code=400, detail=f"Unknown model: {request.model}")
        prompt = "\n".join(f"{message.role}: {message.content}" for message in request.messages)
        try:
            text = generate(
                model,
                tokenizer,
                prompt,
                request.max_tokens,
                request.temperature,
                40,
                0.95,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
