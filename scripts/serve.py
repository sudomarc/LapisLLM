#!/usr/bin/env python3
"""Serve a Lapis checkpoint through a small OpenAI-style HTTP API."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lapis.config.base import resolve_device
from lapis.config.model_config import model_config_kwargs
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "lapis-tiny"
    messages: list[Message]
    temperature: float = Field(default=0.8, ge=0.01)
    max_tokens: int = Field(default=64, ge=1, le=4096)


def load_runtime(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LapisModel(**model_config_kwargs(checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = Tokenizer.load(str(Path(checkpoint_path).parent / "tokenizer"))
    return model, tokenizer


def create_app(checkpoint_path: str, device: torch.device) -> FastAPI:
    model, tokenizer = load_runtime(checkpoint_path, device)
    app = FastAPI(title="LAPIS API", version="0.1.0")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "lapis-tiny", "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
        from scripts.generate import generate
        try:
            text = generate(
                model, tokenizer, prompt, request.max_tokens, request.temperature, 40, 0.95
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "id": "lapis-completion",
            "object": "chat.completion",
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve LAPIS")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    uvicorn.run(create_app(args.checkpoint, device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
