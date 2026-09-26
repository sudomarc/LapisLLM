#!/usr/bin/env python3
"""Serve a Lapis checkpoint through a small OpenAI-style HTTP API."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from lapis.config.base import resolve_device
from lapis.inference.errors import (
    CancellationError,
    CheckpointLoadError,
    ContextLengthExceededError,
    InvalidPromptError,
    InvalidSamplingConfigError,
    LapisInferenceError,
)
from lapis.inference.runtime import LapisRuntime, SamplingConfig


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "lapis-tiny"
    messages: list[Message] = Field(min_length=1)
    temperature: float = Field(default=0.8, ge=0.0, le=5.0)
    max_tokens: int = Field(default=64, ge=1, le=4096)
    top_k: int = Field(default=40, ge=0, le=4096)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = False


def _status_code_for_error(exc: Exception) -> int:
    if isinstance(exc, (InvalidSamplingConfigError, InvalidPromptError, ContextLengthExceededError)):
        return 400
    if isinstance(exc, CancellationError):
        return 408
    if isinstance(exc, LapisInferenceError):
        return 500
    return 500


def _error_payload(exc: Exception) -> dict[str, Any]:
    code = getattr(exc, "code", "runtime_error")
    return {
        "error": {
            "code": code,
            "message": str(exc),
            "type": type(exc).__name__,
        }
    }


def create_app(
    checkpoint_path: str | None = None,
    device: torch.device | str = "auto",
    runtime: LapisRuntime | None = None,
) -> FastAPI:
    if runtime is None:
        if checkpoint_path is None:
            raise ValueError("Either checkpoint_path or runtime must be provided")
        try:
            runtime = LapisRuntime.from_checkpoint(checkpoint_path, str(device))
        except (CheckpointLoadError, FileNotFoundError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Unable to load serving runtime: {exc}") from exc

    app = FastAPI(title="LAPIS API", version="0.2.0")

    @app.get("/v1/models")
    def models():
        info = runtime.get_model_info()
        return {
            "object": "list",
            "data": [
                {
                    "id": "lapis-tiny",
                    "object": "model",
                    "context_length": info["context_length"],
                    "capabilities": info["capabilities"],
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest, http_request: Request):
        prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
        try:
            sampling = SamplingConfig(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )
            sampling.validate()
        except InvalidSamplingConfigError as exc:
            return JSONResponse(
                status_code=_status_code_for_error(exc),
                content=_error_payload(exc),
            )

        def is_cancelled() -> bool:
            return getattr(http_request, "_is_disconnected", False)

        if request.stream:
            def event_generator():
                try:
                    try:
                        stream_iter = runtime.stream_generate(prompt, sampling, is_cancelled=is_cancelled)
                    except TypeError:
                        stream_iter = runtime.stream_generate(prompt, sampling)

                    for chunk in stream_iter:
                        data = {
                            "id": "lapis-completion",
                            "object": "chat.completion.chunk",
                            "model": request.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": chunk},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    done_data = {
                        "id": "lapis-completion",
                        "object": "chat.completion.chunk",
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(done_data)}\n\n"
                    yield "data: [DONE]\n\n"
                except (LapisInferenceError, ValueError, RuntimeError) as exc:
                    yield f"data: {json.dumps(_error_payload(exc))}\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        try:
            try:
                text = runtime.generate(prompt, sampling, is_cancelled=is_cancelled)
            except TypeError:
                text = runtime.generate(prompt, sampling)
        except LapisInferenceError as exc:
            return JSONResponse(
                status_code=_status_code_for_error(exc),
                content=_error_payload(exc),
            )
        except (ValueError, RuntimeError) as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "runtime_error", "message": str(exc), "type": type(exc).__name__}},
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
        parser.error("--port must be in the range 1..65535")

    device = resolve_device(args.device)
    try:
        uvicorn.run(create_app(args.checkpoint, device), host=args.host, port=args.port)
    except RuntimeError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
