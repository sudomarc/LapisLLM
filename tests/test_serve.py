from pathlib import Path

from fastapi.testclient import TestClient

from lapis.inference.errors import CancellationError
from lapis.inference.runtime import LapisRuntime
from scripts.serve import create_app


class FakeTokenizer:
    VERSION = "test-tokenizer"
    vocab_size = 16
    bos_id = 1
    eos_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3] if text else []

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join("hello" for _ in ids)


class FakeModel:
    max_position_embeddings = 64

    def parameters(self):
        return iter(())


def make_mock_runtime() -> LapisRuntime:
    runtime = object.__new__(LapisRuntime)
    runtime.model = FakeModel()
    runtime.tokenizer = FakeTokenizer()
    runtime.device = "cpu"
    runtime.checkpoint = Path("test.pt")
    return runtime


def test_serve_models_endpoint() -> None:
    runtime = make_mock_runtime()
    app = create_app(runtime=runtime)
    client = TestClient(app)

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    model = data["data"][0]
    assert model["id"] == "lapis-tiny"
    assert model["context_length"] == 64
    assert model["capabilities"]["streaming"] is True
    assert model["capabilities"]["cancellation"] is True


def test_serve_chat_completions_non_streaming(monkeypatch) -> None:
    runtime = make_mock_runtime()
    monkeypatch.setattr(runtime, "generate", lambda prompt, sampling: "Hello world!")
    app = create_app(runtime=runtime)
    client = TestClient(app)

    payload = {
        "model": "lapis-tiny",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "max_tokens": 32,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert res["choices"][0]["message"]["content"] == "Hello world!"
    assert res["choices"][0]["finish_reason"] == "stop"


def test_serve_chat_completions_invalid_sampling_parameters() -> None:
    runtime = make_mock_runtime()
    app = create_app(runtime=runtime)
    client = TestClient(app)

    payload = {
        "model": "lapis-tiny",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": -0.5,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422  # Pydantic validation error

    payload_valid_pydantic_invalid_sampling = {
        "model": "lapis-tiny",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.0001,
        "top_k": 0,
        "top_p": 0.0,
    }
    response = client.post("/v1/chat/completions", json=payload_valid_pydantic_invalid_sampling)
    assert response.status_code == 400
    res = response.json()
    assert res["error"]["code"] == "invalid_sampling_config"
    assert "top_p" in res["error"]["message"]


def test_serve_chat_completions_streaming(monkeypatch) -> None:
    runtime = make_mock_runtime()
    monkeypatch.setattr(
        runtime, "stream_generate", lambda prompt, sampling: iter(["Hello", " ", "world!"])
    )
    app = create_app(runtime=runtime)
    client = TestClient(app)

    payload = {
        "model": "lapis-tiny",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    body = response.text
    assert "data: " in body
    assert "[DONE]" in body
    assert "Hello" in body
    assert "world!" in body


def test_serve_chat_completions_streaming_error_handling(monkeypatch) -> None:
    runtime = make_mock_runtime()

    def error_gen(prompt, sampling, is_cancelled=None):
        raise CancellationError("Request was cancelled by client")

    monkeypatch.setattr(runtime, "stream_generate", error_gen)
    app = create_app(runtime=runtime)
    client = TestClient(app)

    payload = {
        "model": "lapis-tiny",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "cancelled" in response.text
    assert "Request was cancelled by client" in response.text
