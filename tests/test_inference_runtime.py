from pathlib import Path

import pytest
import torch

from lapis.inference.runtime import LapisRuntime, SamplingConfig, _sample_next_token


class FakeTokenizer:
    VERSION = "test-tokenizer"
    vocab_size = 8
    bos_id = 1
    eos_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3] if text else []

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join("x" for _ in ids)


class LongPromptTokenizer(FakeTokenizer):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return list(range(20))


class FakeModel:
    max_position_embeddings = 16

    def parameters(self):
        return iter(())


def make_runtime(tokenizer=None) -> LapisRuntime:
    runtime = object.__new__(LapisRuntime)
    runtime.model = FakeModel()
    runtime.tokenizer = tokenizer or FakeTokenizer()
    runtime.device = "cpu"
    runtime.checkpoint = Path("test.pt")
    return runtime


def test_sampling_config_rejects_non_finite_values() -> None:
    for temperature in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="temperature"):
            SamplingConfig(temperature=temperature).validate()
    for top_p in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="top_p"):
            SamplingConfig(top_p=top_p).validate()


def test_sampling_config_rejects_invalid_types() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        SamplingConfig(max_new_tokens=True).validate()
    with pytest.raises(ValueError, match="top_k"):
        SamplingConfig(top_k=1.5).validate()


def test_runtime_exposes_product_agnostic_helpers() -> None:
    runtime = make_runtime()
    assert runtime.tokenize("hello") == [3]
    info = runtime.get_model_info()
    assert info["device"] == "cpu"
    assert info["vocab_size"] == 8
    assert info["context_length"] == 16
    assert info["parameter_count"] == 0
    assert info["tokenizer_version"] == "test-tokenizer"


def test_prepare_ids_truncates_long_prompts_to_context_limit() -> None:
    runtime = make_runtime(LongPromptTokenizer())
    ids = runtime._prepare_ids("long prompt")
    assert ids.shape == (1, 16)
    assert ids.tolist()[0] == list(range(4, 20))


def test_sampling_rejects_non_finite_logits() -> None:
    logits = torch.tensor([[0.0, float("nan"), 1.0]])
    with pytest.raises(RuntimeError, match="non-finite logits"):
        _sample_next_token(logits, SamplingConfig())


def test_stream_generate_yields_incremental_text(monkeypatch) -> None:
    runtime = make_runtime()
    monkeypatch.setattr(runtime, "_iter_generated_token_ids", lambda prompt, config: iter([3, 3]))
    assert list(runtime.stream_generate("hello", SamplingConfig(max_new_tokens=2))) == ["x", "x"]


def test_stream_generate_buffers_unstable_decoding(monkeypatch) -> None:
    runtime = make_runtime()
    decoded = {(3,): "�", (3, 4): "é"}
    monkeypatch.setattr(runtime, "_iter_generated_token_ids", lambda prompt, config: iter([3, 4]))
    runtime.tokenizer.decode = lambda ids, skip_special_tokens=True: decoded[tuple(ids)]
    assert "".join(runtime.stream_generate("hello", SamplingConfig(max_new_tokens=2))) == "é"
