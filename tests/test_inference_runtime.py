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
    with pytest.raises(ValueError, match="seed"):
        SamplingConfig(seed=-1).validate()
    with pytest.raises(ValueError, match="seed"):
        SamplingConfig(seed="123").validate()  # type: ignore[arg-type]


def test_sampling_seed_is_deterministic() -> None:
    logits = torch.randn(1, 10)
    config = SamplingConfig(seed=42)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    t1 = _sample_next_token(logits, config, generator=g1)
    t2 = _sample_next_token(logits, config, generator=g2)
    assert torch.equal(t1, t2)


def test_generate_with_metadata_returns_structured_metrics(monkeypatch) -> None:
    runtime = make_runtime()
    monkeypatch.setattr(runtime, "_iter_generated_token_ids", lambda prompt, config: iter([3, 3, 3]))
    meta = runtime.generate_with_metadata("hello", SamplingConfig(max_new_tokens=3))
    assert meta["text"] == "xxxx"
    assert meta["prompt_tokens"] == 1
    assert meta["completion_tokens"] == 3
    assert meta["total_tokens"] == 4
    assert "time_to_first_token_ms" in meta
    assert "total_time_ms" in meta
    assert "tokens_per_second" in meta


def test_from_checkpoint_supports_quantize(tmp_path: Path) -> None:
    from lapis.tokenizer.tokenizer import Tokenizer

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer.train_from_iterator(["hello world Lapis LLM"], vocab_size=32)
    tokenizer.save(str(tokenizer_dir))

    checkpoint_path = tmp_path / "model.pt"
    config = {
        "model": {
            "vocab_size": tokenizer.vocab_size,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "rope_theta": 10000.0,
            "dropout": 0.0,
            "bias": True,
        }
    }
    from lapis.config.model_config import model_config_kwargs
    from lapis.model.lapis_model import LapisModel

    model = LapisModel(**model_config_kwargs(config))
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
            "tokenizer_version": tokenizer.VERSION,
        },
        checkpoint_path,
    )

    runtime = LapisRuntime.from_checkpoint(checkpoint_path, device="cpu", quantize=True)
    assert runtime.quantized is True
    info = runtime.get_model_info()
    assert info["quantized"] is True
    assert info["capabilities"]["quantized"] is True
    assert "seed" in info["capabilities"]["sampling_parameters"]


def test_runtime_exposes_product_agnostic_helpers() -> None:
    runtime = make_runtime()
    assert runtime.tokenize("hello") == [3]
    info = runtime.get_model_info()
    assert info["device"] == "cpu"
    assert info["vocab_size"] == 8
    assert info["context_length"] == 16
    assert info["parameter_count"] == 0
    assert info["tokenizer_version"] == "test-tokenizer"
    assert info["capabilities"]["streaming"] is True

def test_runtime_get_capabilities() -> None:
    runtime = make_runtime()
    caps = runtime.get_capabilities()
    assert caps["context_length"] == 16
    assert caps["tokenizer_version"] == "test-tokenizer"
    assert caps["vocab_size"] == 8
    assert caps["streaming"] is True
    assert "temperature" in caps["sampling_parameters"]


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
