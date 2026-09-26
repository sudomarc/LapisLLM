"""Tests for mixed-precision inference support in LapisRuntime."""

import pytest
import torch

from lapis.config.model_config import model_config_kwargs
from lapis.inference.errors import CheckpointLoadError
from lapis.inference.runtime import LapisRuntime, SamplingConfig
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer


@pytest.fixture
def dummy_checkpoint(tmp_path):
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

    model = LapisModel(**model_config_kwargs(config))
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
            "tokenizer_version": tokenizer.VERSION,
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_runtime_from_checkpoint_dtypes(dummy_checkpoint):
    runtime_fp32 = LapisRuntime.from_checkpoint(dummy_checkpoint, dtype="float32")
    assert runtime_fp32.dtype == torch.float32
    assert runtime_fp32.get_model_info()["dtype"] == "float32"
    assert "supported_dtypes" in runtime_fp32.get_capabilities()

    runtime_fp16 = LapisRuntime.from_checkpoint(dummy_checkpoint, dtype="float16")
    assert runtime_fp16.dtype == torch.float16
    assert runtime_fp16.get_model_info()["dtype"] == "float16"
    p_dtype = next(runtime_fp16.model.parameters()).dtype
    assert p_dtype == torch.float16

    runtime_bf16 = LapisRuntime.from_checkpoint(dummy_checkpoint, dtype="bfloat16")
    assert runtime_bf16.dtype == torch.bfloat16
    assert runtime_bf16.get_model_info()["dtype"] == "bfloat16"
    p_dtype_bf = next(runtime_bf16.model.parameters()).dtype
    assert p_dtype_bf == torch.bfloat16


def test_runtime_invalid_dtype(dummy_checkpoint):
    with pytest.raises(CheckpointLoadError, match="Unsupported dtype"):
        LapisRuntime.from_checkpoint(dummy_checkpoint, dtype="invalid_dtype")


def test_mixed_precision_generation(dummy_checkpoint):
    runtime = LapisRuntime.from_checkpoint(dummy_checkpoint, dtype="bfloat16")
    sampling = SamplingConfig(max_new_tokens=8, seed=42)
    output = runtime.generate("hello", sampling)
    assert isinstance(output, str)
    assert len(output) > 0

    meta = runtime.generate_with_metadata("hello", sampling)
    assert meta["completion_tokens"] > 0
    assert meta["tokens_per_second"] >= 0.0
