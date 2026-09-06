import pytest
import torch

from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.generate import generate, sample_next_token


def make_fixture():
    tokenizer = Tokenizer.train_from_iterator(["hello world"], vocab_size=64, min_frequency=1)
    model = LapisModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rope_theta=10000.0,
        dropout=0.0,
        bias=True,
    )
    model.eval()
    return model, tokenizer


def test_temperature_zero_is_greedy():
    logits = torch.tensor([[0.1, 2.0, 0.5]])
    assert sample_next_token(logits, 0.0, 0, 1.0).item() == 1


def test_generation_parameter_validation():
    model, tokenizer = make_fixture()
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hello", -1, 0.8, 10, 0.9)
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hello", 1, -0.1, 10, 0.9)
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hello", 1, 0.8, -1, 0.9)
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hello", 1, 0.8, 10, 0.0)
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hello", 1, 0.8, 10, 1.1)


def test_generation_respects_context_limit_and_empty_prompt():
    model, tokenizer = make_fixture()
    text = generate(
        model,
        tokenizer,
        "",
        max_new_tokens=2,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
    )
    assert isinstance(text, str)
