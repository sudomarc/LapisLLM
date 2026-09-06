import pytest
import torch

from lapis.model.lapis_model import LapisModel


def make_model():
    return LapisModel(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=8,
        rope_theta=10000.0,
        dropout=0.0,
        bias=True,
    )


def test_model_rejects_non_long_inputs():
    with pytest.raises(ValueError, match="torch.long"):
        make_model()(torch.zeros(1, 4, dtype=torch.int32))


def test_model_rejects_out_of_range_token_ids():
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        make_model()(torch.tensor([[0, 1, 32, 3]], dtype=torch.long))


def test_model_rejects_invalid_label_ids():
    model = make_model()
    inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    labels = inputs.clone()
    labels[0, 2] = -2
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        model(inputs, labels=labels)


def test_model_rejects_single_token_sequences():
    with pytest.raises(ValueError, match="two tokens"):
        make_model()(torch.tensor([[1]], dtype=torch.long))


def test_model_rejects_odd_rope_head_dimension():
    with pytest.raises(ValueError, match="even for RoPE"):
        LapisModel(
            vocab_size=32,
            hidden_size=12,
            intermediate_size=24,
            num_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=8,
            rope_theta=10000.0,
            dropout=0.0,
            bias=True,
        )
