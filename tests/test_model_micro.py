"""Small, real forward/backward sanity tests for the LAPIS model."""

import torch

from lapis.model.lapis_model import LapisModel


def test_micro_forward_and_backward():
    model = LapisModel(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        dropout=0.0,
    )
    model.train()

    input_ids = torch.randint(0, 256, (2, 16))
    logits, loss = model(input_ids, labels=input_ids)

    assert logits.shape == (2, 16, 256)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()
    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_parameter_count_matches_model():
    model = LapisModel(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        dropout=0.0,
    )
    breakdown = model.parameter_breakdown()
    assert breakdown["total"] == model.num_parameters()
    assert breakdown["total"] == sum(p.numel() for p in model.parameters())
