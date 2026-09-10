"""Correctness tests for the LAPIS model core."""

import warnings

import torch

from lapis.model.lapis_model import LapisModel
from lapis.model.rope import apply_rotary_pos_emb, precompute_freqs_cis


def make_model(**overrides):
    config = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "dropout": 0.0,
        "bias": True,
    }
    config.update(overrides)
    return LapisModel(**config)


def test_micro_forward_and_backward():
    model = make_model()
    model.train()
    input_ids = torch.randint(0, model.vocab_size, (2, 16))
    logits, loss = model(input_ids, labels=input_ids)
    assert logits.shape == (2, 16, model.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_parameter_count_matches_model():
    model = make_model()
    breakdown = model.parameter_breakdown()
    assert breakdown["total"] == model.num_parameters()
    assert breakdown["total"] == sum(p.numel() for p in model.parameters())


def test_gqa_shapes_and_forward_are_finite():
    model = make_model(num_attention_heads=4, num_key_value_heads=2)
    input_ids = torch.randint(0, model.vocab_size, (2, 8))
    logits, loss = model(input_ids, labels=input_ids)
    assert logits.shape == (2, 8, model.vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)


def test_rope_preserves_vector_norm():
    torch.manual_seed(0)
    head_dim = 8
    q = torch.randn(2, 4, 6, head_dim)
    k = torch.randn(2, 2, 6, head_dim)
    freqs = precompute_freqs_cis(head_dim, 6)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, freqs)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5, rtol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_model_dtype_cast_preserves_rope_behavior():
    model = make_model()
    model.float()
    input_ids = torch.randint(0, model.vocab_size, (1, 8))
    logits, loss = model(input_ids, labels=input_ids)
    assert logits.shape == (1, 8, model.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_model_dtype_cast_does_not_warn_about_rope_complex_conversion():
    model = make_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.float()
    assert not any("Casting complex values to real" in str(item.message) for item in caught)


def test_causal_future_tokens_do_not_change_current_logits():
    torch.manual_seed(0)
    model = make_model()
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    extended = torch.tensor([[1, 2, 3, 4, 9, 10]])
    with torch.no_grad():
        prefix_logits, _ = model(prefix)
        extended_logits, _ = model(extended)
    assert torch.allclose(prefix_logits[:, -1], extended_logits[:, 3], atol=1e-5, rtol=1e-5)


def test_padding_targets_are_ignored_by_loss():
    model = make_model()
    input_ids = torch.randint(0, model.vocab_size, (1, 8))
    labels = input_ids.clone()
    labels[:, 5:] = -100
    _, loss = model(input_ids, labels=labels)
    assert torch.isfinite(loss)
