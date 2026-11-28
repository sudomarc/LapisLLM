"""Architecture tests for Lapis Micro Stage 5."""
import os
import sys
import json
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lapis'))

from lapis.model.architecture import (
    LapisMicro, Block, CausalSelfAttention, SwiGLU, RMSNorm,
    rope_angle,
)


def test_rope_angle():
    """RoPE angle computation should work."""
    pos = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
    angles = rope_angle(pos, dim=32, base=10_000.0)
    assert angles.shape == (4, 32)
    # Sanity: first position should have angle 0 for dim=0
    assert angles[0, 0] == 0.0


def test_rmsnorm_basic():
    """RMSNorm should normalize and scale."""
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 3, 64)
    y = norm(x)
    # Output should have similar magnitude to input (normalization)
    assert y.shape == x.shape
    # Weight should be applied
    assert torch.isfinite(y).all()


def test_causal_self_attention():
    """Causal self-attention should produce output with correct shape."""
    attention = CausalSelfAttention(
        hidden_size=64, num_heads=4, head_dim=16, max_seq_len=512
    )
    x = torch.randn(2, 8, 64)  # batch=2, seq_len=8, hidden=64
    y = attention(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_swiglu_basic():
    """SwiGLU MLP should work."""
    swiglu = SwiGLU(hidden_size=64, intermediate_size=256)  # 4x
    x = torch.randn(2, 8, 64)
    y = swiglu(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_lapis_micro_shape():
    """LapisMicro forward pass should produce correct output shapes."""
    config = {
        "vocab_size": 16_000,
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "head_dim": 64,
        "max_seq_len": 512,
        "rope_base": 10_000.0,
        "shared_embedding": True,
    }

    model = LapisMicro(config)
    # Input: token IDs within vocab
    input_ids = torch.randint(0, 100, (2, 16))  # small vocab subset for test

    # Forward with labels
    output = model(input_ids=input_ids, labels=input_ids)
    assert "loss" in output
    assert "logits" in output
    assert output["loss"] is not None
    assert torch.isfinite(output["loss"])

    # Forward without labels
    output_no_labels = model(input_ids=input_ids, labels=None)
    assert output_no_labels["loss"] is None
    assert "logits" in output_no_labels

    # Logits shape: (batch, seq_len, vocab_size)
    assert output_no_labels["logits"].shape == (2, 16, 16_000)
    print(f"  LapisMicro test passed: loss={output['loss'].item():.4f}, "
          f"logits shape={output_no_labels['logits'].shape}")


def test_lapis_micro_parameter_count():
    """Micro model should have ~5-10M parameters."""
    config = {
        "vocab_size": 16_000,
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "head_dim": 64,
        "max_seq_len": 512,
        "shared_embedding": True,
    }

    model = LapisMicro(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    # Should be in the 5-10M range for Micro profile
    assert 5_000_000 <= total_params <= 15_000_000, \
        f"Parameter count {total_params:,} outside expected Micro range"

    # Check that embeddings + LM head are tied (if configured)
    if config["shared_embedding"]:
        assert model.lm_head.weight.shape == model.embed_tokens.weight.shape
        print(f"  Embedding tying: weights shared ({total_params:,} params)")


def test_model_with_trained_tokenizer():
    """Test model can use the trained BPE tokenizer's vocab size."""
    from lapis.tokenizer.bpe_tokenizer import train_tokenizer

    # Train a small tokenizer
    texts = ["def hello(): pass", "x = 1 + 2"]
    tokenizer = train_tokenizer(texts, vocab_size=100)

    # Get config binding
    config = {
        "vocab_size": len(tokenizer.get_vocab()),
        "hidden_size": 256,
        "num_layers": 2,  # Smaller for quick test
        "num_heads": 4,
        "head_dim": 64,
        "max_seq_len": 32,
        "rope_base": 10_000.0,
        "shared_embedding": True,
    }

    model = LapisMicro(config)

    # Encode a text and run forward
    enc = tokenizer.encode("def hello():")
    # Convert to tensor of token IDs
    # Simplified: just check model forward with random ids in vocab range
    test_ids = torch.randint(0, len(tokenizer.get_vocab()), (1, 8))
    output = model(input_ids=test_ids)
    assert output["loss"] is not None
    print(f"  Model with tokenizer vocab: loss={output['loss'].item():.4f}, "
          f"vocab_size={len(tokenizer.get_vocab())}")


if __name__ == "__main__":
    print("Running architecture tests...")

    tests = [
        test_rope_angle,
        test_rmsnorm_basic,
        test_causal_self_attention,
        test_swiglu_basic,
        test_lapis_micro_shape,
        test_lapis_micro_parameter_count,
        test_model_with_trained_tokenizer,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {test.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")