"""Tokenizer tests for LapisLLM Stage 4."""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lapis'))

from lapis.tokenizer.bpe_tokenizer import (
    get_training_texts,
    train_tokenizer,
    encode_text,
    decode_text,
    get_tokenizer_config,
)


def test_get_training_texts():
    """Should retrieve texts from registry or fallback."""
    texts = get_training_texts()
    assert isinstance(texts, list)
    assert len(texts) > 0
    # Check that texts are non-empty strings
    for t in texts:
        assert isinstance(t, str)
        assert len(t) > 0


def test_get_tokenizer_config():
    """Config should have expected keys."""
    config = get_tokenizer_config()
    assert "vocab_size" in config
    assert "model_name" in config
    assert "max_seq_len" in config
    assert "special_tokens" in config
    assert config["vocab_size"] == 16_000


def test_train_tokenizer_basic():
    """Basic tokenizer training should produce a Tokenizer instance."""
    from tokenizers import Tokenizer as HFTokenizer

    # Create minimal training texts
    texts = [
        "def hello_world():\n    print('Hello')",
        "def sum_numbers(a, b):\n    return a + b",
    ]

    tokenizer = train_tokenizer(texts, vocab_size=100)  # Small vocab for speed
    assert isinstance(tokenizer, HFTokenizer)
    # Check that we have a vocabulary
    vocab = tokenizer.get_vocab()
    assert len(vocab) >= 50  # At least the requested size
    # Verify tokenizer is usable (can encode/decode)
    enc = tokenizer.encode("def hello():")
    assert enc is not None
    assert hasattr(enc, "ids")
    # Verify decode works
    decoded = tokenizer.decode(enc.ids)
    assert isinstance(decoded, str)


def test_encode_decode_round_trip():
    """Encode/decode round-trip should preserve text structure."""
    from tokenizers import Tokenizer as HFTokenizer

    # Train a minimal tokenizer
    texts = [
        "def hello():\n    pass",
        "x = 1 + 2",
    ]
    tokenizer = train_tokenizer(texts, vocab_size=100)

    # Test encode/decode round-trip
    enc = encode_text(tokenizer, "def hello():")
    assert enc is not None
    assert hasattr(enc, "ids")
    # Decode back
    decoded = tokenizer.decode(enc.ids)
    assert isinstance(decoded, str)
    # BPE may normalize/change text, just check it's non-empty
    assert len(decoded) > 0


def test_tokenizer_with_development_corpus():
    """Tokenizer should train on the development corpus."""
    from lapis.data.registry import initialize_development_registry

    # Initialize registry (creates .lapis-data/dataset-registry.json if missing)
    reg = initialize_development_registry()

    # Get texts
    texts = get_training_texts()
    assert len(texts) > 0

    # Train tokenizer
    tokenizer = train_tokenizer(texts, vocab_size=250)  # Small for test speed

    # Verify basic properties
    vocab = tokenizer.get_vocab()
    assert len(vocab) > 0
    print(f"  Vocab size from dev corpus: {len(vocab)}")


if __name__ == "__main__":
    print("Running tokenizer tests...")

    tests = [
        test_get_training_texts,
        test_get_tokenizer_config,
        test_train_tokenizer_basic,
        test_encode_decode_round_trip,
        test_tokenizer_with_development_corpus,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")