"""LapisLLM — from-scratch coding language model.

Base model trains from random initialization. No third-party pretrained weights.
"""
__version__ = "0.1.0"

# Ensure tokenizer is importable
try:
    from lapis.tokenizer.bpe_tokenizer import (
        train_tokenizer,
        encode_text,
        decode_text,
        get_tokenizer_config,
    )
except ImportError:
    pass  # tokenizer training may happen in separate step