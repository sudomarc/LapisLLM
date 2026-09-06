from __future__ import annotations

import torch


def pack_tokens(tokenized_documents, max_seq_length: int, pad_id: int = 0):
    """Pack tokenized documents into fixed-length sequences.

    Documents longer than ``max_seq_length`` are truncated. Shorter documents
    are padded with the caller's actual PAD token ID rather than assuming that
    PAD is token ID zero.
    """
    if max_seq_length < 1:
        raise ValueError("max_seq_length must be at least 1")
    if pad_id < 0:
        raise ValueError("pad_id must be non-negative")

    packed = []
    for doc_tokens in tokenized_documents:
        if len(doc_tokens) > max_seq_length:
            packed.append(list(doc_tokens[:max_seq_length]))
        else:
            packed.append(
                list(doc_tokens) + [pad_id] * (max_seq_length - len(doc_tokens))
            )
    return packed


def pack_into_batches(
    tokenized_data, batch_size: int, max_seq_length: int, pad_id: int = 0
):
    """Pack tokenized data into non-empty batches of fixed-length sequences."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    packed_sequences = pack_tokens(tokenized_data, max_seq_length, pad_id=pad_id)
    return [
        packed_sequences[i : i + batch_size]
        for i in range(0, len(packed_sequences), batch_size)
        if packed_sequences[i : i + batch_size]
    ]


def create_attention_mask(seq_len: int, device="cpu"):
    """Create an upper-triangular boolean causal attention mask."""
    if seq_len < 1:
        raise ValueError("seq_len must be at least 1")
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
