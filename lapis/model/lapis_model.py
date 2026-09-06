"""Lapis model - main entry point."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lapis.model.block import Block
from lapis.model.normalization import RMSNorm
from lapis.model.rope import precompute_freqs_cis


class LapisModel(nn.Module):
    """Decoder-only causal Transformer used by LAPIS."""

    def __init__(self, vocab_size=50304, hidden_size=384, intermediate_size=768,
                 num_layers=2, num_attention_heads=6, num_key_value_heads=2,
                 max_position_embeddings=512, rope_theta=10000.0, dropout=0.1,
                 bias=True):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        rope_theta = float(rope_theta)
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.bias = bias

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                hidden_size // num_attention_heads,
                max_position_embeddings,
                rope_theta,
            ),
            persistent=False,
        )
        self.layers = nn.ModuleList([
            Block(
                hidden_size,
                intermediate_size,
                num_attention_heads,
                num_key_value_heads,
                rope_theta,
                max_position_embeddings,
                dropout,
                bias,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=bias)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        seq_len = input_ids.size(1)
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings "
                f"({self.max_position_embeddings})"
            )

        h = self.embed_tokens(input_ids)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for layer in self.layers:
            h = layer(h, mask, self.freqs_cis)

        logits = self.lm_head(self.norm(h))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_parameters(self, trainable_only=False):
        """Return the actual parameter count of the instantiated model."""
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    def parameter_breakdown(self):
        """Return parameter counts by major model component."""
        embedding = sum(p.numel() for p in self.embed_tokens.parameters())
        attention = sum(
            p.numel() for layer in self.layers
            for p in layer.attention.parameters()
        )
        mlp = sum(
            p.numel() for layer in self.layers
            for p in layer.mlp.parameters()
        )
        norm = sum(p.numel() for p in self.norm.parameters())
        norm += sum(
            p.numel()
            for layer in self.layers
            for norm_layer in (layer.norm1, layer.norm2)
            for p in norm_layer.parameters()
        )
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        return {
            "total": self.num_parameters(),
            "trainable": self.num_parameters(trainable_only=True),
            "embedding": embedding,
            "attention": attention,
            "mlp": mlp,
            "norm": norm,
            "lm_head": lm_head,
        }
