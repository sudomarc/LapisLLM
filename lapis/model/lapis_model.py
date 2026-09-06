"""Lapis model - main entry point."""

from __future__ import annotations

import torch
import torch.nn as nn

from lapis.model.block import Block
from lapis.model.normalization import RMSNorm
from lapis.model.rope import precompute_freqs_cis


class LapisModel(nn.Module):
    """Decoder-only causal Transformer used by LAPIS."""

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=384,
        intermediate_size=768,
        num_layers=2,
        num_attention_heads=6,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
        dropout=0.1,
        bias=True,
    ):
        super().__init__()
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if hidden_size < 1 or intermediate_size < 1 or num_layers < 1:
            raise ValueError("hidden_size, intermediate_size, and num_layers must be positive")
        if num_attention_heads < 1 or num_key_value_heads < 1:
            raise ValueError("attention head counts must be positive")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if max_position_embeddings < 2:
            raise ValueError("max_position_embeddings must be at least 2")
        if not torch.isfinite(torch.tensor(float(rope_theta))) or rope_theta <= 0:
            raise ValueError("rope_theta must be finite and greater than 0")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if (hidden_size // num_attention_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_layers = int(num_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.dropout = float(dropout)
        self.bias = bool(bias)

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.hidden_size // self.num_attention_heads,
                self.max_position_embeddings,
                self.rope_theta,
            ),
            persistent=False,
        )
        self.layers = nn.ModuleList([
            Block(
                self.hidden_size,
                self.intermediate_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.rope_theta,
                self.max_position_embeddings,
                self.dropout,
                self.bias,
            )
            for _ in range(self.num_layers)
        ])
        self.norm = RMSNorm(self.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=self.bias)
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
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long dtype")
        if input_ids.size(0) < 1 or input_ids.size(1) < 2:
            raise ValueError("input_ids must contain at least one batch and two tokens")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
            raise ValueError("input_ids contains token IDs outside the model vocabulary")

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
            if not isinstance(labels, torch.Tensor) or labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if labels.dtype != torch.long:
                raise ValueError("labels must use torch.long dtype")
            invalid = (labels < 0) & (labels != -100)
            invalid |= labels >= self.vocab_size
            if torch.any(invalid):
                raise ValueError("labels contains token IDs outside the model vocabulary")
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
