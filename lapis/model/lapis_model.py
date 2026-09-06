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
        self._validate_config(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rope_theta,
            dropout,
        )

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
        self.layers = nn.ModuleList(
            [
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
            ]
        )
        self.norm = RMSNorm(self.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=self.bias)
        self.apply(self._init_weights)

    @staticmethod
    def _validate_config(
        vocab_size,
        hidden_size,
        intermediate_size,
        num_layers,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings,
        rope_theta,
        dropout,
    ) -> None:
        integer_fields = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "max_position_embeddings": max_position_embeddings,
        }
        for name, value in integer_fields.items():
            if int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        head_dim = hidden_size // num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError(
                "head dimension must be even for Rotary Position Embeddings"
            )
        if max_position_embeddings < 2:
            raise ValueError("max_position_embeddings must be at least 2")
        if float(rope_theta) <= 0:
            raise ValueError("rope_theta must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

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
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long dtype")

        seq_len = input_ids.size(1)
        if seq_len == 0:
            raise ValueError("input_ids sequence length must be greater than zero")
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings "
                f"({self.max_position_embeddings})"
            )
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.vocab_size
        ):
            raise ValueError(
                f"input_ids contain values outside [0, {self.vocab_size - 1}]"
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
            if labels.dtype != torch.long:
                raise ValueError("labels must use torch.long dtype")
            shifted_targets = labels[:, 1:].contiguous()
            shifted_logits = logits[:, :-1, :].contiguous()
            valid_targets = shifted_targets.ne(-100)
            if not valid_targets.any():
                raise ValueError("labels contain no valid next-token targets")
            invalid_targets = shifted_targets.masked_select(valid_targets)
            if invalid_targets.numel() and (
                int(invalid_targets.min()) < 0
                or int(invalid_targets.max()) >= self.vocab_size
            ):
                raise ValueError(
                    f"labels contain values outside [0, {self.vocab_size - 1}]"
                )
            loss = nn.functional.cross_entropy(
                shifted_logits.view(-1, self.vocab_size),
                shifted_targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def num_parameters(self, trainable_only=False):
        """Return the actual parameter count of the instantiated model."""
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )

    def parameter_breakdown(self):
        """Return parameter counts by major model component."""
        embedding = sum(p.numel() for p in self.embed_tokens.parameters())
        attention = sum(
            p.numel() for layer in self.layers for p in layer.attention.parameters()
        )
        mlp = sum(
            p.numel() for layer in self.layers for p in layer.mlp.parameters()
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
