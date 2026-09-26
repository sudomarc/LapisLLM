"""Causal self-attention with grouped-query attention support."""

import torch
import torch.nn as nn

from lapis.model.rope import apply_rotary_pos_emb, precompute_freqs_cis


class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional GQA."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, rope_theta,
                 max_position_embeddings, bias):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if num_heads % (num_kv_heads or num_heads) != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def _repeat_kv(self, x, repeats):
        if repeats == 1:
            return x
        batch, kv_heads, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(batch, kv_heads, repeats, seq_len, head_dim)
            .reshape(batch, kv_heads * repeats, seq_len, head_dim)
        )

    def forward(self, x, mask=None, freqs_cis=None, kv_cache=None, start_pos=0):
        batch, seq_len, _ = x.shape
        if start_pos + seq_len > self.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is None:
            freqs_cis = precompute_freqs_cis(
                self.head_dim, self.max_position_embeddings, self.rope_theta
            ).to(device=x.device, dtype=torch.complex64)
        else:
            freqs_cis = freqs_cis.to(device=x.device, dtype=torch.complex64)

        freqs_cis_step = freqs_cis[start_pos : start_pos + seq_len]
        q, k = apply_rotary_pos_emb(q, k, freqs_cis_step)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv_cache = (k, v)

        repeats = self.num_heads // self.num_kv_heads
        k_rep = self._repeat_kv(k, repeats)
        v_rep = self._repeat_kv(v, repeats)

        total_seq_len = k.size(2)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is None and seq_len > 1:
            mask = torch.triu(
                torch.ones(seq_len, total_seq_len, device=x.device, dtype=torch.bool),
                diagonal=total_seq_len - seq_len + 1,
            )
        if mask is not None:
            scores = scores.masked_fill(mask.view(1, 1, seq_len, total_seq_len), float("-inf"))

        weights = torch.softmax(scores, dim=-1).to(v_rep.dtype)
        output = torch.matmul(weights, v_rep)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return self.o_proj(output), new_kv_cache
