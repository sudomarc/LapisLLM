import torch.nn as nn

from lapis.model.attention import CausalSelfAttention
from lapis.model.mlp import MLPDecoder
from lapis.model.normalization import RMSNorm


class Block(nn.Module):
    """Single pre-norm transformer block with causal attention and SwiGLU."""

    def __init__(self, hidden_size, intermediate_size, num_attention_heads,
                 num_key_value_heads, rope_theta, max_position_embeddings, dropout, bias):
        super().__init__()
        self.attention = CausalSelfAttention(
            hidden_size, num_attention_heads, num_key_value_heads,
            rope_theta, max_position_embeddings, bias
        )
        self.mlp = MLPDecoder(hidden_size, intermediate_size, dropout, bias=bias)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, freqs_cis=None, kv_cache=None, start_pos=0):
        residual = x
        x_norm = self.norm1(x)
        attn_out, new_kv_cache = self.attention(
            x_norm, mask=mask, freqs_cis=freqs_cis, kv_cache=kv_cache, start_pos=start_pos
        )
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.norm2(x)
        x = residual + self.dropout(self.mlp(x_norm))
        return x, new_kv_cache
