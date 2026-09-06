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

    def forward(self, x, mask, freqs_cis):
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask, freqs_cis)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + residual
        return x
