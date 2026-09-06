import torch
import torch.nn as nn
from lapis.model.rope import precompute_freqs_cis, apply_rotary_pos_emb


class CausalSelfAttention(nn.Module):
    """Causal self-attention with RoPE support (GQA compatible)."""
    
    def __init__(self, hidden_size, num_heads, num_kv_heads, rope_theta, max_position_embeddings, bias):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        
        # KV caching optimization: n_kv_heads per group
        self.n_kv_heads = self.num_kv_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Rotary position embeddings
        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_position_embeddings, rope_theta)
    
    def _repeat_kv(self, x, n_repeats):
        """Repeat key/value heads to match query heads.
        
        x: [B, n_kv_heads, seq_len, head_dim]
        returns: [B, n_kv_heads * n_repeats, seq_len, head_dim]
        """
        B, n_kv_heads, seq_len, head_dim = x.shape
        # Repeat along the head dimension dimension
        return (
            x[:, :, :, :]
            .unsqueeze(2)  # [B, n_kv_heads, 1, seq_len, head_dim]
            .repeat(1, 1, n_repeats, 1, 1)  # [B, n_kv_heads, n_repeats, seq_len, head_dim]
            .reshape(B, n_kv_heads * n_repeats, seq_len, head_dim)  # [B, n_heads, seq_len, head_dim]
        )
    
    def forward(self, x, mask, freqs_cis):
        B, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, freqs_cis)
        
        # For GQA: repeat K/V heads to match Q heads
        n_repeats = self.num_heads // self.num_kv_heads
        if n_repeats > 1:
            k = self._repeat_kv(k, n_repeats)
            v = self._repeat_kv(v, n_repeats)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Causal mask
        if mask is None:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        
        attn_weights = nn.functional.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output