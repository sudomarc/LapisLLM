import torch


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute rotary position embeddings.
    
    Returns complex64 tensor of shape [max_seq_len, dim] holding
    cos/sin parameters for RoPE applied across the full head dimension.
    """
    # Create inverse frequencies for all positions
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device).float()
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim//2]
    # Interleave cos and sin to get full dimension
    freqs = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
    cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64 [max_seq_len, dim]
    return cis


def rotate_half(x):
    """Rotate half the channels.
    
    For input [..., head_dim], splits last dim in half and swaps/negates:
    [x1, x2] -> [-x2, x1] where x1 and x2 are the two halves.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs_cis):
    """Apply rotary position embeddings to query and key.
    
    Args:
        q: [batch, n_heads_q, seq_len, head_dim]
        k: [batch, n_heads_k, seq_len, head_dim]
        freqs_cis: [max_seq_len, head_dim] complex64
    
    Returns:
        q_rotated, k_rotated with RoPE applied
    """
    seq_len = q.shape[2]
    
    # Apply RoPE to q using q's head count
    freqs_q = freqs_cis[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    freqs_q = freqs_q.expand(q.shape[0], q.shape[1], -1, -1)  # [B, n_heads_q, seq_len, head_dim]
    cos_q = freqs_q.real  # [B, n_heads_q, seq_len, head_dim]
    sin_q = freqs_q.imag  # [B, n_heads_q, seq_len, head_dim]
    
    q_rotated = q * cos_q + rotate_half(q) * sin_q
    
    # Apply RoPE to k using k's head count (may differ from q due to GQA)
    freqs_k = freqs_cis[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    freqs_k = freqs_k.expand(k.shape[0], k.shape[1], -1, -1)  # [B, n_heads_k, seq_len, head_dim]
    cos_k = freqs_k.real  # [B, n_heads_k, seq_len, head_dim]
    sin_k = freqs_k.imag  # [B, n_heads_k, seq_len, head_dim]
    
    k_rotated = k * cos_k + rotate_half(k) * sin_k
    
    return q_rotated, k_rotated