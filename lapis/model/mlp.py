import torch.nn.functional as F
from torch import nn


class MLPDecoder(nn.Module):
    """SwiGLU feed-forward projection used by a Transformer block."""

    def __init__(self, hidden_size, intermediate_size, dropout=0.0, bias=True):
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("hidden_size and intermediate_size must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = float(dropout)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = F.silu

    def forward(self, x):
        """Apply the SwiGLU projection; residual-branch dropout belongs to Block."""
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
