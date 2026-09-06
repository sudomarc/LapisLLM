import torch.nn.functional as F
from torch import nn


class MLPDecoder(nn.Module):
    """MLP decoder block."""

    def __init__(self, hidden_size, intermediate_size, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act = F.silu
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """Apply the SwiGLU-style gated MLP."""
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        return self.dropout_layer(x)
