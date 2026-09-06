import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # SwiGLU-style: gate * up
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout_layer(x)
        return x