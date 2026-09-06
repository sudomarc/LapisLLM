import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Normalize
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x / norm
        # Scale
        return x * self.weight