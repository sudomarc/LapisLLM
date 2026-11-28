# LapisLLM — from-scratch coding language model
# Base model trains from random initialization. No third-party pretrained weights.
__version__ = "0.1.0"

# Optional: PyTorch is installed separately via [torch] extra
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False