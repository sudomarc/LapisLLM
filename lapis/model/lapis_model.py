"""Lapis model - main entry point."""

import torch
import torch.nn as nn

from lapis.model.block import Block
from lapis.model.embeddings import TokenEmbedding
from lapis.model.rope import precompute_freqs_cis, apply_rotary_pos_emb
from lapis.model.attention import CausalSelfAttention
from lapis.model.normalization import RMSNorm
from lapis.model.mlp import MLPDecoder


class LapisModel(nn.Module):
    """Main LAPIS model - decoder-only causal Transformer."""
    
    def __init__(self, vocab_size=50304, hidden_size=384, intermediate_size=768,
                 num_layers=2, num_attention_heads=6, num_key_value_heads=2,
                 max_position_embeddings=512, rope_theta=10000.0, dropout=0.1, bias=True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.bias = bias
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        
        # RoPE precomputation
        self.freqs_cis = precompute_freqs_cis(hidden_size // num_attention_heads, max_position_embeddings, rope_theta)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            Block(hidden_size, intermediate_size, num_attention_heads, num_key_value_heads,
                  rope_theta, max_position_embeddings, dropout, bias)
            for _ in range(num_layers)
        ])
        
        # Final RMSNorm
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        
        # LM Head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=bias)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Standard PyTorch weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        # input_ids: [batch, seq_len]
        
        # Embed tokens
        h = self.embed_tokens(input_ids)  # [B, S, H]
        
        # Rotary position embeddings
        # freqs_cis shape: [max_seq_len, head_dim]
        # We need to apply RoPE to attention
        
        # Causal mask
        device = input_ids.device
        mask = torch.triu(
            torch.ones(input_ids.size(1), input_ids.size(1), device=device, dtype=torch.bool),
            diagonal=1
        )
        
        # Forward through transformer blocks
        for layer in self.layers:
            h = layer(h, mask, self.freqs_cis)
        
        # Final norm
        h = self.norm(h)
        
        # LM head
        logits = self.lm_head(h)  # [B, S, vocab_size]
        
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    @staticmethod
    def count_parameters():
        """Count total parameters."""
        # This will be called during logging
        pass