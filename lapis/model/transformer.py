import torch
import torch.nn as nn


class LapisTransformer(nn.Module):
    """Full Transformer stack."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Positional embeddings (learned) - we'll use RoPE instead of absolute positions
        self.pos_embedding = nn.Embedding(config["max_position_embeddings"], self.hidden_size)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(self.num_layers)
        ])
        
        # Final norm
        self.norm_final = RMSNorm(self.hidden_size, eps=1e-6)
        
        # Language modeling head
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
    
    def forward(self, input_ids, labels=None):
        B, seq_len = input_ids.shape
        
        # Token embeddings
        h = self.embed_tokens(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_embedding(position_ids)
        h = h + pos_emb
        
        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        
        # Forward through transformer blocks
        for layer in self.layers:
            h = layer(h, mask=mask)
        
        # Final norm and LM head
        h = self.norm_final(h)
        logits = self.lm_head(h)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss