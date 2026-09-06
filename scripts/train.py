"""Training script for LAPIS model.

Usage:
    python scripts/train.py --config configs/tiny.yaml
    python scripts/train.py --config configs/tiny.yaml --resume checkpoints/latest
"""

import argparse
import os
import json
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from lapis.config import get_default_config, ModelConfig, TrainingConfig, DataConfig
from lapis.model.lapis_model import LapisModel
from lapis.logging import TrainingLogger


class SimpleDataset(Dataset):
    """Simple dataset for training - loads text and tokenizes it."""
    
    def __init__(self, text, tokenizer, max_length=512):
        self.text = text
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokens = tokenizer.encode(text)
    
    def __len__(self):
        return max(1, len(self.tokens) // self.max_length)
    
    def __getitem__(self, idx):
        # Return a chunk of tokens
        start = idx * self.max_length
        end = start + self.max_length
        chunk = self.tokens[start:end]
        
        # Pad if short
        if len(chunk) < self.max_length:
            chunk = chunk + [self.tokenizer.pad_id] * (self.max_length - len(chunk))
        
        input_ids = torch.tensor(chunk, dtype=torch.long)
        # For causal LM, the target is the same input shifted by 1
        return input_ids, input_ids


def train_one_epoch(model, dataloader, optimizer, scheduler, logger, device, step):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (input_ids, labels) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # Forward
        logits, loss = model(input_ids, labels=labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if hasattr(torch.nn.utils, "clip_grad_norm_"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        step += 1
        total_loss += loss.item()
        num_batches += 1
        
        # Log step
        if step % 10 == 0:
            logger.log_step(step, loss.item(), scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"])
    
    avg_loss = total_loss / max(num_batches, 1)
    return step, avg_loss


def main():
    parser = argparse.ArgumentParser(description="LAPIS Training")
    parser.add_argument("--config", type=str, default="configs/tiny.yaml", help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    # Load config
    config = get_default_config() if args.config is None else ...  # load config
    
    # Setup device
    device = args.device or config["runtime"]["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Setup model
    model_config = ModelConfig(config)
    model = LapisModel(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_layers=model_config.num_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        max_position_embeddings=model_config.max_position_embeddings,
        rope_theta=model_config.rope_theta,
        dropout=model_config.dropout,
        bias=model_config.bias,
    ).to(device)
    
    # Count parameters
    params = model_config.count_parameters()
    print(f"Model: Lapis Tiny")
    print(f"Parameters: {params['total']:,} total ({params['trainable']:,} trainable, {params['non_trainable']:,} non-trainable)")
    print(f"Embedding: {params['embedding']:,}")
    print(f"Attention: {params['attention']:,}")
    print(f"MLP: {params['mlp']:,}")
    print(f"Norm: {params['norm']:,}")
    print(f"LM Head: {params['lm_head']:,}")
    print()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    
    # Setup logger
    logger = TrainingLogger(config)
    logger.start_run()
    
    # Setup dataset (simple example)
    # In a real scenario, load from data/
    train_text = "Hello world this is a test of the LAPIS training pipeline. "
    dataset = SimpleDataset(train_text, tokenizer=None, max_length=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Training loop
    step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        step, avg_loss = train_one_epoch(model, dataloader, optimizer, scheduler, logger, device, step)
        
        # Log epoch end
        ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        logger.log_end(step, avg_loss, optimizer.param_groups[0]["lr"])
        print(f"Epoch {epoch + 1}: Average Loss {avg_loss:.4f} | Perplexity {ppl:.2f}")
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": config,
    }
    torch.save(checkpoint, "checkpoints/latest")
    print(f"Checkpoint saved to checkpoints/latest")


if __name__ == "__main__":
    main()