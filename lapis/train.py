"""LapisLLM Training Script — Stage 4: Training Engine Pipeline.

Full training loop with:
- Config-driven hyperparameters
- Tokenizer loading from HF format
- Model instantiation (LapisMicro)
- Data loading from dataset registry
- Mixed precision support
- Gradient accumulation
- Checkpoint saving
- Logging and metrics
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lapis.config.base import LapisConfig, load_config
from lapis.config.hardware import detect_hardware, get_device
from lapis.model.architecture import LapisMicro
from lapis.data.registry import DatasetRegistry, SourceRecord, initialize_development_registry
from lapis.tokenizer import BPETokenizer


class CodeDataset(Dataset):
    """Dataset for code training from approved registry records."""

    def __init__(
        self,
        records: list[SourceRecord],
        tokenizer: BPETokenizer,
        max_seq_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        # Tokenize all records and concatenate with EOS
        all_tokens = []
        for rec in records:
            tokens = tokenizer.encode(rec.content)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)

        # Create sliding windows of max_seq_len
        stride = max_seq_len // 2  # 50% overlap
        for i in range(0, len(all_tokens) - max_seq_len, stride):
            window = all_tokens[i:i + max_seq_len]
            if len(window) == max_seq_len:
                self.examples.append(torch.tensor(window, dtype=torch.long))

        print(f"Created {len(self.examples)} training examples from {len(records)} records")
        print(f"Total tokens: {len(all_tokens)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def create_dataloader(
    records: list[SourceRecord],
    tokenizer: BPETokenizer,
    config: LapisConfig,
    shuffle: bool = True,
) -> DataLoader:
    """Create DataLoader from registry records."""
    dataset = CodeDataset(records, tokenizer, config.max_seq_len)
    return DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=shuffle,
        num_workers=0,  # Keep simple for now
        pin_memory=torch.cuda.is_available(),
    )


def train_step(
    model: LapisMicro,
    batch: torch.Tensor,
    tokenizer: BPETokenizer,
    device: torch.device,
    config: LapisConfig,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> dict:
    """Single training step with optional mixed precision."""
    input_ids = batch.to(device)
    labels = input_ids.clone()

    # Forward pass with optional autocast
    if scaler is not None:
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
    else:
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

    return {"loss": loss, "logits": outputs["logits"]}


def evaluate(
    model: LapisMicro,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer: BPETokenizer,
    config: LapisConfig,
) -> float:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.to(device)
            labels = input_ids.clone()

            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

            # Accumulate loss weighted by tokens
            n_tokens = (labels != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    model.train()
    return total_loss / max(1, total_tokens)


def save_checkpoint(
    model: LapisMicro,
    optimizer: optim.Optimizer,
    scheduler,
    step: int,
    config: LapisConfig,
    run_id: str,
    loss: float,
    tokenizer: BPETokenizer,
) -> str:
    """Save training checkpoint."""
    run_dir = Path(config.run_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config.model_dump(),
        "loss": loss,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_special_tokens": {
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "unk_token_id": tokenizer.unk_token_id,
        },
    }

    path = run_dir / f"checkpoint-step-{step}.pt"
    torch.save(checkpoint, path)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description="LapisLLM Training — Stage 4")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--run-id", type=str, default="micro-local-001", help="Run identifier")
    parser.add_argument("--steps", type=int, default=10_000, help="Number of training steps")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=100, help="Log metrics every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16", "fp16"], default="fp32")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = LapisConfig()

    # Override precision from CLI
    config.precision = args.precision

    # Detect hardware
    hw = detect_hardware()
    device = get_device()
    print(f"LapisLLM training on {device}")
    print(f"Hardware: {hw['pytorch']['version']} | CUDA available: {hw['pytorch']['cuda_available']}")
    print(f"CPU cores: {hw['cpu']['cores']} | System RAM: {hw['memory_gb']} GB")

    # Set precision
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        print("BF16 not supported, falling back to FP32")
        config.precision = "fp32"
    if config.precision == "fp16" and not torch.cuda.is_available():
        print("FP16 requested but no CUDA, using FP32")
        config.precision = "fp32"

    # Set random seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Load tokenizer
    tokenizer_path = Path(".lapis-data/tokenizer/tokenizer.json")
    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Run tokenizer training first: python -m lapis.tokenizer.bpe_tokenizer")
        sys.exit(1)

    tokenizer = BPETokenizer.from_file(str(tokenizer_path))
    print(f"Loaded tokenizer: vocab_size={tokenizer.vocab_size}")

    # Update config vocab_size to match tokenizer
    config.vocab_size = tokenizer.vocab_size

    # Initialize model
    model = LapisMicro(config.model_dump())
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load dataset
    print("Loading dataset...")
    registry = initialize_development_registry()
    train_records = [r for r in registry.get_approved() if r.split == "train"]
    val_records = [r for r in registry.get_approved() if r.split in ("validation", "val", "test")]

    if not train_records:
        print("No training data found! Add data to registry first.")
        sys.exit(1)

    print(f"Train records: {len(train_records)}")
    print(f"Val records: {len(val_records)}")

    train_loader = create_dataloader(train_records, tokenizer, config, shuffle=True)
    val_loader = create_dataloader(val_records, tokenizer, config, shuffle=False) if val_records else None

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # LR scheduler: cosine with warmup
    total_steps = args.steps
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision scaler
    scaler = None
    if config.precision in ("fp16", "bf16") and device.type == "cuda":
        dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
        scaler = torch.amp.GradScaler(device_type="cuda")
        print(f"Using mixed precision: {config.precision}")

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint.get("step", 0)
        print(f"Resumed from step {start_step}")

    # Training loop
    print(f"\nStarting training: {args.steps} steps")
    print(f"Batch size: {config.batch_size} (micro={config.micro_batch_size}, acc={config.gradient_accumulation_steps})")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Precision: {config.precision}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Gradient clip: {config.grad_clip}")
    print("=" * 60)

    model.train()
    step = start_step
    epoch = 0
    accumulated_loss = 0.0
    accumulated_tokens = 0

    while step < total_steps:
        epoch += 1
        for batch in train_loader:
            if step >= total_steps:
                break

            # Training step
            result = train_step(model, batch, tokenizer, device, config, scaler)
            loss = result["loss"]

            # Gradient accumulation
            loss = loss / config.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate metrics
            with torch.no_grad():
                tokens = (batch != tokenizer.pad_token_id).sum().item()
                accumulated_loss += loss.item() * tokens * config.gradient_accumulation_steps
                accumulated_tokens += tokens

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if (step + 1) % args.log_every == 0:
                avg_loss = accumulated_loss / max(1, accumulated_tokens)
                ppl = math.exp(min(avg_loss, 20))  # Clamp for numerical stability
                lr = scheduler.get_last_lr()[0]
                print(f"Step {step + 1}/{total_steps} | "
                      f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e} | "
                      f"Epoch: {epoch}")

                accumulated_loss = 0.0
                accumulated_tokens = 0

            # Evaluation
            if val_loader and (step + 1) % args.eval_every == 0:
                val_loss = evaluate(model, val_loader, device, tokenizer, config)
                val_ppl = math.exp(min(val_loss, 20))
                print(f"  → Validation: Loss={val_loss:.4f} PPL={val_ppl:.2f}")
                model.train()

            # Checkpoint
            if (step + 1) % args.save_every == 0:
                avg_loss = accumulated_loss / max(1, accumulated_tokens) if accumulated_tokens > 0 else 0
                path = save_checkpoint(model, optimizer, scheduler, step + 1, config, args.run_id, avg_loss, tokenizer)
                print(f"  → Saved checkpoint: {path}")

            step += 1

    # Final checkpoint
    final_loss = accumulated_loss / max(1, accumulated_tokens) if accumulated_tokens > 0 else 0
    final_path = save_checkpoint(model, optimizer, scheduler, step, config, args.run_id, final_loss, tokenizer)
    print(f"\nTraining complete! Final checkpoint: {final_path}")

    # Final evaluation
    if val_loader:
        val_loss = evaluate(model, val_loader, device, tokenizer, config)
        val_ppl = math.exp(min(val_loss, 20))
        print(f"Final validation: Loss={val_loss:.4f} PPL={val_ppl:.2f}")


if __name__ == "__main__":
    main()