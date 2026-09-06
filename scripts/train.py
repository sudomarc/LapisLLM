#!/usr/bin/env python3
"""Train a Lapis model from a YAML configuration."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from lapis.config.model_config import ModelConfig
from lapis.config.training_config import TrainingConfig
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer

DEFAULT_CORPUS = """
LAPIS is an independent language model research project.
We build the tokenizer, transformer, training loop, evaluation pipeline and inference stack ourselves.
A language model learns to predict the next token from context.
The purpose of this tiny training run is to validate the complete learning pipeline.
This corpus is intentionally small and repetitive for local smoke training.
""".strip()


class TextDataset(Dataset):
    """Fixed-length causal language-modeling examples."""

    def __init__(self, tokens: list[int], seq_len: int, pad_id: int):
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.seq_len = seq_len
        self.pad_id = pad_id
        needed = seq_len + 1
        self.samples: list[torch.Tensor] = []
        for start in range(0, max(0, len(tokens) - 1), seq_len):
            chunk = tokens[start : start + needed]
            if len(chunk) < needed:
                chunk += [pad_id] * (needed - len(chunk))
            self.samples.append(torch.tensor(chunk[:needed], dtype=torch.long))
        if not self.samples:
            self.samples.append(torch.full((needed,), pad_id, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        inputs = sample[:-1]
        labels = sample[1:].clone()
        labels[labels == self.pad_id] = -100
        return inputs, labels


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(config: dict, requested: str | None) -> torch.device:
    value = requested or config.get("runtime", {}).get("device", "auto")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(value)


def resolve_dtype(config: dict, device: torch.device) -> torch.dtype:
    requested = str(config.get("runtime", {}).get("dtype", "float32")).lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype = mapping.get(requested, torch.float32)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def save_checkpoint(path: Path, model, optimizer, scheduler, step, config, tokenizer):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "step": step,
            "config": config,
        },
        path,
    )
    tokenizer.save(str(path.parent / "tokenizer"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LAPIS")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=None, help="Optional UTF-8 text file")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_yaml(args.config)
    model_config = ModelConfig(config)
    training_config = TrainingConfig(config)
    device = resolve_device(config, args.device)
    dtype = resolve_dtype(config, device)
    seq_len = min(
        model_config.max_position_embeddings,
        int(config.get("data", {}).get("max_seq_length", model_config.max_position_embeddings)),
    )

    tokenizer = Tokenizer()
    corpus = Path(args.data).read_text(encoding="utf-8") if args.data else DEFAULT_CORPUS
    tokens = tokenizer.encode(corpus, add_special_tokens=True)
    if max(tokens, default=0) >= model_config.vocab_size:
        raise ValueError(
            f"Tokenizer produced id {max(tokens)} but model vocab_size is {model_config.vocab_size}"
        )

    dataset = TextDataset(tokens, seq_len=seq_len, pad_id=tokenizer.pad_id)
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=True,
        drop_last=False,
    )

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
    ).to(device=device, dtype=dtype)

    breakdown = model.parameter_breakdown()
    print(f"Model: Lapis Tiny")
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Parameters: {breakdown['total']:,}")
    print(f"Dataset samples: {len(dataset)} | sequence length: {seq_len}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, training_config.max_steps)
    )

    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", optimizer.state_dict()))
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        print(f"Resumed from step {start_step}")

    model.train()
    step = start_step
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    while step < training_config.max_steps and args.epochs > 0:
        args.epochs -= 1
        for input_ids, labels in dataloader:
            if step >= training_config.max_steps:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            _, loss = model(input_ids, labels=labels)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")

            (loss / training_config.gradient_accumulation_steps).backward()
            running_loss += loss.item()

            if (step + 1) % training_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 10 == 0:
                avg_loss = running_loss / 10
                ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                print(
                    f"step={step:04d} loss={avg_loss:.4f} "
                    f"ppl={ppl:.2f} lr={optimizer.param_groups[0]['lr']:.6g}"
                )
                running_loss = 0.0

    checkpoint_path = Path("checkpoints/latest.pt")
    save_checkpoint(checkpoint_path, model, optimizer, scheduler, step, config, tokenizer)
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Tokenizer saved: {checkpoint_path.parent / 'tokenizer'}")


if __name__ == "__main__":
    main()
