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
This tiny local run validates the complete learning pipeline on CPU.
LAPIS learns from examples by repeatedly predicting the next token.
The checkpoint can then be loaded by the generation and chat interfaces.
""".strip()


class TextDataset(Dataset):
    """Fixed-length causal language-modeling examples.

    ``seq_len`` is the number of input positions consumed by the model. The
    dataset keeps one extra token so the model can perform the causal shift
    internally and predict the final target in each sample.
    """

    def __init__(self, tokens: list[int], seq_len: int, pad_id: int):
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        needed = seq_len + 1
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, max(1, len(tokens) - 1), seq_len):
            chunk = tokens[start : start + needed]
            real_length = len(chunk)
            if real_length < needed:
                chunk += [pad_id] * (needed - real_length)
            sample = torch.tensor(chunk[:needed], dtype=torch.long)
            labels = sample.clone()
            if real_length < needed:
                labels[real_length:] = -100
            self.samples.append((sample, labels))
        if not self.samples:
            self.samples.append(
                (
                    torch.full((needed,), pad_id, dtype=torch.long),
                    torch.full((needed,), -100, dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


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


def load_or_train_tokenizer(config: dict, corpus: str) -> Tokenizer:
    tokenizer_cfg = config.get("tokenizer", {})
    path = Path(tokenizer_cfg.get("path", "artifacts/tokenizer"))
    tokenizer_file = path / "tokenizer.json"
    if tokenizer_file.exists():
        return Tokenizer.load(str(tokenizer_file))

    tokenizer = Tokenizer.train_from_iterator(
        [corpus],
        vocab_size=int(tokenizer_cfg.get("vocab_size", config["model"]["vocab_size"])),
        min_frequency=int(tokenizer_cfg.get("min_frequency", 1)),
    )
    tokenizer.save(str(path))
    return tokenizer


def save_checkpoint(path: Path, model, optimizer, scheduler, step, epoch, config, tokenizer):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "step": step,
            "epoch": epoch,
            "config": config,
            "tokenizer_version": tokenizer.VERSION,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )
    tokenizer.save(str(path.parent / "tokenizer"))


def build_scheduler(optimizer, training_config):
    """Build linear warmup followed by cosine decay."""
    warmup_steps = int(training_config.warmup_steps)
    total_steps = max(1, int(training_config.max_steps))

    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-12, float(step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_training_seq_len(model_config: ModelConfig, config: dict) -> int:
    """Resolve a dataset sequence length that leaves room for the target token."""
    configured = int(
        config.get("data", {}).get(
            "max_seq_length", model_config.max_position_embeddings - 1
        )
    )
    max_input_length = model_config.max_position_embeddings - 1
    if max_input_length < 2:
        raise ValueError("max_position_embeddings must be at least 3")
    return min(configured, max_input_length)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LAPIS")
    parser.add_argument("--config", default="configs/local-dev.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=None, help="Optional UTF-8 text file")
    parser.add_argument("--tokenizer", default=None, help="Optional trained tokenizer directory")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latest.pt",
        help="Output checkpoint path (defaults to checkpoints/latest.pt)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_yaml(args.config)
    corpus = Path(args.data).read_text(encoding="utf-8") if args.data else DEFAULT_CORPUS
    tokenizer = Tokenizer.load(args.tokenizer) if args.tokenizer else load_or_train_tokenizer(config, corpus)

    config.setdefault("model", {})["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(config)
    training_config = TrainingConfig(config)
    device = resolve_device(config, args.device)
    dtype = resolve_dtype(config, device)
    seq_len = resolve_training_seq_len(model_config, config)

    tokens = tokenizer.encode(corpus, add_special_tokens=True)
    if max(tokens, default=0) >= model_config.vocab_size:
        raise ValueError(
            f"Tokenizer produced id {max(tokens)} but model vocab_size is {model_config.vocab_size}"
        )

    dataset = TextDataset(tokens, seq_len=seq_len, pad_id=tokenizer.pad_id)
    dataloader = DataLoader(dataset, batch_size=training_config.micro_batch_size, shuffle=True, drop_last=False)

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
    print("Model: Lapis Local Dev")
    print(f"Tokenizer: {tokenizer}")
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Parameters: {breakdown['total']:,}")
    print(f"Vocabulary: {tokenizer.vocab_size:,}")
    print(f"Dataset samples: {len(dataset)} | sequence length: {seq_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    scheduler = build_scheduler(optimizer, training_config)

    optimizer_step = 0
    epoch = 0
    accumulation_count = 0
    running_loss = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", optimizer.state_dict()))
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer_step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("epoch", 0))
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if checkpoint.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        print(f"Resumed from optimizer step {optimizer_step}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    while optimizer_step < training_config.max_steps and args.epochs > 0:
        args.epochs -= 1
        epoch += 1
        for batch_index, (input_ids, labels) in enumerate(dataloader):
            if optimizer_step >= training_config.max_steps:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            _, loss = model(input_ids, labels=labels)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")

            (loss / training_config.gradient_accumulation_steps).backward()
            running_loss += loss.item()
            accumulation_count += 1

            end_of_epoch = batch_index == len(dataloader) - 1
            should_step = accumulation_count == training_config.gradient_accumulation_steps
            if should_step or end_of_epoch:
                if accumulation_count < training_config.gradient_accumulation_steps:
                    correction = training_config.gradient_accumulation_steps / accumulation_count
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)

                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if optimizer_step % 10 == 0 or optimizer_step == 1:
                    avg_loss = running_loss / max(1, accumulation_count)
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                    print(
                        f"step={optimizer_step:04d} loss={avg_loss:.4f} "
                        f"ppl={ppl:.2f} lr={optimizer.param_groups[0]['lr']:.6g}"
                    )
                running_loss = 0.0
                accumulation_count = 0

    checkpoint_path = Path(args.checkpoint)
    save_checkpoint(checkpoint_path, model, optimizer, scheduler, optimizer_step, epoch, config, tokenizer)
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Tokenizer saved: {checkpoint_path.parent / 'tokenizer'}")


if __name__ == "__main__":
    main()
