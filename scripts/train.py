#!/usr/bin/env python3
"""Train a Lapis model from a YAML configuration."""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from lapis.config.base import resolve_device as resolve_requested_device
from lapis.config.model_config import ModelConfig
from lapis.config.runtime_config import (
    apply_cpu_fast_profile,
    configure_cpu_runtime,
    get_dataloader_options,
    validate_fast_profile,
)
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
    """Fixed-length causal language-modeling examples."""

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
    return resolve_requested_device(value)


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
    target_vocab = int(tokenizer_cfg.get("vocab_size", config["model"]["vocab_size"]))
    min_frequency = int(tokenizer_cfg.get("min_frequency", 1))

    if tokenizer_file.exists():
        tokenizer = Tokenizer.load(str(tokenizer_file))
        if tokenizer.vocab_size == target_vocab:
            return tokenizer
        print(
            f"Tokenizer vocab mismatch ({tokenizer.vocab_size} != {target_vocab}); "
            "retraining tokenizer from the current corpus."
        )

    tokenizer = Tokenizer.train_from_iterator(
        [corpus],
        vocab_size=target_vocab,
        min_frequency=min_frequency,
    )
    tokenizer.save(str(path))
    return tokenizer


def save_checkpoint(path: Path, model, optimizer, scheduler, step, epoch, config, tokenizer):
    """Stage checkpoint and tokenizer together, then publish with rollback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    generation_dir = Path(tempfile.mkdtemp(prefix=".generation-", dir=path.parent))
    staged_checkpoint = generation_dir / path.name
    staged_tokenizer = generation_dir / "tokenizer"
    visible_tokenizer = path.parent / "tokenizer"
    checkpoint_backup = path.parent / f".{path.name}.backup"
    tokenizer_backup = path.parent / ".tokenizer.backup"
    published_checkpoint = False
    published_tokenizer = False
    old_checkpoint = path.exists()
    old_tokenizer = visible_tokenizer.exists()

    try:
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
                "torch_rng_state": torch.get_rng_state().clone().cpu(),
                "cuda_rng_state": [state.clone().cpu() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None,
            },
            staged_checkpoint,
        )
        tokenizer.save(str(staged_tokenizer))

        if checkpoint_backup.exists():
            checkpoint_backup.unlink()
        if tokenizer_backup.exists():
            shutil.rmtree(tokenizer_backup)
        if old_checkpoint:
            os.replace(path, checkpoint_backup)
        if old_tokenizer:
            os.replace(visible_tokenizer, tokenizer_backup)

        os.replace(staged_checkpoint, path)
        published_checkpoint = True
        os.replace(staged_tokenizer, visible_tokenizer)
        published_tokenizer = True

        if checkpoint_backup.exists():
            checkpoint_backup.unlink()
        if tokenizer_backup.exists():
            shutil.rmtree(tokenizer_backup)
    except Exception:
        if published_tokenizer and visible_tokenizer.exists():
            shutil.rmtree(visible_tokenizer)
        if old_tokenizer and tokenizer_backup.exists():
            os.replace(tokenizer_backup, visible_tokenizer)
        if published_checkpoint and path.exists():
            path.unlink()
        if old_checkpoint and checkpoint_backup.exists():
            os.replace(checkpoint_backup, path)
        raise
    finally:
        shutil.rmtree(generation_dir, ignore_errors=True)
        if checkpoint_backup.exists():
            checkpoint_backup.unlink()
        if tokenizer_backup.exists():
            shutil.rmtree(tokenizer_backup)


def restore_torch_rng_state(state) -> None:
    """Restore RNG state with compatibility for older/corrupted checkpoints."""
    if state is None:
        return
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state, dtype=torch.uint8)
    else:
        state = state.detach().to(device="cpu", dtype=torch.uint8)
    if state.numel() == 0:
        raise ValueError("Checkpoint contains an empty torch RNG state.")
    torch.set_rng_state(state.contiguous())


def restore_cuda_rng_state(state) -> None:
    """Restore CUDA RNG state while accepting tensor/list checkpoint formats."""
    if state is None or not torch.cuda.is_available():
        return
    if isinstance(state, torch.Tensor):
        states = [state]
    elif isinstance(state, (list, tuple)):
        states = list(state)
    else:
        raise TypeError("Checkpoint contains an invalid CUDA RNG state.")

    normalized = []
    for item in states:
        if not isinstance(item, torch.Tensor):
            item = torch.as_tensor(item, dtype=torch.uint8)
        else:
            item = item.detach().to(device="cpu", dtype=torch.uint8)
        normalized.append(item.contiguous())
    torch.cuda.set_rng_state_all(normalized)


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
    configured = int(
        config.get("data", {}).get(
            "max_seq_length", model_config.max_position_embeddings - 1
        )
    )
    max_input_length = model_config.max_position_embeddings - 1
    if max_input_length < 2:
        raise ValueError("max_position_embeddings must be at least 3")
    return min(configured, max_input_length)


def resolve_tokenizer(config: dict, corpus: str, resume_path: str | None, explicit_path: str | None) -> Tokenizer:
    """Load the checkpoint tokenizer on resume; only train one for fresh runs."""
    if explicit_path:
        return Tokenizer.load(explicit_path)

    if resume_path:
        checkpoint_tokenizer = Path(resume_path).parent / "tokenizer"
        if (checkpoint_tokenizer / "tokenizer.json").exists():
            print(f"Loading tokenizer from checkpoint: {checkpoint_tokenizer}")
            return Tokenizer.load(str(checkpoint_tokenizer))

    return load_or_train_tokenizer(config, corpus)


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
        "--cpu-fast",
        action="store_true",
        help="Use a small CPU-oriented model/profile for fast training iterations; incompatible with --resume.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/latest.pt",
        help="Output checkpoint path (defaults to checkpoints/latest.pt)",
    )
    args = parser.parse_args()

    if args.cpu_fast and args.resume:
        raise ValueError("--cpu-fast cannot resume a checkpoint with a different model architecture")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_yaml(args.config)
    if args.cpu_fast:
        config = apply_cpu_fast_profile(config)
        validate_fast_profile(config)
        print("CPU-fast profile enabled: using a reduced model and sequence length for iteration speed.")

    corpus = Path(args.data).read_text(encoding="utf-8") if args.data else DEFAULT_CORPUS
    tokenizer = resolve_tokenizer(config, corpus, args.resume, args.tokenizer)

    config.setdefault("model", {})["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(config)
    training_config = TrainingConfig(config)
    device = resolve_device(config, args.device)
    if args.cpu_fast and device.type != "cpu":
        raise ValueError("--cpu-fast requires CPU; remove --device cuda or set --device cpu")
    if device.type == "cpu":
        configure_cpu_runtime(config, device)
    dtype = resolve_dtype(config, device)
    seq_len = resolve_training_seq_len(model_config, config)

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
        **get_dataloader_options(config, device),
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
    print("Model: Lapis Small")
    print(f"Tokenizer: {tokenizer}")
    print(f"Device: {device} | dtype: {dtype}")
    if device.type == "cpu":
        print(f"CPU threads: {torch.get_num_threads()} | interop threads: {torch.get_num_interop_threads()}")
    print(f"Parameters: {breakdown['total']:,}")
    print(f"Vocabulary: {tokenizer.vocab_size:,}")
    print(f"Tokens: {len(tokens):,} | dataset samples: {len(dataset)} | sequence length: {seq_len}")
    print(
        f"Micro-batch: {training_config.micro_batch_size} | "
        f"gradient accumulation: {training_config.gradient_accumulation_steps} | "
        f"effective batch: {training_config.get_effective_batch_size()}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = build_scheduler(optimizer, training_config)

    optimizer_step = 0
    epoch = 0
    accumulation_count = 0
    running_loss = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_vocab = checkpoint_config.get("model", {}).get("vocab_size")
        if checkpoint_vocab is not None and int(checkpoint_vocab) != model_config.vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size={checkpoint_vocab} does not match current model "
                f"vocab_size={model_config.vocab_size}. Start a fresh run after changing tokenizer/model size."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", optimizer.state_dict()))
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer_step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("epoch", 0))
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        restore_torch_rng_state(checkpoint.get("torch_rng_state"))
        restore_cuda_rng_state(checkpoint.get("cuda_rng_state"))
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
