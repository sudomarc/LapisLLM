#!/usr/bin/env python3
"""Train a Lapis model from a YAML configuration."""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from lapis.checkpoint import (
    CheckpointError,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    validate_resume_compatibility,
)
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
    """Fixed-length causal language-modeling examples."""

    def __init__(self, tokens: list[int], seq_len: int, pad_id: int):
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        if len(tokens) < 2:
            raise ValueError("At least two tokens are required to train a causal model")
        needed = seq_len + 1
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, len(tokens) - 1, seq_len):
            chunk = tokens[start : start + needed]
            real_length = len(chunk)
            if real_length < 2:
                continue
            if real_length < needed:
                chunk += [pad_id] * (needed - real_length)
            sample = torch.tensor(chunk[:needed], dtype=torch.long)
            labels = sample.clone()
            if real_length < needed:
                labels[real_length:] = -100
            self.samples.append((sample, labels))
        if not self.samples:
            raise ValueError("Training data produced no valid causal examples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def load_yaml(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return config


def resolve_device(config: dict, requested: str | None) -> torch.device:
    value = requested or config.get("runtime", {}).get("device", "auto")
    if not isinstance(value, str):
        raise ValueError("runtime.device must be a string")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested, but CUDA is unavailable")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested CUDA device {device.index}, but only "
                f"{torch.cuda.device_count()} device(s) are available"
            )
    return device


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
    if requested not in mapping:
        raise ValueError(f"Unsupported runtime.dtype: {requested}")
    dtype = mapping[requested]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError(
            f"runtime.dtype={requested} is not supported by the release CPU path; use float32"
        )
    return dtype


def split_corpus(corpus: str, validation_fraction: float) -> tuple[str, str]:
    """Deterministically split raw text before tokenizer training to avoid validation leakage."""
    if not corpus.strip():
        raise ValueError("Training corpus is empty")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("data.validation_split must be greater than 0 and less than 0.5")
    split = int(len(corpus) * (1.0 - validation_fraction))
    split = max(1, min(len(corpus) - 1, split))
    train_text = corpus[:split]
    validation_text = corpus[split:]
    if not train_text.strip() or not validation_text.strip():
        raise ValueError("Training/validation split produced an empty partition")
    return train_text, validation_text


def corpus_fingerprint(corpus: str) -> str:
    return hashlib.sha256(corpus.encode("utf-8")).hexdigest()


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
            "retraining tokenizer from the current training partition."
        )

    tokenizer = Tokenizer.train_from_iterator(
        [corpus],
        vocab_size=target_vocab,
        min_frequency=min_frequency,
    )
    tokenizer.save(str(path))
    return tokenizer


def build_scheduler(optimizer, training_config):
    """Build linear warmup followed by cosine decay."""
    warmup_steps = training_config.warmup_steps
    total_steps = training_config.max_steps

    if warmup_steps == 0:
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
    if configured < 2:
        raise ValueError("data.max_seq_length must be at least 2")
    return min(configured, max_input_length)


def resolve_tokenizer(
    config: dict,
    corpus: str,
    resume_path: str | None,
    explicit_path: str | None,
) -> Tokenizer:
    """Load the exact checkpoint tokenizer on resume; only train one for fresh runs."""
    if explicit_path:
        return Tokenizer.load(explicit_path)

    if resume_path:
        checkpoint = load_checkpoint(resume_path, map_location="cpu")
        embedded = checkpoint.get("tokenizer_json")
        if embedded:
            return Tokenizer.from_json(embedded)
        checkpoint_tokenizer = Path(resume_path).parent / "tokenizer"
        if (checkpoint_tokenizer / "tokenizer.json").exists():
            print(f"Loading legacy checkpoint tokenizer: {checkpoint_tokenizer}")
            return Tokenizer.load(str(checkpoint_tokenizer))
        raise CheckpointError(
            "Resume checkpoint does not contain an embedded tokenizer and no legacy "
            "checkpoint tokenizer exists."
        )

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
        "--checkpoint",
        default="checkpoints/latest.pt",
        help="Output checkpoint path (defaults to checkpoints/latest.pt)",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_yaml(args.config)
    validation_fraction = float(config.get("data", {}).get("validation_split", 0.1))
    corpus = (
        Path(args.data).read_text(encoding="utf-8") if args.data else DEFAULT_CORPUS
    )
    train_text, _ = split_corpus(corpus, validation_fraction)
    tokenizer = resolve_tokenizer(config, train_text, args.resume, args.tokenizer)

    if args.resume:
        configured_vocab = int(config["model"]["vocab_size"])
        if tokenizer.vocab_size != configured_vocab:
            raise CheckpointError(
                "Resume tokenizer vocabulary size "
                f"({tokenizer.vocab_size}) does not match configured model vocabulary "
                f"({configured_vocab})."
            )
    else:
        config.setdefault("model", {})["vocab_size"] = tokenizer.vocab_size

    model_config = ModelConfig(config)
    training_config = TrainingConfig(config)
    device = resolve_device(config, args.device)
    dtype = resolve_dtype(config, device)
    seq_len = resolve_training_seq_len(model_config, config)

    tokens = tokenizer.encode(train_text, add_special_tokens=True)
    if max(tokens, default=-1) >= model_config.vocab_size:
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
    print(f"Model: {Path(args.config).stem.title()}")
    print(f"Tokenizer: {tokenizer}")
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Parameters: {breakdown['total']:,}")
    print(f"Vocabulary: {tokenizer.vocab_size:,}")
    print(
        f"Tokens: {len(tokens):,} | dataset samples: {len(dataset)} | "
        f"sequence length: {seq_len}"
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
        checkpoint = load_checkpoint(args.resume, map_location=device)
        checkpoint_config = checkpoint["config"]
        validate_resume_compatibility(
            checkpoint_config,
            config,
            checkpoint.get("tokenizer_version"),
            tokenizer.VERSION,
            checkpoint.get("tokenizer_vocab_size"),
            tokenizer.vocab_size,
        )
        checkpoint_fingerprint = checkpoint.get("data_fingerprint")
        if checkpoint_fingerprint and checkpoint_fingerprint != corpus_fingerprint(corpus):
            raise CheckpointError(
                "Resume data does not match the checkpoint data fingerprint. "
                "Use the original dataset or start a fresh run."
            )

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise CheckpointError(
                "Checkpoint model/optimizer state is incompatible with the current configuration."
            ) from exc

        optimizer_step = int(checkpoint["step"])
        epoch = int(checkpoint["epoch"])
        restore_rng_state(checkpoint)
        print(f"Resumed from optimizer step {optimizer_step}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    while optimizer_step < training_config.max_steps and args.epochs > 0:
        args.epochs -= 1
        epoch += 1
        for batch_index, (input_ids, labels) in enumerate(dataloader):
            if optimizer_step >= training_config.max_steps:
                break
            input_ids = input_ids.to(device=device)
            labels = labels.to(device=device)
            _, loss = model(input_ids, labels=labels)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(
                    f"Training produced a non-finite loss at optimizer step {optimizer_step}"
                )

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

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.gradient_clip
                )
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Training produced a non-finite gradient norm at step {optimizer_step}"
                    )
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
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=optimizer_step,
        epoch=epoch,
        config=config,
        tokenizer=tokenizer,
        data_fingerprint=corpus_fingerprint(corpus),
    )
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Tokenizer saved: {checkpoint_path.parent / 'tokenizer'}")


if __name__ == "__main__":
    main()
