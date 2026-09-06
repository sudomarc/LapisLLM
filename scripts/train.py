#!/usr/bin/env python3
"""Train a Lapis model from a YAML configuration."""

from __future__ import annotations

import argparse
import math
import pickle
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from lapis.config.base import load_config, resolve_path
from lapis.config.model_config import ModelConfig
from lapis.config.training_config import TrainingConfig
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer

CHECKPOINT_FORMAT_VERSION = 2
MODEL_CONFIG_KEYS = (
    "vocab_size", "hidden_size", "intermediate_size", "num_layers",
    "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
    "rope_theta", "dropout", "bias",
)
TRAINING_CONFIG_KEYS = (
    "learning_rate", "weight_decay", "warmup_steps", "max_steps", "batch_size",
    "micro_batch_size", "gradient_accumulation_steps", "gradient_clip",
)

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
        if pad_id < 0:
            raise ValueError("pad_id must be non-negative")
        if not tokens:
            raise ValueError("training data must contain at least one token")
        needed = seq_len + 1
        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for start in range(0, len(tokens), seq_len):
            chunk = list(tokens[start : start + needed])
            real_length = len(chunk)
            if real_length < 2:
                break
            if real_length < needed:
                chunk.extend([pad_id] * (needed - real_length))
            sample = torch.tensor(chunk, dtype=torch.long)
            labels = sample.clone()
            if real_length < needed:
                labels[real_length:] = -100
            self.samples.append((sample, labels))
        if not self.samples:
            raise ValueError("training data must contain at least two tokens")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def load_yaml(path: str) -> dict:
    """Compatibility wrapper around the canonical configuration loader."""
    return load_config(path)


def resolve_device(config: dict, requested: str | None) -> torch.device:
    value = str(requested or config.get("runtime", {}).get("device", "auto")).lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(value)
    except RuntimeError as exc:
        raise ValueError(f"Invalid device: {value}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)"
            )
    return device


def resolve_dtype(config: dict, device: torch.device) -> torch.dtype:
    requested = str(config.get("runtime", {}).get("dtype", "float32")).lower()
    mapping = {
        "float32": torch.float32, "fp32": torch.float32,
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    }
    if requested not in mapping:
        raise ValueError(f"Unsupported runtime.dtype: {requested}")
    if device.type == "cpu" and mapping[requested] != torch.float32:
        raise ValueError("CPU training currently supports runtime.dtype=float32 only")
    return mapping[requested]


def validate_runtime_config(config: dict) -> None:
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("configuration.runtime must be a mapping")
    if runtime.get("compile", False):
        raise ValueError("runtime.compile is not supported by this release")


def load_or_train_tokenizer(config: dict, corpus: str) -> Tokenizer:
    tokenizer_cfg = config.get("tokenizer", {})
    path = resolve_path(tokenizer_cfg.get("path", "artifacts/tokenizer"))
    tokenizer_file = path / "tokenizer.json"
    target_vocab = int(tokenizer_cfg.get("vocab_size", config["model"]["vocab_size"]))
    min_frequency = int(tokenizer_cfg.get("min_frequency", 1))
    if target_vocab < len(Tokenizer.SPECIAL_TOKENS):
        raise ValueError("tokenizer.vocab_size is smaller than the required special-token set")
    if min_frequency <= 0:
        raise ValueError("tokenizer.min_frequency must be positive")
    if tokenizer_file.exists():
        tokenizer = Tokenizer.load(str(tokenizer_file))
        if tokenizer.vocab_size == target_vocab:
            return tokenizer
        print(
            f"Tokenizer vocab mismatch ({tokenizer.vocab_size} != {target_vocab}); "
            "retraining tokenizer from the current corpus."
        )
    tokenizer = Tokenizer.train_from_iterator(
        [corpus], vocab_size=target_vocab, min_frequency=min_frequency
    )
    tokenizer.save(str(path))
    return tokenizer


def _checkpoint_payload(model, optimizer, scheduler, step, epoch, batch_index, config, tokenizer):
    return {
        "checkpoint_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": int(step),
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "config": config,
        "tokenizer_version": tokenizer.VERSION,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state().clone().cpu(),
        "cuda_rng_state": (
            [state.clone().cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else None
        ),
    }


def save_checkpoint(path, model, optimizer, scheduler, step, epoch, batch_index, config, tokenizer):
    """Write a complete checkpoint atomically after persisting its tokenizer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path.parent / "tokenizer"))
    payload = _checkpoint_payload(model, optimizer, scheduler, step, epoch, batch_index, config, tokenizer)
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def restore_torch_rng_state(state) -> None:
    if state is None:
        return
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state, dtype=torch.uint8)
    else:
        state = state.detach().to(device="cpu", dtype=torch.uint8)
    if state.numel() == 0:
        raise ValueError("Checkpoint contains an empty torch RNG state")
    torch.set_rng_state(state.contiguous())


def restore_cuda_rng_state(state) -> None:
    if state is None or not torch.cuda.is_available():
        return
    if isinstance(state, torch.Tensor):
        states = [state]
    elif isinstance(state, (list, tuple)):
        states = list(state)
    else:
        raise TypeError("Checkpoint contains an invalid CUDA RNG state")
    normalized = []
    for item in states:
        if not isinstance(item, torch.Tensor):
            item = torch.as_tensor(item, dtype=torch.uint8)
        else:
            item = item.detach().to(device="cpu", dtype=torch.uint8)
        if item.numel() == 0:
            raise ValueError("Checkpoint contains an empty CUDA RNG state")
        normalized.append(item.contiguous())
    if len(normalized) != torch.cuda.device_count():
        raise ValueError("Checkpoint CUDA RNG state does not match the current CUDA device count")
    torch.cuda.set_rng_state_all(normalized)


def build_scheduler(optimizer, training_config):
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
    configured = int(config.get("data", {}).get("max_seq_length", model_config.max_position_embeddings - 1))
    if configured < 2:
        raise ValueError("data.max_seq_length must be at least 2")
    max_input_length = model_config.max_position_embeddings - 1
    if max_input_length < 2:
        raise ValueError("max_position_embeddings must be at least 3")
    return min(configured, max_input_length)


def resolve_tokenizer(config, corpus, resume_path, explicit_path):
    if resume_path and explicit_path:
        raise ValueError("--resume and --tokenizer cannot be used together")
    if resume_path:
        checkpoint_tokenizer = resolve_path(resume_path).parent / "tokenizer"
        if not (checkpoint_tokenizer / "tokenizer.json").exists():
            raise FileNotFoundError(f"Resume checkpoint tokenizer not found: {checkpoint_tokenizer}")
        print(f"Loading tokenizer from checkpoint: {checkpoint_tokenizer}")
        return Tokenizer.load(str(checkpoint_tokenizer))
    if explicit_path:
        return Tokenizer.load(str(resolve_path(explicit_path)))
    return load_or_train_tokenizer(config, corpus)


def _assert_resume_compatibility(checkpoint, config, tokenizer):
    version = int(checkpoint.get("checkpoint_version", 1))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format version {version}; expected {CHECKPOINT_FORMAT_VERSION}")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint is missing its configuration metadata")
    checkpoint_model = checkpoint_config.get("model", {})
    current_model = config.get("model", {})
    for key in MODEL_CONFIG_KEYS:
        if key in checkpoint_model and key in current_model and checkpoint_model[key] != current_model[key]:
            raise ValueError(
                f"Checkpoint model.{key}={checkpoint_model[key]!r} does not match current value {current_model[key]!r}"
            )
    checkpoint_training = checkpoint_config.get("training", {})
    current_training = config.get("training", {})
    for key in TRAINING_CONFIG_KEYS:
        if checkpoint_training.get(key) != current_training.get(key):
            raise ValueError(
                f"Checkpoint training.{key}={checkpoint_training.get(key)!r} does not match current value {current_training.get(key)!r}"
            )
    checkpoint_data = checkpoint_config.get("data", {})
    current_data = config.get("data", {})
    if checkpoint_data.get("max_seq_length") != current_data.get("max_seq_length"):
        raise ValueError("Checkpoint data.max_seq_length does not match current configuration")
    if checkpoint.get("tokenizer_vocab_size") != tokenizer.vocab_size:
        raise ValueError(
            f"Checkpoint tokenizer vocabulary size {checkpoint.get('tokenizer_vocab_size')} does not match loaded tokenizer vocabulary size {tokenizer.vocab_size}"
        )
    if checkpoint.get("tokenizer_version") != tokenizer.VERSION:
        raise ValueError("Checkpoint tokenizer version does not match the installed tokenizer implementation")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except (RuntimeError, OSError, EOFError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Unable to load checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a mapping")
    required = {"checkpoint_version", "model_state_dict", "optimizer_state_dict", "config"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {', '.join(missing)}")
    return checkpoint


def _build_dataloader(dataset, batch_size, seed, epoch):
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(epoch))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator)


def _build_model(model_config, device, dtype):
    model = LapisModel(
        vocab_size=model_config.vocab_size, hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size, num_layers=model_config.num_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        max_position_embeddings=model_config.max_position_embeddings,
        rope_theta=model_config.rope_theta, dropout=model_config.dropout, bias=model_config.bias,
    )
    return model.to(device=device, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LAPIS")
    parser.add_argument("--config", default="configs/local-dev.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default=None, help="Optional UTF-8 text file")
    parser.add_argument("--tokenizer", default=None, help="Optional trained tokenizer directory")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt", help="Output checkpoint path")
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_yaml(args.config)
    validate_runtime_config(config)
    corpus = resolve_path(args.data).read_text(encoding="utf-8") if args.data else DEFAULT_CORPUS
    resume_path = str(resolve_path(args.resume)) if args.resume else None
    checkpoint_path = resolve_path(args.checkpoint)

    tokenizer = resolve_tokenizer(config, corpus, resume_path, args.tokenizer)
    config.setdefault("model", {})["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(config)
    training_config = TrainingConfig(config)
    device = resolve_device(config, args.device)
    dtype = resolve_dtype(config, device)
    seq_len = resolve_training_seq_len(model_config, config)

    tokens = tokenizer.encode(corpus, add_special_tokens=True)
    if not tokens:
        raise ValueError("Tokenization produced no training tokens")
    if max(tokens) >= model_config.vocab_size:
        raise ValueError(f"Tokenizer produced id {max(tokens)} but model vocab_size is {model_config.vocab_size}")

    dataset = TextDataset(tokens, seq_len=seq_len, pad_id=tokenizer.pad_id)
    model = _build_model(model_config, device, dtype)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Tokenizer: {tokenizer}")
    print(f"Device: {device} | dtype: {dtype}")
    print(f"Tokens: {len(tokens):,} | dataset samples: {len(dataset)} | sequence length: {seq_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    scheduler = build_scheduler(optimizer, training_config)

    optimizer_step = 0
    epoch = 1
    start_batch_index = 0

    if resume_path:
        checkpoint = load_checkpoint(Path(resume_path), device)
        _assert_resume_compatibility(checkpoint, config, tokenizer)
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler_state = checkpoint.get("scheduler_state_dict")
            if scheduler_state is None:
                raise ValueError("Checkpoint is missing scheduler state")
            scheduler.load_state_dict(scheduler_state)
        except (RuntimeError, KeyError, ValueError) as exc:
            raise ValueError(f"Checkpoint state is incompatible with the current model: {exc}") from exc
        optimizer_step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("epoch", 1))
        start_batch_index = int(checkpoint.get("batch_index", 0))
        if optimizer_step < 0 or epoch < 1 or start_batch_index < 0:
            raise ValueError("Checkpoint contains invalid training progress")
        if optimizer_step > training_config.max_steps:
            raise ValueError(f"Checkpoint step {optimizer_step} exceeds configured max_steps {training_config.max_steps}")
        if checkpoint.get("python_rng_state") is not None:
            try:
                random.setstate(checkpoint["python_rng_state"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Checkpoint contains an invalid Python RNG state") from exc
        restore_torch_rng_state(checkpoint.get("torch_rng_state"))
        restore_cuda_rng_state(checkpoint.get("cuda_rng_state"))
        print(f"Resumed from optimizer step {optimizer_step}, epoch {epoch}, batch {start_batch_index}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    accumulation_count = 0
    epochs_completed = 0

    while optimizer_step < training_config.max_steps and epochs_completed < args.epochs:
        dataloader = _build_dataloader(dataset, training_config.micro_batch_size, args.seed, epoch)
        num_batches = len(dataloader)
        if num_batches == 0:
            raise ValueError("Training dataloader produced no batches")
        if start_batch_index >= num_batches:
            epoch += 1
            start_batch_index = 0
            dataloader = _build_dataloader(dataset, training_config.micro_batch_size, args.seed, epoch)
            num_batches = len(dataloader)

        saw_batch = False
        for batch_index, (input_ids, labels) in enumerate(dataloader):
            if batch_index < start_batch_index:
                continue
            saw_batch = True
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            _, loss = model(input_ids, labels=labels)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")
            (loss / training_config.gradient_accumulation_steps).backward()
            running_loss += loss.item()
            accumulation_count += 1

            end_of_epoch = batch_index == num_batches - 1
            should_step = accumulation_count == training_config.gradient_accumulation_steps
            if should_step or end_of_epoch:
                if accumulation_count < training_config.gradient_accumulation_steps:
                    correction = training_config.gradient_accumulation_steps / accumulation_count
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("Training produced a non-finite gradient norm")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                start_batch_index = batch_index + 1

                if optimizer_step % 10 == 0 or optimizer_step == 1:
                    avg_loss = running_loss / max(1, accumulation_count)
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                    print(f"step={optimizer_step:04d} loss={avg_loss:.4f} ppl={ppl:.2f} lr={optimizer.param_groups[0]['lr']:.6g}")
                running_loss = 0.0
                accumulation_count = 0
                if optimizer_step >= training_config.max_steps:
                    break

        if not saw_batch:
            raise RuntimeError("Resume batch position produced no training batches")
        if start_batch_index >= num_batches:
            epoch += 1
            start_batch_index = 0
        epochs_completed += 1

    save_checkpoint(checkpoint_path, model, optimizer, scheduler, optimizer_step, epoch, start_batch_index, config, tokenizer)
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Tokenizer saved: {checkpoint_path.parent / 'tokenizer'}")


if __name__ == "__main__":
    main()
