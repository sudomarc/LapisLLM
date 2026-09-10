#!/usr/bin/env python3
"""Memory-bounded training path for file-backed corpora."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from lapis.config.base import resolve_device
from lapis.config.model_config import ModelConfig
from lapis.config.runtime_config import configure_cpu_runtime
from lapis.config.training_config import TrainingConfig
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from lapis.training.learning_monitor import LearningMonitor
from scripts._train_impl import (
    build_scheduler,
    load_yaml,
    parse_monitor_prompts,
    resolve_dtype,
    resolve_training_seq_len,
    restore_cuda_rng_state,
    restore_torch_rng_state,
    save_checkpoint,
)


class StreamingTextDataset(IterableDataset):
    """Read a UTF-8 corpus incrementally and yield fixed-length causal samples."""

    def __init__(self, path: str | Path, tokenizer: Tokenizer, seq_len: int) -> None:
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Training data not found: {self.path}")
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.needed = seq_len + 1

    def __iter__(self):
        buffer: list[int] = [self.tokenizer.bos_id]
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                ids = self.tokenizer.encode(line, add_special_tokens=False)
                if not ids:
                    continue
                buffer.extend(ids)
                while len(buffer) >= self.needed:
                    sample_ids = buffer[: self.needed]
                    del buffer[: self.seq_len]
                    sample = torch.tensor(sample_ids, dtype=torch.long)
                    yield sample, sample.clone()

        buffer.append(self.tokenizer.eos_id)
        if len(buffer) > 1:
            real_length = len(buffer)
            sample_ids = buffer[: self.needed]
            if len(sample_ids) < self.needed:
                sample_ids.extend(
                    [self.tokenizer.pad_id] * (self.needed - len(sample_ids))
                )
            sample = torch.tensor(sample_ids, dtype=torch.long)
            labels = sample.clone()
            labels[min(real_length, self.needed) :] = -100
            yield sample, labels


def _load_stream_tokenizer(
    config: dict,
    data_path: Path,
    resume_path: str | None,
    explicit_path: str | None,
) -> Tokenizer:
    tokenizer_cfg = config.get("tokenizer", {})
    target_vocab = int(
        tokenizer_cfg.get("vocab_size", config["model"]["vocab_size"])
    )
    min_frequency = int(tokenizer_cfg.get("min_frequency", 1))

    if explicit_path:
        return Tokenizer.load(explicit_path)

    if resume_path:
        checkpoint_tokenizer = Path(resume_path).parent / "tokenizer"
        if (checkpoint_tokenizer / "tokenizer.json").is_file():
            print(f"Loading tokenizer from checkpoint: {checkpoint_tokenizer}")
            return Tokenizer.load(str(checkpoint_tokenizer))

    path = Path(tokenizer_cfg.get("path", "artifacts/tokenizer"))
    tokenizer_file = path / "tokenizer.json"
    if tokenizer_file.is_file():
        tokenizer = Tokenizer.load(str(tokenizer_file))
        if tokenizer.vocab_size == target_vocab:
            return tokenizer
        print(
            f"Tokenizer vocab mismatch ({tokenizer.vocab_size} != {target_vocab}); "
            "retraining tokenizer from the training file."
        )

    print(f"Training tokenizer from file: {data_path}")
    tokenizer = Tokenizer.train_from_files(
        [str(data_path)],
        vocab_size=target_vocab,
        min_frequency=min_frequency,
    )
    tokenizer.save(str(path))
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LAPIS from a streamed text corpus"
    )
    parser.add_argument("--config", default="configs/local-dev.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--cpu-fast", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--monitor-interval", type=int, default=500)
    parser.add_argument("--monitor-sample-tokens", type=int, default=48)
    parser.add_argument("--monitor-prompts", default=None)
    parser.add_argument("--monitor-log", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.monitor_interval < 0:
        raise ValueError("monitor-interval must be >= 0")
    if args.monitor_sample_tokens < 1:
        raise ValueError("monitor-sample-tokens must be at least 1")
    if args.cpu_fast and args.resume:
        raise ValueError(
            "--cpu-fast cannot resume a checkpoint with a different model architecture"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = Path(args.data).resolve()
    config = load_yaml(args.config)
    if args.cpu_fast:
        from lapis.config.runtime_config import apply_cpu_fast_profile, validate_fast_profile

        config = apply_cpu_fast_profile(config)
        validate_fast_profile(config)
        print("CPU-fast profile enabled")

    tokenizer = _load_stream_tokenizer(
        config, data_path, args.resume, args.tokenizer
    )
    config.setdefault("model", {})["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(config)
    training_config = TrainingConfig(config)
    device = resolve_device(
        args.device or config.get("runtime", {}).get("device", "auto")
    )
    if args.cpu_fast and device.type != "cpu":
        raise ValueError("--cpu-fast requires CPU")
    if device.type == "cpu":
        configure_cpu_runtime(config, device)
    dtype = resolve_dtype(config, device)
    seq_len = resolve_training_seq_len(model_config, config)

    dataset = StreamingTextDataset(data_path, tokenizer, seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.micro_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
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
    print(f"Parameters: {breakdown['total']:,}")
    print(
        f"Corpus: {data_path} | size={data_path.stat().st_size / 1024 / 1024:.1f} MiB | "
        "dataset=streaming"
    )
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
    tokens_seen = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_config = checkpoint.get("config", {})
        checkpoint_vocab = checkpoint_config.get("model", {}).get("vocab_size")
        if checkpoint_vocab is not None and int(checkpoint_vocab) != model_config.vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size={checkpoint_vocab} does not match current model "
                f"vocab_size={model_config.vocab_size}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(
            checkpoint.get("optimizer_state_dict", optimizer.state_dict())
        )
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer_step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("epoch", 0))
        if checkpoint.get("python_rng_state") is not None:
            random.setstate(checkpoint["python_rng_state"])
        restore_torch_rng_state(checkpoint.get("torch_rng_state"))
        restore_cuda_rng_state(checkpoint.get("cuda_rng_state"))
        print(f"Resumed from optimizer step {optimizer_step}")

    monitor = None
    if args.monitor_interval > 0:
        checkpoint_path = Path(args.checkpoint)
        monitor_path = (
            Path(args.monitor_log)
            if args.monitor_log
            else checkpoint_path.parent / "learning_monitor.jsonl"
        )
        monitor = LearningMonitor(
            model=model,
            tokenizer=tokenizer,
            device=device,
            output_path=monitor_path,
            interval=args.monitor_interval,
            sample_tokens=args.monitor_sample_tokens,
            prompts=parse_monitor_prompts(args.monitor_prompts),
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)

    while optimizer_step < training_config.max_steps and epoch < args.epochs:
        epoch += 1
        for input_ids, labels in dataloader:
            if optimizer_step >= training_config.max_steps:
                break
            input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            valid_tokens = int(labels.ne(-100).sum().item())
            tokens_seen += valid_tokens
            _, loss = model(input_ids, labels=labels)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss")
            (loss / training_config.gradient_accumulation_steps).backward()
            running_loss += loss.item()
            accumulation_count += 1

            if accumulation_count == training_config.gradient_accumulation_steps:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.gradient_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                avg_loss = running_loss / accumulation_count
                if optimizer_step % 10 == 0 or optimizer_step == 1:
                    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                    print(
                        f"step={optimizer_step:04d} loss={avg_loss:.4f} "
                        f"ppl={ppl:.2f} lr={optimizer.param_groups[0]['lr']:.6g} "
                        f"tokens={tokens_seen:,}",
                        flush=True,
                    )
                if monitor is not None:
                    monitor.observe(
                        step=optimizer_step,
                        epoch=epoch,
                        loss=avg_loss,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        tokens_seen=tokens_seen,
                    )
                running_loss = 0.0
                accumulation_count = 0

        if accumulation_count:
            correction = (
                training_config.gradient_accumulation_steps / accumulation_count
            )
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training_config.gradient_clip
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            avg_loss = running_loss / accumulation_count
            if optimizer_step % 10 == 0 or optimizer_step == 1:
                ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
                print(
                    f"step={optimizer_step:04d} loss={avg_loss:.4f} "
                    f"ppl={ppl:.2f} lr={optimizer.param_groups[0]['lr']:.6g} "
                    f"tokens={tokens_seen:,}",
                    flush=True,
                )
            if monitor is not None:
                monitor.observe(
                    step=optimizer_step,
                    epoch=epoch,
                    loss=avg_loss,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    tokens_seen=tokens_seen,
                )
            running_loss = 0.0
            accumulation_count = 0

    if optimizer_step < training_config.max_steps:
        raise RuntimeError(
            f"Training exhausted epochs before reaching max_steps: "
            f"{optimizer_step} < {training_config.max_steps}"
        )

    checkpoint_path = Path(args.checkpoint)
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        optimizer_step,
        epoch,
        config,
        tokenizer,
    )
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Tokenizer saved: {checkpoint_path.parent / 'tokenizer'}")
    if monitor is not None:
        print(f"Learning monitor log: {monitor.output_path}")


if __name__ == "__main__":
    main()
