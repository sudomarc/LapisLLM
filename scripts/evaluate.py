#!/usr/bin/env python3
"""Evaluate a Lapis checkpoint on an explicitly supplied corpus."""

from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from lapis.config.base import resolve_path
from lapis.config.model_config import ModelConfig
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.train import DEFAULT_CORPUS, load_checkpoint, resolve_training_seq_len


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LAPIS")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument(
        "--data",
        default=None,
        help="UTF-8 evaluation corpus; defaults to the built-in smoke-test corpus",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(args.device)
        except RuntimeError as exc:
            raise ValueError(f"Invalid device: {args.device}") from exc
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")

    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    model = LapisModel(**config["model"]).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, KeyError, ValueError) as exc:
        raise ValueError(f"Checkpoint model weights are incompatible: {exc}") from exc
    model.eval()

    tokenizer = Tokenizer.load(str(checkpoint_path.parent / "tokenizer"))
    if tokenizer.vocab_size != model.vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary size {tokenizer.vocab_size} does not match "
            f"model vocabulary size {model.vocab_size}"
        )
    if checkpoint.get("tokenizer_vocab_size") is not None and int(
        checkpoint["tokenizer_vocab_size"]
    ) != tokenizer.vocab_size:
        raise ValueError("Checkpoint tokenizer metadata is inconsistent")

    corpus = (
        resolve_path(args.data).read_text(encoding="utf-8")
        if args.data
        else DEFAULT_CORPUS
    )
    model_config = ModelConfig(config)
    seq_len = resolve_training_seq_len(model_config, config)
    dataset = __import__("scripts.train", fromlist=["TextDataset"]).TextDataset(
        tokenizer.encode(corpus), seq_len, tokenizer.pad_id
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_loss = 0.0
    total_batches = 0
    with torch.inference_mode():
        for input_ids, labels in loader:
            _, loss = model(input_ids.to(device), labels=labels.to(device))
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Evaluation produced a non-finite loss")
            total_loss += float(loss.item())
            total_batches += 1

    if total_batches == 0:
        raise ValueError("Evaluation corpus produced no batches")
    loss = total_loss / total_batches
    ppl = math.exp(loss) if loss < 20 else float("inf")
    print(f"Evaluation loss: {loss:.6f}")
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
