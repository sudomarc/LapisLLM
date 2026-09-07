#!/usr/bin/env python3
"""Evaluate validation loss and perplexity for a Lapis checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lapis.config.base import resolve_device
from lapis.config.model_config import ModelConfig, model_config_kwargs
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.generate import validate_checkpoint_tokenizer
from scripts.train import DEFAULT_CORPUS, TextDataset, resolve_training_seq_len


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LAPIS")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer_path = checkpoint_path.parent / "tokenizer"
    if not tokenizer_path.exists():
        parser.error(f"Tokenizer directory not found: {tokenizer_path}")
    tokenizer = Tokenizer.load(str(tokenizer_path))
    validate_checkpoint_tokenizer(checkpoint, tokenizer)

    config = checkpoint["config"]
    model = LapisModel(**model_config_kwargs(config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    corpus = (
        Path(args.data).read_text(encoding="utf-8")
        if args.data
        else DEFAULT_CORPUS
    )
    model_config = ModelConfig(config)
    seq_len = resolve_training_seq_len(model_config, config)
    dataset = TextDataset(tokenizer.encode(corpus), seq_len, tokenizer.pad_id)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for input_ids, labels in loader:
            valid_tokens = int(labels.ne(-100).sum().item())
            if valid_tokens == 0:
                continue
            _, batch_loss = model(input_ids.to(device), labels=labels.to(device))
            if batch_loss is None or not torch.isfinite(batch_loss):
                raise RuntimeError("Evaluation produced a non-finite loss")
            total_loss += float(batch_loss.item()) * valid_tokens
            total_tokens += valid_tokens

    if total_tokens == 0:
        raise RuntimeError("Evaluation dataset contains no valid target tokens")

    loss = total_loss / total_tokens
    ppl = math.exp(loss) if loss < 20 else float("inf")
    print(f"Validation loss: {loss:.6f}")
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
