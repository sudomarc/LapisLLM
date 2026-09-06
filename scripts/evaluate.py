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

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tokenizer = Tokenizer.load(str(Path(args.checkpoint).parent / "tokenizer"))
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

    total = 0.0
    count = 0
    with torch.no_grad():
        for input_ids, labels in loader:
            _, loss = model(input_ids.to(device), labels=labels.to(device))
            total += float(loss.item())
            count += 1

    loss = total / max(count, 1)
    ppl = math.exp(loss) if loss < 20 else float("inf")
    print(f"Validation loss: {loss:.6f}")
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
