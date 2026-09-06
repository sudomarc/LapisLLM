#!/usr/bin/env python3
"""Evaluate validation loss and perplexity for a Lapis checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lapis.checkpoint import CheckpointError, load_checkpoint
from lapis.config.model_config import ModelConfig
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.train import (
    DEFAULT_CORPUS,
    TextDataset,
    load_yaml,
    resolve_device,
    resolve_training_seq_len,
    split_corpus,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LAPIS")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    if not isinstance(config, dict):
        raise CheckpointError("Checkpoint does not contain a valid configuration")
    device = resolve_device(config, args.device)
    if device.type != "cpu":
        checkpoint = load_checkpoint(args.checkpoint, map_location=device)
        config = checkpoint["config"]

    try:
        model = LapisModel(**config["model"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    except (RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointError("Checkpoint model state does not match checkpoint config") from exc
    model.eval()

    embedded = checkpoint.get("tokenizer_json")
    tokenizer = (
        Tokenizer.from_json(embedded)
        if embedded
        else Tokenizer.load(str(Path(args.checkpoint).parent / "tokenizer"))
    )
    expected_vocab = checkpoint.get("tokenizer_vocab_size")
    if expected_vocab is not None and int(expected_vocab) != tokenizer.vocab_size:
        raise CheckpointError("Checkpoint tokenizer metadata does not match tokenizer")

    corpus = Path(args.data).read_text(encoding="utf-8") if args.data else DEFAULT_CORPUS
    validation_fraction = float(config.get("data", {}).get("validation_split", 0.1))
    _, validation_text = split_corpus(corpus, validation_fraction)
    model_config = ModelConfig(config)
    seq_len = resolve_training_seq_len(model_config, config)
    dataset = TextDataset(tokenizer.encode(validation_text), seq_len, tokenizer.pad_id)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_nll = 0.0
    token_count = 0
    with torch.inference_mode():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device=device)
            labels = labels.to(device=device)
            logits, _ = model(input_ids, labels=None)
            target = labels[:, 1:].contiguous().view(-1)
            prediction = logits[:, :-1, :].contiguous().view(-1, model.vocab_size)
            valid = target != -100
            if not torch.any(valid):
                continue
            nll = F.cross_entropy(prediction[valid], target[valid], reduction="sum")
            if not torch.isfinite(nll):
                raise RuntimeError("Evaluation produced a non-finite loss")
            total_nll += float(nll.item())
            token_count += int(valid.sum().item())

    if token_count == 0:
        raise RuntimeError("Validation set contains no valid target tokens")
    loss = total_nll / token_count
    ppl = math.exp(loss) if loss < 20 else float("inf")
    print(f"Validation loss: {loss:.6f}")
    print(f"Perplexity: {ppl:.4f}")


if __name__ == "__main__":
    main()
