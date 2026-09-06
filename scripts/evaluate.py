#!/usr/bin/env python3
"""Evaluate validation loss and perplexity for a Lapis checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from scripts.train import DEFAULT_CORPUS, TextDataset
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LAPIS")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--data", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = LapisModel(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = Tokenizer.load(str(Path(args.checkpoint).parent / "tokenizer"))
    corpus = Path(args.data).read_text(encoding="utf-8") if args.data else DEFAULT_CORPUS
    seq_len = min(model.max_position_embeddings, int(config.get("data", {}).get("max_seq_length", model.max_position_embeddings)))
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
