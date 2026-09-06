#!/usr/bin/env python3
"""Interactive chat with a trained Lapis checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.generate import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with LAPIS")
    parser.add_argument("--checkpoint", default="checkpoints/latest.pt")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LapisModel(**checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = Tokenizer.load(str(Path(args.checkpoint).parent / "tokenizer"))

    print("LAPIS Chat — type 'quit' to exit")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"quit", "exit"}:
            break
        response = generate(model, tokenizer, prompt, 64, 0.8, 40, 0.95)
        print(f"Lapis: {response}")


if __name__ == "__main__":
    main()
