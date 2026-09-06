"""Chat script for LAPIS model.

Usage:
    python scripts/chat.py --checkpoint checkpoints/latest
"""

import argparse
import torch

from lapis.model.lapis_model import LapisModel


@torch.no_grad()
def chat_loop(model, checkpoint_path, device="cpu"):
    """Run an interactive chat loop."""
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    model.eval()
    model.to(device)
    
    print("LAPIS Chat (type 'quit' to exit)")
    print("=" * 40)
    
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() in ["quit", "exit", ""]:
                break
            
            # Generate response
            input_ids = torch.tensor([prompt.encode() if hasattr(prompt, 'encode') else prompt], 
                                      dtype=torch.long).to(device)
            logits, _ = model(input_ids)
            logits = logits[:, -1, :] / 0.7  # temperature
            
            # Greedy sampling
            next_token = torch.argmax(logits, dim=-1)
            
            # Simple response
            response_token = next_token.item()
            print(f"Lapis: token {response_token}")
            
        except KeyboardInterrupt:
            break
    
    print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="LAPIS Chat")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest", help="Path to checkpoint")
    args = parser.parse_args()
    
    model = LapisModel(vocab_size=50304, hidden_size=384, num_layers=2)
    chat_loop(model, args.checkpoint)


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__) + "/../")
    main()