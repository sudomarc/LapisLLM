"""Micro-test: random input -> model -> logits -> loss -> backpropagation."""

import torch
import torch.nn as nn
from lapis.model.lapis_model import LapisModel


def test_micro():
    model = LapisModel(vocab_size=50304, hidden_size=384, num_layers=2)
    model.train()

    # Random input tokens
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 50304, (batch_size, seq_len))

    # Forward (model returns (logits, loss))
    logits, loss = model(input_ids)
    assert logits.shape == (batch_size, seq_len, 50304)

    # Compute loss (next-token prediction) - use the model's loss if available
    if loss is None:
        target_ids = input_ids[:, 1:]
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits[:, :-1, :].reshape(-1, 50304), target_ids.reshape(-1))

    # Backward
    loss.backward()

    # Check loss is finite
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    print(f"Micro-test passed: loss = {loss.item():.4f}")