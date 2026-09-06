import random
from pathlib import Path

import pytest
import torch

from lapis.checkpoint import (
    CheckpointError,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    validate_resume_compatibility,
)
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.train import build_scheduler, make_epoch_loader, resolve_tokenizer


def make_components():
    tokenizer = Tokenizer.train_from_iterator(
        ["hello world hello world"], vocab_size=64, min_frequency=1
    )
    model = LapisModel(
        vocab_size=tokenizer.vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        rope_theta=10000.0,
        dropout=0.0,
        bias=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(
        optimizer,
        type("TrainingConfigProxy", (), {"warmup_steps": 0, "max_steps": 4})(),
    )
    return tokenizer, model, optimizer, scheduler


def checkpoint_config(model):
    return {
        "model": {
            "vocab_size": model.vocab_size,
            "hidden_size": model.hidden_size,
            "intermediate_size": model.intermediate_size,
            "num_layers": model.num_layers,
            "num_attention_heads": model.num_attention_heads,
            "num_key_value_heads": model.num_key_value_heads,
            "max_position_embeddings": model.max_position_embeddings,
            "rope_theta": model.rope_theta,
            "dropout": model.dropout,
            "bias": model.bias,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "warmup_steps": 0,
            "max_steps": 4,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "gradient_clip": 1.0,
        },
    }


def test_checkpoint_is_self_contained_and_round_trips(tmp_path: Path):
    tokenizer, model, optimizer, scheduler = make_components()
    input_ids = torch.tensor([tokenizer.encode("hello world")])
    _, loss = model(input_ids, labels=input_ids)
    loss.backward()
    optimizer.step()
    scheduler.step()

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        epoch=2,
        batch_in_epoch=3,
        seed=123,
        config=checkpoint_config(model),
        tokenizer=tokenizer,
        data_fingerprint="abc",
    )

    checkpoint = load_checkpoint(path)
    assert checkpoint["checkpoint_version"] == 3
    assert checkpoint["step"] == 1
    assert checkpoint["epoch"] == 2
    assert checkpoint["batch_in_epoch"] == 3
    assert checkpoint["seed"] == 123
    assert checkpoint["tokenizer_json"]
    restored = Tokenizer.from_json(checkpoint["tokenizer_json"])
    assert restored.encode("hello world") == tokenizer.encode("hello world")


def test_checkpoint_corruption_fails_clearly(tmp_path: Path):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a pytorch checkpoint")
    with pytest.raises(CheckpointError):
        load_checkpoint(path)


def test_resume_rejects_model_training_and_seed_mismatch():
    config = checkpoint_config(make_components()[1])
    changed = {"model": dict(config["model"]), "training": dict(config["training"])}
    changed["model"]["hidden_size"] = 32
    with pytest.raises(CheckpointError, match="hidden_size"):
        validate_resume_compatibility(config, changed, "v", "v", 64, 64, 42, 42)

    changed = {"model": dict(config["model"]), "training": dict(config["training"])}
    changed["training"]["learning_rate"] = 2e-3
    with pytest.raises(CheckpointError, match="learning_rate"):
        validate_resume_compatibility(config, changed, "v", "v", 64, 64, 42, 42)

    with pytest.raises(CheckpointError, match="seed"):
        validate_resume_compatibility(config, config, "v", "v", 64, 64, 42, 7)


def test_rng_restore_reproduces_python_and_torch_sequences(tmp_path: Path):
    tokenizer, model, optimizer, scheduler = make_components()
    path = tmp_path / "checkpoint.pt"
    random.seed(123)
    torch.manual_seed(456)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=0,
        epoch=0,
        seed=123,
        config=checkpoint_config(model),
        tokenizer=tokenizer,
    )
    checkpoint = load_checkpoint(path)
    expected_py = random.random()
    expected_torch = torch.rand(3)

    random.seed(999)
    torch.manual_seed(999)
    restore_rng_state(checkpoint)
    assert random.random() == expected_py
    assert torch.equal(torch.rand(3), expected_torch)


def test_epoch_loader_order_is_stable_for_resume_cursor():
    dataset = list(range(20))
    first = [index for batch in make_epoch_loader(dataset, 2, seed=42, epoch=3) for index in batch]
    second = [index for batch in make_epoch_loader(dataset, 2, seed=42, epoch=3) for index in batch]
    different_epoch = [index for batch in make_epoch_loader(dataset, 2, seed=42, epoch=4) for index in batch]
    assert first == second
    assert first != different_epoch


def test_resume_rejects_explicit_tokenizer_override():
    with pytest.raises(CheckpointError, match="--tokenizer"):
        resolve_tokenizer({}, "text", "checkpoint.pt", "tokenizer")
