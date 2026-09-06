from pathlib import Path

import pytest
import torch

from lapis.config.base import load_config
from lapis.config.training_config import TrainingConfig
from lapis.data.cleaner import clean_dataset
from lapis.model.lapis_model import LapisModel
from lapis.tokenizer.tokenizer import Tokenizer
from scripts.generate import sample_next_token, validate_generation_parameters
from scripts.prepare_data import prepare_data
from scripts.train import (
    TextDataset,
    _assert_resume_compatibility,
    _build_dataloader,
    _checkpoint_payload,
    build_scheduler,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
)


def make_model(**overrides):
    config = {
        "vocab_size": 32,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 16,
        "rope_theta": 10000.0,
        "dropout": 0.0,
        "bias": True,
    }
    config.update(overrides)
    return LapisModel(**config)


def test_config_loader_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config("configs/tiny.yaml")
    assert config["_config_path"].endswith("configs/tiny.yaml")
    assert config["model"]["hidden_size"] == 256


def test_model_rejects_odd_rope_head_dimension():
    with pytest.raises(ValueError, match="head dimension must be even"):
        make_model(hidden_size=24, num_attention_heads=8)


def test_model_rejects_out_of_range_tokens():
    model = make_model()
    with pytest.raises(ValueError, match="outside"):
        model(torch.tensor([[0, model.vocab_size]]))


def test_model_rejects_all_ignored_targets():
    model = make_model()
    input_ids = torch.zeros((1, 4), dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    with pytest.raises(ValueError, match="no valid next-token targets"):
        model(input_ids, labels=labels)


def test_training_config_rejects_batch_size_mismatch():
    config = load_config("configs/tiny.yaml")
    config["training"]["batch_size"] = 7
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(config)


def test_generation_parameter_validation_and_greedy_zero_temperature():
    validate_generation_parameters(0, 0.0, 0, 1.0)
    with pytest.raises(ValueError, match="top_p"):
        validate_generation_parameters(1, 0.8, 1, 0.0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        validate_generation_parameters(-1, 0.8, 1, 0.95)

    logits = torch.tensor([[0.1, 2.0, -1.0]])
    token = sample_next_token(logits, 0.0, 0, 1.0)
    assert token.item() == 1


def test_explicit_cuda_request_fails_cleanly_when_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this runner")
    with pytest.raises(RuntimeError, match="CUDA was explicitly requested"):
        resolve_device({"runtime": {"device": "cuda"}}, None)


def test_epoch_dataloader_order_is_reproducible():
    dataset = TextDataset(list(range(30)), seq_len=4, pad_id=0)
    first = list(_build_dataloader(dataset, batch_size=3, seed=123, epoch=4))
    second = list(_build_dataloader(dataset, batch_size=3, seed=123, epoch=4))
    assert torch.equal(first[0][0], second[0][0])
    assert torch.equal(first[-1][0], second[-1][0])


def test_targetless_trailing_fragment_is_not_emitted():
    dataset = TextDataset(list(range(5)), seq_len=4, pad_id=0)
    assert len(dataset) == 1
    inputs, labels = dataset[0]
    assert inputs.tolist() == [0, 1, 2, 3, 4]
    assert torch.equal(labels, inputs)


def test_checkpoint_contains_progress_and_scheduler_state(tmp_path: Path):
    tokenizer = Tokenizer.train_from_iterator(["hello world"], vocab_size=32, min_frequency=1)
    model = make_model(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training = TrainingConfig(load_config("configs/local-dev.yaml"))
    scheduler = build_scheduler(optimizer, training)
    config = load_config("configs/local-dev.yaml")
    config["model"]["vocab_size"] = tokenizer.vocab_size
    payload = _checkpoint_payload(model, optimizer, scheduler, 3, 2, 5, config, tokenizer)
    assert payload["checkpoint_version"] == 2
    assert payload["step"] == 3
    assert payload["epoch"] == 2
    assert payload["batch_index"] == 5
    assert payload["optimizer_state_dict"] is not None
    assert payload["scheduler_state_dict"] is not None

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, 3, 2, 5, config, tokenizer)
    loaded = load_checkpoint(path, torch.device("cpu"))
    assert loaded["checkpoint_version"] == 2
    assert loaded["batch_index"] == 5
    assert (tmp_path / "tokenizer" / "tokenizer.json").is_file()

    bad_config = load_config("configs/local-dev.yaml")
    bad_config["training"]["max_steps"] += 1
    with pytest.raises(ValueError, match="max_steps"):
        _assert_resume_compatibility(loaded, bad_config, tokenizer)


def test_prepare_data_is_deterministic_and_rejects_empty_input(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "b.txt").write_text("second\n", encoding="utf-8")
    (source / "a.txt").write_text("first\n", encoding="utf-8")
    output = tmp_path / "out.txt"
    assert prepare_data(source, output) == 2
    assert output.read_text(encoding="utf-8") == "first\n\nsecond"

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "blank.txt").write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        prepare_data(empty, tmp_path / "empty.txt")


def test_clean_dataset_failures_are_not_silenced(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "bad.json").write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON input"):
        clean_dataset(source, tmp_path / "out")
