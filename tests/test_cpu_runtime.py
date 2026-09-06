import pytest
import torch

from lapis.config.runtime_config import (
    apply_cpu_fast_profile,
    get_dataloader_options,
    validate_fast_profile,
)


def test_cpu_fast_profile_reduces_training_cost_without_mutating_input():
    config = {
        "model": {
            "vocab_size": 4096,
            "hidden_size": 256,
            "intermediate_size": 1024,
            "num_layers": 6,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "max_position_embeddings": 512,
        },
        "training": {
            "micro_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_steps": 10000,
        },
        "runtime": {},
        "data": {"max_seq_length": 511},
    }

    tuned = apply_cpu_fast_profile(config)

    assert config["model"]["num_layers"] == 6
    assert tuned["model"]["num_layers"] == 2
    assert tuned["model"]["hidden_size"] == 128
    assert tuned["model"]["max_position_embeddings"] == 128
    assert tuned["data"]["max_seq_length"] == 127
    assert tuned["training"]["micro_batch_size"] == 1
    assert tuned["runtime"]["device"] == "cpu"
    validate_fast_profile(tuned)


def test_fast_profile_shape_validation_rejects_invalid_heads():
    config = apply_cpu_fast_profile({"model": {}, "training": {}, "runtime": {}, "data": {}})
    config["model"]["hidden_size"] = 127
    with pytest.raises(ValueError, match="hidden_size"):
        validate_fast_profile(config)


def test_dataloader_options_are_cpu_safe():
    cpu = torch.device("cpu")
    config = {"runtime": {"cpu": {"dataloader_workers": 0, "pin_memory": False}}}
    options = get_dataloader_options(config, cpu)
    assert options["num_workers"] == 0
    assert options["pin_memory"] is False
    assert "persistent_workers" not in options
