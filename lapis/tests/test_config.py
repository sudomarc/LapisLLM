"""Configuration tests for LapisLLM Stage 1."""
import pytest
from lapis.config.base import load_config, LapisConfig


def test_default_config_loads():
    """Default LapisConfig should load without error."""
    config = LapisConfig()
    assert config.model_name == "lapis-micro"
    assert config.vocab_size == 16_000
    assert config.hidden_size == 256
    assert config.num_layers == 4
    assert config.max_seq_len == 512


def test_load_config_from_yaml():
    """Config should load from YAML file."""
    import tempfile
    import os
    config_data = {
        "model_name": "test-model",
        "vocab_size": 10_000,
        "hidden_size": 128,
        "num_layers": 2,
        "max_seq_len": 256,
        "seed": 123,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        f.flush()
        config = load_config(f.name)
    assert config.model_name == "test-model"
    assert config.vocab_size == 10_000
    assert config.hidden_size == 128
    assert config.num_layers == 2
    assert config.max_seq_len == 256
    os.unlink(f.name)


def test_config_has_all_required_fields():
    """Config should have all required fields after loading."""
    config = LapisConfig()
    required_fields = [
        "model_name", "vocab_size", "hidden_size", "num_layers",
        "num_heads", "head_dim", "max_seq_len", "rope_base",
        "tokenizer_name", "batch_size", "micro_batch_size",
        "gradient_accumulation_steps", "max_steps", "learning_rate",
        "weight_decay", "beta1", "beta2", "grad_clip",
        "precision", "device", "data_root", "run_root", "seed"
    ]
    for field in required_fields:
        assert hasattr(config, field), f"Missing field: {field}"