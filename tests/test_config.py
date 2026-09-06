import pytest

from lapis.config import DataConfig, ModelConfig, TrainingConfig, get_default_config
from lapis.config.base import load_config
from lapis.tokenizer.tokenizer import Tokenizer


def test_load_default_config():
    config = get_default_config()
    assert config is not None
    assert "model" in config
    assert "training" in config
    assert "runtime" in config


def test_config_resolution_does_not_depend_on_current_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = load_config("configs/tiny.yaml")
    assert config["model"]["vocab_size"] == 512


def test_model_config():
    mc = ModelConfig()
    params = mc.count_parameters()
    assert "total" in params
    assert "trainable" in params
    assert "non_trainable" in params


def test_training_config():
    tc = TrainingConfig()
    assert tc.learning_rate > 0
    assert tc.max_steps > 0
    assert tc.batch_size == tc.get_effective_batch_size()


def test_data_config_matches_current_open_mixture_dataset():
    dc = DataConfig()
    assert dc.dataset_name == "lapis-open-mixture"


def test_invalid_model_configuration_fails_early():
    config = get_default_config()
    config["model"]["num_key_value_heads"] = 3
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(config)


def test_invalid_training_batch_configuration_fails_early():
    config = get_default_config()
    config["training"]["batch_size"] = 3
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(config)


def test_bpe_tokenizer_round_trip(tmp_path):
    text = "Lapis local test: 123!\nUnicode café — 世界"
    tokenizer = Tokenizer.train_from_iterator(
        [text, "Lapis learns next-token prediction."],
        vocab_size=128,
        min_frequency=1,
    )
    ids = tokenizer.encode(text)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert max(ids) < tokenizer.vocab_size
    assert tokenizer.decode(ids) == text

    tokenizer.save(str(tmp_path / "tokenizer"))
    loaded = Tokenizer.load(str(tmp_path / "tokenizer"))
    assert loaded.VERSION == tokenizer.VERSION
    assert loaded.encode(text) == ids
    assert loaded.decode(ids) == text
