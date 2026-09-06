from lapis.config import get_default_config, ModelConfig, TrainingConfig, DataConfig
from lapis.tokenizer.tokenizer import Tokenizer


def test_load_default_config():
    config = get_default_config()
    assert config is not None
    assert "model" in config
    assert "training" in config
    assert "runtime" in config


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


def test_data_config():
    dc = DataConfig()
    assert dc.dataset_name == "lapis"


def test_local_tokenizer_round_trip():
    tokenizer = Tokenizer()
    text = "Lapis local test: 123!\n"
    ids = tokenizer.encode(text)
    assert tokenizer.vocab_size <= 128
    assert max(ids) < 128
    assert tokenizer.decode(ids) == text
