from lapis.config import DataConfig, ModelConfig, TrainingConfig, get_default_config
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


def test_data_config_matches_current_smoke_dataset():
    dc = DataConfig()
    assert dc.dataset_name == "lapis-smoke"


def test_bpe_tokenizer_round_trip(tmp_path):
    text = "Lapis local test: 123!\nUnicode café — 世界"
    tokenizer = Tokenizer.train_from_iterator([text, "Lapis learns next-token prediction."], vocab_size=128, min_frequency=1)
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
