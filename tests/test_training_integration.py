import torch

from lapis.config.model_config import ModelConfig
from lapis.model.lapis_model import LapisModel
from scripts.train import DEFAULT_CORPUS, TextDataset, load_yaml, resolve_training_seq_len
from lapis.tokenizer.tokenizer import Tokenizer


def test_resolved_training_sequence_fits_model_context():
    config = load_yaml("configs/tiny.yaml")
    model_config = ModelConfig(config)
    seq_len = resolve_training_seq_len(model_config, config)
    dataset = TextDataset([1, 2, 3, 4, 5], seq_len=seq_len, pad_id=0)
    inputs, labels = dataset[0]

    assert inputs.shape[-1] == seq_len + 1
    assert inputs.shape[-1] <= model_config.max_position_embeddings
    assert labels.shape == inputs.shape


def test_tiny_model_accepts_dataset_batch_and_backpropagates():
    config = load_yaml("configs/tiny.yaml")
    tokenizer = Tokenizer.train_from_iterator(
        [DEFAULT_CORPUS],
        vocab_size=config["tokenizer"].get("vocab_size", config["model"]["vocab_size"]),
        min_frequency=1,
    )
    config["model"]["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig(config)
    seq_len = resolve_training_seq_len(model_config, config)
    tokens = tokenizer.encode(DEFAULT_CORPUS)
    dataset = TextDataset(tokens, seq_len=seq_len, pad_id=tokenizer.pad_id)
    input_ids, labels = dataset[0]
    model = LapisModel(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_layers=model_config.num_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        max_position_embeddings=model_config.max_position_embeddings,
        rope_theta=model_config.rope_theta,
        dropout=model_config.dropout,
        bias=model_config.bias,
    )

    _, loss = model(input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()

    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
