from pathlib import Path

import torch

from lapis.tokenizer.tokenizer import Tokenizer
from scripts.streaming_train import StreamingTextDataset


def test_streaming_dataset_yields_fixed_length_samples_without_materializing_dataset(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Lapis learns language from data.\n"
        "A transformer predicts the next token.\n",
        encoding="utf-8",
    )
    tokenizer = Tokenizer.train_from_files(
        [str(corpus)],
        vocab_size=128,
        min_frequency=1,
    )

    dataset = StreamingTextDataset(corpus, tokenizer, seq_len=8)
    samples = list(dataset)

    assert samples
    assert len(samples) >= 2
    for inputs, labels in samples:
        assert inputs.shape == (9,)
        assert labels.shape == inputs.shape
        assert inputs.dtype == torch.long
        assert labels.dtype == inputs.dtype

    assert not hasattr(dataset, "samples")


def test_streaming_dataset_masks_only_final_padding(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("tiny corpus", encoding="utf-8")
    tokenizer = Tokenizer.train_from_files(
        [str(corpus)],
        vocab_size=64,
        min_frequency=1,
    )

    samples = list(StreamingTextDataset(corpus, tokenizer, seq_len=16))
    assert len(samples) == 1
    inputs, labels = samples[0]
    valid = int(labels.ne(-100).sum().item())

    assert inputs.shape == (17,)
    assert 1 < valid < 17
    assert (labels[valid:] == -100).all()
    assert labels[:valid].equal(inputs[:valid])
