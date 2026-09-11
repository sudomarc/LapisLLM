import torch

from lapis.tokenizer.tokenizer import Tokenizer
from scripts import build_colab_corpus
from scripts.streaming_train import StreamingTextDataset
from scripts.train import TextDataset


def test_text_dataset_keeps_full_sequence_for_model_shift():
    dataset = TextDataset([1, 2, 3, 4, 5], seq_len=4, pad_id=0)
    inputs, labels = dataset[0]
    assert inputs.tolist() == [1, 2, 3, 4, 5]
    assert labels.tolist() == [1, 2, 3, 4, 5]


def test_text_dataset_masks_only_padding_targets():
    dataset = TextDataset([1, 2, 3], seq_len=4, pad_id=0)
    inputs, labels = dataset[0]
    assert inputs.tolist() == [1, 2, 3, 0, 0]
    assert labels.tolist() == [1, 2, 3, -100, -100]
    assert inputs.dtype == torch.long
    assert labels.dtype == torch.long


def test_streaming_corpus_source_retries_after_transient_open_error(monkeypatch):
    attempts = 0

    def fake_load_dataset(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("transient transport failure")
        return "dataset"

    monkeypatch.setattr(build_colab_corpus, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(build_colab_corpus, "RETRY_BACKOFF_SECONDS", 0)

    dataset = build_colab_corpus.load_source(
        "fineweb_edu",
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        "train",
    )

    assert dataset == "dataset"
    assert attempts == 3
