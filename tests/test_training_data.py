import torch

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
