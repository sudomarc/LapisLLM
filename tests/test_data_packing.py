import pytest

from lapis.data.packing import create_attention_mask, pack_into_batches, pack_tokens


def test_pack_tokens_uses_explicit_pad_id():
    assert pack_tokens([[1, 2]], max_seq_length=4, pad_id=7) == [[1, 2, 7, 7]]


def test_pack_tokens_preserves_long_document_truncation():
    assert pack_tokens([[1, 2, 3, 4]], max_seq_length=3, pad_id=9) == [[1, 2, 3]]


def test_pack_helpers_reject_invalid_sizes():
    with pytest.raises(ValueError):
        pack_tokens([[1]], max_seq_length=0)
    with pytest.raises(ValueError):
        pack_into_batches([[1]], batch_size=0, max_seq_length=2)
    with pytest.raises(ValueError):
        create_attention_mask(0)


def test_create_attention_mask_is_strictly_causal():
    mask = create_attention_mask(4)
    assert mask.tolist() == [
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ]
