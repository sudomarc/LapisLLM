from pathlib import Path

from lapis.tokenizer.tokenizer import Tokenizer


def test_bpe_round_trip(tmp_path: Path):
    corpus = [
        "LAPIS builds language models.",
        "This tokenizer learns reusable subwords.",
        "Bonjour le monde. Lapis fonctionne.",
        "UTF-8: café, naïve, 中文, العربية.",
    ]

    tokenizer = Tokenizer.train_from_iterator(corpus, vocab_size=512, min_frequency=1)
    text = "Bonjour le monde. café 中文"
    ids = tokenizer.encode(text)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == text

    output = tmp_path / "tokenizer"
    tokenizer.save(str(output))
    restored = Tokenizer.load(str(output))
    assert restored.decode(restored.encode(text)) == text
    assert restored.vocab_size == tokenizer.vocab_size


def test_special_tokens_have_stable_ids():
    tokenizer = Tokenizer.train_from_iterator(["hello world"], vocab_size=512)
    assert tokenizer.bos_id == 0
    assert tokenizer.eos_id == 1
    assert tokenizer.pad_id == 2
    assert tokenizer.unk_id == 3
