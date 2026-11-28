"""LapisLLM Tokenizer — HuggingFace tokenizers wrapper."""

from tokenizers import Tokenizer as HFTokenizer


class BPETokenizer:
    """Wrapper around HuggingFace BPE tokenizer for LapisLLM."""

    def __init__(self, tokenizer: HFTokenizer):
        self._tokenizer = tokenizer

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from HuggingFace tokenizer.json file."""
        return cls(HFTokenizer.from_file(path))

    @classmethod
    def train(cls, training_texts: list, vocab_size: int = 16_000) -> "BPETokenizer":
        """Train a new BPE tokenizer."""
        from lapis.tokenizer.bpe_tokenizer import train_tokenizer
        tokenizer = train_tokenizer(training_texts, vocab_size=vocab_size)
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        return self._tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.token_to_id("<PAD>")

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.token_to_id("<EOS>")

    @property
    def unk_token_id(self) -> int:
        return self._tokenizer.token_to_id("<UNK>")

    @property
    def bos_token_id(self) -> int:
        return self._tokenizer.token_to_id("<BEGIN_OF_TOKEN>")

    @property
    def tokenizer_json(self) -> str:
        """Return tokenizer as JSON string."""
        return self._tokenizer.to_str()

    def save(self, path: str):
        """Save tokenizer to file."""
        self._tokenizer.save(path)

    # Alias for compatibility
    tokenize = encode