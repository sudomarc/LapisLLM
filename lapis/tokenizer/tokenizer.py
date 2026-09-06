from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

try:
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFKC
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
except ImportError as exc:  # pragma: no cover - exercised only on missing dependency
    HFTokenizer = None
    BPE = None
    ByteLevelDecoder = None
    ByteLevel = None
    NFKC = None
    BpeTrainer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class Tokenizer:
    """LAPIS BPE tokenizer.

    Uses Hugging Face Tokenizers' native BPE implementation with ByteLevel
    pre-tokenization so the vocabulary can grow with the training corpus while
    remaining byte-complete for arbitrary UTF-8 input.
    """

    VERSION = "lapis-tokenizer-v3-bpe-bytelevel"
    SPECIAL_TOKENS = ("<s>", "</s>", "<p>", "<u>")

    def __init__(self, backend: HFTokenizer):
        self._tokenizer = backend
        self.bos_token, self.eos_token, self.pad_token, self.unk_token = self.SPECIAL_TOKENS
        self.bos_id = self._required_id(self.bos_token)
        self.eos_id = self._required_id(self.eos_token)
        self.pad_id = self._required_id(self.pad_token)
        self.unk_id = self._required_id(self.unk_token)
        self.vocab_size = int(self._tokenizer.get_vocab_size())

    @staticmethod
    def _require_dependency() -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "The 'tokenizers' package is required for the LAPIS BPE tokenizer. "
                "Install the project dependencies first."
            ) from _IMPORT_ERROR

    def _required_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Missing required special token: {token}")
        return int(token_id)

    @classmethod
    def create_untrained(cls) -> "Tokenizer":
        cls._require_dependency()
        tokenizer = HFTokenizer(BPE(unk_token="<u>", byte_fallback=True))
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = ByteLevelDecoder()
        tokenizer.add_special_tokens(list(cls.SPECIAL_TOKENS))
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int = 512,
        min_frequency: int = 1,
    ) -> "Tokenizer":
        if vocab_size < len(cls.SPECIAL_TOKENS):
            raise ValueError("vocab_size must be at least the number of special tokens")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        tokenizer = cls.create_untrained()
        trainer = BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=int(min_frequency),
            special_tokens=list(cls.SPECIAL_TOKENS),
            limit_alphabet=256,
            show_progress=True,
        )
        tokenizer._tokenizer.train_from_iterator(texts, trainer=trainer)
        tokenizer._tokenizer.decoder = ByteLevelDecoder()
        return cls(tokenizer._tokenizer)

    @classmethod
    def train_from_files(
        cls,
        files: list[str],
        vocab_size: int = 32000,
        min_frequency: int = 2,
    ) -> "Tokenizer":
        if not files:
            raise ValueError("At least one tokenizer training file is required")
        if vocab_size < len(cls.SPECIAL_TOKENS):
            raise ValueError("vocab_size must be at least the number of special tokens")
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        tokenizer = cls.create_untrained()
        trainer = BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=int(min_frequency),
            special_tokens=list(cls.SPECIAL_TOKENS),
            limit_alphabet=256,
            show_progress=True,
        )
        tokenizer._tokenizer.train(files, trainer=trainer)
        tokenizer._tokenizer.decoder = ByteLevelDecoder()
        return cls(tokenizer._tokenizer)

    @classmethod
    def from_json(cls, serialized: str) -> "Tokenizer":
        cls._require_dependency()
        if not serialized:
            raise ValueError("Serialized tokenizer is empty")
        try:
            backend = HFTokenizer.from_str(serialized)
        except Exception as exc:
            raise ValueError("Could not deserialize the embedded tokenizer") from exc
        return cls(backend)

    def to_json(self) -> str:
        """Return the exact backend tokenizer definition for checkpoint embedding."""
        return self._tokenizer.to_str()

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        cls._require_dependency()
        tokenizer_path = Path(path)
        if tokenizer_path.is_dir():
            tokenizer_path = tokenizer_path / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        backend = HFTokenizer.from_file(str(tokenizer_path))
        return cls(backend)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids = list(encoding.ids)
        if add_special_tokens:
            ids = [self.bos_id, *ids, self.eos_id]
        return ids

    def batch_encode(self, texts: Iterable[str], add_special_tokens: bool = True) -> list[list[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        ids = [int(idx) for idx in token_ids]
        if skip_special_tokens:
            ids = [
                idx
                for idx in ids
                if idx not in {self.bos_id, self.eos_id, self.pad_id}
            ]
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, path: str) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(directory / "tokenizer.json"))
        metadata = {
            "tokenizer_version": self.VERSION,
            "type": "bpe",
            "pre_tokenizer": "ByteLevel",
            "normalizer": "NFKC",
            "model": "BPE",
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "bos": self.bos_token,
                "eos": self.eos_token,
                "pad": self.pad_token,
                "unk": self.unk_token,
            },
        }
        with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return self.vocab_size

    def __getitem__(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        return self.unk_id if token_id is None else int(token_id)

    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self.vocab_size}, type=bpe-bytelevel)"
