from __future__ import annotations

import json
from pathlib import Path


class Tokenizer:
    """Small deterministic character-level tokenizer used by Lapis development builds.

    This is intentionally simple. It is a pipeline tokenizer, not the final Lapis BPE tokenizer.
    """

    VERSION = "lapis-tokenizer-v2-char"

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        pad_token: str = "<p>",
        unk_token: str = "<u>",
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.vocab = self._build_default_vocab() if vocab is None else {
            str(token): int(idx) for token, idx in vocab.items()
        }
        self.itos = {idx: token for token, idx in self.vocab.items()}

        for token in (bos_token, eos_token, pad_token, unk_token):
            if token not in self.vocab:
                raise ValueError(f"Missing required special token: {token}")

        self.bos_id = self.vocab[bos_token]
        self.eos_id = self.vocab[eos_token]
        self.pad_id = self.vocab[pad_token]
        self.unk_id = self.vocab[unk_token]
        self.vocab_size = len(self.vocab)

    def _build_default_vocab(self) -> dict[str, int]:
        vocab: dict[str, int] = {
            self.bos_token: 0,
            self.eos_token: 1,
            self.pad_token: 2,
            self.unk_token: 3,
        }
        alphabet = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,;:!?\"'()[]{}<>_-/@#$%^&*+=|\\~`\n\t"
        )
        for char in alphabet:
            if char not in vocab:
                vocab[char] = len(vocab)
        return vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)
        ids.extend(self.vocab.get(char, self.unk_id) for char in text)
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        special_ids = {self.bos_id, self.eos_id, self.pad_id}
        output: list[str] = []
        for idx in token_ids:
            idx = int(idx)
            if skip_special_tokens and idx in special_ids:
                continue
            output.append(self.itos.get(idx, self.unk_token))
        return "".join(output)

    def batch_encode(self, texts, add_special_tokens: bool = True) -> list[list[int]]:
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def save(self, path: str) -> None:
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "tokenizer.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "vocab": self.vocab,
                    "bos_token": self.bos_token,
                    "eos_token": self.eos_token,
                    "pad_token": self.pad_token,
                    "unk_token": self.unk_token,
                    "vocab_size": self.vocab_size,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        with (directory / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "tokenizer_version": self.VERSION,
                    "type": "character",
                    "model_arch": "decoder-only transformer",
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        vocab_path = Path(path) / "tokenizer.json"
        with vocab_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(
            vocab=data["vocab"],
            bos_token=data.get("bos_token", "<s>"),
            eos_token=data.get("eos_token", "</s>"),
            pad_token=data.get("pad_token", "<p>"),
            unk_token=data.get("unk_token", "<u>"),
        )

    def __len__(self) -> int:
        return self.vocab_size

    def __getitem__(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, "
            f"type=character, bos={self.bos_token!r}, eos={self.eos_token!r})"
        )
