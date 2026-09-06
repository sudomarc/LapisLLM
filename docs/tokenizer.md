# LAPIS tokenizer pipeline

LAPIS now uses a real BPE tokenizer based on the Hugging Face `tokenizers` library.

The tokenizer uses BPE with a ByteLevel pre-tokenizer and NFKC normalization. ByteLevel keeps the tokenizer byte-complete for arbitrary UTF-8 input, while BPE learns reusable subword units from the training corpus. The special token order is fixed as `<s>`, `</s>`, `<p>`, `<u>`.

## Train a tokenizer

```bash
python scripts/train_tokenizer.py --data data/raw/corpus.txt --vocab-size 32000 --min-frequency 2
```

Or use a directory of `.txt` files:

```bash
python scripts/train_tokenizer.py --data-dir data/raw --vocab-size 32000 --min-frequency 2
```

The output contains:

```text
artifacts/tokenizer/
├── tokenizer.json
└── metadata.json
```

The training script automatically loads an existing tokenizer artifact when present. Otherwise it trains one from the configured corpus, so the model embedding vocabulary always matches the actual tokenizer vocabulary.
