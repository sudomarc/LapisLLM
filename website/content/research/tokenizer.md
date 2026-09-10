# Tokenizer

Lapis uses the Hugging Face `tokenizers` BPE backend with ByteLevel pre-tokenization and NFKC normalization. The decoder is also ByteLevel and the BPE model uses byte fallback.

The required special tokens are `<s>`, `</s>`, `<p>`, and `<u>`. The implementation identifies itself as `lapis-tokenizer-v3-bpe-bytelevel` and stores tokenizer metadata beside `tokenizer.json`.

Inference checkpoints keep the tokenizer with the model artifact. Loading validates tokenizer version and vocabulary size before constructing the model so incompatible weights and tokenization cannot silently mix.
