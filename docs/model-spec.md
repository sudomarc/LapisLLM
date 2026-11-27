# Proposed model specification

**Status: proposed baseline.** This is the first implementation target, not an implemented or benchmarked model.

## Objective

Train a coding-focused base model from random initialization using causal next-token prediction. The initial pre-training mixture prioritizes source code and relevant technical text. Instruction following, repository editing and tool use are later post-training stages.

## Architecture proposal

| Component | Initial choice |
|---|---|
| Family | Decoder-only Transformer |
| Attention | Causal self-attention |
| Position encoding | RoPE |
| Normalization | Pre-norm RMSNorm |
| MLP | SwiGLU |
| Tokenizer | Code-oriented, byte-aware BPE |
| Objective | Next-token cross-entropy |
| Initial context | 512 tokens |
| Initial vocabulary | 16,000 tokens, subject to tokenizer evaluation |
| Embeddings | Tied token embedding and LM head for Micro baseline |
| Framework | PyTorch |

Use conventional, inspectable components for the first baseline. Do not add custom kernels, mixture-of-experts, speculative decoding or distributed complexity before correctness and baseline throughput are measured.

## Model profiles

| Profile | Target parameters | Purpose |
|---|---:|---|
| Micro | 5–10M | Prove end-to-end correctness and first learning experiment |
| Small | 30–50M | First scale-up if compute and data justify it |
| Medium | 100–300M | Future research target, not a commitment |

A parameter target is not a performance promise. Compute the actual parameter count from the instantiated model and save it in the run manifest.

## Micro configuration proposal

- Vocabulary: 16,000.
- Hidden size: 256.
- Transformer layers: 4.
- Attention heads: 4; head dimension 64.
- MLP intermediate size: approximately 704.
- Maximum sequence length: 512.
- RoPE base: 10,000.
- Pre-norm RMSNorm and SwiGLU.
- Tied input and output token embeddings.
- Bias disabled unless controlled tests show a benefit.

The implementation must calculate the actual parameter count. If the profile falls outside 5–10M, revise it and record why.

## Tokenizer contract

Train on the training partition only. Compare byte-aware BPE settings against representative languages and file types. Record vocabulary size, special-token IDs, normalization, pre-tokenization, implementation version, training-data manifest and artifact hash.

A checkpoint must declare the exact tokenizer identity it expects. Never silently load a tokenizer with a different vocabulary or special-token mapping.

## Precision and hardware

- FP32 is the reference path for correctness tests.
- Use BF16 only when the detected accelerator supports it reliably.
- Use FP16 with gradient scaling when appropriate.
- CPU remains supported for tiny tests and smoke runs.
- Never assume a fixed Colab GPU model.

## Architecture change gate

Any change to architecture or tokenizer requires shape and parameter-count tests, a fixed evaluation sample, measured memory and tokens/second, comparison with the prior baseline, and an ADR when it affects a public model contract.
