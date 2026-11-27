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

Use conventional, inspectable components for the first baseline. Do not add custom kernels, mixture-of-experts, speculative decoding or distributed complexity before correctness and local throughput are measured.

## Model profiles

| Profile | Target parameters | Purpose |
|---|---:|---|
| Micro | 5–10M | Prove end-to-end correctness and first learning experiment |
| Small | 30–50M | First scale-up if local compute and data justify it |
| Medium | 100–300M | Future research target, not a commitment |

These targets are provisional. The actual PC specifications are not yet recorded. Do not freeze hidden size, layers, batch size, sequence length or precision until hardware discovery and memory/throughput tests are complete. A parameter target is not a performance promise.

## Micro configuration proposal

A candidate starting point is vocabulary 16k, hidden size 256, 4 layers, 4 heads, head dimension 64, sequence length 512, RoPE base 10,000, pre-norm RMSNorm and SwiGLU, with tied input/output embeddings.

This is a configuration hypothesis, not a guarantee that the machine can train it efficiently. The implementation must calculate actual parameter count and benchmark memory/throughput. Reduce the profile when the measured host budget requires it.

## Tokenizer contract

Train on the training partition only. Compare byte-aware BPE settings against representative languages and file types. Record vocabulary size, special-token IDs, normalization, pre-tokenization, implementation version, training-data manifest and artifact hash.

A checkpoint must declare the exact tokenizer identity it expects. Never silently load a tokenizer with a different vocabulary or special-token mapping.

## Precision and hardware

- FP32 is the reference path for correctness tests.
- BF16 is used only when the detected device supports it reliably.
- FP16 uses gradient scaling where appropriate.
- CPU remains supported for tiny tests and bounded experiments.
- Accelerator selection is based on actual PyTorch backend availability and tested compatibility, never on a hardcoded device assumption.

## Architecture change gate

Any change to architecture or tokenizer requires shape and parameter-count tests, a fixed evaluation sample, measured peak memory and tokens/second, comparison with the prior baseline, and an ADR when it affects a public model contract.
