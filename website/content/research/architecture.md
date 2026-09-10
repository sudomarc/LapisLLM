# Architecture

Lapis is a decoder-only causal Transformer implemented directly in PyTorch. The current `LapisModel` combines token embeddings, repeated Transformer blocks, a final RMSNorm, and a language-model head.

Each block contains RMSNorm, causal grouped-query attention with Rotary Position Embeddings (RoPE), a residual connection, then RMSNorm, a SwiGLU MLP, and another residual connection.

The current Tiny configuration uses 8 query attention heads and 4 key/value heads. RoPE frequencies are kept as a separate non-persistent complex64 buffer when the model is cast or moved between devices.

The forward pass produces next-token logits and can compute causal cross-entropy against shifted labels. The instantiated model can also report its actual parameter count rather than relying on a hard-coded headline number.
