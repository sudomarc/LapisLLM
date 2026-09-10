# Models

Lapis currently publishes one concrete development configuration: Lapis Tiny. The configuration is experimental and is not presented as a competitive pretrained foundation model.

## Lapis Tiny

The canonical configuration is `configs/tiny.yaml`.

```text
vocab size              4096
auto hidden size         256
intermediate size       1024
transformer layers        6
query attention heads     8
key/value heads           4
context length          512
RoPE theta            10000
```

The architecture is a decoder-only causal Transformer using token embeddings, RMSNorm, grouped-query causal attention, RoPE, SwiGLU MLP blocks, a final RMSNorm, and a language-model head.

The actual instantiated parameter count is exposed by the model/runtime code rather than hard-coded into public documentation.

## Planned scales

Lapis Small, Lapis 1B, Lapis 3B, and Lapis 7B are planning targets. Their exact configurations and public artifacts are not published yet.

## Model publication

Software versions and model versions are separate. Do not infer a model release from a software release. Check the repository configuration, checkpoint metadata, and release notes for a concrete artifact.
