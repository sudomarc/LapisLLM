# Reference

## Public inference API

The stable Python inference surface is exposed from `lapis.inference`.

```python
from lapis.inference import LapisRuntime, SamplingConfig

runtime = LapisRuntime.from_checkpoint("checkpoints/latest.pt")
text = runtime.generate("Explain DNS.", SamplingConfig(max_new_tokens=64))
```

`LapisRuntime` loads a checkpoint and its paired tokenizer read-only. It exposes tokenization, model metadata, normal generation, and streaming generation. Sampling validates maximum tokens, temperature, top-k, top-p, and model probability tensors.

## Local HTTP API

Start the server with:

```text
lapis api serve
```

The underlying server also remains available through `scripts/serve.py`. The API currently exposes:

```text
GET  /v1/models
POST /v1/chat/completions
```

Chat requests contain a model id, one or more role/content messages, and sampling controls for temperature, maximum tokens, top-k, and top-p. The service is explicitly a local experimentation and integration surface, not production serving infrastructure.

## Checkpoint contract

Inference uses safe read-only checkpoint loading and verifies tokenizer version and vocabulary compatibility. A missing checkpoint, missing tokenizer, incompatible vocabulary, or incompatible tokenizer version is rejected with an explicit error.

## Current limits

Distributed training, high-performance KV-cache inference, broad public benchmark coverage, production serving hardening, and large-scale model releases are not currently presented as implemented capabilities.
