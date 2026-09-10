# LapisLLM runtime boundary

LapisLLM has one product role: it is the language-model engine and developer/research platform.

The user-facing conversational application is CHAD, a separate repository. LapisLLM exposes model and runtime capabilities; CHAD owns conversation UX, user workflows, history, accounts, settings, and presentation.

## Developer inference console

A lightweight interactive console is retained for checkpoint and generation testing:

```bash
lapis dev chat --checkpoint checkpoints/latest.pt
```

This console is a developer tool. It exists for:

- checkpoint smoke tests
- tokenizer and prompt debugging
- generation and sampling experiments
- CPU/CUDA runtime testing
- regression investigation
- rapid local experimentation

It must not acquire consumer onboarding, persistent product conversations, account management, or user-facing application settings.

## Stable runtime API

External applications should consume the product-agnostic inference surface:

```python
from lapis.inference import LapisRuntime

runtime = LapisRuntime.from_checkpoint("checkpoints/latest.pt")
response = runtime.generate("Explain DNS.")
```

The runtime owns checkpoint loading, tokenizer/model compatibility, device selection, generation, and sampling. It does not know about CHAD-specific product behavior.

## Developer CLI

Developer operations are explicitly namespaced:

```bash
lapis dev --help
lapis dev train --config configs/tiny.yaml
lapis dev evaluate --checkpoint checkpoints/latest.pt
lapis dev generate "A language model learns by"
lapis dev chat --checkpoint checkpoints/latest.pt
lapis dev inspect --checkpoint checkpoints/latest.pt
lapis dev benchmark --checkpoint checkpoints/latest.pt
lapis dev checkpoint inspect --checkpoint checkpoints/latest.pt
```

Legacy top-level training, evaluation, and generation commands may remain temporarily for compatibility. New integrations and documentation should prefer `lapis dev ...` or the stable Python runtime API.

## Boundary rule

```text
LAPISLLM
  model + tokenizer + data + training + evaluation
  + checkpoints + inference + generation + developer tools
                         │
                         ▼
                product-agnostic runtime
                         │
                         ▼
                       CHAD
                         │
                         ▼
                       USERS
```

No first-class consumer mode exists in LapisLLM.
