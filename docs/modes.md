# LapisLLM modes

LapisLLM uses one shared model/tokenizer/checkpoint core with two command surfaces.

## USER mode

The default command is:

```bash
lapis
```

Equivalent explicit command:

```bash
lapis chat
```

USER mode is inference-only. It loads the configured approved/default checkpoint and exposes chat, generation settings, and a clean runtime error message. User configuration lives in `configs/user/default.yaml` and contains only inference settings.

The stable Python surface is:

```python
from lapis.inference import LapisRuntime

runtime = LapisRuntime.from_checkpoint("checkpoints/latest.pt")
response = runtime.generate("Explain DNS.")
```

## DEV mode

Developer tooling is explicitly namespaced:

```bash
lapis dev --help
lapis dev train --config configs/tiny.yaml
lapis dev evaluate --checkpoint checkpoints/latest.pt
lapis dev generate "A language model learns by"
lapis dev inspect --checkpoint checkpoints/latest.pt
lapis dev benchmark --checkpoint checkpoints/latest.pt
lapis dev checkpoint inspect --checkpoint checkpoints/latest.pt
```

DEV mode owns training, evaluation, benchmarking, checkpoint inspection, datasets and the detailed training console. Existing lower-level scripts remain available for compatibility.

## Separation rules

- USER runtime does not import or expose optimizer/training APIs.
- USER checkpoint loading is read-only and uses PyTorch `weights_only=True`.
- DEV commands explicitly invoke training/evaluation utilities.
- Detailed startup diagnostics are emitted only when `LAPIS_DEV=1`; normal USER errors remain concise.
- The existing top-level `train`, `evaluate`, `generate`, and script entry points remain available as compatibility paths. New documentation should use `lapis dev ...` for developer operations.
