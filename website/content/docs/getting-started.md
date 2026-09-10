# Getting Started

LapisLLM 0.2.0 is an experimental language-model engine and developer/research platform. It owns the model, tokenizer, training, evaluation, checkpointing, inference runtime, API, and developer tooling. The user-facing conversational application is CHAD, a separate project.

## Install

Use Python 3.11 or newer, create a virtual environment, then install the repository in editable mode.

```text
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development dependencies, install the project with its development extras according to your environment.

## Verify

Run the correctness suite and static analysis.

```text
python -m pytest
ruff check .
```

## Developer CLI

The supported developer namespace is explicit.

```text
lapis --help
lapis dev --help
lapis dev train
lapis dev evaluate --checkpoint checkpoints/latest.pt
lapis dev generate "A language model learns by"
lapis dev chat
lapis dev inspect --checkpoint checkpoints/latest.pt
lapis dev benchmark --checkpoint checkpoints/latest.pt
lapis dev checkpoint inspect --checkpoint checkpoints/latest.pt
```

There is no consumer `lapis chat` mode. The interactive chat command under `lapis dev chat` is a developer/research inference console for checkpoint smoke tests, prompt debugging, sampling experiments, and runtime verification.

## Stable Python runtime

External applications should use the public inference surface rather than importing model internals.

```python
from lapis.inference import LapisRuntime

runtime = LapisRuntime.from_checkpoint("checkpoints/latest.pt")
response = runtime.generate("Explain DNS.")
```

The runtime loads the checkpoint read-only, verifies tokenizer compatibility, and exposes generation-oriented operations.
