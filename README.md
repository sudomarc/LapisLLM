# LapisLLM

**An independent, from-scratch language model project focused on software development.**

LapisLLM aims to own the complete model lifecycle: tokenizer, decoder-only Transformer, data pipeline, pre-training, checkpoints, evaluation and local inference. Its base model starts from random initialization; a third-party pretrained checkpoint is not the foundation.

> **Status: documentation-first rebuild.** The current default branch has no implementation code. The architecture and workflows below are specifications, not claims that the software already works.

## Project decisions

- Strategy: train the base model from scratch.
- Primary ambition: coding and software-development assistance.
- Identity: independent LapisLLM architecture, training lineage and weights.
- Compute: limited; begin with a small Micro model and measure before scaling.
- Training environment: Google Colab for experimental GPU runs.
- Automation: GitHub Actions for CI and bounded CPU checks, not long-running GPU pre-training.

LapisLLM is an experimental engineering project. It does not claim parity with Claude, GPT or Gemini. Any capability claim must be supported by reproducible evaluation.

## Planned technology

| Area | Initial direction |
|---|---|
| Language and framework | Python and PyTorch |
| Model | Decoder-only causal Transformer |
| Tokenizer | Code-oriented, byte-aware BPE |
| Objective | Causal next-token prediction |
| First profile | Micro, target 5–10M parameters |
| Compute | Colab, with CPU fallback for tests |
| Artifact storage | Durable storage outside Git; Drive supported for manual notebooks |
| CI | GitHub Actions |

## Documentation

- [Project status](docs/status.md)
- [Architecture](docs/architecture.md)
- [Proposed model specification](docs/model-spec.md)
- [Colab training guide](docs/training/colab.md)
- [Checkpoint and reproducibility contract](docs/training/reproducibility.md)
- [Dataset governance](docs/data-policy.md)
- [Evaluation policy](docs/evaluation.md)
- [Roadmap](docs/roadmap.md)
- [Architecture decisions](docs/decisions/)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Agent instructions](AGENTS.md)

## Boundaries

LapisLLM owns model engineering and inference. Consumer chat UX, accounts, conversation storage, user memory and agent orchestration belong in a separate application. External applications should integrate through a stable, product-agnostic inference boundary.

## License

The repository software is under the MIT License. Model weights, datasets and third-party assets are separate artifacts; see [MODEL_LICENSE.md](MODEL_LICENSE.md) and [the data policy](docs/data-policy.md).
