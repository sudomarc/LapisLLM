# LapisLLM

**An independent, from-scratch language model project focused on software development.**

LapisLLM aims to own the complete model lifecycle: tokenizer, decoder-only Transformer, data pipeline, pre-training, checkpoints, evaluation and local inference. Its base model starts from random initialization; a third-party pretrained checkpoint is not the foundation.

> **Status: documentation-first rebuild.** The current default branch has no implementation code. The architecture and workflows below are specifications, not claims that the software already works.

## Project decisions

- Strategy: train the base model from scratch.
- Primary ambition: coding and software-development assistance.
- Identity: independent LapisLLM architecture, training lineage and weights.
- Compute: the maintainer's own PC; no hosted training runtime is part of the plan.
- Initial scale: a small Micro model for validating the full pipeline; scale only after measuring local hardware.
- Development: Python and PyTorch.
- CI: GitHub Actions for deterministic CPU tests only.

LapisLLM is an experimental engineering project. It does not claim parity with Claude, GPT or Gemini. Any capability claim must be supported by reproducible evaluation.

## Planned technology

| Area | Initial direction |
|---|---|
| Language and framework | Python and PyTorch |
| Model | Decoder-only causal Transformer |
| Tokenizer | Code-oriented, byte-aware BPE |
| Objective | Causal next-token prediction |
| First profile | Micro, initially targeting 5–10M parameters |
| Training | Local PC, with hardware-aware CPU/accelerator selection |
| Artifacts | Local datasets, run manifests and checkpoints outside Git |
| CI | GitHub Actions, CPU-only |

The exact model size, batch size, sequence length, precision and expected training time must be calibrated against the actual PC. Do not assume a particular GPU or VRAM capacity.

## Documentation

- [Project status](docs/status.md)
- [Architecture](docs/architecture.md)
- [Proposed model specification](docs/model-spec.md)
- [Local PC training guide](docs/training/local.md)
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
