# LAPIS

> A language model built from the ground up.

LAPIS is an independent LLM research project focused on building the complete language-model stack: data, tokenization, Transformer architecture, training, evaluation, inference and distribution.

## Project status

**Lapis 0.1 — Research / In development**

The first milestone is a small, real Transformer model that can be trained, checkpointed, evaluated and used for generation. The architecture is intentionally designed to scale into a family of models over time.

## Model family

- Lapis Tiny — architecture and training validation
- Lapis Small — planned
- Lapis 1B — planned
- Lapis 3B — planned
- Lapis 7B+ — long-term

## Stack

Python · PyTorch · safetensors · FastAPI · CUDA-ready

## Roadmap

Foundation → Pretraining → Post-training → Evaluation → Scaling → Local runtimes & API ecosystem

## Website

The repository contains the project landing page and documentation site. Enable GitHub Pages on the `main` branch using the repository root to publish it.

## Philosophy

Build the machine to understand the machine. Measure capabilities from real experiments, keep the stack modular, and scale deliberately.

## License

See [LICENSE](./LICENSE).
