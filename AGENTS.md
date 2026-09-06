# LAPIS — AGENTS.md

## Project

LAPIS is a from-scratch decoder-only Transformer language model and training stack implemented with Python and PyTorch. The project is experimental and prioritizes correctness, reproducibility, and inspectability.

## Actual repository structure

```text
lapis/
├── config/
├── data/
├── model/
├── tokenizer/
└── checkpoint.py

scripts/
├── train_tokenizer.py
├── train.py
├── evaluate.py
├── generate.py
├── chat.py
├── serve.py
└── data-fetching utilities

tests/
configs/
docs/
website/
training_data/
```

There is currently no `src/` package layout and no `scripts/prepare_data.py`. Do not document or configure either as existing functionality until they are implemented.

## Architecture rules

- Correctness > simplicity > modularity > optimization.
- Preserve working behavior; prefer the smallest correct architectural fix.
- The model remains a real decoder-only causal Transformer.
- Model dimensions are configuration-driven; parameter counts are computed from instantiated modules.
- Tokenizer behavior and vocabulary are versioned with checkpoints.
- Training uses causal next-token cross-entropy.
- Checkpoints must preserve model weights, optimizer state, scheduler state, step, epoch, configuration, tokenizer reference, and RNG state.
- CPU execution must remain supported for tests and the Tiny profile.
- CUDA is optional and must never be assumed to exist.
- Do not claim unsupported performance, model capability, determinism, or benchmark results.

## Tokenizer rules

The tokenizer must provide `encode()`, `decode()`, `batch_encode()`, `save()`, and `load()`, plus BOS/EOS/PAD/UNK identifiers as applicable. Training and inference must use the exact tokenizer associated with the checkpoint.

## Training rules

Training must explicitly manage forward pass, loss, gradient accumulation, clipping, optimizer step, scheduler step, checkpointing, and numerical validation. Non-finite losses or gradient norms are fatal errors. Invalid configuration must fail before training starts.

Resume must restore model, optimizer, scheduler, training position, and RNG state and must reject incompatible architecture/training/tokenizer/data state. Exact mid-epoch DataLoader replay is not currently supported and must not be claimed.

## Checkpoint security

Treat model/checkpoint files as untrusted input. Prefer `weights_only=True` loading and explicit validation. Avoid arbitrary object deserialization. Checkpoint writes must be atomic so an interruption cannot replace a valid checkpoint with a partial file.

## Testing rules

Critical components require behavioral tests, including tokenizer round trips, model shapes, causal masking, loss alignment, generation/sampling, checkpoint save/load, resume compatibility, and CPU training. Every significant bug fix should include a regression test.

CI must exercise the real project entry points. Never delete, weaken, or skip a failing test solely to make CI green.

## Configuration rules

Do not keep configuration keys that the runtime silently ignores. Configuration names must match the code. `batch_size` must equal `micro_batch_size * gradient_accumulation_steps`. Unsupported features must be removed from active configs or fail explicitly.

## Data rules

Validate empty or malformed inputs early. Training and validation partitions must be separated before tokenizer training to avoid tokenizer leakage into validation. Dataset and experiment metadata should remain reproducible.

## CLI rules

CLI errors should identify the actual configuration or artifact problem rather than exposing avoidable low-level exceptions. Relative paths should work from documented execution contexts.

## Git rules

- Keep commits focused and descriptive.
- Never commit secrets, `.env` files, temporary files, caches, or accidental large artifacts.
- Checkpoint artifacts are generated outputs and should remain ignored unless intentionally distributed.
- Before a PR, review the full diff from the base branch and verify tests, lint, package build, and smoke tests.

## Security rules

Inspect subprocess construction, filesystem writes, environment variables, HTTP surfaces, dependency handling, and model/checkpoint deserialization. Never log credentials or silently suppress errors.

## Release gate

A release is acceptable only when the repository passes unit, integration, and end-to-end smoke tests; linting; packaging/build validation; CPU training; checkpoint save/load/resume; deterministic generation; and the available CI workflows. CUDA-specific paths must be reported as unvalidated when no CUDA runner is available.
