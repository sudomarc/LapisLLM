# LapisLLM architecture

## System pipeline

~~~text
Approved dataset sources
        |
        v
Provenance, license and privacy checks
        |
        v
Cleaning, filtering and deduplication
        |
        v
Train / validation / test manifests
        |
        v
Tokenizer training and encoding
        |
        v
Packed token blocks
        |
        v
Decoder-only Transformer
        |
        v
Next-token loss and optimizer
        |
        v
Local resumable checkpoint
        |
        v
Evaluation and local inference
        |
        v
Versioned model artifact
~~~

## Planned repository layout

~~~text
lapis/
  config/          typed model, data, hardware and training configuration
  data/            ingestion, filtering, deduplication and dataset readers
  tokenizer/       tokenizer training and compatibility
  model/           Transformer modules and parameter counting
  training/        optimizer loop, metrics, scheduling and resume
  checkpoint/      save, validate, index and restore artifacts
  evaluation/      loss, coding tasks and benchmark adapters
  inference/       checkpoint-backed generation
  cli/             developer and hardware-diagnostic commands
configs/           versioned experiment profiles
scripts/            thin executable entry points
tests/              unit, integration and regression coverage
docs/               canonical technical documentation
.github/            CI, issue forms and PR templates
~~~

This is a target layout, not a claim that these modules exist. Avoid creating empty scaffolding without an implementation issue.

## Platform roles

**Maintainer's PC** is the only planned training compute environment. The runtime must detect operating system, CPU, system RAM, accelerator backend, GPU model, available accelerator memory, driver/runtime versions and free disk space. Training configurations must be selected only after this report and a bounded throughput/memory test.

**GitHub** is the source-control and governance plane. It stores code, configs, docs, small legally safe fixtures, experiment manifests and CI results. Large corpora and checkpoints stay outside Git.

**GitHub Actions** runs deterministic CPU validation: lint, unit tests, config checks, documentation checks, packaging and tiny smoke tests. It must not assume access to the maintainer's GPU, private datasets or host machine.

**Local artifact storage** holds downloaded corpora, tokenizer artifacts, run logs and checkpoints. Use a configurable data/output root outside the source tree where possible. Storage operations must be isolated behind an interface to allow backups or another store later without changing the trainer.

## Hardware support policy

- CPU is the reference path for correctness tests and very small smoke runs.
- NVIDIA CUDA is the first intended accelerator path, subject to installed PyTorch/driver compatibility.
- AMD ROCm, Apple MPS and other accelerators are not promised until specifically tested.
- Never assume VRAM, RAM or storage capacity.
- Never silently fall back to CPU for a long run requested for an accelerator.
- Record device, precision, peak memory and throughput with every experiment.

The exact model size and batch profile remain open until the actual PC is audited.

## Module boundaries

- Model: forward computation and architecture only.
- Tokenizer: encode/decode and version metadata.
- Data: source ingestion, transformation and manifests.
- Trainer: optimization, gradient scaling, accumulation, scheduling and progress.
- Checkpoint: serialization, integrity, compatibility and recovery.
- Evaluation: frozen-model evaluation, isolated from training side effects.
- Inference: load model-only artifacts and generate within explicit limits.
- CLI: orchestrate public operations without duplicating module logic.

## External application boundary

LapisLLM provides model and inference capabilities. Consumer accounts, chat history, user memory, agent orchestration and product UX belong in a separate application. External consumers should use a stable inference interface, not import training internals.
