# Training LapisLLM on the local PC

## Decision

LapisLLM training runs on the maintainer's own computer. No hosted notebook runtime or external training GPU is part of the planned workflow. GitHub is used for source control and CI, not to execute long training jobs.

The exact PC specifications are not yet recorded. Do not assume that a useful coding model can be trained until CPU, RAM, GPU/VRAM, drivers, storage and measured throughput are known.

## Platform strategy

- Python and PyTorch are the training stack.
- CPU is required for correctness tests and tiny smoke experiments.
- NVIDIA CUDA is the first intended accelerator path when a compatible GPU and PyTorch build are available.
- AMD ROCm, Apple MPS and other backends are not promised until explicitly tested.
- Windows and Linux are the initial developer environments to validate. Keep path handling and process control portable.

## Hardware discovery

Before selecting model or training settings, implement a diagnostic command that reports:

- operating system and architecture;
- CPU model and logical/physical core counts;
- available system RAM;
- GPU model and backend;
- accelerator memory / VRAM;
- PyTorch, CUDA/ROCm and driver versions where exposed;
- supported dtypes (FP32, FP16, BF16);
- free space on the selected dataset and checkpoint volumes.

The diagnostic must not collect serial numbers, machine account names, private directory names or unrelated personal information. The report should be safe to share after the owner reviews it.

Then run a bounded benchmark with the candidate Micro model and representative sequence length. Record peak memory, tokens per second, step time and CPU/GPU utilization. Select batch size, gradient accumulation, precision and sequence length from measured results, not guesses.

## Intended local workflow

The following commands define the desired CLI contract. They are **not executable until the corresponding implementation issues are completed**.

~~~powershell
# Windows PowerShell — after package bootstrap is implemented
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -e ".[dev]"

lapis doctor
lapis data validate
lapis train --config configs/micro.yaml --run-id micro-local-001
lapis train --config configs/micro.yaml --run-id micro-local-001 --resume auto
lapis evaluate --run-id micro-local-001
~~~

~~~bash
# Linux — after package bootstrap is implemented
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

lapis doctor
lapis data validate
lapis train --config configs/micro.yaml --run-id micro-local-001
lapis train --config configs/micro.yaml --run-id micro-local-001 --resume auto
lapis evaluate --run-id micro-local-001
~~~

## Local storage layout

Keep generated material out of Git:

~~~text
.lapis-data/
  raw/                 # source snapshots or local downloads
  processed/           # cleaned/deduplicated records
  tokenized/           # encoded shards
  tokenizer/           # tokenizer artifacts
  runs/<run-id>/
    config.yaml
    run-manifest.json
    metrics.jsonl
    checkpoints/
    evaluation/
~~~

Allow an explicit data root and output root so the user can place large artifacts on a disk with sufficient free space. Check available disk before downloads, tokenization and checkpoints. Never write multi-gigabyte outputs into the source tree by default.

## Long-running training process

- Print useful step, loss, learning rate, tokens processed, elapsed time and tokens/second metrics.
- Save complete checkpoints at configurable step/time intervals.
- Handle Ctrl+C as a graceful stop: finish or safely abort the current step, save a valid checkpoint and exit.
- On power loss or process crash, resume from the last complete checkpoint.
- Never overwrite the only known-good checkpoint with a partial write.
- Validate configuration, tokenizer identity, dataset manifest and free disk before starting.
- Do not start long runs automatically during installation, tests or CI.

## GitHub Actions

Actions runs CPU tests, lint, static checks and bounded smoke runs. It must not require access to the maintainer's machine, local dataset, GPU or private filesystem. The local PC is not configured as a self-hosted runner by default; training is launched explicitly by the maintainer.

## Resource limits and realistic expectations

A small model can validate the engineering pipeline, but it is not automatically a useful coding assistant. CPU-only training may be extremely slow. Even a discrete GPU can be constrained by VRAM and the amount of legally usable code data. Report actual tokens/second and wall-clock cost before increasing model size or corpus volume.

If the device cannot support the candidate profile, reduce context length, micro-batch size, hidden size or layer count. Use gradient accumulation and checkpointing where they help, but do not claim that these techniques remove the fundamental compute cost.

## Completion criteria

The local workflow is not complete until a bounded Micro run starts from random weights, produces finite metrics, saves a complete checkpoint, resumes after deliberate interruption, and can be evaluated from the saved artifact without relying on the training process still running.
