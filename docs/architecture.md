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
Full resumable checkpoint
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
  config/          typed model, data and training configuration
  data/            ingestion, filtering, deduplication and dataset readers
  tokenizer/       tokenizer training and compatibility
  model/           Transformer modules and parameter counting
  training/        optimizer loop, metrics and scheduler
  checkpoint/      save, validate, index and restore artifacts
  evaluation/      loss, coding tasks and benchmark adapters
  inference/       checkpoint-backed generation
  cli/             developer commands
configs/           versioned experiment profiles
scripts/            thin executable entry points
notebooks/          thin Colab notebooks
tests/              unit, integration and regression coverage
docs/               canonical technical documentation
.github/            CI, issue forms and PR templates
~~~

This is a target layout, not a claim that these modules exist. Avoid creating empty scaffolding without an implementation issue.

## Platform roles

**GitHub** is the source-control and governance plane. It stores code, configs, docs, small fixtures, experiment manifests and CI results. Large corpora and checkpoints stay outside Git.

**GitHub Actions** runs deterministic checks that do not need a paid or availability-sensitive accelerator: lint, unit tests, config checks, docs checks, packaging and tiny CPU smoke tests. It is not the default GPU-training service.

**Google Colab** provides temporary compute for manual GPU/TPU experiments. Hardware must be detected at runtime and each run must tolerate interruption.

**Artifact storage** persists complete checkpoints and datasets. For manual hosted notebooks, Google Drive is the initial practical option. A future controller may use Drive API, Hugging Face Hub or Cloud Storage after authentication, quota, cost and reliability are tested. Keep storage behind an interface.

## Module boundaries

- Model: forward computation and architecture only.
- Tokenizer: encode/decode and version metadata.
- Data: source ingestion, transformation and manifests.
- Trainer: optimization, gradient scaling, accumulation, scheduling and progress.
- Checkpoint: serialization, integrity, compatibility and recovery.
- Evaluation: frozen model evaluation, isolated from training side effects.
- Inference: load model-only artifacts and generate within explicit limits.
- CLI: orchestrate public operations without duplicating module logic.

## External application boundary

LapisLLM provides model and inference capabilities. Consumer accounts, chat history, user memory, agent orchestration and product UX belong in a separate application. External consumers should use a stable inference interface, not import training internals.
