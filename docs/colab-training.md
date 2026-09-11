# LapisLLM — Colab GPU training

Use a GPU runtime in Google Colab.

## Quick start

```python
!git clone https://github.com/sudomarc/LapisLLM.git
```

```python
%cd /content/LapisLLM
```

Run the trainer as a Python module:

```python
!python -m scripts.colab_train
```

This is the **canonical non-interactive Colab entry point**. A normal run must never call `input()`, wait for notebook keyboard input, open a consumer chat prompt, or silently wait for a manual choice. The default is exactly one training run; run selection is controlled by explicit command-line arguments only.

The execution contract is:

```text
preflight
  → dependencies
  → GPU/CPU selection
  → corpus validation/build
  → tokenizer/model initialization
  → training
  → checkpoint verification
  → post-training generation checks
  → training-history recording
  → canonical checkpoint publication
```

A long-running phase must remain observable. The orchestrator emits explicit phase messages and periodic heartbeats when the child process is otherwise quiet. A heartbeat is diagnostic output only; it is never a prompt for input.

## Training performance contract

The normal Colab path does **not** run generated-text samples inside the optimizer loop. Loss, learning rate, token count, and step progress may be reported during training, but model generation is deferred to the post-training verification stage unless an explicit developer override enables the learning monitor.

This separation matters because generation is additional model inference work and must not accidentally turn a training run into a repeated train-plus-inference loop. The default Colab path therefore prioritizes optimizer throughput and continuous progress reporting.

The training path for file-backed corpora is memory-bounded: the corpus is not loaded into one Python string, tokenized samples are not all materialized in RAM, and the DataLoader consumes fixed-length sequences incrementally from the file. Hugging Face documents `streaming=True` as progressive iteration without downloading the complete dataset, and PyTorch documents iterable-style datasets/DataLoaders as a supported pattern for iterative training workloads.

## Corpus composition and provenance

The Colab corpus is globally bounded by an explicit `--max-chars` budget. The budget is distributed progressively across the configured sources so that the first source cannot consume the entire budget and starve later sources. If an earlier source fails, its unused budget remains available to later sources.

Each generated manifest records the selected sources, per-source records and character counts, source failures, builder settings, and output path. A cached corpus is reusable only when its manifest is structurally valid and consistent with the current corpus-builder contract.

The Colab corpus downloader uses the standard Hugging Face Hub transport by default (`HF_HUB_DISABLE_XET=1`) and has a bounded retry policy for transient source/network failures. A source that remains unavailable is recorded in the manifest and skipped so another configured corpus source can still provide training data. Partial output from a failed source is discarded before retrying, so a retry cannot duplicate records in the corpus.

## Refreshing an existing runtime

If an older Colab runtime already cloned the repository, refresh it before running training:

```python
%cd /content/LapisLLM
!git pull --ff-only
```

Do not run an older notebook cell that embeds an interactive training menu. The repository runner, not the notebook UI, owns the workflow.

## Smoke test

Use a small corpus before the full run:

```python
!python -m scripts.colab_train --smoke-test
```

The smoke path is intentionally CPU-only, bounded, non-interactive, and must not push training artifacts.

## Resume

```python
!python -m scripts.colab_train --resume
```

Resume selects incomplete runs from repository training history. It does not ask how many runs to execute.

## Explicit controls

The runner supports explicit controls such as `--runs`, `--max-chars`, `--device`, and `--monitor-interval`. These are automation parameters, not interactive prompts.

For example:

```python
!python -m scripts.colab_train --runs 2 --monitor-interval 0
```

`--monitor-interval 0` is the default for the canonical Colab training path and disables in-training generation sampling.

## Output and publication

The runner writes training data under `training_data/`, verified run checkpoints under `checkpoints/colab-runs/`, and synchronizes the newest verified model to:

```text
checkpoints/latest.pt
checkpoints/tokenizer/
```

A checkpoint is not considered publishable until its file, optimizer step, model/config metadata, tokenizer artifact, and tokenizer-version metadata pass verification. Training history is written only after the training process and checkpoint validation succeed.

The publication step refuses to proceed when unrelated local source changes are present. Git authentication failures fail clearly rather than opening an interactive password/token prompt.

## Checkpoint publishing

To publish the newest locally available Colab checkpoint manually:

```bash
lapis dev publish
```

The command verifies that the checkpoint is readable and contains model/config metadata, updates the canonical `latest.pt` and tokenizer paths, commits only checkpoint changes, and pushes them to `origin/main`.

GitHub write authentication must be configured in the Colab runtime before the training workflow can push changes.

## Evidence and verification

Every change to the Colab workflow must follow the repository agent contract and the testing policy. At minimum, verify the no-input contract, the affected corpus/training behavior, checkpoint publication behavior, and the final repository checks appropriate to the risk of the change.

Authoritative external references used by this workflow:

- Hugging Face Datasets streaming: https://huggingface.co/docs/datasets/stream
- Hugging Face `load_dataset(..., streaming=True)`: https://huggingface.co/docs/datasets/package_reference/loading_methods
- PyTorch data loading and `IterableDataset`: https://docs.pytorch.org/docs/stable/data
