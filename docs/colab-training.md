# LapisLLM — Colab GPU training

Use a GPU runtime in Google Colab.

## Quick start

```python
!git clone https://github.com/sudomarc/LapisLLM.git
```

```python
%cd /content/LapisLLM
```

Run the trainer as a Python module so the repository root is always on the
import path:

```python
!python -m scripts.colab_train
```

The script installs the project dependencies, checks CUDA, builds the bounded
pretraining corpus, launches the training job with `configs/colab.yaml`, and
publishes each verified checkpoint into the repository checkout.

If an older Colab runtime already cloned the repository, refresh it before
running training:

```python
%cd /content/LapisLLM
!git pull --ff-only
```

## Smoke test

Use a small corpus before the full run:

```python
!python -m scripts.colab_train --smoke-test
```

## Resume

```python
!python -m scripts.colab_train --resume
```

## Limit corpus size

```python
!python -m scripts.colab_train --max-chars 50000000
```

The entry point also supports direct execution with `python scripts/colab_train.py`.

## Output

The runner writes training data under `training_data/`, verified run checkpoints
under `checkpoints/colab-runs/`, and synchronizes the newest verified model to:

```text
checkpoints/latest.pt
checkpoints/tokenizer/
```

The checkpoint is committed and pushed to `origin/main` together with the
lightweight training history. Generated training data remains excluded from Git.

The checkpoint publication step refuses to proceed when unrelated local source
changes are present, so Colab cannot silently overwrite user-authored work.

## Checkpoint publishing

To publish the newest locally available Colab checkpoint manually:

```bash
lapis dev publish
```

The command verifies that the checkpoint is readable and contains model/config
metadata, updates the canonical `latest.pt` and tokenizer paths, commits only
checkpoint changes, and pushes them to `origin/main`.

GitHub write authentication must be configured in the Colab runtime before the
training workflow can push changes.
