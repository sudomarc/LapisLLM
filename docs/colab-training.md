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
pretraining corpus, and launches the training job with `configs/colab.yaml`.

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

The runner writes the corpus and provenance manifest under `training_data/` and
the trained checkpoint under `checkpoints/`.

Do not commit generated training data, checkpoints, or tokenizer artifacts.
