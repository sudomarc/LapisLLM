# LapisLLM — Colab GPU training

Use a GPU runtime in Google Colab.

## Quick start

```python
!git clone https://github.com/sudomarc/LapisLLM.git
```

```python
%cd LapisLLM
```

```python
!python scripts/colab_train.py
```

The script installs the project dependencies, checks CUDA, builds the bounded
pretraining corpus, and launches the training job with `configs/colab.yaml`.

## Smoke test

Use a small corpus before the full run:

```python
!python scripts/colab_train.py --smoke-test
```

## Resume

```python
!python scripts/colab_train.py --resume
```

## Limit corpus size

```python
!python scripts/colab_train.py --max-chars 50000000
```

## Output

The runner writes the corpus and provenance manifest under `training_data/` and
the trained checkpoint under `checkpoints/`.

Do not commit generated training data, checkpoints, or tokenizer artifacts.
