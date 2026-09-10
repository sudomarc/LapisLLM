# Training

Training is implemented around next-token prediction, explicit configuration, checkpointing, and reproducible monitoring.

## Canonical development configuration

`configs/tiny.yaml` currently uses a 4096-token vocabulary, 256 hidden dimensions, 1024 intermediate dimensions, 6 layers, 8 query heads, 4 key/value heads, and a 512-token context. Its training section targets 10,000 optimizer steps with learning rate 3e-4, weight decay 0.1, 100 warmup steps, batch size 8, micro-batch size 4, two accumulation steps, and gradient clipping at 1.0.

## Core training behavior

The model is trained with causal cross-entropy for next-token prediction. The training stack supports gradient accumulation, clipping, linear warmup, cosine decay, checkpoint save/load, and validation-oriented evaluation.

## Learning Monitor

The training monitor can record loss, perplexity, learning rate, tokens seen, and generated samples from fixed prompts during training. Custom monitor prompts use `||` as the separator.

```text
python scripts/train.py --monitor-prompts "Machine learning is||The transformer architecture"
```

## Checkpoints

A successful training run verifies that the checkpoint exists and that its paired tokenizer artifact exists. The inference runtime also verifies tokenizer version and vocabulary compatibility before loading a checkpoint.

Checkpoint publication is an explicit developer workflow and is used to make a verified checkpoint available to external consumers such as CHAD. Generated training data and large artifacts are not automatically treated as source code.

## Colab

`python scripts/colab_train.py` is the repository's non-interactive Colab entry point. Recent fixes make it invoke the actual trainer directly, verify the resulting checkpoint, generate preview samples, write experiment history, and stage the newest verified checkpoint for publication.
