# Training

Training is implemented around next-token prediction, explicit configuration, checkpointing, and reproducible monitoring.

## Canonical development configuration

`configs/tiny.yaml` currently uses a 4096-token vocabulary, 256 hidden dimensions, 1024 intermediate dimensions, 6 layers, 8 query heads, 4 key/value heads, and a 512-token context. Its training section targets 10,000 optimizer steps with learning rate 3e-4, weight decay 0.1, 100 warmup steps, batch size 8, micro-batch size 4, two accumulation steps, and gradient clipping at 1.0.

## Core training behavior

The model is trained with causal cross-entropy for next-token prediction. The training stack supports gradient accumulation, clipping, linear warmup, cosine decay, checkpoint save/load, and validation-oriented evaluation.

## Learning Monitor

The training monitor can record loss, perplexity, learning rate, tokens seen, and generated samples from fixed prompts during training. Custom monitor prompts use `||` as the separator.

The canonical Colab runner disables generated-text sampling during optimization (`--monitor-interval 0`) so the training loop is not implicitly turned into repeated train-plus-inference work. Developers can explicitly enable the monitor when investigating learning behavior.

```text
python scripts/train.py --monitor-prompts "Machine learning is||The transformer architecture"
```

## Checkpoints

A successful training run verifies that the checkpoint exists and that its paired tokenizer artifact exists. The inference runtime also verifies tokenizer version and vocabulary compatibility before loading a checkpoint.

Training checkpoints and published inference checkpoints are deliberately different artifacts. A training checkpoint may contain optimizer, scheduler, epoch, and RNG state required for trusted resume. The published `checkpoints/latest.pt` artifact contains only inference-required model/configuration state plus tokenizer metadata and an inference-format marker, and is paired with `checkpoints/tokenizer/`.

Checkpoint publication is an explicit developer workflow and is used to make a verified inference checkpoint available to external consumers such as CHAD. Generated training data remains experiment input and is not automatically treated as source code.

## Colab

`python -m scripts.colab_train` is the canonical non-interactive Colab entry point. It executes the real streaming trainer, reports progress without requesting notebook input, verifies the resulting training checkpoint, runs post-training preview generations, records training history, updates the canonical inference checkpoint paths, and publishes the permitted generated outputs when GitHub authentication is available.

The Colab corpus builder uses Hugging Face streaming and divides its explicit character budget across configured sources so one source cannot consume the whole budget before later sources are considered. Network transport uses bounded retries and explicit Hugging Face Hub timeouts.
