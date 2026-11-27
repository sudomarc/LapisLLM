# Glossary

| Term | Meaning |
|---|---|
| Base model | The model produced by initial next-token pre-training. |
| From scratch | Train the base model from random initialization, not third-party pretrained weights. |
| Micro | First experimental profile targeting 5–10M parameters. |
| Small | Follow-up profile targeting 30–50M parameters. |
| Tokenizer | Maps source text/code to token IDs and back. |
| Corpus | Approved and processed collection of training records. |
| Manifest | Machine-readable record of source revisions, transformations, settings and hashes. |
| Pre-training | Initial causal next-token training. |
| SFT | Supervised fine-tuning of LapisLLM's own checkpoint on instructions and examples. |
| Checkpoint | Saved model/training state used for recovery or evaluation. |
| Inference artifact | Weights and metadata needed for generation without optimizer state. |
| Resume state | State needed to continue optimization, including optimizer, scheduler and RNG. |
| Effective batch size | Number of examples/tokens per optimizer step after micro-batching and accumulation. |
| Colab runtime | Temporary managed Jupyter environment that can be reclaimed or disconnected. |
| pass@k | Probability at least one of k generated candidates passes tests under a specified sampling protocol. |
