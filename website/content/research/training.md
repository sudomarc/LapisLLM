# Training Pipeline

The training objective is causal next-token prediction with cross-entropy loss. The current configuration supports gradient accumulation, gradient clipping, linear warmup, cosine learning-rate decay, and checkpoint persistence.

The development trainer records training progress and can run the Learning Monitor at configured intervals. Monitor records include loss, perplexity, learning rate, tokens seen, and generated samples from fixed prompts.

Training artifacts are paired with tokenizer artifacts. A successful run verifies the checkpoint, its metadata, optimizer/training state, and the companion tokenizer before publication or reuse by the inference runtime.
