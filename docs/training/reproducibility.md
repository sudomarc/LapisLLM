# Training reproducibility and checkpoint contract

## Run manifest

Every experiment receives a unique run ID and immutable manifest containing:

- model architecture and serialized config;
- tokenizer ID, config and hash;
- dataset source IDs, revisions, splits, filtering and manifest hashes;
- training code commit SHA;
- Python, PyTorch, CUDA and relevant library versions;
- hardware model, accelerator, VRAM and precision;
- optimizer and scheduler settings;
- global step, consumed tokens and effective batch size;
- seed and determinism settings;
- start/end timestamps and termination reason;
- evaluation results and artifact hashes.

Never write secrets or credential-bearing environment values into a manifest.

## Full resumable checkpoint

A full training checkpoint must contain:

- model state;
- optimizer state;
- scheduler state;
- gradient-scaler state when FP16 scaling is used;
- global optimizer step and consumed-token count;
- Python RNG state;
- NumPy RNG state when used;
- PyTorch CPU RNG state;
- CUDA RNG states when applicable;
- sampler/data-loader state sufficient for the advertised resume level;
- tokenizer identity;
- model/training config and dataset manifest IDs;
- source commit and checkpoint format version.

If any required state is missing, label recovery approximate rather than exact.

## Safe persistence sequence

1. Write a complete checkpoint to a temporary local path.
2. Flush and close the file.
3. Compute a cryptographic hash.
4. Reopen and validate metadata.
5. Copy to a unique temporary location in durable storage.
6. Verify the copied artifact hash.
7. Publish a completion manifest or pointer only after verification.
8. Retain the previous known-good checkpoint until the new one is confirmed.

Do not assume every mounted filesystem provides identical atomic rename semantics. The storage adapter must document its guarantees. A partial copy must never overwrite the only usable checkpoint.

## Resume validation

Before restore, verify architecture and tensor shapes, tokenizer vocabulary and special-token IDs, dataset manifests, optimizer/scheduler compatibility, checkpoint format version, checksum and trusted origin. Incompatible state must fail loudly. Loading model weights alone is not a full training resume.

## Determinism levels

Seed Python, NumPy and PyTorch and record determinism settings. Different accelerators, CUDA versions, kernels and data-loader schedules may prevent bitwise replay. Distinguish:
- restart from last complete checkpoint;
- deterministic replay in a fixed environment where supported;
- statistical reproducibility across different environments.

Never claim a stronger guarantee than tests establish.

## Proposed artifact layout

~~~text
runs/<run-id>/
  run-manifest.json
  config.yaml
  metrics.jsonl
  checkpoints/
    step-00001000/
      model.safetensors
      training-state.pt
      metadata.json
      SHA256SUMS
  final/
    model.safetensors
    tokenizer/
    model-card.md
~~~

This is a proposal. Keep safe inference weights distinct from trusted training-resume state. Prefer safetensors for model-only distribution; treat training-state serialization as trusted input.
