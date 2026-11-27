# Training reproducibility and checkpoint contract

## Run manifest

Every experiment receives a unique run ID and immutable manifest containing:

- model architecture and serialized config;
- tokenizer ID, config and hash;
- dataset source IDs, revisions, splits, filtering and manifest hashes;
- training code commit SHA;
- Python, PyTorch, accelerator backend, driver and relevant library versions;
- hardware model, device, VRAM/system RAM and precision;
- optimizer and scheduler settings;
- global step, consumed tokens and effective batch size;
- seed and determinism settings;
- start/end timestamps and termination reason;
- evaluation results and artifact hashes.

Never write secrets, machine serial numbers, account names or credential-bearing environment values into a manifest.

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
- accelerator RNG states when applicable;
- sampler/data-loader state sufficient for the advertised resume level;
- tokenizer identity;
- model/training config and dataset manifest IDs;
- source commit and checkpoint format version.

If any required state is missing, label recovery approximate rather than exact.

## Safe local persistence sequence

1. Write a complete checkpoint to a temporary file on the target local volume.
2. Flush and close the file.
3. Compute a cryptographic hash.
4. Reopen and validate metadata and tensor shapes.
5. Move the completed file into a versioned checkpoint directory using a filesystem operation with documented semantics.
6. Verify the final artifact hash.
7. Update the latest-checkpoint pointer/manifest only after verification.
8. Retain a previous known-good checkpoint until the new one is confirmed.

Filesystem atomicity differs across platforms and volumes. Implement and test the persistence behavior on supported Windows and Linux filesystems. A partial write must never overwrite the only usable checkpoint.

## Resume validation

Before restore, verify architecture and tensor shapes, tokenizer vocabulary and special-token IDs, dataset manifests, optimizer/scheduler compatibility, checkpoint format version, checksum and trusted origin. Incompatible state must fail loudly. Loading model weights alone is not a full training resume.

## Determinism levels

Seed Python, NumPy and PyTorch and record determinism settings. Different accelerators, drivers, kernels and data-loader schedules may prevent bitwise replay. Distinguish:
- restart from last complete checkpoint;
- deterministic replay in a fixed environment where supported;
- statistical reproducibility across different environments.

Never claim a stronger guarantee than tests establish.

## Proposed local artifact layout

~~~text
.lapis-data/runs/<run-id>/
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

This is a proposal. Keep safe inference weights distinct from trusted training-resume state. Prefer safetensors for model-only distribution; treat training-state serialization as trusted input. Keep generated artifacts outside Git and maintain a backup before destructive cleanup.
