"""Versioned, validated checkpoint utilities for Lapis."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

CHECKPOINT_VERSION = 2


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is missing, invalid, or incompatible."""


def _json_safe_rng_state(state: tuple[Any, ...]) -> str:
    return json.dumps(state, separators=(",", ":"))


def _restore_python_rng_state(value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        raw = json.loads(value)
        if not isinstance(raw, list) or len(raw) != 3:
            raise CheckpointError("Invalid Python RNG state in checkpoint.")
        version, internal_state, gauss_next = raw
        state = (int(version), tuple(int(x) for x in internal_state), gauss_next)
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        version, internal_state, gauss_next = value
        state = (int(version), tuple(int(x) for x in internal_state), gauss_next)
    else:
        raise CheckpointError("Invalid Python RNG state in checkpoint.")
    random.setstate(state)


def _restore_torch_rng_state(state: Any) -> None:
    if state is None:
        return
    if not isinstance(state, torch.Tensor):
        state = torch.as_tensor(state, dtype=torch.uint8)
    else:
        state = state.detach().to(device="cpu", dtype=torch.uint8)
    if state.numel() == 0:
        raise CheckpointError("Checkpoint contains an empty torch RNG state.")
    torch.set_rng_state(state.contiguous())


def _restore_cuda_rng_state(state: Any) -> None:
    if state is None or not torch.cuda.is_available():
        return
    if isinstance(state, torch.Tensor):
        states = [state]
    elif isinstance(state, (list, tuple)):
        states = list(state)
    else:
        raise CheckpointError("Invalid CUDA RNG state in checkpoint.")

    normalized: list[torch.Tensor] = []
    for item in states:
        if not isinstance(item, torch.Tensor):
            item = torch.as_tensor(item, dtype=torch.uint8)
        else:
            item = item.detach().to(device="cpu", dtype=torch.uint8)
        if item.numel() == 0:
            raise CheckpointError("Checkpoint contains an empty CUDA RNG state.")
        normalized.append(item.contiguous())
    torch.cuda.set_rng_state_all(normalized)


def _validate_required_fields(checkpoint: Mapping[str, Any]) -> None:
    required = {"model_state_dict", "config", "step", "epoch"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise CheckpointError(
            "Checkpoint is missing required field(s): " + ", ".join(missing)
        )

    if not isinstance(checkpoint["config"], Mapping):
        raise CheckpointError("Checkpoint config must be a mapping.")

    version = int(checkpoint.get("checkpoint_version", 1))
    if version > CHECKPOINT_VERSION:
        raise CheckpointError(
            f"Checkpoint version {version} is newer than supported version "
            f"{CHECKPOINT_VERSION}. Update Lapis before loading it."
        )
    if int(checkpoint["step"]) < 0 or int(checkpoint["epoch"]) < 0:
        raise CheckpointError("Checkpoint step and epoch must be non-negative.")


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate a Lapis checkpoint without unsafe object deserialization."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise CheckpointError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except RuntimeError as exc:
        raise CheckpointError(
            f"Could not safely load checkpoint '{checkpoint_path}'. "
            "The file may be corrupt or use unsupported legacy serialization."
        ) from exc
    except Exception as exc:
        raise CheckpointError(
            f"Could not load checkpoint '{checkpoint_path}': {exc}"
        ) from exc

    if not isinstance(checkpoint, Mapping):
        raise CheckpointError("Checkpoint root object must be a mapping.")
    _validate_required_fields(checkpoint)
    return dict(checkpoint)


def restore_rng_state(checkpoint: Mapping[str, Any]) -> None:
    """Restore Python, CPU torch, and available CUDA RNG state."""
    _restore_python_rng_state(checkpoint.get("python_rng_state"))
    _restore_torch_rng_state(checkpoint.get("torch_rng_state"))
    _restore_cuda_rng_state(checkpoint.get("cuda_rng_state"))


def validate_resume_compatibility(
    checkpoint_config: Mapping[str, Any],
    current_config: Mapping[str, Any],
    checkpoint_tokenizer_version: str | None,
    current_tokenizer_version: str,
    checkpoint_tokenizer_vocab_size: int | None,
    current_tokenizer_vocab_size: int,
) -> None:
    """Reject resume attempts that would silently change model/training semantics."""
    checkpoint_model = checkpoint_config.get("model")
    current_model = current_config.get("model")
    if not isinstance(checkpoint_model, Mapping) or not isinstance(current_model, Mapping):
        raise CheckpointError("Both checkpoint and current config must define model settings.")

    model_keys = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "rope_theta",
        "dropout",
        "bias",
    )
    mismatches = [
        key
        for key in model_keys
        if checkpoint_model.get(key) != current_model.get(key)
    ]
    if mismatches:
        raise CheckpointError(
            "Checkpoint model configuration is incompatible for resume; "
            "mismatched fields: " + ", ".join(mismatches)
        )

    checkpoint_training = checkpoint_config.get("training")
    current_training = current_config.get("training")
    if not isinstance(checkpoint_training, Mapping) or not isinstance(current_training, Mapping):
        raise CheckpointError("Both checkpoint and current config must define training settings.")

    training_keys = (
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "max_steps",
        "micro_batch_size",
        "gradient_accumulation_steps",
        "gradient_clip",
    )
    mismatched_training = [
        key
        for key in training_keys
        if checkpoint_training.get(key) != current_training.get(key)
    ]
    if mismatched_training:
        raise CheckpointError(
            "Checkpoint training configuration is incompatible for resume; "
            "mismatched fields: " + ", ".join(mismatched_training)
        )

    if checkpoint_tokenizer_version not in (None, current_tokenizer_version):
        raise CheckpointError(
            "Checkpoint tokenizer version does not match the current tokenizer."
        )
    if (
        checkpoint_tokenizer_vocab_size is not None
        and int(checkpoint_tokenizer_vocab_size) != int(current_tokenizer_vocab_size)
    ):
        raise CheckpointError(
            "Checkpoint tokenizer vocabulary size does not match the current tokenizer."
        )


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    config: Mapping[str, Any],
    tokenizer: Any,
    data_fingerprint: str | None = None,
) -> None:
    """Atomically save a self-describing checkpoint."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if step < 0 or epoch < 0:
        raise ValueError("step and epoch must be non-negative")

    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "step": int(step),
        "epoch": int(epoch),
        "config": dict(config),
        "tokenizer_version": tokenizer.VERSION,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_json": tokenizer.to_json(),
        "data_fingerprint": data_fingerprint,
        "python_rng_state": _json_safe_rng_state(random.getstate()),
        "torch_rng_state": torch.get_rng_state().clone().cpu(),
        "cuda_rng_state": [state.clone().cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None,
    }

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
        dir=checkpoint_path.parent,
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(payload, temp_path)
        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temp_path, checkpoint_path)
    finally:
        temp_path.unlink(missing_ok=True)

    # Keep the tokenizer as a convenient standalone artifact, while the checkpoint
    # itself remains self-contained so the two can never be required as a pair.
    tokenizer.save(str(checkpoint_path.parent / "tokenizer"))
