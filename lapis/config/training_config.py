from __future__ import annotations

import math

from lapis.config.base import load_config


def _default_config():
    return load_config("configs/tiny.yaml")


class TrainingConfig:
    """Validated optimizer, scheduler, and batch-training settings."""

    def __init__(self, config=None):
        config = _default_config() if config is None else config
        if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
            raise ValueError("Configuration must contain a 'training' mapping")
        training = config["training"]
        required = (
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "max_steps",
            "batch_size",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "gradient_clip",
        )
        missing = [key for key in required if key not in training]
        if missing:
            raise ValueError(
                "Missing training configuration field(s): " + ", ".join(missing)
            )

        self.learning_rate = float(training["learning_rate"])
        self.weight_decay = float(training["weight_decay"])
        self.warmup_steps = int(training["warmup_steps"])
        self.max_steps = int(training["max_steps"])
        self.batch_size = int(training["batch_size"])
        self.micro_batch_size = int(training["micro_batch_size"])
        self.gradient_accumulation_steps = int(training["gradient_accumulation_steps"])
        self.gradient_clip = float(training["gradient_clip"])

        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and greater than 0")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than 0")
        if self.batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError("batch_size and micro_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and greater than 0")
        effective = self.get_effective_batch_size()
        if self.batch_size != effective:
            raise ValueError(
                "batch_size must equal micro_batch_size * gradient_accumulation_steps "
                f"({self.batch_size} != {effective})"
            )
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")

    def get_effective_batch_size(self):
        return self.micro_batch_size * self.gradient_accumulation_steps


class DataConfig:
    """Validated data-pipeline configuration."""

    def __init__(self, config=None):
        config = _default_config() if config is None else config
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a mapping")
        data = config.get("data", {})
        if not isinstance(data, dict):
            raise ValueError("Configuration field 'data' must be a mapping")
        self.dataset_name = data.get("dataset_name", "lapis")
        self.max_seq_length = int(data.get("max_seq_length", 512))
        self.shard_size = int(data.get("shard_size", 1024 * 1024))
        validation_split = float(data.get("validation_split", 0.1))
        self.validation_split = validation_split
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty")
        if self.max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2")
        if self.shard_size < 1:
            raise ValueError("shard_size must be positive")
        if not math.isfinite(validation_split) or not 0.0 < validation_split < 0.5:
            raise ValueError("validation_split must be greater than 0 and less than 0.5")
