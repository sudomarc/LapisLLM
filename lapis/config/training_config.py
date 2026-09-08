from __future__ import annotations

import math

from lapis.config.base import load_config


def _default_config():
    return load_config("configs/tiny.yaml")


class TrainingConfig:
    def __init__(self, config=None):
        config = _default_config() if config is None else config
        training = config["training"]
        self.learning_rate = float(training["learning_rate"])
        self.weight_decay = float(training["weight_decay"])
        self.warmup_steps = int(training["warmup_steps"])
        self.max_steps = int(training["max_steps"])
        self.batch_size = int(training["batch_size"])
        self.micro_batch_size = int(training["micro_batch_size"])
        self.gradient_accumulation_steps = int(training["gradient_accumulation_steps"])
        self.gradient_clip = float(training["gradient_clip"])

        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.micro_batch_size < 1:
            raise ValueError("micro_batch_size must be at least 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")

        effective_batch_size = self.get_effective_batch_size()
        if self.batch_size != effective_batch_size:
            raise ValueError(
                "batch_size must equal micro_batch_size * gradient_accumulation_steps "
                f"({self.micro_batch_size} * {self.gradient_accumulation_steps} = "
                f"{effective_batch_size}, got {self.batch_size})"
            )

    def get_effective_batch_size(self):
        return self.micro_batch_size * self.gradient_accumulation_steps


class DataConfig:
    def __init__(self, config=None):
        config = _default_config() if config is None else config
        data = config.get("data", {})
        self.dataset_name = data.get("dataset_name", "lapis")
        self.max_seq_length = int(data.get("max_seq_length", 512))
        self.shard_size = int(data.get("shard_size", 1024 * 1024))
