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
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be finite and positive")

    def get_effective_batch_size(self):
        return self.micro_batch_size * self.gradient_accumulation_steps


class DataConfig:
    def __init__(self, config=None):
        config = _default_config() if config is None else config
        data = config.get("data", {})
        self.dataset_name = data.get("dataset_name", "lapis")
        self.max_seq_length = int(data.get("max_seq_length", 512))
        self.shard_size = int(data.get("shard_size", 1024 * 1024))
