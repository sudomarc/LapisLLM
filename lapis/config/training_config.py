from __future__ import annotations

from lapis.config.base import load_config


def _default_config():
    return load_config("configs/tiny.yaml")


class TrainingConfig:
    def __init__(self, config=None):
        config = _default_config() if config is None else config
        training = config.get("training")
        if not isinstance(training, dict):
            raise ValueError("configuration must define a 'training' mapping")

        self.learning_rate = float(training["learning_rate"])
        self.weight_decay = float(training["weight_decay"])
        self.warmup_steps = int(training["warmup_steps"])
        self.max_steps = int(training["max_steps"])
        self.batch_size = int(training["batch_size"])
        self.micro_batch_size = int(training["micro_batch_size"])
        self.gradient_accumulation_steps = int(training["gradient_accumulation_steps"])
        self.gradient_clip = float(training["gradient_clip"])

        if self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("training.weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("training.warmup_steps must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("training.max_steps must be positive")
        if self.batch_size <= 0 or self.micro_batch_size <= 0:
            raise ValueError("training batch sizes must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        expected_batch = self.micro_batch_size * self.gradient_accumulation_steps
        if self.batch_size != expected_batch:
            raise ValueError(
                "training.batch_size must equal micro_batch_size * "
                "gradient_accumulation_steps"
            )
        if self.gradient_clip <= 0:
            raise ValueError("training.gradient_clip must be positive")
        if self.warmup_steps > self.max_steps:
            raise ValueError("training.warmup_steps cannot exceed training.max_steps")

    def get_effective_batch_size(self):
        return self.micro_batch_size * self.gradient_accumulation_steps


class DataConfig:
    def __init__(self, config=None):
        config = _default_config() if config is None else config
        data = config.get("data", {})
        self.dataset_name = data.get("dataset_name", "lapis")
        self.max_seq_length = int(data.get("max_seq_length", 512))
        self.shard_size = int(data.get("shard_size", 1024 * 1024))
        if self.max_seq_length < 2:
            raise ValueError("data.max_seq_length must be at least 2")
        if self.shard_size <= 0:
            raise ValueError("data.shard_size must be positive")
