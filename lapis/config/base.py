# Base configuration for LapisLLM
from pydantic import BaseModel, Field
from typing import Literal, Optional


class LapisConfig(BaseModel):
    """Core configuration for the LapisLLM model and training."""

    # Model architecture
    model_name: str = "lapis-micro"
    vocab_size: int = 16_000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    head_dim: int = 64
    max_seq_len: int = 512
    rope_base: float = 10_000.0

    # Tokenizer
    tokenizer_name: str = "lapis-code"

    # Training
    batch_size: int = 8
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_steps: int = 10_000
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Precision
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"

    # Hardware
    device: Literal["cpu", "cuda", "auto"] = "auto"

    # Paths
    data_root: str = "./.lapis-data"
    run_root: str = "./.lapis-data/runs"

    # Reproducibility
    seed: int = 42


def load_config(path: str) -> LapisConfig:
    """Load configuration from a YAML file."""
    import yaml
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return LapisConfig(**data)