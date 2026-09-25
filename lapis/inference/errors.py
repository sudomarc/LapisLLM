"""Structured error taxonomy for LapisLLM inference and runtime operations."""

from __future__ import annotations


class LapisInferenceError(Exception):
    """Base exception class for all LapisLLM inference and runtime errors."""

    code: str = "inference_error"

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class CheckpointLoadError(LapisInferenceError, RuntimeError):
    """Raised when an inference checkpoint cannot be loaded safely."""

    code: str = "checkpoint_load_error"


class InvalidSamplingConfigError(LapisInferenceError, ValueError):
    """Raised when sampling parameters fail validation."""

    code: str = "invalid_sampling_config"


class InvalidPromptError(LapisInferenceError, ValueError):
    """Raised when an inference prompt is invalid."""

    code: str = "invalid_prompt"


class ContextLengthExceededError(LapisInferenceError, ValueError):
    """Raised when context length exceeds maximum position embeddings."""

    code: str = "context_length_exceeded"


class GenerationError(LapisInferenceError, RuntimeError):
    """Base class for errors occurring during model generation."""

    code: str = "generation_error"


class NonFiniteLogitsError(GenerationError):
    """Raised when non-finite logits or probabilities occur during generation."""

    code: str = "non_finite_logits"


class CancellationError(LapisInferenceError, RuntimeError):
    """Raised when generation is cancelled before or during execution."""

    code: str = "cancelled"
