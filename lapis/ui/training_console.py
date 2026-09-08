"""Public training console module.

The implementation lives in ``training_console_fixed``; the legacy runner is
kept separately so existing telemetry behavior remains reusable while the
interactive UI stays compatible with current Textual TCSS.
"""

from .training_console_fixed import (
    CheckpointInfo,
    Sample,
    TrainingAppController,
    TrainingConsoleRunner,
    TrainingState,
    TrainingTelemetry,
    can_use_tui,
    format_duration,
    run_training_console,
    sanitize_output,
)

__all__ = [
    "CheckpointInfo",
    "Sample",
    "TrainingAppController",
    "TrainingConsoleRunner",
    "TrainingState",
    "TrainingTelemetry",
    "can_use_tui",
    "format_duration",
    "run_training_console",
    "sanitize_output",
]
