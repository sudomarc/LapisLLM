import copy
import inspect

import pytest

from lapis.config import TrainingConfig, get_default_config
from scripts.lapis import is_history_path, parse_training_line, train_one_run


def test_training_config_rejects_ignored_batch_size_mismatch():
    config = copy.deepcopy(get_default_config())
    config["training"]["batch_size"] = 7
    with pytest.raises(ValueError, match="batch_size must equal"):
        TrainingConfig(config)


def test_legacy_console_accepts_scientific_loss_values():
    assert parse_training_line("step=0010 loss=3.9e-04 ppl=1.0004") == (10, 3.9e-4)
    assert parse_training_line("step=0010 loss=1.25E+01 ppl=...") == (10, 12.5)


def test_legacy_console_uses_auto_device_for_training_and_chat():
    training_source = inspect.getsource(train_one_run)
    assert '"--device",\n        "auto"' in training_source


def test_training_history_is_classified_as_syncable_history():
    assert is_history_path("training_history/run-1/summary.json")
    assert not is_history_path("README.md")
