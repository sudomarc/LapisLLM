import os

import yaml


def load_config(path):
    """Load configuration from a YAML file path."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["_base_path"] = os.path.dirname(os.path.abspath(path))
    return config
