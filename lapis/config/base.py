import yaml
import os


def load_config(path):
    """Load configuration from a YAML file path."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    # Set the base path for relative references
    config["_base_path"] = os.path.dirname(os.path.abspath(path))
    
    return config