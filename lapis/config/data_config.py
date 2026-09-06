from lapis.config.base import load_config

class DataConfig:
    def __init__(self, config=None):
        if config is None:
            config = get_default_config()
        
        self.dataset_name = config.get("data", {}).get("dataset_name", "lapis")
        self.max_seq_length = config.get("data", {}).get("max_seq_length", 512)
        self.shard_size = config.get("data", {}).get("shard_size", 1024 * 1024)