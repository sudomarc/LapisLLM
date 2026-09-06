from lapis.config.model_config import get_default_config


class DataConfig:
    def __init__(self, config=None):
        config = get_default_config() if config is None else config
        data = config.get("data", {})
        self.dataset_name = data.get("dataset_name", "lapis")
        self.max_seq_length = int(data.get("max_seq_length", 512))
        self.shard_size = int(data.get("shard_size", 1024 * 1024))
