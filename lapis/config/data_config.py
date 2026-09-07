from lapis.config.model_config import get_default_config


class DataConfig:
    """Configuration for the Lapis data preparation/training pipeline."""

    def __init__(self, config=None):
        config = get_default_config() if config is None else config
        data = config.get("data", {})
        self.dataset_name = data.get("dataset_name", "lapis")
        self.max_seq_length = int(data.get("max_seq_length", 512))
        self.shard_size = int(data.get("shard_size", 1024 * 1024))
        self.catalog = data.get("catalog", "configs/data/datasets.yaml")
        self.raw_dir = data.get("raw_dir", "data/raw")
        self.cleaned_dir = data.get("cleaned_dir", "data/cleaned")
        self.deduplicated_dir = data.get("deduplicated_dir", "data/deduplicated")
        self.tokenized_dir = data.get("tokenized_dir", "data/tokenized")
        self.manifests_dir = data.get("manifests_dir", "data/manifests")
        self.profile = data.get("profile", "recommended")
        self.sources = list(data.get("sources", []))
        self.seed = int(data.get("seed", 42))
        self.streaming = bool(data.get("streaming", False))
