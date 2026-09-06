from lapis.config.base import load_config
from lapis.config import get_default_config


class TrainingConfig:
    def __init__(self, config=None):
        if config is None:
            config = get_default_config()
        
        self.learning_rate = float(config["training"]["learning_rate"])
        self.weight_decay = float(config["training"]["weight_decay"])
        self.warmup_steps = int(config["training"]["warmup_steps"])
        self.max_steps = int(config["training"]["max_steps"])
        self.batch_size = int(config["training"]["batch_size"])
        self.micro_batch_size = int(config["training"]["micro_batch_size"])
        self.gradient_accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
        self.gradient_clip = float(config["training"]["gradient_clip"])
    
    def get_effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation_steps


class DataConfig:
    def __init__(self, config=None):
        if config is None:
            config = get_default_config()
        
        self.dataset_name = config.get("data", {}).get("dataset_name", "lapis")
        self.max_seq_length = int(config.get("data", {}).get("max_seq_length", 512))
        self.shard_size = int(config.get("data", {}).get("shard_size", 1024 * 1024))