from lapis.config.base import load_config


def get_default_config():
    """Get the default model configuration (tiny)."""
    return load_config("configs/tiny.yaml")


def get_config(path):
    """Load a configuration from a file path."""
    return load_config(path)


class ModelConfig:
    def __init__(self, config=None):
        if config is None:
            config = get_default_config()
        
        self.vocab_size = config["model"]["vocab_size"]
        self.hidden_size = config["model"]["hidden_size"]
        self.intermediate_size = config["model"]["intermediate_size"]
        self.num_layers = config["model"]["num_layers"]
        self.num_attention_heads = config["model"]["num_attention_heads"]
        self.num_key_value_heads = config["model"]["num_key_value_heads"]
        self.max_position_embeddings = config["model"]["max_position_embeddings"]
        self.rope_theta = config["model"]["rope_theta"]
        self.dropout = config["model"]["dropout"]
        self.bias = config["model"]["bias"]
    
    def count_parameters(self):
        """Count total parameters in the model."""
        # Embedding: vocab_size * hidden_size
        embedding_params = self.vocab_size * self.hidden_size
        
        # Transformer blocks
        layer_params = (6 * self.hidden_size**2 + 3 * self.hidden_size * self.intermediate_size + 2 * self.hidden_size)
        attention_params = self.num_layers * layer_params
        
        # LM Head
        lm_head_params = self.vocab_size * self.hidden_size
        
        total = embedding_params + attention_params + lm_head_params
        
        # Non-trainable: embeddings + lm_head (if not tied)
        non_trainable = embedding_params + lm_head_params
        trainable = total - non_trainable
        
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": non_trainable,
            "embedding": embedding_params,
            "attention": attention_params,
            "mlp": self.num_layers * 3 * self.hidden_size * self.intermediate_size,
            "norm": self.num_layers * 2 * self.hidden_size,
            "lm_head": lm_head_params,
        }


class TrainingConfig:
    def __init__(self, config=None):
        if config is None:
            config = get_default_config()
        
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.warmup_steps = config["training"]["warmup_steps"]
        self.max_steps = config["training"]["max_steps"]
        self.batch_size = config["training"]["batch_size"]
        self.micro_batch_size = config["training"]["micro_batch_size"]
        self.gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
        self.gradient_clip = config["training"]["gradient_clip"]
    
    def get_effective_batch_size(self):
        return self.batch_size * self.gradient_accumulation_steps


class DataConfig:
    def __init__(self, config=None):
        if config is None:
            config = get_default_config()
        
        self.dataset_name = config.get("data", {}).get("dataset_name", "lapis")
        self.max_seq_length = config.get("data", {}).get("max_seq_length", 512)
        self.shard_size = config.get("data", {}).get("shard_size", 1024 * 1024)