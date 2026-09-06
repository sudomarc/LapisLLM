from __future__ import annotations

from lapis.config.base import load_config


def get_default_config():
    """Load the default Lapis Tiny configuration."""
    return load_config("configs/tiny.yaml")


def get_config(path):
    """Load a configuration from a YAML file path."""
    return load_config(path)


class ModelConfig:
    def __init__(self, config=None):
        config = get_default_config() if config is None else config
        model = config["model"]
        self.vocab_size = int(model["vocab_size"])
        self.hidden_size = int(model["hidden_size"])
        self.intermediate_size = int(model["intermediate_size"])
        self.num_layers = int(model["num_layers"])
        self.num_attention_heads = int(model["num_attention_heads"])
        self.num_key_value_heads = int(model.get("num_key_value_heads", self.num_attention_heads))
        self.max_position_embeddings = int(model["max_position_embeddings"])
        self.rope_theta = float(model["rope_theta"])
        self.dropout = float(model["dropout"])
        self.bias = bool(model["bias"])

        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    def count_parameters(self):
        """Estimate parameter counts from the architecture configuration."""
        h = self.hidden_size
        kv = self.num_key_value_heads
        heads = self.num_attention_heads
        d = h // heads
        linear_bias = h if self.bias else 0
        q = h * h + linear_bias
        k = h * (kv * d) + (kv * d if self.bias else 0)
        v = k
        o = h * h + linear_bias
        attention_per_layer = q + k + v + o

        gate = h * self.intermediate_size + (self.intermediate_size if self.bias else 0)
        up = gate
        down = self.intermediate_size * h + (h if self.bias else 0)
        mlp_per_layer = gate + up + down

        norm_per_layer = 2 * h
        embedding = self.vocab_size * h
        final_norm = h
        lm_head = h * self.vocab_size + (self.vocab_size if self.bias else 0)
        total = (
            embedding
            + self.num_layers * (attention_per_layer + mlp_per_layer + norm_per_layer)
            + final_norm
            + lm_head
        )
        return {
            "total": total,
            "trainable": total,
            "non_trainable": 0,
            "embedding": embedding,
            "attention": self.num_layers * attention_per_layer,
            "mlp": self.num_layers * mlp_per_layer,
            "norm": self.num_layers * norm_per_layer + final_norm,
            "lm_head": lm_head,
        }
