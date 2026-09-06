import sys
import time
from lapis.config.base import load_config

class TrainingLogger:
    def __init__(self, config=None):
        if config is None:
            config = load_config("configs/tiny.yaml")
        self.config = config
        self.start_time = None
        self.step_count = 0
    
    def start_run(self):
        self.start_time = time.time()
        self.step_count = 0
        print("LAPIS TRAINING")
        print(f"Model: {self.config['model']['hidden_size']}-hidden, {self.config['model']['num_layers']} layers")
        from lapis.config.model_config import ModelConfig
        mc = ModelConfig()
        params = mc.count_parameters()
        print(f"Parameters: {params:,} total ({params['trainable']:,} trainable, {params['non_trainable']:,} non-trainable)")
        print(f"Dataset: unspecified")
        print(f"Device: {self.config['runtime']['device']}")
        print(f"Precision: {self.config['runtime']['dtype']}")
        print()
    
    def log_step(self, step, loss, learning_rate):
        self.step_count += 1
        elapsed = time.time() - self.start_time if self.start_time else 0
        tokens_per_sec = self.step_count / elapsed if elapsed > 0 else 0
        ppl = float(loss.exp()) if hasattr(loss, 'exp') else float('inf')
        
        print(f"Step {step}: Loss {loss:.4f} | Perplexity {ppl:.2f} | LR {learning_rate:.6f} | Tokens/sec {tokens_per_sec:.1f}")
    
    def log_end(self, step, loss, learning_rate):
        elapsed = time.time() - self.start_time if self.start_time else 0
        print()
        print(f"Training complete. Final step: {step}")
        print(f"Final loss: {loss:.4f}")
        print(f"Final learning rate: {learning_rate:.6f}")
        print(f"Total time: {elapsed:.1f}s")