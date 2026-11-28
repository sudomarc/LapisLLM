"""Minimal training entry point for LapisLLM Stage 1."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="LapisLLM training (Stage 1)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--run-id", type=str, default="micro-local-001", help="Run identifier")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    args = parser.parse_args()

    # Import config (PyTorch optional)
    try:
        from lapis.config.base import load_config
        from lapis.config.hardware import detect_hardware, get_device
    except ImportError as e:
        print(f"Import error: {e}")
        print("Try: pip install -e .[torch]")
        sys.exit(1)

    # Load config
    from lapis.config.base import LapisConfig

    if args.config:
        config = load_config(args.config)
    else:
        config = LapisConfig()

    # Detect hardware
    hw = detect_hardware()
    device = get_device()
    print(f"LapisLLM training on {device}")
    print(f"Hardware: {hw['pytorch']['version']} | CUDA available: {hw['pytorch']['cuda_available']}")
    print(f"CPU cores: {hw['cpu']['cores']} | System RAM: {hw['memory_gb']} GB")

    if config.precision == "fp16":
        print("WARNING: FP16 requested but may not be supported on CPU-only. Using FP32.")

    print(f"Config: hidden_size={config.hidden_size}, layers={config.num_layers}, "
          f"vocab={config.vocab_size}, max_seq={config.max_seq_len}")
    print(f"Training steps: {args.steps} | batch_size={config.batch_size} "
          f"(micro={config.micro_batch_size}, acc={config.gradient_accumulation_steps})")
    print("Stage 1: Package and CI foundation — no model training yet.")
    print("Done.")


if __name__ == "__main__":
    main()