"""Live training observability for Lapis.

The learning monitor records scalar metrics and periodic generations from the
current in-memory model so training logs show both optimization progress and
observable changes in model behavior.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Iterable

import torch

from scripts.generate import sample_next_token

DEFAULT_PROMPTS = (
    "Machine learning is",
    "A neural network can",
    "The transformer architecture",
    "Language models learn",
)


class LearningMonitor:
    """Print and persist training metrics plus deterministic-ish sample generations."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        device: torch.device,
        output_path: str | Path,
        interval: int = 500,
        sample_tokens: int = 48,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.95,
        prompts: Iterable[str] = DEFAULT_PROMPTS,
    ) -> None:
        if interval < 1:
            raise ValueError("monitor interval must be at least 1")
        if sample_tokens < 1:
            raise ValueError("monitor sample tokens must be at least 1")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.output_path = Path(output_path)
        self.interval = interval
        self.sample_tokens = sample_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.prompts = tuple(prompt for prompt in prompts if prompt.strip())
        if not self.prompts:
            raise ValueError("learning monitor requires at least one prompt")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def should_sample(self, step: int) -> bool:
        return step == 1 or step % self.interval == 0

    def log_metrics(
        self,
        *,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        tokens_seen: int,
    ) -> None:
        perplexity = math.exp(loss) if loss < 20 else float("inf")
        record = {
            "type": "metrics",
            "timestamp": time.time(),
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "tokens_seen": tokens_seen,
        }
        self._write(record)

        print("\n" + "─" * 70)
        print("LEARNING MONITOR")
        print(
            f"step={step:05d}  loss={loss:.4f}  "
            f"ppl={perplexity:.2f}  lr={learning_rate:.6g}  "
            f"tokens={tokens_seen:,}"
        )

    def sample(self, *, step: int) -> None:
        was_training = self.model.training
        self.model.eval()
        samples: list[dict[str, str]] = []

        try:
            with torch.inference_mode():
                for prompt in self.prompts:
                    token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                    if not token_ids:
                        token_ids = [self.tokenizer.bos_id]
                    ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                    generated: list[int] = []
                    for _ in range(self.sample_tokens):
                        context = ids[:, -self.model.max_position_embeddings :]
                        logits, _ = self.model(context)
                        next_id = sample_next_token(
                            logits[:, -1, :],
                            self.temperature,
                            self.top_k,
                            self.top_p,
                        )
                        ids = torch.cat([ids, next_id], dim=1)
                        token_id = int(next_id.item())
                        if token_id == self.tokenizer.eos_id:
                            break
                        generated.append(token_id)

                    continuation = self.tokenizer.decode(
                        generated, skip_special_tokens=True
                    )
                    samples.append({"prompt": prompt, "completion": continuation})
        finally:
            if was_training:
                self.model.train()

        record = {
            "type": "samples",
            "timestamp": time.time(),
            "step": step,
            "samples": samples,
        }
        self._write(record)

        print("\nWHAT LAPIS IS LEARNING")
        for item in samples:
            print(f"\nPrompt : {item['prompt']}")
            print(f"Lapis  : {item['completion'] or '<EOS>'}")
        print("─" * 70)

    def observe(
        self,
        *,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float,
        tokens_seen: int,
    ) -> None:
        if not self.should_sample(step):
            return
        self.log_metrics(
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            tokens_seen=tokens_seen,
        )
        self.sample(step=step)

    def _write(self, record: dict) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
