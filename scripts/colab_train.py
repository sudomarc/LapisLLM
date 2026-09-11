#!/usr/bin/env python3
"""Non-interactive Colab training entry point for LapisLLM.

One command runs the complete training pipeline without waiting for notebook
input. The actual corpus, tokenizer, model, and optimizer implementations remain
in their dedicated Lapis modules.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "colab.yaml"
CORPUS = ROOT / "training_data" / "colab_pretrain.txt"
MANIFEST = ROOT / "training_data" / "colab_pretrain_manifest.json"
HISTORY = ROOT / "training_history"
CHECKPOINTS = ROOT / "checkpoints" / "colab-runs"
LATEST_CHECKPOINT = ROOT / "checkpoints" / "latest.pt"
HEARTBEAT_SECONDS = 10.0
CORPUS_BUILDER_MARKER = "HF_HUB_DISABLE_XET"
METRIC_RE = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<loss>[0-9.eE+-]+).*?"
    r"ppl=(?P<ppl>[0-9.eE+-]+).*?lr=(?P<lr>[0-9.eE+-]+)"
)
TOKEN_RE = re.compile(r"tokens=(?P<tokens>[0-9,]+)")
DEFAULT_PROMPTS = (
    "Explain a transformer in simple terms.",
    "Write a Python function that reverses a string.",
    "Explique les réseaux de neurones.",
)


def phase(name: str, message: str) -> None:
    print(f"[{name}] {message}", flush=True)


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def get_github_token() -> str | None:
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "LAPIS_GITHUB_TOKEN"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    try:
        from google.colab import userdata  # type: ignore
    except (ImportError, ModuleNotFoundError):
        return None
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "LAPIS_GITHUB_TOKEN"):
        try:
            value = userdata.get(name)
        except Exception:
            continue
        if value and str(value).strip():
            return str(value).strip()
    return None


@contextmanager
def github_auth_env(token: str | None):
    """Use an ephemeral askpass helper; never place a token in argv or logs."""
    if not token:
        yield {"GIT_TERMINAL_PROMPT": "0"}
        return
    with tempfile.TemporaryDirectory(prefix="lapis-git-auth-") as tmp:
        askpass = Path(tmp) / "askpass.sh"
        askpass.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *[Uu]sername*) printf '%s\\n' 'x-access-token' ;;\n"
            "  *) printf '%s\\n' \"$LAPIS_GIT_TOKEN\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        yield {
            "GIT_ASKPASS": str(askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "LAPIS_GIT_TOKEN": token,
        }


def run(
    command: list[str],
    *,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    full_env["PYTHONUNBUFFERED"] = "1"
    if env:
        full_env.update(env)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=full_env,
        check=True,
        text=True,
        capture_output=capture,
    )


def ensure_dependencies() -> None:
    required = ("torch", "yaml", "datasets", "tokenizers")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        phase("DEPENDENCIES", "READY")
        return
    phase("DEPENDENCIES", f"INSTALLING | {', '.join(missing)}")
    run([sys.executable, "-m", "pip", "install", "-e", ".[data]"])
    phase("DEPENDENCIES", "READY")


def resolve_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def target_steps() -> int:
    import yaml

    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    steps = int(data["training"]["max_steps"])
    if steps < 1:
        raise ValueError("configs/colab.yaml training.max_steps must be >= 1")
    return steps


def corpus_valid(max_chars: int) -> bool:
    if not CORPUS.is_file() or CORPUS.stat().st_size <= 0 or not MANIFEST.is_file():
        return False
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        int(manifest.get("actual_chars", 0)) > 0
        and int(manifest.get("max_chars", 0)) == max_chars
        and bool(manifest.get("output"))
        and CORPUS_BUILDER_MARKER in manifest
    )


def prepare_corpus(max_chars: int, max_records_per_source: int) -> None:
    phase("CORPUS", "CHECK")
    if corpus_valid(max_chars):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        phase(
            "CORPUS",
            f"REUSED | size={CORPUS.stat().st_size / 1024 / 1024:.1f} MiB | chars={manifest['actual_chars']:,}",
        )
        return

    phase("CORPUS", f"BUILD | target={max_chars:,} chars")
    command = [
        sys.executable,
        "scripts/build_colab_corpus.py",
        "--output",
        str(CORPUS.relative_to(ROOT)),
        "--manifest",
        str(MANIFEST.relative_to(ROOT)),
        "--max-chars",
        str(max_chars),
    ]
    if max_records_per_source:
        command.extend(["--max-records-per-source", str(max_records_per_source)])
    run(command)
    if not corpus_valid(max_chars):
        raise RuntimeError("Corpus build completed but validation failed.")
    phase("CORPUS", f"READY | size={CORPUS.stat().st_size / 1024 / 1024:.1f} MiB")


def completed_runs() -> set[int]:
    result: set[int] = set()
    if not HISTORY.is_dir():
        return result
    for summary in HISTORY.glob("run-*/summary.json"):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            if data.get("status") == "completed":
                result.add(int(data["run_number"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return result


def verify_checkpoint(path: Path, expected_steps: int) -> None:
    import torch

    phase("CHECKPOINT", f"VERIFY | {path.relative_to(ROOT)}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Checkpoint missing or empty: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in state or "config" not in state:
        raise RuntimeError("Checkpoint is missing model/config metadata.")
    if int(state.get("step", -1)) != expected_steps:
        raise RuntimeError(
            f"Checkpoint step mismatch: got {state.get('step')}, expected {expected_steps}."
        )
    tokenizer = path.parent / "tokenizer" / "tokenizer.json"
    if not tokenizer.is_file() or tokenizer.stat().st_size <= 0:
        raise RuntimeError(f"Checkpoint tokenizer missing or empty: {tokenizer}")
    if state.get("tokenizer_version") is None:
        raise RuntimeError("Checkpoint tokenizer metadata is missing.")
    phase("CHECKPOINT", f"VERIFIED | {path.stat().st_size / 1024 / 1024:.1f} MiB")


def stream_training(
    command: list[str],
    *,
    run_number: int,
    runs: int,
    steps: int,
) -> dict[str, float]:
    started = time.monotonic()
    latest: dict[str, float] = {
        "step": 0,
        "loss": float("nan"),
        "ppl": float("nan"),
        "lr": float("nan"),
        "tokens": 0,
    }
    tail: list[str] = []
    phase("TRAINING", f"START | run={run_number}/{runs} | steps={steps:,}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    last_output = time.monotonic()
    stop = threading.Event()

    def heartbeat() -> None:
        nonlocal last_output
        while not stop.wait(HEARTBEAT_SECONDS):
            now = time.monotonic()
            if now - last_output < HEARTBEAT_SECONDS:
                continue
            elapsed = now - started
            step = int(latest["step"])
            rate = step / elapsed if elapsed else 0.0
            eta = (steps - step) / rate if rate > 0 else None
            phase(
                "TRAINING",
                f"HEARTBEAT | run={run_number}/{runs} | step={step:,}/{steps:,} | "
                f"elapsed={duration(elapsed)} | ETA={duration(eta)}",
            )
            last_output = now

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            print(line, flush=True)
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
            last_output = time.monotonic()

            token_match = TOKEN_RE.search(line)
            if token_match:
                latest["tokens"] = float(token_match.group("tokens").replace(",", ""))
            metric = METRIC_RE.search(line)
            if metric:
                latest["step"] = float(metric.group("step"))
                latest["loss"] = float(metric.group("loss"))
                latest["ppl"] = float(metric.group("ppl"))
                latest["lr"] = float(metric.group("lr"))
                elapsed = max(0.001, time.monotonic() - started)
                rate = latest["step"] / elapsed
                eta = (steps - latest["step"]) / rate if rate > 0 else None
                phase(
                    "TRAINING",
                    f"run={run_number}/{runs} | step={int(latest['step']):,}/{steps:,} | "
                    f"loss={latest['loss']:.4f} | ppl={latest['ppl']:.2f} | "
                    f"tok/s={latest['tokens'] / elapsed:,.0f} | ETA={duration(eta)}",
                )
    finally:
        stop.set()
        thread.join(timeout=HEARTBEAT_SECONDS)
        if process.stdout is not None:
            process.stdout.close()

    code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"Training failed with exit code {code}.\nLast output:\n" + "\n".join(tail[-12:])
        )
    phase("TRAINING", f"COMPLETE | run={run_number}/{runs} | elapsed={duration(time.monotonic() - started)}")
    return latest


def preview(checkpoint: Path, device: str) -> list[dict[str, str]]:
    phase("PREVIEW", "START | post-training only")
    samples: list[dict[str, str]] = []
    for prompt in DEFAULT_PROMPTS:
        result = run(
            [
                sys.executable,
                "scripts/generate.py",
                "--checkpoint",
                str(checkpoint.relative_to(ROOT)),
                "--prompt",
                prompt,
                "--max-new-tokens",
                "32",
                "--device",
                device,
            ],
            capture=True,
        )
        completion = result.stdout.strip()
        print(f"\nPrompt: {prompt}\nLapis: {completion or '<EMPTY>'}", flush=True)
        samples.append({"prompt": prompt, "completion": completion})
    phase("PREVIEW", "COMPLETE")
    return samples


def write_history(
    run_number: int,
    runs: int,
    checkpoint: Path,
    metrics: dict[str, float],
    samples: list[dict[str, str]],
    started: float,
    device: str,
) -> None:
    run_id = f"run-{run_number:03d}"
    run_dir = HISTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "run_number": run_number,
        "total_runs": runs,
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "device": device,
        "steps": int(metrics.get("step", 0)),
        "loss": metrics.get("loss"),
        "perplexity": metrics.get("ppl"),
        "learning_rate": metrics.get("lr"),
        "tokens_seen": int(metrics.get("tokens", 0)),
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "dataset_manifest": str(MANIFEST.relative_to(ROOT)),
        "samples": samples,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def publish_outputs(token: str | None) -> None:
    phase("PUBLISH", "START | checkpoint + training history")
    auth: dict[str, str]
    with github_auth_env(token) as auth:
        run([sys.executable, "-m", "scripts.publish_checkpoint"], env=auth)
        status = run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            capture=True,
            env=auth,
        ).stdout.splitlines()
        history = [line for line in status if line[3:].replace("\\", "/").startswith("training_history/")]
        unrelated = [line for line in status if line not in history]
        if unrelated:
            raise RuntimeError(
                "Refusing to publish training history because unrelated local changes exist:\n"
                + "\n".join(unrelated)
            )
        if history:
            run(["git", "add", "training_history"], env=auth)
            staged = run(
                ["git", "diff", "--cached", "--name-only"],
                capture=True,
                env=auth,
            ).stdout.splitlines()
            staged_history = [
                path for path in staged if path.replace("\\", "/").startswith("training_history/")
            ]
            if staged_history:
                run(["git", "commit", "-m", "chore: save Colab training history"], env=auth)
                run(["git", "push", "origin", "main"], env=auth)
    phase("PUBLISH", f"COMPLETE | latest={LATEST_CHECKPOINT}")


def run_one(
    run_number: int,
    runs: int,
    device: str,
    max_chars: int,
    max_records_per_source: int,
    monitor_interval: int,
) -> None:
    started = time.monotonic()
    prepare_corpus(max_chars, max_records_per_source)
    steps = target_steps()
    run_dir = CHECKPOINTS / f"run-{run_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"

    command = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(CONFIG.relative_to(ROOT)),
        "--data",
        str(CORPUS.relative_to(ROOT)),
        "--checkpoint",
        str(checkpoint.relative_to(ROOT)),
        "--device",
        device,
        "--epochs",
        "1000",
        "--monitor-interval",
        str(monitor_interval),
    ]
    metrics = stream_training(command, run_number=run_number, runs=runs, steps=steps)
    metrics["device"] = device
    verify_checkpoint(checkpoint, steps)
    samples = preview(checkpoint, device)
    write_history(run_number, runs, checkpoint, metrics, samples, started, device)
    publish_outputs(get_github_token())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LapisLLM Colab training")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=200_000_000)
    parser.add_argument("--max-records-per-source", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=0,
        help="Enable expensive in-training samples only for debugging; 0 disables them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.max_chars < 1:
        raise SystemExit("--max-chars must be >= 1")
    if args.max_records_per_source < 0:
        raise SystemExit("--max-records-per-source must be >= 0")
    if args.monitor_interval < 0:
        raise SystemExit("--monitor-interval must be >= 0")

    print("\nLAPIS COLAB TRAINING", flush=True)
    print("non-interactive | no notebook input | no training-time generation by default", flush=True)
    print(
        f"runs={args.runs} | max_chars={args.max_chars:,} | "
        f"monitor_interval={args.monitor_interval}",
        flush=True,
    )

    ensure_dependencies()
    device = resolve_device(args.device)
    if device == "cuda":
        import torch
        phase("GPU", f"READY | {torch.cuda.get_device_name(0)}")
    else:
        phase("GPU", "CPU")

    done = completed_runs()
    remaining = [number for number in range(1, args.runs + 1) if number not in done]
    phase("PLAN", f"requested={args.runs} | completed={len(done)} | remaining={len(remaining)}")

    for run_number in remaining:
        run_one(
            run_number=run_number,
            runs=args.runs,
            device=device,
            max_chars=args.max_chars,
            max_records_per_source=args.max_records_per_source,
            monitor_interval=args.monitor_interval,
        )

    phase("DONE", f"all requested runs completed | latest={LATEST_CHECKPOINT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
