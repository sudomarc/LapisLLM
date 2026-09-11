#!/usr/bin/env python3
"""Canonical non-interactive Colab training orchestrator for LapisLLM.

The orchestrator owns preflight, corpus preparation, run management, bounded
observability, checkpoint verification, post-training previews, history, and
GitHub publication. The optimizer loop remains in ``scripts.train``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "colab.yaml"
SMOKE_CONFIG = ROOT / "configs" / "local-dev.yaml"
CORPUS = ROOT / "training_data" / "colab_pretrain.txt"
MANIFEST = ROOT / "training_data" / "colab_pretrain_manifest.json"
HISTORY = ROOT / "training_history"
CHECKPOINTS = ROOT / "checkpoints" / "colab-runs"
HEARTBEAT_SECONDS = 10.0
METRIC_RE = re.compile(
    r"step=(?P<step>\d+)\s+loss=(?P<loss>[0-9.eE+-]+).*?"
    r"ppl=(?P<ppl>[0-9.eE+-]+).*?lr=(?P<lr>[0-9.eE+-]+)"
)
TOKEN_RE = re.compile(r"tokens=(?P<tokens>[0-9,]+)")
CORPUS_CONTRACT_VERSION = "lapis-colab-corpus-v2"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def phase(name: str, message: str) -> None:
    print(f"[{name}] {message}", flush=True)


def format_duration(seconds: float | None) -> str:
    seconds = max(0, int(seconds or 0))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run observable LapisLLM Colab training")
    parser.add_argument("--runs", type=int, default=1, help="Total number of runs")
    parser.add_argument("--resume", action="store_true", help="Resume at the first incomplete run")
    parser.add_argument("--smoke-test", action="store_true", help="Run one CPU-only end-to-end smoke test")
    parser.add_argument("--monitor-interval", type=int, default=0, help="In-training generation interval; 0 disables generation")
    parser.add_argument("--monitor-sample-tokens", type=int, default=32)
    parser.add_argument("--monitor-prompts", default=None)
    parser.add_argument("--max-chars", type=int, default=200_000_000)
    parser.add_argument("--max-records-per-source", type=int, default=0)
    parser.add_argument("--heartbeat-seconds", type=float, default=HEARTBEAT_SECONDS)
    parser.add_argument("--no-push", action="store_true", help="Do not push history/checkpoint outputs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv if argv is not None else [])


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


def run_command(command: list[str], *, label: str | None = None) -> None:
    if label:
        phase(label, "START")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if label:
        phase(label, "COMPLETE")


def ensure_dependencies() -> None:
    required = ("torch", "yaml", "datasets", "tokenizers", "typer")
    import importlib.util

    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if not missing:
        phase("DEPENDENCIES", "COMPLETE | required packages already installed")
        return
    phase("DEPENDENCIES", f"INSTALLING | missing={', '.join(missing)}")
    run_command([sys.executable, "-m", "pip", "install", "-e", ".[data]"])
    phase("DEPENDENCIES", "COMPLETE")


def check_gpu(requested: str) -> str:
    import torch

    phase("GPU CHECK", "START")
    if requested == "cpu":
        phase("GPU CHECK", "CPU explicitly selected")
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        phase("GPU CHECK", f"READY | device=cuda | name={name}")
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    phase("GPU CHECK", "NO CUDA DEVICE | falling back to cpu")
    return "cpu"


def corpus_valid(max_chars: int) -> bool:
    if not CORPUS.is_file() or CORPUS.stat().st_size <= 0 or not MANIFEST.is_file():
        return False
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(manifest, dict):
        return False
    try:
        return (
            manifest.get("contract_version") == CORPUS_CONTRACT_VERSION
            and int(manifest.get("max_chars", 0)) == max_chars
            and int(manifest.get("actual_chars", 0)) > 0
            and bool(manifest.get("output"))
            and bool(manifest.get("sources"))
        )
    except (TypeError, ValueError, OverflowError):
        return False


def prepare_corpus(args: argparse.Namespace) -> None:
    phase("CORPUS", "START")
    effective_max_chars = 50_000 if args.smoke_test else args.max_chars
    if corpus_valid(effective_max_chars):
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            manifest = {}
        size = CORPUS.stat().st_size
        phase(
            "CORPUS",
            f"REUSED | size={size / 1024 / 1024:.1f} MiB | chars={manifest.get('actual_chars', 0):,} | "
            f"contract={manifest.get('contract_version')}",
        )
        phase("MANIFEST", f"COMPLETE | {MANIFEST}")
        return

    command = [
        sys.executable,
        "scripts/build_colab_corpus.py",
        "--output", str(CORPUS.relative_to(ROOT)),
        "--manifest", str(MANIFEST.relative_to(ROOT)),
        "--max-chars", str(effective_max_chars),
    ]
    if args.max_records_per_source > 0:
        command += ["--max-records-per-source", str(args.max_records_per_source)]
    if args.smoke_test:
        command += ["--source", "wikipedia"]
    run_command(command)
    if not corpus_valid(effective_max_chars):
        raise RuntimeError("Corpus/manifest verification failed after corpus build.")
    phase("CORPUS", f"COMPLETE | {CORPUS.stat().st_size / 1024 / 1024:.1f} MiB")
    phase("MANIFEST", f"COMPLETE | {MANIFEST}")


def completed_run_numbers() -> set[int]:
    numbers: set[int] = set()
    if not HISTORY.is_dir():
        return numbers
    for summary in HISTORY.glob("run-*/summary.json"):
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            number = int(data["run_number"])
            if data.get("status", "completed") == "completed":
                numbers.add(number)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
            continue
    return numbers


def monitor_child(
    process: subprocess.Popen[str],
    *,
    run_number: int,
    total_runs: int,
    target_steps: int,
    started: float,
    heartbeat_seconds: float,
) -> dict[str, float]:
    latest: dict[str, float] = {
        "step": 0,
        "loss": float("nan"),
        "ppl": float("nan"),
        "lr": float("nan"),
        "tokens": 0,
    }
    last_output = time.monotonic()
    stop = threading.Event()

    def heartbeat() -> None:
        nonlocal last_output
        while not stop.wait(heartbeat_seconds):
            now = time.monotonic()
            if now - last_output < heartbeat_seconds:
                continue
            elapsed = now - started
            step = int(latest["step"])
            speed = step / elapsed if elapsed > 0 else 0.0
            remaining = max(0, target_steps - step)
            eta = remaining / speed if speed > 0 else None
            phase(
                "TRAINING",
                f"HEARTBEAT | run={run_number}/{total_runs} | child active | "
                f"step={step:,}/{target_steps:,} | elapsed={format_duration(elapsed)}"
                + (f" | ETA={format_duration(eta)}" if eta is not None else ""),
            )
            last_output = now

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    assert process.stdout is not None
    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue
            print(line, flush=True)
            last_output = time.monotonic()
            metric = METRIC_RE.search(line)
            token_match = TOKEN_RE.search(line)
            if token_match:
                latest["tokens"] = float(token_match.group("tokens").replace(",", ""))
            if metric:
                latest["step"] = float(metric.group("step"))
                latest["loss"] = float(metric.group("loss"))
                latest["ppl"] = float(metric.group("ppl"))
                latest["lr"] = float(metric.group("lr"))
                elapsed = max(0.001, time.monotonic() - started)
                speed = latest["step"] / elapsed
                remaining = max(0.0, target_steps - latest["step"])
                eta = remaining / speed if speed > 0 else None
                tokens = int(latest["tokens"])
                token_speed = tokens / elapsed if tokens else 0.0
                phase(
                    "TRAINING",
                    f"run={run_number}/{total_runs} | step={int(latest['step']):,}/{target_steps:,} | "
                    f"progress={latest['step'] / target_steps * 100:.2f}% | loss={latest['loss']:.4f} | "
                    f"ppl={latest['ppl']:.2f} | lr={latest['lr']:.6g} | tokens={tokens:,} | "
                    f"tok/s={token_speed:,.0f}"
                    + (f" | ETA={format_duration(eta)}" if eta is not None else ""),
                )
            if "Checkpoint saved:" in line:
                phase("CHECKPOINT", "SAVED | trainer reported checkpoint write complete")
    finally:
        stop.set()
        thread.join(timeout=max(1.0, heartbeat_seconds))
        process.stdout.close()
    code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"[TRAINING] ERROR | run={run_number}/{total_runs} | exit_code={code} | "
            f"last_step={int(latest['step'])}"
        )
    return latest


def target_steps(config: Path) -> int:
    import yaml

    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    steps = int(data["training"]["max_steps"])
    if steps < 1:
        raise ValueError("training.max_steps must be >= 1")
    return steps


def write_failure_history(run_number: int, total_runs: int, error: Exception, started: float) -> None:
    run_id = f"run-{run_number:03d}"
    run_dir = HISTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "run_number": run_number,
        "total_runs": total_runs,
        "status": "failed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - started,
        "error": str(error),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "error.log").write_text(str(error) + "\n", encoding="utf-8")


def verify_checkpoint(checkpoint: Path, expected_steps: int) -> None:
    phase("CHECKPOINT", f"VERIFYING | path={checkpoint}")
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise RuntimeError(f"Checkpoint missing or empty: {checkpoint}")
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = int(state.get("step", -1))
    if step != expected_steps:
        raise RuntimeError(f"Checkpoint step mismatch: got {step}, expected {expected_steps}")
    if "model_state_dict" not in state or "config" not in state:
        raise RuntimeError("Checkpoint is missing model/config metadata")
    tokenizer = checkpoint.parent / "tokenizer" / "tokenizer.json"
    if not tokenizer.is_file() or tokenizer.stat().st_size <= 0:
        raise RuntimeError(f"Checkpoint tokenizer missing or empty: {tokenizer}")
    if state.get("tokenizer_version") is None:
        raise RuntimeError("Checkpoint tokenizer metadata missing")
    phase(
        "CHECKPOINT",
        f"VERIFIED | size={checkpoint.stat().st_size / 1024 / 1024:.1f} MiB | "
        f"step={step:,} | tokenizer=OK",
    )


def generate_preview(checkpoint: Path, device: str, prompts: tuple[str, ...]) -> list[dict[str, str]]:
    phase("CHATBOT PREVIEW", "START")
    samples: list[dict[str, str]] = []
    for prompt in prompts:
        command = [
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
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        completion = result.stdout.strip()
        print(f"\nPrompt: {prompt}\nLapisLLM: {completion or '<EMPTY>'}", flush=True)
        samples.append({"prompt": prompt, "completion": completion})
    phase("CHATBOT PREVIEW", "COMPLETE")
    return samples


def write_history(
    run_number: int,
    total_runs: int,
    checkpoint: Path,
    metrics: dict[str, float],
    samples: list[dict[str, str]],
    started: float,
    config: Path,
) -> Path:
    run_id = f"run-{run_number:03d}"
    run_dir = HISTORY / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    duration = time.monotonic() - started
    summary = {
        "run_id": run_id,
        "run_number": run_number,
        "total_runs": total_runs,
        "status": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "steps": int(metrics.get("step", 0)),
        "loss": metrics.get("loss"),
        "perplexity": metrics.get("ppl"),
        "learning_rate": metrics.get("lr"),
        "tokens_seen": int(metrics.get("tokens", 0)),
        "duration_seconds": duration,
        "device": metrics.get("device", "unknown"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "config": str(config),
        "dataset_manifest": str(MANIFEST),
        "samples": samples,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {"run_id": run_id, "checkpoint": str(checkpoint), "manifest": str(MANIFEST)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "samples.md").write_text(
        "\n".join(
            [
                f"# {run_id}",
                "",
                *[
                    f"## {item['prompt']}\n\n{item['completion'] or '<EMPTY>'}\n"
                    for item in samples
                ],
            ]
        ),
        encoding="utf-8",
    )
    phase("HISTORY", f"COMPLETE | {run_dir}")
    return run_dir


def train_one_run(
    run_number: int,
    total_runs: int,
    args: argparse.Namespace,
    device: str,
    config: Path,
) -> dict[str, float]:
    run_dir = CHECKPOINTS / f"run-{run_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "checkpoint.pt"
    monitor_log = run_dir / "learning_monitor.jsonl"
    steps = target_steps(config)
    phase("RUN", f"{run_number} / {total_runs}")
    phase("RUN", f"target_steps={steps:,} | checkpoint={checkpoint}")
    phase("TRAINING", "START | generation monitor is disabled by default")

    command = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(config.relative_to(ROOT)),
        "--data",
        str(CORPUS.relative_to(ROOT)),
        "--checkpoint",
        str(checkpoint.relative_to(ROOT)),
        "--device",
        device,
        "--epochs",
        "1000",
        "--monitor-interval",
        str(args.monitor_interval),
        "--monitor-sample-tokens",
        str(args.monitor_sample_tokens),
        "--monitor-log",
        str(monitor_log.relative_to(ROOT)),
    ]
    if args.monitor_prompts:
        command += ["--monitor-prompts", args.monitor_prompts]

    started = time.monotonic()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    try:
        metrics = monitor_child(
            process,
            run_number=run_number,
            total_runs=total_runs,
            target_steps=steps,
            started=started,
            heartbeat_seconds=args.heartbeat_seconds,
        )
        metrics["device"] = device
        phase(
            "TRAINING",
            f"COMPLETE | run={run_number}/{total_runs} | elapsed={format_duration(time.monotonic() - started)}",
        )
        verify_checkpoint(checkpoint, steps)
        prompts = tuple(
            item.strip()
            for item in (
                args.monitor_prompts
                or "Explain a transformer.||Write a Python function to reverse a string.||Explique les réseaux de neurones."
            ).split("||")
            if item.strip()
        )
        samples = generate_preview(checkpoint, device, prompts)
        write_history(run_number, total_runs, checkpoint, metrics, samples, started, config)
        publish_checkpoint_locally(checkpoint)
        return metrics
    except Exception as exc:
        write_failure_history(run_number, total_runs, exc, started)
        phase("RUN", f"FAILED | run={run_number}/{total_runs} | error={exc}")
        raise


def publish_checkpoint_locally(checkpoint: Path) -> None:
    phase("PUBLISH", "CHECKPOINT | preparing canonical latest.pt")
    command = [
        sys.executable,
        "-m",
        "scripts.publish_checkpoint",
        str(checkpoint.relative_to(ROOT)),
        "--no-push",
    ]
    run_command(command)
    phase("PUBLISH", "CHECKPOINT | canonical paths updated")


def git_push(token: str | None, smoke: bool) -> bool:
    if smoke:
        phase("GITHUB", "SKIPPED | smoke test")
        return True

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout.splitlines()
    protected = [
        line
        for line in status
        if line[3:].lstrip().replace("\\", "/")
        and not (
            line[3:].lstrip().replace("\\", "/").startswith("training_history/")
            or line[3:].lstrip().replace("\\", "/").startswith("checkpoints/")
        )
    ]
    if protected:
        raise RuntimeError(
            "Refusing GitHub push because unrelated local changes exist:\n"
            + "\n".join(protected)
        )

    canonical = (
        "training_history/",
        "checkpoints/latest.pt",
        "checkpoints/tokenizer/",
    )
    pushable = [
        line[3:].lstrip().replace("\\", "/")
        for line in status
        if line[3:].lstrip().replace("\\", "/").startswith(canonical)
    ]
    if pushable:
        phase("GITHUB", f"COMMIT | files={len(pushable)}")
        subprocess.run(
            [
                "git",
                "add",
                "training_history",
                "checkpoints/latest.pt",
                "checkpoints/tokenizer",
            ],
            cwd=ROOT,
            check=True,
            timeout=30,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        if staged:
            subprocess.run(
                ["git", "commit", "-m", "chore: record Colab training outputs"],
                cwd=ROOT,
                check=True,
                timeout=60,
            )

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "PYTHONUNBUFFERED": "1"}
    if token:
        import stat
        import tempfile

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
            env.update({"GIT_ASKPASS": str(askpass), "LAPIS_GIT_TOKEN": token})
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=ROOT,
                env=env,
                check=True,
                timeout=120,
            )
    else:
        credential_helper = subprocess.run(
            ["git", "config", "--get", "credential.helper"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        ).stdout.strip()
        if not credential_helper:
            phase("GITHUB", "AUTHENTICATION MISSING | failing before modifying Git state")
            raise RuntimeError("No GitHub credentials available for non-interactive Colab push.")
        phase("GITHUB", "NO TOKEN | attempting existing non-interactive Git credentials")
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=ROOT,
            env=env,
            check=True,
            timeout=120,
        )

    phase("GITHUB", "PUSH COMPLETE")
    return True


def main() -> int:
    args = parse_args(sys.argv[1:])
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    if args.monitor_interval < 0:
        raise SystemExit("--monitor-interval must be >= 0")
    if args.monitor_sample_tokens < 1:
        raise SystemExit("--monitor-sample-tokens must be >= 1")
    if args.heartbeat_seconds <= 0:
        raise SystemExit("--heartbeat-seconds must be > 0")
    if args.max_chars < 1:
        raise SystemExit("--max-chars must be >= 1")
    if args.max_records_per_source < 0:
        raise SystemExit("--max-records-per-source must be >= 0")

    if args.smoke_test:
        args.runs = 1
        args.no_push = True
        args.device = "cpu"

    banner("LAPISLLM COLAB TRAINING")
    phase("INITIALIZATION", f"root={ROOT}")
    ensure_dependencies()
    device = check_gpu(args.device)
    prepare_corpus(args)
    phase("INITIALIZATION", "COMPLETE")

    completed = completed_run_numbers() if args.resume else set()
    remaining = [number for number in range(1, args.runs + 1) if number not in completed]
    phase(
        "PLAN",
        f"runs requested={args.runs} | completed={len(completed)} | remaining={len(remaining)}",
    )
    if args.resume:
        phase(
            "RESUME",
            f"previous completed={len(completed)} | remaining={len(remaining)} | "
            f"next={remaining[0] if remaining else 'none'}",
        )

    config = SMOKE_CONFIG if args.smoke_test else CONFIG
    for run_number in remaining:
        train_one_run(run_number, args.runs, args, device, config)

    banner("FINAL SUMMARY")
    completed_summaries = []
    for summary_file in sorted(HISTORY.glob("run-*/summary.json")):
        try:
            data = json.loads(summary_file.read_text(encoding="utf-8"))
            if data.get("status") == "completed" and int(data.get("run_number", 0)) <= args.runs:
                completed_summaries.append(data)
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            continue
    print(f"GPU              : {device}", flush=True)
    print(f"Runs requested   : {args.runs}", flush=True)
    print(f"Runs completed   : {len(completed_summaries)}", flush=True)
    if completed_summaries:
        final = completed_summaries[-1]
        print(f"Final loss       : {final.get('loss')}", flush=True)
        print(f"Final ppl        : {final.get('perplexity')}", flush=True)
        print(f"Tokens seen      : {int(final.get('tokens_seen', 0)):,}", flush=True)
    print(f"History          : {HISTORY}", flush=True)
    print(f"Latest checkpoint: {ROOT / 'checkpoints' / 'latest.pt'}", flush=True)

    if args.no_push:
        phase("GITHUB", "SKIPPED | --no-push")
    else:
        token = get_github_token()
        phase("GITHUB", "TOKEN READY" if token else "TOKEN NOT SET")
        git_push(token, args.smoke_test)

    banner("LAPISLLM TRAINING COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
