#!/usr/bin/env python3
"""One-command Colab training orchestrator for LapisLLM.

Authentication for GitHub is deliberately kept out of command lines and remotes.
The preferred source in Google Colab is a secret named ``GITHUB_TOKEN``.

Flow:
    clone/reuse checkout -> authenticate -> pull -> install -> validate -> tests
    -> build/reuse corpus -> run verified training runs -> write history
    -> commit history -> push a dedicated branch

Generated artifacts such as ``training_data/`` and ``checkpoints/`` stay local.
Only lightweight training history is committed.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import stat
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/sudomarc/LapisLLM.git"
REPO_ROOT = Path("/content/LapisLLM")
HISTORY_ROOT = REPO_ROOT / "training_history"
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints" / "colab-runs"
TRAINING_DATA_ROOT = REPO_ROOT / "training_data"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tiny.yaml"
DEFAULT_DATA = TRAINING_DATA_ROOT / "combined.txt"
REQUIRED_FILES = (
    "pyproject.toml",
    "configs/tiny.yaml",
    "scripts/train.py",
    "scripts/fetch_training_data.py",
)


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run a command without ever printing environment variables."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("$", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=capture,
        env=merged_env,
    )


def get_github_token() -> str | None:
    """Resolve the GitHub PAT from Colab Secret first, then environment aliases."""
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
def github_auth_env(token: str):
    """Expose a token to Git through a short-lived ``GIT_ASKPASS`` helper."""
    if not token:
        yield {}
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


def git_run(
    cmd: list[str],
    *,
    token: str | None = None,
    check: bool = True,
    capture: bool = False,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run a Git command with optional ephemeral token authentication."""
    with github_auth_env(token or "") as auth_env:
        return run(cmd, check=check, capture=capture, env=auth_env, cwd=cwd)


def prompt_int(label: str, default: int, minimum: int = 0) -> int:
    raw = input(f"{label} [{default}] : ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def bootstrap_repo(token: str | None) -> None:
    print("\n" + "=" * 70)
    print("LAPISLLM — INITIALISATION")
    print("=" * 70)
    if not REPO_ROOT.exists():
        print("Clonage du dépôt...")
        git_run(
            [
                "git",
                "-c",
                "http.version=HTTP/1.1",
                "-c",
                "http.lowSpeedLimit=1000",
                "-c",
                "http.lowSpeedTime=30",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--branch",
                "main",
                REPO_URL,
                str(REPO_ROOT),
            ],
            token=token,
            cwd=Path("/content"),
        )
    elif not (REPO_ROOT / ".git").exists():
        raise RuntimeError(f"{REPO_ROOT} existe mais n'est pas un dépôt Git.")
    print(f"Répertoire : {REPO_ROOT}")


def path_from_status(status_line: str) -> str:
    path = status_line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.replace("\\", "/")


def sync_pull(token: str | None) -> None:
    print("\n" + "=" * 70)
    print("GIT — SYNCHRONISATION")
    print("=" * 70)
    status = git_run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture=True,
    ).stdout.splitlines()
    protected = [
        line
        for line in status
        if not path_from_status(line).startswith("training_history/")
    ]
    if protected:
        raise RuntimeError(
            "Refus de synchroniser : modifications locales hors training_history :\n"
            + "\n".join(protected)
        )

    history_changes = [
        line for line in status if path_from_status(line).startswith("training_history/")
    ]
    if history_changes:
        git_run(["git", "add", "training_history"])
        staged = git_run(
            ["git", "diff", "--cached", "--name-only"],
            capture=True,
        ).stdout.strip()
        if staged:
            git_run(["git", "commit", "-m", "chore: save local training history"])

    git_run(["git", "fetch", "origin", "main"], token=token)
    git_run(["git", "pull", "--rebase", "origin", "main"], token=token)
    print("✅ Git pull terminé.")


def validate_project() -> None:
    missing = [path for path in REQUIRED_FILES if not (REPO_ROOT / path).exists()]
    if missing:
        raise RuntimeError("Fichiers requis absents :\n" + "\n".join(missing))
    for path in REQUIRED_FILES:
        print(f"✅ {path}")


def install_project() -> None:
    run([os.sys.executable, "-m", "pip", "install", "-e", ".[data]", "-q"])
    print("✅ Dépendances installées.")


def validate_tests() -> None:
    run([os.sys.executable, "-m", "pytest", "-q"])
    print("✅ Tests réussis.")


def parse_max_steps(config: Path) -> int:
    text = config.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*max_steps:\s*(\d+)\s*$", text)
    if not match:
        raise RuntimeError(f"Impossible de trouver training.max_steps dans {config}")
    steps = int(match.group(1))
    if steps < 1:
        raise ValueError("training.max_steps doit être >= 1")
    return steps


def build_corpus(max_wiki_articles: int = 40, max_doc_files: int = 20) -> None:
    print("\n" + "=" * 70)
    print("DONNÉES")
    print("=" * 70)
    TRAINING_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if DEFAULT_DATA.exists() and DEFAULT_DATA.stat().st_size >= 100_000:
        print(f"✅ Corpus existant : {DEFAULT_DATA.stat().st_size:,} octets")
        return
    run(
        [
            os.sys.executable,
            "scripts/fetch_training_data.py",
            "--output-dir",
            "training_data",
            "--max-wiki-articles",
            str(max_wiki_articles),
            "--max-doc-files",
            str(max_doc_files),
        ]
    )
    if not DEFAULT_DATA.exists() or DEFAULT_DATA.stat().st_size < 100_000:
        raise RuntimeError("Le corpus n'a pas été généré correctement.")
    print(f"✅ Corpus prêt : {DEFAULT_DATA.stat().st_size:,} octets")


def parse_training_line(line: str) -> tuple[int, float] | None:
    match = re.search(r"step=(\d+)\s+loss=([0-9.eE+-]+)", line)
    if not match:
        return None
    return int(match.group(1)), float(match.group(2))


def train_one_run(
    *,
    run_number: int,
    total_runs: int,
    monitor_interval: int,
    monitor_sample_tokens: int,
    prompts: str | None,
    device: str,
    seed: int,
) -> dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"run-{timestamp}-{uuid.uuid4().hex[:6]}"
    local_dir = CHECKPOINT_ROOT / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = local_dir / "checkpoint.pt"
    monitor_log = local_dir / "learning_monitor.jsonl"
    target_steps = parse_max_steps(DEFAULT_CONFIG)

    cmd = [
        os.sys.executable,
        "scripts/train.py",
        "--config",
        str(DEFAULT_CONFIG.relative_to(REPO_ROOT)),
        "--data",
        str(DEFAULT_DATA.relative_to(REPO_ROOT)),
        "--checkpoint",
        str(checkpoint.relative_to(REPO_ROOT)),
        "--device",
        device,
        "--seed",
        str(seed),
        "--monitor-interval",
        str(monitor_interval),
        "--monitor-sample-tokens",
        str(monitor_sample_tokens),
        "--monitor-log",
        str(monitor_log.relative_to(REPO_ROOT)),
    ]
    if prompts:
        cmd += ["--monitor-prompts", prompts]

    print("\n" + "=" * 70)
    print(f"RUN {run_number}/{total_runs}")
    print("=" * 70)
    print(f"id          {run_id}")
    print(f"config      {DEFAULT_CONFIG.relative_to(REPO_ROOT)}")
    print(f"target      {target_steps:,} optimizer steps")
    print(f"checkpoint  {checkpoint}")
    print(f"seed        {seed}")

    start = time.monotonic()
    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    last_loss: float | None = None
    assert process.stdout is not None
    try:
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if line:
                print(line)
            parsed = parse_training_line(line)
            if parsed:
                _, last_loss = parsed
    finally:
        process.stdout.close()
    return_code = process.wait()
    elapsed = time.monotonic() - start
    if return_code != 0:
        raise RuntimeError(f"Training failed with exit code {return_code}")

    verification = verify_run(checkpoint, monitor_log, target_steps)
    metrics = verification["final_metrics"]
    final_loss = float(metrics.get("loss", last_loss or 0.0))
    final_perplexity = float(metrics.get("perplexity", 2.718281828 ** final_loss))
    tokens_seen = int(metrics.get("tokens_seen", 0))
    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_number": run_number,
        "total_runs": total_runs,
        "recorded_step": verification["recorded_step"],
        "final_loss": final_loss,
        "final_perplexity": final_perplexity,
        "tokens_seen": tokens_seen,
        "elapsed_seconds": elapsed,
        "device": device,
        "checkpoint_size_bytes": verification["checkpoint_size_bytes"],
        "monitor_records": verification["monitor_records"],
        "samples": verification["final_samples"],
    }
    write_history(summary)
    print("✅ RUN VERIFIED")
    return summary


def verify_run(checkpoint: Path, monitor_log: Path, expected_steps: int) -> dict:
    if not checkpoint.exists():
        raise RuntimeError(f"Checkpoint absent : {checkpoint}")
    size = checkpoint.stat().st_size
    if size < 1_000_000:
        raise RuntimeError(f"Checkpoint suspectement petit : {size} octets")

    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    recorded_step = int(state.get("step", -1))
    if recorded_step != expected_steps:
        raise RuntimeError(
            f"Checkpoint à l'étape {recorded_step}, attendu {expected_steps}"
        )

    records: list[dict] = []
    if monitor_log.exists():
        for line in monitor_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    metrics = [item for item in records if item.get("type") == "metrics"]
    samples = [item for item in records if item.get("type") == "samples"]
    return {
        "recorded_step": recorded_step,
        "checkpoint_size_bytes": size,
        "monitor_records": len(records),
        "final_metrics": metrics[-1] if metrics else {},
        "final_samples": samples[-1].get("samples", []) if samples else [],
    }


def write_history(summary: dict) -> Path:
    run_dir = HISTORY_ROOT / summary["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {summary['run_id']}",
        "",
        f"- Date (UTC): `{summary['timestamp_utc']}`",
        f"- Run: `{summary['run_number']}/{summary['total_runs']}`",
        f"- Steps: `{summary['recorded_step']:,}`",
        f"- Final loss: `{summary['final_loss']}`",
        f"- Final perplexity: `{summary['final_perplexity']}`",
        f"- Tokens seen: `{summary['tokens_seen']}`",
        f"- Duration: `{summary['elapsed_seconds']:.1f}s`",
        f"- Device: `{summary['device']}`",
        "",
        "## Final learning samples",
        "",
    ]
    for sample in summary.get("samples", []):
        lines.extend(
            [
                f"### {sample.get('prompt', '<unknown>')}",
                "",
                sample.get("completion") or "<EOS>",
                "",
            ]
        )
    (run_dir / "samples.md").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        print(f"CUDA : True | GPU : {torch.cuda.get_device_name(0)}")
        return "cuda"
    print("CUDA : False | GPU : aucun")
    return "cpu"


def push_history(token: str | None) -> bool:
    print("\n" + "=" * 70)
    print("GIT — PUSH DE L'HISTORIQUE")
    print("=" * 70)
    status = git_run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        capture=True,
    ).stdout.splitlines()
    history_changes = [
        line for line in status if path_from_status(line).startswith("training_history/")
    ]
    protected = [
        line for line in status if not path_from_status(line).startswith("training_history/")
    ]
    if protected:
        raise RuntimeError(
            "Refus de pousser : modifications locales hors training_history :\n"
            + "\n".join(protected)
        )
    if history_changes:
        git_run(["git", "add", "training_history"])
        staged = git_run(
            ["git", "diff", "--cached", "--name-only"],
            capture=True,
        ).stdout.strip()
        if staged:
            git_run(["git", "commit", "-m", "chore: record automated training run history"])

    branch = git_run(["git", "branch", "--show-current"], capture=True).stdout.strip()
    if branch in {"main", "master"}:
        branch = f"training-history/{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        result = git_run(["git", "push", "origin", f"HEAD:{branch}"], token=token, check=False)
    else:
        result = git_run(["git", "push", "origin", branch], token=token, check=False)

    if result.returncode == 0:
        print(f"✅ Historique poussé vers origin/{branch}.")
        return True
    print(f"❌ git push a échoué (code {result.returncode}).")
    if token:
        return False

    token = getpass.getpass("GitHub PAT pour pousser l'historique (entrée masquée) : ").strip()
    if not token:
        print("⚠️ Aucun PAT fourni ; push non effectué.")
        return False
    if branch in {"main", "master"}:
        raise RuntimeError("Refus de pousser vers main/master")
    result = git_run(["git", "push", "origin", branch], token=token, check=False)
    if result.returncode != 0:
        raise RuntimeError("git push a échoué après authentification PAT.")
    print(f"✅ Historique poussé vers origin/{branch}.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="LapisLLM Colab training orchestrator")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--monitor-interval", type=int, default=500)
    parser.add_argument("--monitor-sample-tokens", type=int, default=48)
    parser.add_argument("--monitor-prompts", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-wiki-articles", type=int, default=40)
    parser.add_argument("--max-doc-files", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.runs is not None and args.runs < 1:
        parser.error("--runs doit être >= 1")
    if args.monitor_interval < 0:
        parser.error("--monitor-interval doit être >= 0")
    if args.monitor_sample_tokens < 1:
        parser.error("--monitor-sample-tokens doit être >= 1")

    token = get_github_token()
    print("GitHub auth: secret détecté (valeur masquée)." if token else "GitHub auth: aucun secret détecté.")
    bootstrap_repo(token)
    sync_pull(token)
    validate_project()
    install_project()
    validate_tests()
    build_corpus(args.max_wiki_articles, args.max_doc_files)

    runs = args.runs if args.runs is not None else prompt_int("Combien de runs d'entraînement ?", 1, 1)
    device = detect_device() if args.device == "auto" else args.device
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda demandé mais CUDA n'est pas disponible.")

    summaries = []
    for index in range(1, runs + 1):
        summaries.append(
            train_one_run(
                run_number=index,
                total_runs=runs,
                monitor_interval=args.monitor_interval,
                monitor_sample_tokens=args.monitor_sample_tokens,
                prompts=args.monitor_prompts,
                device=device,
                seed=args.seed + index - 1,
            )
        )

    print("\n" + "=" * 70)
    print("PROCESSUS TERMINÉ")
    print("=" * 70)
    print(f"Runs vérifiés : {len(summaries)}/{runs}")
    pushed = push_history(token)
    if not pushed:
        print("⚠️ Training terminé, mais l'historique GitHub n'a pas été poussé.")
