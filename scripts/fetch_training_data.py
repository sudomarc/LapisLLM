#!/usr/bin/env python3
"""Build a small, reproducible LapisLLM training corpus from Wikipedia and GitHub.

The collector deliberately uses public APIs/raw files instead of cloning entire
repositories. It collects topic-focused Wikipedia articles and public README /
documentation files from selected ML repositories, then writes:

    training_data/wikipedia.txt
    training_data/github.txt
    training_data/combined.txt
    training_data/manifest.json

Examples (Colab or a local checkout):
    !python scripts/fetch_training_data.py
    !python scripts/fetch_training_data.py --sample-chars 100000

Network failures are isolated per source so one unavailable article/repository
does not abort the complete corpus build.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover - exercised in minimal environments
    print("Missing dependency: requests. Install it with: pip install requests", file=sys.stderr)
    raise SystemExit(2)


LOGGER = logging.getLogger("lapis.fetch_training_data")
USER_AGENT = "LapisLLM-training-data/0.1 (+https://github.com/sudomarc/LapisLLM)"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
GITHUB_API = "https://api.github.com"

WIKIPEDIA_TOPICS = [
    "machine learning",
    "supervised learning",
    "unsupervised learning",
    "neural network",
    "deep learning",
    "transformer (machine learning model)",
    "attention (machine learning)",
    "natural language processing",
    "large language model",
    "language model",
    "generative artificial intelligence",
    "reinforcement learning",
    "representation learning",
    "word embedding",
]

GITHUB_REPOS = [
    "pytorch/pytorch",
    "huggingface/transformers",
    "openai/gpt-2",
    "karpathy/nanoGPT",
    "facebookresearch/llama",
]

# Keep the collector focused. The API can expose very large documentation trees.
DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}
MAX_DOC_FILES_PER_REPO = 20
MAX_FILE_CHARS = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and clean LapisLLM training data")
    parser.add_argument("--output-dir", default="training_data", help="Output directory")
    parser.add_argument("--wiki-limit", type=int, default=3, help="Articles per topic/search query")
    parser.add_argument("--max-wiki-articles", type=int, default=40, help="Maximum unique Wikipedia articles")
    parser.add_argument("--max-doc-files", type=int, default=MAX_DOC_FILES_PER_REPO, help="Maximum GitHub docs per repository")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retries for transient HTTP failures")
    parser.add_argument("--sample-chars", type=int, default=0, help="Also write combined_sample.txt with this many characters; 0 disables")
    parser.add_argument("--ascii-only", action="store_true", help="Drop non-ASCII characters after Unicode normalization")
    parser.add_argument("--keep-markdown", action="store_true", help="Keep GitHub Markdown instead of converting common Markdown syntax to text")
    return parser.parse_args()


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/plain;q=0.9, */*;q=0.8"})
    return session


def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    retries: int,
    params: dict[str, Any] | None = None,
) -> requests.Response | None:
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"transient HTTP {response.status_code}")
            response.raise_for_status()
            return response
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                LOGGER.warning("Request failed after %d attempts: %s", retries, exc)
                return None
            delay = min(8.0, 1.5 * (2 ** (attempt - 1)))
            LOGGER.warning("Request failed (%s); retrying in %.1fs", exc, delay)
            time.sleep(delay)
    return None


def clean_text(text: str, *, ascii_only: bool = False) -> str:
    """Normalize text while preserving punctuation useful to a language model."""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove control characters except tab/newline, then normalize whitespace.
    text = "".join(ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))
    if ascii_only:
        text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def markdown_to_text(text: str) -> str:
    """Conservatively remove presentation syntax without destroying code/content."""
    text = re.sub(r"^\s*```[^\n]*\n", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*```\s*$", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~`]+", "", text)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    return text


def wikipedia_search(session: requests.Session, query: str, limit: int, *, timeout: float, retries: int) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "utf8": 1,
    }
    response = request_with_retries(session, WIKIPEDIA_API, timeout=timeout, retries=retries, params=params)
    if response is None:
        return []
    try:
        return [item["title"] for item in response.json().get("query", {}).get("search", [])]
    except (ValueError, KeyError, TypeError) as exc:
        LOGGER.warning("Invalid Wikipedia search response for %r: %s", query, exc)
        return []


def wikipedia_extract(session: requests.Session, title: str, *, timeout: float, retries: int) -> str | None:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "titles": title,
        "format": "json",
        "formatversion": 2,
    }
    response = request_with_retries(session, WIKIPEDIA_API, timeout=timeout, retries=retries, params=params)
    if response is None:
        return None
    try:
        pages = response.json().get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            return None
        return pages[0].get("extract") or None
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        LOGGER.warning("Invalid Wikipedia article response for %r: %s", title, exc)
        return None


def collect_wikipedia(session: requests.Session, args: argparse.Namespace) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    titles: list[str] = []
    seen: set[str] = set()
    for topic in WIKIPEDIA_TOPICS:
        LOGGER.info("Wikipedia search: %s", topic)
        for title in wikipedia_search(session, topic, args.wiki_limit, timeout=args.timeout, retries=args.retries):
            key = title.casefold()
            if key not in seen:
                seen.add(key)
                titles.append(title)
            if len(titles) >= args.max_wiki_articles:
                break
        if len(titles) >= args.max_wiki_articles:
            break

    articles: list[tuple[str, str]] = []
    for index, title in enumerate(titles, 1):
        LOGGER.info("Wikipedia [%d/%d]: %s", index, len(titles), title)
        raw = wikipedia_extract(session, title, timeout=args.timeout, retries=args.retries)
        if not raw:
            continue
        text = clean_text(raw, ascii_only=args.ascii_only)
        if text:
            articles.append((title, text))

    return articles, {
        "search_topics": WIKIPEDIA_TOPICS,
        "articles_requested": len(titles),
        "articles_collected": len(articles),
    }


def github_repo_metadata(session: requests.Session, repo: str, *, timeout: float, retries: int) -> dict[str, Any] | None:
    response = request_with_retries(session, f"{GITHUB_API}/repos/{repo}", timeout=timeout, retries=retries)
    if response is None:
        return None
    try:
        return response.json()
    except ValueError as exc:
        LOGGER.warning("Invalid GitHub metadata for %s: %s", repo, exc)
        return None


def github_tree_docs(session: requests.Session, repo: str, branch: str, max_files: int, *, timeout: float, retries: int) -> list[str]:
    url = f"{GITHUB_API}/repos/{repo}/git/trees/{branch}"
    response = request_with_retries(session, url, timeout=timeout, retries=retries, params={"recursive": 1})
    if response is None:
        return []
    try:
        tree = response.json().get("tree", [])
    except ValueError:
        return []
    candidates: list[str] = []
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        path = str(entry.get("path", ""))
        lower = path.lower()
        suffix = Path(path).suffix.lower()
        is_doc = suffix in DOC_EXTENSIONS and (
            Path(path).name.lower().startswith(("readme", "contributing", "install", "getting-started"))
            or "/docs/" in f"/{lower}/"
            or lower.startswith("docs/")
        )
        if is_doc:
            candidates.append(path)
    # README and top-level docs first, then remaining docs in stable order.
    candidates.sort(key=lambda p: (0 if Path(p).parent == Path(".") else 1, len(p), p.lower()))
    return candidates[:max_files]


def github_raw_file(session: requests.Session, repo: str, branch: str, path: str, *, timeout: float, retries: int) -> str | None:
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    response = request_with_retries(session, url, timeout=timeout, retries=retries)
    if response is None:
        return None
    try:
        content = response.content.decode("utf-8", errors="replace")
    except Exception as exc:  # defensive: requests may expose unusual content implementations
        LOGGER.warning("Could not decode %s/%s: %s", repo, path, exc)
        return None
    return content[:MAX_FILE_CHARS]


def collect_github(session: requests.Session, args: argparse.Namespace) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    documents: list[tuple[str, str]] = []
    repo_stats: list[dict[str, Any]] = []
    for repo in GITHUB_REPOS:
        LOGGER.info("GitHub repository: %s", repo)
        metadata = github_repo_metadata(session, repo, timeout=args.timeout, retries=args.retries)
        if not metadata:
            repo_stats.append({"repo": repo, "status": "metadata_failed", "files": 0})
            continue
        branch = metadata.get("default_branch") or "main"
        paths = github_tree_docs(session, repo, branch, args.max_doc_files, timeout=args.timeout, retries=args.retries)
        collected = 0
        for path in paths:
            LOGGER.info("GitHub %s: %s", repo, path)
            raw = github_raw_file(session, repo, branch, path, timeout=args.timeout, retries=args.retries)
            if raw is None:
                continue
            text = raw if args.keep_markdown else markdown_to_text(raw)
            text = clean_text(text, ascii_only=args.ascii_only)
            if text:
                documents.append((f"{repo}:{path}", text))
                collected += 1
        repo_stats.append({"repo": repo, "branch": branch, "files_requested": len(paths), "files_collected": collected})

    return documents, {"repositories": repo_stats, "documents_collected": len(documents)}


def write_corpus(path: Path, records: list[tuple[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for source, text in records:
            handle.write(f"\n\n===== {source} =====\n\n{text}\n")
    return path.stat().st_size


def approx_tokens(text: str) -> int:
    """Fast dependency-free estimate; actual Lapis BPE token count may differ."""
    return len(re.findall(r"\S+", text))


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = build_session()

    LOGGER.info("Starting LapisLLM training-data collection")
    LOGGER.info("Output directory: %s", output_dir.resolve())

    wikipedia, wiki_stats = collect_wikipedia(session, args)
    github, github_stats = collect_github(session, args)

    wiki_path = output_dir / "wikipedia.txt"
    github_path = output_dir / "github.txt"
    combined_path = output_dir / "combined.txt"
    wiki_size = write_corpus(wiki_path, wikipedia)
    github_size = write_corpus(github_path, github)
    combined_records = [(f"wikipedia/{source}", text) for source, text in wikipedia]
    combined_records += [(f"github/{source}", text) for source, text in github]
    combined_size = write_corpus(combined_path, combined_records)

    combined_text = combined_path.read_text(encoding="utf-8")
    sample_path: Path | None = None
    if args.sample_chars > 0:
        sample_path = output_dir / "combined_sample.txt"
        sample_path.write_text(combined_text[: args.sample_chars], encoding="utf-8")

    manifest = {
        "project": "LapisLLM",
        "collector": "scripts/fetch_training_data.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "ascii_only": args.ascii_only,
            "keep_markdown": args.keep_markdown,
            "wiki_limit": args.wiki_limit,
            "max_wiki_articles": args.max_wiki_articles,
            "max_doc_files": args.max_doc_files,
        },
        "wikipedia": wiki_stats,
        "github": github_stats,
        "outputs": {
            "wikipedia.txt": {"bytes": wiki_size, "records": len(wikipedia)},
            "github.txt": {"bytes": github_size, "records": len(github)},
            "combined.txt": {"bytes": combined_size, "records": len(combined_records), "approx_tokens": approx_tokens(combined_text)},
        },
    }
    if sample_path is not None:
        manifest["outputs"][sample_path.name] = {"bytes": sample_path.stat().st_size, "records": 1}

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    LOGGER.info("Collection complete")
    LOGGER.info("Wikipedia: %d articles | %.1f KiB", len(wikipedia), wiki_size / 1024)
    LOGGER.info("GitHub: %d documents | %.1f KiB", len(github), github_size / 1024)
    LOGGER.info("Combined: %.1f KiB | ~%d whitespace tokens", combined_size / 1024, approx_tokens(combined_text))
    LOGGER.info("Training input: %s", combined_path)
    LOGGER.info("Note: token count is an estimate until Lapis's BPE tokenizer is applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
