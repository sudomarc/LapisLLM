#!/usr/bin/env python3
"""Collect explicitly allowlisted public .onion text sources through Tor.

This collector is intentionally bounded and non-discovering:

* it accepts only explicitly supplied .onion URLs;
* it never crawls links or searches for onion services;
* redirects must remain on the same .onion host;
* credentials, fragments, non-onion hosts, and non-HTTP(S) URLs are rejected;
* response bodies are size-limited and restricted to text-like content;
* provenance is recorded for every collected source.

A local Tor client must be running with its SOCKS proxy available. Tor Project
currently documents the default SOCKS5 endpoint as ``socks5://127.0.0.1:9050``
and recommends using the ``socks5`` variant that prevents DNS leaks for the
application path. See: https://support.torproject.org/little-t-tor/troubleshooting/check-for-leaks/

Example:
    python scripts/fetch_onion_training_data.py \
        --source-file configs/data/onion_sources.txt

The source file is deliberately user-supplied. Do not add arbitrary onion
indexes or untrusted bulk URL lists to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

try:
    import requests
except ImportError:  # pragma: no cover
    raise SystemExit("Missing dependency: requests. Install with: pip install 'requests[socks]'")


LOGGER = logging.getLogger("lapis.fetch_onion_training_data")
DEFAULT_PROXY = "socks5h://127.0.0.1:9050"
ONION_SUFFIX = ".onion"
MAX_REDIRECTS = 3
DEFAULT_MAX_CHARS = 2_000_000
DEFAULT_TIMEOUT = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch explicitly allowlisted .onion training sources through Tor")
    parser.add_argument("--source-file", required=True, help="Text file containing one public .onion URL per line")
    parser.add_argument("--output-dir", default="training_data", help="Output directory")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Maximum response characters per source")
    parser.add_argument("--proxy", default=DEFAULT_PROXY, help="Tor SOCKS5 proxy URL")
    parser.add_argument("--append-to-combined", action="store_true", help="Append collected onion records to training_data/combined.txt")
    return parser.parse_args()


def normalize_onion_url(value: str) -> str:
    """Validate and normalize one explicit public onion HTTP(S) URL."""
    value = value.strip()
    if not value or value.startswith("#"):
        raise ValueError("empty/comment source")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("source must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("credentials are not allowed in onion sources")
    if parsed.fragment:
        raise ValueError("URL fragments are not allowed")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname.endswith(ONION_SUFFIX):
        raise ValueError("source host must end in .onion")
    label = hostname[: -len(ONION_SUFFIX)]
    if len(label) != 56 or not re.fullmatch(r"[a-z2-7]{56}", label):
        raise ValueError("source must use a valid v3 56-character onion address")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("non-standard ports are not allowed")
    return urlunsplit((parsed.scheme, hostname if parsed.port is None else f"{hostname}:{parsed.port}", parsed.path or "/", parsed.query, ""))


def load_sources(path: Path) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            source = normalize_onion_url(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if source not in seen:
            seen.add(source)
            sources.append(source)
    return sources


def clean_text(text: str) -> str:
    """Extract readable text without executing or preserving HTML markup."""
    text = re.sub(r"(?is)<(script|style|noscript|template)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_text_response(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    return content_type in {"text/html", "text/plain", "application/xhtml+xml", "application/json", "application/xml", "text/xml"}


def fetch_source(session: requests.Session, source: str, *, timeout: float, max_chars: int) -> tuple[str, str]:
    """Fetch one source, following only same-host redirects."""
    current = normalize_onion_url(source)
    original_host = urlsplit(current).hostname

    for _ in range(MAX_REDIRECTS + 1):
        response = session.get(current, timeout=timeout, allow_redirects=False, stream=True)
        if 300 <= response.status_code < 400:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise RuntimeError("redirect without Location header")
            next_url = normalize_onion_url(requests.compat.urljoin(current, location))
            if urlsplit(next_url).hostname != original_host:
                raise RuntimeError("redirect leaves the original .onion host")
            current = next_url
            continue

        response.raise_for_status()
        if not is_text_response(response):
            response.close()
            raise RuntimeError("response is not a supported text content type")

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_chars * 4:
                response.close()
                raise RuntimeError("response exceeds the configured size limit")
            chunks.append(chunk)
        response.close()
        raw = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
        text = clean_text(raw)[:max_chars].strip()
        if not text:
            raise RuntimeError("source contained no readable text")
        return current, text

    raise RuntimeError("too many redirects")


def write_record_file(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f"\n\n===== {record['url']} =====\n\n{record['text']}\n")


def append_combined(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(f"\n\n===== onion/{record['url']} =====\n\n{record['text']}\n")


def main() -> int:
    args = parse_args()
    if args.max_chars <= 0:
        raise SystemExit("--max-chars must be positive")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    source_file = Path(args.source_file)
    sources = load_sources(source_file)
    if not sources:
        raise SystemExit("No valid onion sources were supplied")

    session = requests.Session()
    session.proxies.update({"http": args.proxy, "https": args.proxy})
    session.headers.update({"User-Agent": "LapisLLM-training-data/0.2", "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1"})

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, source in enumerate(sources, 1):
        LOGGER.info("Onion source [%d/%d]: %s", index, len(sources), source)
        try:
            final_url, text = fetch_source(session, source, timeout=args.timeout, max_chars=args.max_chars)
            records.append({
                "url": final_url,
                "source_url": source,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "chars": len(text),
                "text": text,
            })
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            LOGGER.warning("Skipped %s: %s", source, exc)
            failures.append({"url": source, "error": str(exc)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_record_file(output_dir / "onion.txt", records)

    manifest = {
        "project": "LapisLLM",
        "collector": "scripts/fetch_onion_training_data.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_file": str(source_file),
        "proxy": args.proxy,
        "settings": {"timeout": args.timeout, "max_chars": args.max_chars},
        "sources_requested": len(sources),
        "sources_collected": len(records),
        "failures": failures,
        "records": [
            {key: value for key, value in record.items() if key != "text"}
            for record in records
        ],
    }
    (output_dir / "onion_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.append_to_combined and records:
        append_combined(output_dir / "combined.txt", records)

    LOGGER.info("Collected %d/%d onion sources", len(records), len(sources))
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
