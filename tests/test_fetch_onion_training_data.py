from pathlib import Path

import pytest

from scripts.fetch_onion_training_data import clean_text, load_sources, normalize_onion_url


VALID_ONION = "a" * 56 + ".onion"


def test_normalize_onion_url_accepts_v3_source() -> None:
    assert normalize_onion_url(f"HTTPS://{VALID_ONION}/docs#ignored".replace("#ignored", "")) == f"https://{VALID_ONION}/docs"


def test_normalize_onion_url_rejects_non_onion_host() -> None:
    with pytest.raises(ValueError, match="must end in .onion"):
        normalize_onion_url("https://example.com/")


def test_normalize_onion_url_rejects_invalid_onion_length() -> None:
    with pytest.raises(ValueError, match="valid v3"):
        normalize_onion_url("https://abc.onion/")


def test_normalize_onion_url_rejects_credentials() -> None:
    with pytest.raises(ValueError, match="credentials"):
        normalize_onion_url(f"https://user:pass@{VALID_ONION}/")


def test_load_sources_deduplicates_and_skips_comments(tmp_path: Path) -> None:
    source_file = tmp_path / "sources.txt"
    source_file.write_text(
        f"# trusted sources\nhttps://{VALID_ONION}/\nhttps://{VALID_ONION}/\n",
        encoding="utf-8",
    )

    assert load_sources(source_file) == [f"https://{VALID_ONION}/"]


def test_clean_text_removes_active_html_content() -> None:
    text = clean_text("<h1>Title</h1><script>alert('x')</script><p>Hello</p>")
    assert text == "Title Hello"
