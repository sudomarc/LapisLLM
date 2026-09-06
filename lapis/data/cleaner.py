import glob
import json
import re
import unicodedata
from pathlib import Path


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def normalize_unicode(text):
    """Normalize Unicode text to NFKC form."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return unicodedata.normalize("NFKC", text)


def remove_extra_whitespace(text):
    """Collapse runs of whitespace while preserving clean text boundaries."""
    return re.sub(r"\s+", " ", text).strip()


def remove_special_chars(text, pattern=None):
    """Remove control characters while preserving printable Unicode by default."""
    if pattern is None:
        return _CONTROL_CHARS.sub("", text)
    return re.sub(pattern, "", text)


def clean_text(text):
    """Run the full text cleaning pipeline."""
    return remove_special_chars(remove_extra_whitespace(normalize_unicode(text)))


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 input: {path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read input file: {path}") from exc


def clean_dataset(input_dir, output_dir, clean_func=clean_text):
    """Clean supported text files and fail if any input cannot be processed."""
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in (".jsonl", ".txt", ".json", ".csv"):
        files.extend(input_path.glob(f"*{ext}"))
    files = sorted(files)
    if not files:
        raise ValueError(f"No supported input files found in {input_path}")

    for path in files:
        content = _read_text_file(path)
        destination = output_path / path.name
        try:
            if path.suffix == ".json":
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    cleaned = [clean_func(str(item)) for item in parsed]
                    destination.write_text(
                        json.dumps(cleaned, ensure_ascii=False), encoding="utf-8"
                    )
                    continue
                if isinstance(parsed, str):
                    content = parsed
                else:
                    raise ValueError("JSON input must be a string or array")
            destination.write_text(clean_func(content), encoding="utf-8")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON input: {path}") from exc
        except OSError as exc:
            raise OSError(f"Unable to write cleaned file: {destination}") from exc
