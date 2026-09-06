import glob
import json
import os
import re
import unicodedata


def normalize_unicode(text):
    """Normalize unicode text to NFKC form."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return unicodedata.normalize("NFKC", text)


def remove_extra_whitespace(text):
    """Collapse runs of whitespace while preserving clean text boundaries."""
    return re.sub(r"\s+", " ", text).strip()


def remove_special_chars(text, pattern=None):
    """Remove characters excluded by the configured pattern."""
    if pattern is None:
        pattern = r"[^\x20-\x7E\n\r\t]"
    return re.sub(pattern, "", text)


def clean_text(text):
    """Run the full text cleaning pipeline."""
    return remove_special_chars(remove_extra_whitespace(normalize_unicode(text)))


def _read_text_file(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 input: {path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read input file: {path}") from exc


def clean_dataset(input_dir, output_dir, clean_func=clean_text):
    """Clean supported text files and fail if any input cannot be processed."""
    input_path = os.path.abspath(input_dir)
    output_path = os.path.abspath(output_dir)
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input directory not found: {input_path}")
    os.makedirs(output_path, exist_ok=True)

    files = []
    for ext in (".jsonl", ".txt", ".json", ".csv"):
        files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
    files = sorted(files)
    if not files:
        raise ValueError(f"No supported input files found in {input_path}")

    for filepath in files:
        source = os.path.abspath(filepath)
        destination = os.path.join(output_path, os.path.basename(filepath))
        content = _read_text_file(PathLike(source))
        try:
            if source.endswith(".json"):
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    cleaned = [clean_func(str(item)) for item in parsed]
                    with open(destination, "w", encoding="utf-8") as handle:
                        json.dump(cleaned, handle, ensure_ascii=False)
                    continue
                if isinstance(parsed, str):
                    content = parsed
                else:
                    raise ValueError("JSON input must be a string or array")
            cleaned = clean_func(content)
            with open(destination, "w", encoding="utf-8") as handle:
                handle.write(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON input: {source}") from exc
        except OSError as exc:
            raise OSError(f"Unable to write cleaned file: {destination}") from exc


class PathLike(str):
    """Small path adapter retained for Python-version-compatible file reading."""

    def read_text(self, *args, **kwargs):
        with open(self, *args, **kwargs) as handle:
            return handle.read()
