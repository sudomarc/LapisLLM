import glob
import json
import os
import re
import unicodedata


def normalize_unicode(text):
    """Normalize unicode text to NFKC form."""
    return unicodedata.normalize("NFKC", text)


def remove_extra_whitespace(text):
    """Remove extra whitespace and normalize."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def remove_special_chars(text, pattern=None):
    """Remove special characters from text."""
    if pattern is None:
        # Remove non-printable characters
        pattern = r"[^\x20-\x7E\n\r\t]"
    return re.sub(pattern, "", text)


def clean_text(text):
    """Full text cleaning pipeline."""
    text = normalize_unicode(text)
    text = remove_extra_whitespace(text)
    text = remove_special_chars(text)
    return text


def clean_dataset(input_dir, output_dir, clean_func=clean_text):
    """Clean a dataset of text files."""
    os.makedirs(output_dir, exist_ok=True)
    
    files = []
    for ext in [".jsonl", ".txt", ".json", ".csv"]:
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    for filepath in sorted(files):
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            
            if isinstance(content, list):
                # Handle JSON arrays
                cleaned = [clean_func(str(item)) for item in content]
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, ensure_ascii=False)
            else:
                cleaned = clean_func(content)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                    
        except (IOError, ValueError, json.JSONDecodeError) as e:
            print(f"Error cleaning {filepath}: {e}")