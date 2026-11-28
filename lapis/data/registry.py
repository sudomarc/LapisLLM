"""Dataset source registry and approval gate for LapisLLM.

Manages approved training data sources with license evidence,
provenance tracking, and redistribution constraints.
"""
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any


class SourceRecord:
    """A single approved training data source record."""

    # Required fields per data-policy.md§9-19
    # attribution has default "" so it is not in REQUIRED_FIELDS
    # content_sha256 computed from content if not provided
    REQUIRED_FIELDS = [
        "source_id",
        "source_revision",
        "record_id",
        "language",
        "path",
        "license",
        "split",
        "content",
        "reviewer",
        "approval_state",
    ]

    # Known incompatible licenses (blocked by default)
    BLOCKED_LICENSE_PATTERNS = [
        "private",
        "proprietary",
        "non-commercial-only",
        "personal-data",
    ]

    def __init__(self, data: Dict[str, Any]):
        """Initialize from dict, validating required fields."""
        missing = [f for f in self.REQUIRED_FIELDS if f not in data or not data[f]]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Normalize and store
        self.source_id = data["source_id"]
        self.source_revision = data["source_revision"]
        self.record_id = data["record_id"]
        # Compute SHA256 from content if not provided
        self.content_sha256 = data.get("content_sha256", "")
        if not self.content_sha256:
            self.content_sha256 = compute_content_sha256(data["content"])
        self.language = data["language"]
        self.path = data["path"]
        self.license = data["license"]
        self.content = data["content"]
        # Attribution with default empty string
        self.attribution = data.get("attribution", "")
        self.reviewer = data.get("reviewer", "unreviewed")
        self.approval_state = data["approval_state"]
        # Split for train/val/test partitioning
        self.split = data["split"]
        self.retrieval_date = datetime.utcnow().isoformat() + "Z"
        self._normalize_license()

    def _normalize_license(self):
        """Standardize license format and check for blocks."""
        lic = self.license.lower() if self.license else ""

        # Check for blocked patterns
        for pattern in self.BLOCKED_LICENSE_PATTERNS:
            if pattern in lic:
                self.approval_state = "rejected"
                break

    def is_approved(self) -> bool:
        """Check if this source is approved for training."""
        return self.approval_state == "approved"

    def is_blocked(self) -> bool:
        """Check if this source is blocked (uncompatible license)."""
        return self.approval_state == "rejected"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "record_id": self.record_id,
            "content_sha256": self.content_sha256,
            "language": self.language,
            "path": self.path,
            "license": self.license,
            "split": self.split,
            "attribution": self.attribution,
            "reviewer": self.reviewer,
            "approval_state": self.approval_state,
            "retrieval_date": self.retrieval_date,
            "content": self.content,
        }

    @property
    def content_length(self) -> int:
        """Length of content in characters."""
        return len(self.content) if self.content else 0

    @property
    def is_code(self) -> bool:
        """Check if this is a code language record."""
        return self.language in {"python", "javascript", "typescript", "rust", "go", "java"}


class DatasetRegistry:
    """Manages a collection of approved source records."""

    def __init__(self, registry_path: str = ".lapis-data/dataset-registry.json"):
        self.registry_path = registry_path
        self.records: Dict[str, SourceRecord] = {}
        self.load()

    def load(self):
        """Load registry from disk."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                for rec_data in data.get("records", []):
                    try:
                        rec = SourceRecord(rec_data)
                        self.records[rec_data["source_id"]] = rec
                    except ValueError as e:
                        print(f"⚠️  Skipping invalid record: {e}")
            except (json.JSONDecodeError, KeyError):
                # Empty or corrupt registry — start fresh
                self.records = {}

    def save(self):
        """Persist registry to disk."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        data = {
            "version": "1.0",
            "updated": datetime.utcnow().isoformat() + "Z",
            "records": [rec.to_dict() for rec in self.records.values()]
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_source(self, data: Dict[str, Any]) -> SourceRecord:
        """Add a new source record. Returns the record."""
        rec = SourceRecord(data)
        self.records[rec.source_id] = rec
        self.save()
        return rec

    def get_source(self, source_id: str) -> Optional[SourceRecord]:
        """Get a source by ID."""
        return self.records.get(source_id)

    def get_approved(self) -> List[SourceRecord]:
        """Get all approved sources."""
        return [r for r in self.records.values() if r.is_approved()]

    def get_by_language(self, language: str) -> List[SourceRecord]:
        """Get all approved sources for a given language."""
        return [r for r in self.records.values()
                if r.language == language and r.is_approved()]

    def get_by_split(self, split: str) -> List[SourceRecord]:
        """Get all approved sources for a given split."""
        return [r for r in self.records.values()
                if r.split == split and r.is_approved()]

    def reject_source(self, source_id: str, reason: str = ""):
        """Mark a source as rejected."""
        if source_id in self.records:
            self.records[source_id].approval_state = "rejected"
            self.save()

    def update_source(self, source_id: str, data: Dict[str, Any]) -> bool:
        """Update a source record. Returns True if successful."""
        if source_id not in self.records:
            return False
        try:
            # Preserve existing fields not being updated
            existing = self.records[source_id].to_dict()
            existing.update(data)
            self.records[source_id] = SourceRecord(existing)
            self.save()
            return True
        except ValueError:
            return False


def validate_license(license_str: str) -> bool:
    """Validate a license string is compatible with training.

    Returns True if the license appears compatible.
    Returns False if clearly incompatible.
    Returns None if requires manual review.
    """
    if not license_str:
        return False

    lic = license_str.lower()

    # Block known incompatible categories
    blocked = [
        "private", "proprietary",
        "non-commercial", "personal",
    ]

    for pattern in blocked:
        if pattern in lic:
            return False

    # Common compatible licenses (brief check)
    compatible_indicators = [
        "mit", "apache", "bsd", "ipache", "gpl", "lgpl",
        "unlicense", "cc-by", "cc-0", "public domain",
    ]

    # If it contains a known compatible indicator, accept
    for indicator in compatible_indicators:
        if indicator in lic:
            return True

    # Otherwise, require explicit review
    return None


def compute_content_sha256(content: str) -> str:
    """Compute SHA256 hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# Convenient small corpus for local development
# This is a legally safe fixture — CC0 / public domain inspired
DEVELOPMENT_CORPUS: List[SourceRecord] = [
    # Python language examples
    SourceRecord({
        "source_id": "lapis-foundation-code",
        "source_revision": "v0.1.0-release",
        "record_id": "python-examples-001",
        "content_sha256": "",
        "language": "python",
        "path": "examples/basic.py",
        "license": "MIT",
        "split": "train",
        "content": "# LapisLLM foundation code examples\n\n"
        "def hello_world():\n"
        "    \"\"\"Print a greeting.\"\"\"\n"
        "print(\"Hello, World!\")\n\n"
        "if __name__ == \"__main__\":\n"
        "    hello_world()",
        "attribution": "Internal LapisLLM foundation examples",
        "reviewer": "system",
        "approval_state": "approved",
    }),
    SourceRecord({
        "source_id": "lapis-foundation-code",
        "source_revision": "v0.1.0-release",
        "record_id": "python-functions-001",
        "content_sha256": "",
        "language": "python",
        "path": "examples/sum.py",
        "license": "MIT",
        "split": "train",
        "content": "# Sum function example\n\ndef sum_numbers(a: int, b: int) -> int:\n"
        "    \"\"\"Return the sum of two numbers.\"\"\"\n"
        "return a + b\n\n"
        "# Tests\n"
        "assert sum_numbers(2, 3) == 5\n"
        "assert sum_numbers(0, 0) == 0",
        "attribution": "Internal LapisLLM foundation examples",
        "reviewer": "system",
        "approval_state": "approved",
    }),
    # Simple code explanation
    SourceRecord({
        "source_id": "lapis-foundation-code",
        "source_revision": "v0.1.0-release",
        "record_id": "python-explain-001",
        "content_sha256": "",
        "language": "python",
        "path": "examples/explain.py",
        "license": "MIT",
        "split": "train",
        "content": "# Explain code function\ndef explain_factorial(n: int) -> int:\n"
        "    \"\"\"Compute factorial of n.\"\"\"\n"
        "if n <= 1:\n"
        "    return 1\n"
        "return n * explain_factorial(n - 1)\n\n"
        "# factorial(5) = 120",
        "attribution": "Internal LapisLLM foundation examples",
        "reviewer": "system",
        "approval_state": "approved",
    }),
]


def initialize_development_registry(registry_path: str = ".lapis-data/dataset-registry.json"):
    """Initialize the dataset registry with a safe development corpus.

    Creates a registry file with approved sources that can be used
    for initial tokenizer training. All sources are MIT-licensed,
    publicly authored code examples.
    """
    reg = DatasetRegistry(registry_path)

    # Add development corpus sources
    for rec in DEVELOPMENT_CORPUS:
        existing = reg.get_source(rec.source_id)
        if existing is None:
            reg.add_source(rec.to_dict())
            print(f"  ✓ Added: {rec.source_id}")
        else:
            print(f"  ~ already exists: {rec.source_id}")

    # Print summary
    approved = reg.get_approved()
    print(f"\nDataset registry summary:")
    print(f"  Total sources: {len(reg.records)}")
    print(f"  Approved: {len(approved)}")
    blocked = [r for r in reg.records.values() if r.is_blocked()]
    print(f"  Blocked: {len(blocked)}")

    # By language
    languages = set(r.language for r in reg.records.values())
    print(f"  Languages: {', '.join(sorted(languages))}")

    # By split
    splits = set(r.split for r in reg.records.values())
    print(f"  Splits: {', '.join(sorted(splits))}")

    return reg


if __name__ == "__main__":
    # Standalone initialization
    print("LapisLLM Dataset Registry — Stage 3")
    print("=" * 50)
    reg = initialize_development_registry()
    print(f"\nRegistry saved to: {reg.registry_path}")