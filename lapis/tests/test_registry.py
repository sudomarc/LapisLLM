"""Dataset registry tests for LapisLLM Stage 3."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lapis'))

from lapis.data.registry import (
    SourceRecord, DatasetRegistry, validate_license,
    compute_content_sha256, DEVELOPMENT_CORPUS, initialize_development_registry
)


def test_source_record_creation():
    """SourceRecord should create and validate properly."""
    rec = SourceRecord({
        "source_id": "test-source",
        "source_revision": "v1.0",
        "record_id": "rec-001",
        "content_sha256": "abc123",
        "language": "python",
        "path": "test.py",
        "license": "MIT",
        "split": "train",
        "content": "def foo(): pass",
        "attribution": "Test attribution",
        "reviewer": "tester",
        "approval_state": "approved",
    })
    assert rec.source_id == "test-source"
    assert rec.is_approved() == True
    assert rec.is_blocked() == False
    assert rec.language == "python"
    assert rec.split == "train"  # split is stored


def test_source_record_blocks():
    """SourceRecord should block incompatible licenses."""
    rec = SourceRecord({
        "source_id": "bad-source",
        "source_revision": "v1.0",
        "record_id": "rec-002",
        "content_sha256": "def456",
        "language": "python",
        "path": "test.py",
        "license": "proprietary",
        "split": "train",
        "content": "def foo(): pass",
        "attribution": "",  # required field
        "reviewer": "system",
        "approval_state": "approved",  # will be overridden
    })
    assert rec.is_blocked() == True
    assert rec.approval_state == "rejected"


def test_source_record_code_detection():
    """SourceRecord should detect code languages."""
    rec = SourceRecord({
        "source_id": "code-src",
        "source_revision": "v1.0",
        "record_id": "rec-003",
        "content_sha256": "",
        "language": "python",
        "path": "code.py",
        "license": "MIT",
        "split": "train",
        "content": "def foo(): pass",
        "attribution": "",
        "reviewer": "system",
        "approval_state": "approved",
    })
    assert rec.is_code == True


def test_sha256_computation():
    """SHA256 should compute correctly and be deterministic."""
    content = "hello world"
    h1 = compute_content_sha256(content)
    h2 = compute_content_sha256(content)
    assert h1 == h2
    assert len(h1) == 64  # hex SHA256

    # Different content should give different hash
    h3 = compute_content_sha256("goodbye world")
    assert h1 != h3


def test_registry_load_save():
    """Registry should load and save correctly."""
    from lapis.data.registry import DatasetRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "test-registry.json")
        reg = DatasetRegistry(registry_path)

        # Add a source
        rec = SourceRecord({
            "source_id": "test-src",
            "source_revision": "v1.0",
            "record_id": "rec-001",
            "content_sha256": "abc123",
            "language": "python",
            "path": "test.py",
            "license": "MIT",
            "split": "train",
            "content": "def foo(): pass",  # required field
            "attribution": "Test",
            "reviewer": "system",
            "approval_state": "approved",
        })
        reg.add_source(rec.to_dict())

        # Check it loaded
        assert len(reg.records) == 1
        loaded = reg.get_source("test-src")
        assert loaded is not None
        assert loaded.is_approved()


def test_registry_approval_states():
    """Registry should manage approval states."""
    from lapis.data.registry import DatasetRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "test-registry.json")
        reg = DatasetRegistry(registry_path)

        # Add approved
        rec1 = SourceRecord({
            "source_id": "good-src",
            "source_revision": "v1.0",
            "record_id": "rec-001",
            "content_sha256": "abc123",
            "language": "python",
            "path": "test.py",
            "license": "MIT",
            "split": "train",
            "content": "def foo(): pass",
            "attribution": "",
            "reviewer": "system",
            "approval_state": "approved",
        })
        reg.add_source(rec1.to_dict())

        # Add blocked
        rec2 = SourceRecord({
            "source_id": "bad-src",
            "source_revision": "v1.0",
            "record_id": "rec-002",
            "content_sha256": "def456",
            "language": "python",
            "path": "test.py",
            "license": "proprietary",
            "split": "train",
            "content": "def bar(): pass",
            "attribution": "",
            "reviewer": "system",
            "approval_state": "approved",  # will be overridden by _normalize_license
        })
        reg.add_source(rec2.to_dict())

        # Check states
        assert reg.get_source("good-src").is_approved() == True
        assert reg.get_source("bad-src").is_blocked() == True

        # Reject a previously approved source
        reg.reject_source("good-src", "license review")
        assert reg.get_source("good-src").is_approved() == False


def test_validate_license():
    """License validation function."""
    # MIT should be compatible
    result = validate_license("MIT")
    assert result == True or result is None  # None means requires manual review

    # Proprietary should be blocked
    result = validate_license("proprietary")
    assert result == False

    # GPL should be compatible
    result = validate_license("GPL-3.0")
    assert result == True or result is None

    # None/empty should be invalid
    result = validate_license("")
    assert result == False


def test_development_corpus():
    """Development corpus should have valid records."""
    for rec in DEVELOPMENT_CORPUS:
        assert rec.source_id is not None
        assert rec.is_approved() == True
        # language check - dev corpus has python
        assert rec.language in {"python", }
        assert rec.split == "train"


def test_initialize_development_registry():
    """Initialize development registry should work."""
    from lapis.data.registry import initialize_development_registry
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        registry_path = os.path.join(tmpdir, "dev-registry.json")
        reg = initialize_development_registry(registry_path)
        assert reg is not None
        assert len(reg.records) > 0
        # Check at least one is approved
        approved = reg.get_approved()
        assert len(approved) > 0


def test_content_deduplication():
    """Content deduplication should work at pipeline level."""
    from lapis.data.preprocess import deduplicate_records
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        records = [
            SourceRecord({
                "source_id": "a", "source_revision": "v1", "record_id": "r1",
                "content_sha256": "abc", "language": "python", "path": "a.py",
                "license": "MIT", "split": "train",
                "content": "def foo(): pass",
                "attribution": "", "reviewer": "system", "approval_state": "approved",
            }),
            SourceRecord({
                "source_id": "b", "source_revision": "v1", "record_id": "r2",
                "content_sha256": "abc", "language": "python", "path": "b.py",
                "license": "MIT", "split": "train",
                "content": "def foo(): pass",  # duplicate content
                "attribution": "", "reviewer": "system", "approval_state": "approved",
            }),
            SourceRecord({
                "source_id": "c", "source_revision": "v1", "record_id": "r3",
                "content_sha256": "xyz", "language": "python", "path": "c.py",
                "license": "MIT", "split": "train",
                "content": "def bar(): pass",  # unique
                "attribution": "", "reviewer": "system", "approval_state": "approved",
            }),
        ]

        unique = deduplicate_records(records)
        assert len(unique) == 2  # one duplicate removed


def test_license_property_normalization():
    """License normalization should work in SourceRecord."""
    rec = SourceRecord({
        "source_id": "test-noncommercial",
        "source_revision": "v1.0",
        "record_id": "rec-001",
        "content_sha256": "abc123",
        "language": "python",
        "path": "test.py",
        "license": "non-commercial-only",
        "split": "train",
        "content": "def foo(): pass",
        "attribution": "",
        "reviewer": "system",
        "approval_state": "approved",
    })
    assert rec.is_blocked() == True
    assert rec.approval_state == "rejected"


def test_normalize_text():
    """Text normalization should strip and collapse whitespace."""
    # Multiple spaces collapsed
    result = normalize_text("hello   world")
    # Should have at most one space between words
    assert "hello  world" not in result  # no double space

    # Leading/trailing whitespace stripped
    result = normalize_text("  hello  ")
    assert result == "hello"

    # Newlines preserved but cleaned
    result = normalize_text("def foo():\n    pass")
    assert "def" in result


if __name__ == "__main__":
    print("Running registry tests...")
    tests = [
        test_source_record_creation,
        test_source_record_blocks,
        test_source_record_code_detection,
        test_sha256_computation,
        test_registry_load_save,
        test_registry_approval_states,
        test_validate_license,
        test_development_corpus,
        test_initialize_development_registry,
        test_content_deduplication,
        test_license_property_normalization,
        test_normalize_text,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")