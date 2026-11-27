# Dataset governance and data pipeline

## Principle

Training data is a model dependency. Every source must be traceable, reproducible and compatible with intended training and distribution. Public availability does not itself grant permission to train on or redistribute content.

## Source approval gate

Before ingestion, each source record must identify:
- stable source ID and canonical URL;
- upstream owner/publisher;
- immutable revision, snapshot or retrieval date;
- license identifier and link to terms;
- permitted uses and known restrictions;
- redistribution status;
- fields selected and transformations;
- attribution requirements;
- privacy/data classification review;
- reviewer and approval state.

Unknown, missing or incompatible license evidence is blocked by default. Review source-specific and file-level exceptions; do not infer permission from a dataset name.

## Code corpus restrictions

- Never ingest private repositories or private mirrors.
- Never ingest credentials, API keys, private keys, tokens or other secrets.
- Exclude vendored dependencies, minified files and generated artifacts unless explicitly justified.
- Preserve copyright and license notices where required.
- Respect repository-level and file-level licensing metadata.
- Record source revision and file path.
- Do not execute code during ingestion.

## Pipeline

1. discover reviewed sources;
2. fetch pinned revisions;
3. normalize records;
4. reject unapproved licenses and malformed content;
5. remove secrets and prohibited personal data;
6. filter binaries, generated files and low-quality records;
7. normalize without damaging code semantics;
8. exact and near-duplicate detection;
9. split train/validation/test without leakage;
10. tokenize and pack sequences;
11. emit immutable manifests and summary statistics.

Every stage should be deterministic for the same inputs and config where supported.

## Normalized record proposal

~~~json
{
  "source_id": "source-name",
  "source_revision": "immutable-revision",
  "record_id": "stable-record-id",
  "content_sha256": "hex-digest",
  "language": "python",
  "path": "relative/path.py",
  "license": "SPDX-or-reviewed-expression",
  "split": "train",
  "content": "..."
}
~~~

This schema is illustrative and must be validated before implementation.

## Leakage, storage and release

Deduplicate before splitting where appropriate and check overlap afterward. Keep hidden benchmark tests and final evaluation answers out of training. Large raw and processed datasets stay outside Git; Git may contain registries, schemas, legally safe fixtures and manifests without raw content.

Every model release must identify dataset manifests used in pre-training and post-training. Document any restrictions on source-level disclosure.

## Open decisions

The initial programming-language mix, tokenizer normalization and vocabulary size, approved datasets, external storage backend, deduplication algorithm and thresholds remain open. Resolve them through issues and recorded evidence before large-scale ingestion.
