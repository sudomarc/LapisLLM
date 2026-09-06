---
name: release-manager
description: Prepare changelogs, releases and versioning.
version: 0.1.0
status: experimental
category: Development
---

# Release Manager

## Purpose

Prepare changelogs, releases and versioning.

## When to use

Use when the commit history, release scope, and versioning policy are available.

## Inputs

- Changes since the previous release.
- Versioning policy.
- Release checklist.

## Outputs

- Changelog draft.
- Release notes.
- Validation and compatibility checklist.

## Workflow

1. Inspect commits and user-visible changes.
2. Group additions, changes, fixes, performance, and breaking changes.
3. Cross-check version numbers and documentation.
4. Record known limitations.

## Examples

Generate release notes from verified commits without inventing metrics or features.

## Limitations

The skill cannot establish whether a release is safe without repository-specific validation.

## Safety

Never publish secrets or unsupported capability claims. Verify release artifacts before shipping.
