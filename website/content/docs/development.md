# Development

Lapis follows an evidence-first engineering workflow. Repository code and tests are the source of truth; public documentation must not claim behavior that cannot be reproduced.

## CLI boundary

The top-level `lapis` command is the developer/research CLI. Use explicit namespaces for development operations.

```text
lapis dev train
lapis dev evaluate
lapis dev generate "A language model learns by"
lapis dev chat
lapis dev inspect
lapis dev benchmark
lapis dev checkpoint inspect
lapis dev publish
lapis api serve
```

Legacy top-level developer commands may remain for compatibility, but new integrations and documentation should use the namespaced commands.

## Runtime boundary

The stable Python inference surface is `lapis.inference`, centered on `LapisRuntime` and `SamplingConfig`. External applications should depend on that surface instead of importing internal model classes.

## Testing

Run the repository test suite and static analysis before shipping behavioral changes.

```text
python -m pytest
ruff check .
```

Meaningful behavior changes should include regression coverage. Do not weaken tests to make an implementation pass.

## Configuration

YAML configuration is validated at boundaries. Invalid non-finite and impossible values must fail before model or training construction.
