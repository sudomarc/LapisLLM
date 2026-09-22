# CHAD / LapisLLM integration

LapisLLM is the model-engine/runtime half of the three-repository ecosystem.

```text
vibe-coding-instructions
        ↓
   agent policies
        ↓
      CHAD
        ↓
  model gateway
        ↓
    LapisLLM
```

## Lapis responsibility

Lapis owns:

- model architecture;
- tokenizer;
- model/training data workflows;
- training and checkpointing;
- evaluation of model behavior;
- inference/runtime;
- generation and streaming runtime primitives;
- product-agnostic model metadata;
- developer/research tooling.

Lapis does not own:

- consumer accounts;
- conversation persistence;
- agent orchestration;
- user/project memory;
- product UI;
- consumer tool permissions.

## Public integration boundary

CHAD must consume Lapis through the stable inference/runtime/API surface.

CHAD must not import:

- `lapis.model.*`;
- `lapis.training.*`;
- `lapis.tokenizer.*`;

for product behavior.

Lapis may evolve internal implementation freely when the public contract remains compatible.

## Required capability metadata

The integration should expose, where known:

- model identifier;
- software/runtime version;
- context length;
- modalities;
- streaming support;
- tool-calling support where implemented;
- cancellation support where implemented;
- tokenizer compatibility/version;
- parameter count and other diagnostic metadata where useful.

Unknown capabilities must remain unknown rather than being inferred.

## Integration tests

Cross-repository compatibility should be tested against the public Lapis interface.

At minimum:

```text
health/model discovery
      ↓
generation
      ↓
context-limit handling
      ↓
streaming when advertised
      ↓
error mapping
```

A capability is not considered active because a route or class exists in source code. The representative public operation must work in the target environment.

## Change propagation

A breaking or material runtime contract change requires:

1. update Lapis contract/documentation;
2. update CHAD adapter/tests;
3. update compatibility documentation;
4. run representative integration verification.

Do not add CHAD-specific response fields or concepts to Lapis merely for UI convenience.

## Ecosystem references

- CHAD: https://github.com/sudomarc/CHAD
- Vibe Coding Instructions: https://github.com/sudomarc/vibe-coding-instructions

The ecosystem remains three repositories with explicit boundaries, not a monorepo.
