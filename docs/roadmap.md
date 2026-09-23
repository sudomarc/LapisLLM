# LapisLLM roadmap

LapisLLM is the model-engineering and runtime layer of the CHAD agentic AI ecosystem.

The roadmap therefore prioritizes **model quality, runtime correctness, serving capability and measurable progress**, while CHAD owns the product/agent layer.

## Phase 0 — Foundation [mostly complete]

- [x] Decoder-only Transformer.
- [x] ByteLevel BPE tokenizer.
- [x] RoPE.
- [x] GQA.
- [x] RMSNorm.
- [x] SwiGLU.
- [x] Configuration validation.
- [x] Checkpoint save/load.
- [x] Generation.
- [x] Public runtime.
- [x] Developer CLI.
- [x] Basic evaluation and regression testing.

## Phase 1 — Runtime contract for CHAD

- [x] Versioned public runtime contract.
- [x] Stable model metadata schema.
- [x] Explicit context-length metadata.
- [x] Capability metadata.
- [ ] Consistent usage metadata.
- [ ] Structured error taxonomy.
- [x] Streaming endpoint/runtime integration.
- [ ] Cancellation support.
- [ ] Contract tests consumed by CHAD.
- [x] Compatibility documentation.

## Phase 2 — Inference correctness and efficiency

- [x] Efficient KV-cache generation.
- [ ] Better batching where justified.
- [ ] Memory/throughput benchmarks.
- [ ] Mixed-precision inference.
- [ ] Quantized inference.
- [ ] Long-context validation.
- [ ] Deterministic generation controls.
- [ ] p50/p95 latency tracking.
- [ ] Regression benchmarks.

## Phase 3 — Training reproducibility

- [ ] Exact mid-epoch replay.
- [ ] Stronger checkpoint resume semantics.
- [ ] Data-loader/sampler state persistence.
- [ ] Full experiment manifests.
- [ ] Dataset provenance validation.
- [ ] Seed/RNG reproducibility verification.
- [ ] Failure/recovery tests.

## Phase 4 — Data and corpus engineering

- [ ] Scalable ingestion.
- [ ] Deduplication.
- [ ] Data quality scoring.
- [ ] Domain mixture controls.
- [ ] Better multilingual data coverage.
- [ ] Provenance/license tracking.
- [ ] Repeatable dataset builds.
- [ ] Training-data evaluation.

## Phase 5 — Instruction and tool-use post-training

- [ ] Instruction-tuning pipeline.
- [ ] Curated instruction dataset.
- [ ] Tool-use dataset.
- [ ] Function/tool schema training.
- [ ] Preference data pipeline.
- [ ] Preference optimization.
- [ ] Safety tuning.
- [ ] Agent-oriented evaluation tasks.
- [ ] CHAD integration benchmark.

## Phase 6 — Model family expansion

- [ ] Larger development checkpoints.
- [ ] Memory/context scaling experiments.
- [ ] Multiple quality/latency tiers.
- [ ] Specialist checkpoints where justified.
- [ ] Quantized release variants.
- [ ] Reproducible release manifests.

Model-size milestones are gated by available compute, data quality, evaluation evidence and serving feasibility. A larger parameter count is not itself a quality claim.

## Phase 7 — Multimodal model capability

- [ ] Multimodal architecture research.
- [ ] Image input pipeline.
- [ ] Multimodal evaluation.
- [ ] Vision-text serving contract.
- [ ] CHAD multimodal integration.

This phase starts only when the model/runtime architecture can support the modality without compromising the stable runtime boundary.

## Phase 8 — Distributed scale

- [ ] Mixed-precision training.
- [ ] Multi-GPU training.
- [ ] Distributed data parallelism.
- [ ] Gradient checkpointing where beneficial.
- [ ] Data sharding.
- [ ] Checkpoint sharding.
- [ ] Training throughput benchmarks.
- [ ] Fault/recovery testing.

## Phase 9 — Production inference

- [ ] Hardened serving process.
- [ ] Request concurrency controls.
- [ ] Queueing/backpressure.
- [ ] Graceful shutdown.
- [ ] Health/readiness endpoints.
- [ ] Load testing.
- [ ] Autoscaling integration.
- [ ] Resource-aware scheduling.
- [ ] Provider-independent observability.

## Phase 10 — Evaluation as a first-class subsystem

- [ ] Versioned benchmark registry.
- [ ] General QA.
- [ ] Reasoning.
- [ ] Coding.
- [ ] Mathematics.
- [ ] Long-context.
- [ ] Tool use.
- [ ] Instruction following.
- [ ] Multilingual.
- [ ] Safety.
- [ ] Regression dashboards.
- [ ] Release gates.

Every reported score must include task set, model/runtime version, configuration, date, methodology and limitations.

## Phase 11 — Ecosystem compatibility

- [ ] Compatibility matrix for CHAD releases.
- [ ] Backward-compatibility policy.
- [ ] Contract-test fixtures.
- [ ] Migration guides.
- [ ] Version negotiation where justified.
- [ ] Integration incident runbook.

## Phase 12 — Research frontier

- [ ] Architecture experiments.
- [ ] Attention/context research.
- [ ] Efficiency research.
- [ ] Distillation where legally and technically appropriate.
- [ ] Better post-training methods.
- [ ] Tool/reasoning experiments.
- [ ] New model-family evaluations.

Research changes must remain isolated from production contracts until validated.

## Out of scope

Lapis must not absorb:

- consumer chat UX;
- user accounts;
- agent orchestration;
- user memory;
- consumer project management;
- consumer billing;
- external side-effect permissions.

Those belong to CHAD.
