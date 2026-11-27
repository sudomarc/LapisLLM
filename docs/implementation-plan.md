# LapisLLM implementation plan

This page is the execution index for the GitHub issue backlog. Work proceeds in dependency order. Do not skip a gate because a later issue is marked high priority.

## Stage 0 — Adopt the project foundation

**Documentation PR:** [#73](https://github.com/sudomarc/LapisLLM/pull/73)

Review and merge the repository policy, local-training architecture, model specification, data policy, security policy and roadmap before beginning implementation. This stage adds no executable model code.

## Stage 1 — Establish a clean Python project

**Issue:** [#58 — Python package and CPU-only CI](https://github.com/sudomarc/LapisLLM/issues/58)

Deliver a reproducible Python package, typed configuration foundation, development dependencies, lint/test/build checks and CPU-only GitHub Actions. This is the prerequisite for the implementation tasks.

## Stage 2 — Measure the target PC

**Issue:** [#74 — Hardware audit and measured Micro profile](https://github.com/sudomarc/LapisLLM/issues/74)

Collect a redacted local hardware report and verify PyTorch device support. Measure actual memory and throughput once the model/trainer can run. Do not freeze the model profile, batch size, context length or precision before this evidence exists.

## Stage 3 — Define lawful, useful training data

**Issues:** [#59 — Dataset registry and license gate](https://github.com/sudomarc/LapisLLM/issues/59), [#72 — Programming-language and dataset mixture](https://github.com/sudomarc/LapisLLM/issues/72)

First build the source registry and approval checks. Then select a small initial language/data mixture. Do not ingest or train on sources with unknown or incompatible terms. Keep raw data outside Git.

## Stage 4 — Train the tokenizer

**Issue:** [#60 — Code-oriented BPE tokenizer](https://github.com/sudomarc/LapisLLM/issues/60)

Train only on the approved training partition. Freeze the tokenizer metadata and hash before model training. Check round-trip behavior, special tokens and code tokenization.

## Stage 5 — Implement Lapis Micro

**Issue:** [#61 — Decoder-only Transformer](https://github.com/sudomarc/LapisLLM/issues/61)

Implement the model from random initialization. Add shape, causal-mask, gradient and numerical tests. Finalize the Micro configuration only after Stage 2 measurements.

## Stage 6 — Implement the training engine

**Issue:** [#62 — Next-token training engine](https://github.com/sudomarc/LapisLLM/issues/62)

Implement the loss, optimizer, scheduler, accumulation, clipping, precision selection and progress metrics. Keep long training out of CI.

## Stage 7 — Make training recoverable

**Issues:** [#63 — Resumable checkpoints](https://github.com/sudomarc/LapisLLM/issues/63), [#65 — Local artifact storage](https://github.com/sudomarc/LapisLLM/issues/65)

These can proceed in parallel once the model/config contract is stable. Define complete training state, safe local writes, checksums, retention and restore compatibility. Test process termination and restart.

## Stage 8 — Build the local training command

**Issue:** [#64 — Local PC training CLI and preflight](https://github.com/sudomarc/LapisLLM/issues/64)

Compose the hardware diagnostic, data validation, training engine and checkpoint manager into one command. Support explicit configs, run IDs, data/output roots, progress, Ctrl+C and resume. Do not use notebooks or hosted runners as the canonical path.

## Stage 9 — Evaluation and local generation

**Issues:** [#66 — Coding evaluation harness](https://github.com/sudomarc/LapisLLM/issues/66), [#69 — Local inference and developer CLI](https://github.com/sudomarc/LapisLLM/issues/69)

Implement checkpoint-backed generation and an evaluation harness with fixed prompts, versioned tasks and sandboxed code execution. Evaluation must run against held-out data and retain failures as well as successes.

## Stage 10 — First end-to-end local experiment

**Issue:** [#67 — First Lapis Micro experiment](https://github.com/sudomarc/LapisLLM/issues/67)

Train from random weights on the maintainer's PC. Produce a complete run manifest and verified checkpoint. Deliberately interrupt and resume the process. Report actual device, precision, peak memory, tokens, throughput, wall time and evaluation results.

## Stage 11 — Coding post-training

**Issue:** [#68 — Coding instruction post-training](https://github.com/sudomarc/LapisLLM/issues/68)

Only after a baseline has been measured, apply supervised instruction tuning to LapisLLM's own checkpoint. Compare against the base model on the same evaluation suite and report regressions.

## Stage 12 — Release gate

**Issue:** [#71 — Model-weight license](https://github.com/sudomarc/LapisLLM/issues/71)

Resolve model-artifact licensing, dataset obligations, model card, provenance and evaluation disclosure before publishing weights. The source-code MIT license does not automatically license weights or training data.

## Dependency summary

~~~text
#58 package + CPU CI
 ├── #74 hardware discovery
 └── #59 dataset registry
      └── #72 data mixture
           └── #60 tokenizer
                └── #61 Micro model
                     └── #62 training engine
                          ├── #63 checkpoint state
                          ├── #65 local artifact storage
                          └── #64 local training command
                               ├── #66 evaluation
                               ├── #69 inference CLI
                               └── #67 end-to-end Micro run
                                    └── #68 coding post-training

#71 model-weight license is a release blocker.
~~~

The diagram shows the logical order, not a requirement that every independent task be done by one person. #63 and #65 are intentionally parallelizable. #74 can begin once the package foundation exists, but its model benchmark portion waits for the Micro model and trainer.

## Working rules

- Each issue should be delivered through a focused pull request referencing the issue.
- A PR is complete only when acceptance criteria, tests and documentation are satisfied.
- Do not close an issue merely because a draft implementation exists.
- Record hardware and training metrics from actual runs. Never invent estimates as results.
- Keep datasets, logs, checkpoints and generated model artifacts out of Git.
- Do not increase model size until the prior profile's memory, throughput and quality have been measured.
