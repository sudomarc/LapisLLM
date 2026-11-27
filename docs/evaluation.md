# Evaluation policy

## Principle

Training loss measures fit to the training objective; it does not prove that a model can solve programming tasks, follow instructions or edit repositories. LapisLLM must publish evidence, failures and limitations rather than describe intended capabilities as existing behavior.

## Evaluation stages

**Stage 1 — engineering correctness**
- forward pass produces finite logits with expected shapes;
- causal masking blocks future-token information;
- loss decreases on a tiny known corpus;
- gradients are finite and non-zero where expected;
- checkpoints save, validate and restore;
- CPU and accelerator paths agree within documented tolerances.

**Stage 2 — language-model quality**
- held-out validation loss and perplexity;
- loss by language/file type when sample sizes permit;
- duplicate and contamination checks;
- generated samples inspected using fixed prompts and decoding settings.

**Stage 3 — coding behavior**
Build a versioned internal task set for code completion, function generation, bug diagnosis/fixing, code explanation, test generation and small refactoring. Use hidden tests or isolated execution where appropriate.

**Stage 4 — external benchmarks**
Potential families include HumanEval, MBPP and later repository-level suites such as SWE-bench. Review their licenses and terms, use the appropriate harness, and explain that repository-agent benchmarks are not raw language-model benchmarks.

## Required result metadata

Every reported result includes model revision and checkpoint hash, tokenizer hash, benchmark/dataset revision, prompt template, few-shot policy, decoding config, random seeds, harness version, hardware/software environment, date, run ID, sample count, exclusions, metric and aggregation method, contamination checks and limitations.

## Fair comparison

Use equivalent task sets, prompt policies, execution budgets, decoding settings and scoring rules when comparing checkpoints. Report pass@k only with its exact sampling protocol. Do not compare different harnesses as though the scores were directly equivalent.

## Release gate

A public model card must disclose training lineage, data limitations, intended use, failure modes, evaluation setup and limitations. Do not claim parity with Claude, GPT, Gemini or another model unless a transparent, reproducible evaluation supports the exact bounded claim.
