# Training LapisLLM on Google Colab

## Purpose and operating model

Google Colab is the primary experimental compute environment because local compute is limited. Colab runtimes are temporary and accelerator availability is dynamic. Training must be interruption-safe.

- GitHub stores source, configs, tests and experiment definitions.
- Colab executes bounded experiments.
- /content is temporary staging storage.
- A durable artifact backend stores complete checkpoints and run manifests.
- GitHub Actions runs CPU CI and bounded smoke tests.

For a manually operated hosted Colab notebook, Google Drive is the initial practical checkpoint destination. Do not commit checkpoints or datasets to Git. Keep artifact persistence behind an interface.

## First-run sequence

1. Open a trusted notebook from the official repository.
2. Select a Colab runtime; request a GPU only when needed.
3. Run preflight and record Python, PyTorch, CUDA, accelerator, VRAM, RAM and free disk.
4. Authenticate to the artifact store using Colab Secrets or an approved OAuth flow.
5. Fetch the exact source commit for the experiment.
6. Install pinned project dependencies.
7. Validate dataset license/provenance manifests, tokenizer and model config.
8. Run a bounded smoke test.
9. Start training with an explicit config and run ID.
10. Save full checkpoints at a configured step or elapsed-time interval.
11. Verify checksums and test restoration.
12. Record metrics and release the runtime when finished.

The notebook must be thin. Reusable logic belongs in importable Python modules.

## Runtime preflight

Record Python/PyTorch versions, CUDA details, accelerator name and VRAM, device dtype support, system RAM, free disk, source commit SHA, configuration, dataset manifest IDs and tokenizer ID.

If no accelerator is available, fail clearly for a GPU-required run or explicitly switch to a CPU smoke profile. Never silently run a long intended GPU job on CPU.

## Checkpoints and interruption

Assume a runtime can be reclaimed, disconnected or stopped without graceful shutdown. Save complete resumable checkpoints to durable storage at bounded intervals. Keep a prior known-good checkpoint. On restart, verify integrity and compatibility before resuming.

Do not write every batch to Drive. Stage writes locally, validate a complete checkpoint and then copy it to durable storage. A partially copied checkpoint must not become the latest valid checkpoint. See [reproducibility.md](reproducibility.md).

## Official Colab API

Google publishes a Colab REST API at https://colaboratory.googleapis.com.

Official references:
- API overview: https://developers.google.com/colab/api/reference/rest
- Runtime specs: https://developers.google.com/colab/api/reference/rest/v1beta/runtimespecs/list
- Create runtime: https://developers.google.com/colab/api/reference/rest/v1beta/runtimes/create
- Runtime resource: https://developers.google.com/colab/api/reference/rest/v1beta/runtimes

The v1beta API can list runtime specifications and their eligibility, create runtime assignments, retrieve runtime details and delete a runtime. Listing specifications supports the read-only OAuth scope https://www.googleapis.com/auth/colaboratory.readonly. Creating/managing runtimes requires https://www.googleapis.com/auth/colaboratory.

A created runtime is a managed Jupyter server. Runtime connection information includes a URL, short-lived proxy token and expiry. Clients send the token in the X-Colab-Runtime-Proxy-Token header and refresh connection information through the runtime get method near expiry. A controller then uses the standard Jupyter REST API/client.

**The Colab API is not a one-call endpoint that accepts a Python script and runs training.** Runtime provisioning and executing notebook code are separate operations. LapisLLM must implement and test the Jupyter interaction layer before claiming that GitHub Actions can launch training end to end.

Runtime creation is quota- and capacity-controlled. An eligible GPU specification is not a guarantee of allocation at request time. Handle permission, quota, capacity, rate-limit and token-expiry errors distinctly.

## Automation stages

**Stage 1 — manual notebook:** the user starts the notebook, selects a runtime, runs preflight and training, and persists checkpoints.

**Stage 2 — API-assisted controller:** after Stage 1 is reliable, implement OAuth with least privilege, runtime discovery, reuse of owned runtimes, idempotent creation, operation polling, authenticated Jupyter interaction, controlled command submission, log streaming, checkpoint verification and cleanup.

Never build Stage 2 before a Micro training run can recover from forced termination.

## Limits and policy

Google states that resource availability and usage limits vary and are not guaranteed. Paid plans may provide increased compute availability but do not make Colab unlimited. Do not bypass quotas or account restrictions. Check the current Colab FAQ and terms before changing the operating model.

Official FAQ: https://research.google.com/colaboratory/faq.html
