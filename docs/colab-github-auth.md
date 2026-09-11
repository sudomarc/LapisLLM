# Colab + GitHub authentication

The LapisLLM Colab runner uses a GitHub token only for authenticated Git operations such as publishing `training_history/` and verified checkpoint outputs under `checkpoints/`.

## Recommended setup

In Google Colab, open the Secrets panel and create:

```text
Name: GITHUB_TOKEN
Value: <GitHub fine-grained PAT>
```

The token should have only the repository permissions required by the workflow. For a repository where the runner must write to `main`, grant the minimum contents permission needed for that repository.

Run the canonical orchestrator:

```python
%cd /content/LapisLLM
!git pull --ff-only
!python -m scripts.colab_train
```

The canonical path is non-interactive. It must not prompt for a run count, password, token, or consumer chat input.

## Authentication precedence

The runner checks these sources in order:

1. `GITHUB_TOKEN` environment variable
2. `GH_TOKEN` environment variable
3. `LAPIS_GITHUB_TOKEN` environment variable
4. Colab Secret `GITHUB_TOKEN`
5. Colab Secret `GH_TOKEN`
6. Colab Secret `LAPIS_GITHUB_TOKEN`
7. Existing Git credentials are not converted into an interactive prompt; when no usable authentication is available, the runner fails clearly.

## Token handling

The token is never written to repository configuration, never inserted into `origin`, and never passed as a command-line argument. Git authentication uses a temporary `GIT_ASKPASS` helper and an environment variable that exists only for the lifetime of the Git command.

The helper is deleted immediately after the Git operation. The runner also avoids printing its authentication environment.

## What is pushed

The runner may publish only the generated paths owned by the training workflow:

```text
training_history/
checkpoints/latest.pt
checkpoints/tokenizer/
```

Unrelated source changes abort synchronization instead of being silently overwritten. Generated corpus text remains excluded from Git.

## Security notes

Do not put a PAT directly in a notebook cell, shell command, Python source file, YAML configuration, or Git remote URL. Do not commit a token or a copied Colab secret anywhere in the repository.

If a token is ever exposed, revoke it immediately and create a replacement with the minimum required permissions.
