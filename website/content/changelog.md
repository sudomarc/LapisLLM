# Changelog

## 0.2.0

Lapis moved to an explicit developer/research platform boundary. The current release centers the stable `lapis.inference` runtime, developer-only CLI operations, a local OpenAI-style API, verified checkpoint publication, and hardened non-interactive Colab training.

Recent fixes include invoking the actual trainer from Colab, verifying paired checkpoint/tokenizer artifacts, and keeping generated experiment state separated from unrelated source changes.
