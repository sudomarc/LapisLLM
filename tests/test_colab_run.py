from __future__ import annotations

import os
import sys
import types

from scripts import colab_run


def test_get_github_token_prefers_environment(monkeypatch):
    monkeypatch.setenv("GH_TOKEN", "  env-token  ")
    assert colab_run.get_github_token() == "env-token"


def test_get_github_token_reads_colab_secret(monkeypatch):
    for name in ("GITHUB_TOKEN", "GH_TOKEN", "LAPIS_GITHUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    userdata = types.SimpleNamespace(get=lambda name: "colab-token" if name == "GITHUB_TOKEN" else None)
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.userdata = userdata
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    assert colab_run.get_github_token() == "colab-token"


def test_github_auth_env_does_not_modify_environment(monkeypatch):
    monkeypatch.delenv("GIT_ASKPASS", raising=False)
    with colab_run.github_auth_env("secret-token") as env:
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["LAPIS_GIT_TOKEN"] == "secret-token"
        assert os.environ.get("GIT_ASKPASS") is None
