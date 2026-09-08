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


def test_push_history_uses_dedicated_branch_from_main(monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stdout = "main\n"

    def fake_git_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[:3] == ["git", "status", "--porcelain"]:
            result = Result()
            result.stdout = " M training_history/run/summary.json\n"
            return result
        if cmd[:3] == ["git", "diff", "--cached"]:
            result = Result()
            result.stdout = "training_history/run/summary.json\n"
            return result
        if cmd[:3] == ["git", "branch", "--show-current"]:
            return Result()
        return Result()

    monkeypatch.setattr(colab_run, "git_run", fake_git_run)
    assert colab_run.push_history("token") is True

    push_commands = [cmd for cmd, _ in calls if cmd[:3] == ["git", "push", "origin"]]
    assert len(push_commands) == 1
    assert push_commands[0][3].startswith("HEAD:training-history/")
    assert push_commands[0][3] != "main"
