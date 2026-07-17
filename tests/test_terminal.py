import os
import sys
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.terminal_tools import (
    UserRejectedCommandError,
    _guard_command,
    _redact_command,
    _truncate,
    run_shell,
)


def test_truncate_short():
    assert _truncate("abc", 10) == "abc"


def test_truncate_long():
    out = _truncate("a" * 10, 5)
    assert out.startswith("a" * 5)
    assert "截断" in out


def test_guard_deny():
    r = _guard_command("rm -rf /")
    assert r is not None and r["success"] is False


def test_guard_allow():
    assert _guard_command("echo hi") is None


def test_guard_stops_turn_when_user_rejects(monkeypatch):
    # 用户输入 N 后必须抛出终止信号，不能把普通失败结果交回模型继续重试。
    monkeypatch.setattr("tools.terminal_tools.confirm", lambda prompt: False)

    with pytest.raises(UserRejectedCommandError):
        _guard_command("python cleanup.py")


@pytest.mark.parametrize(
    ("command", "secret"),
    [
        ("python deploy.py --api-key sk-secret", "sk-secret"),
        ("python deploy.py --token=token-secret", "token-secret"),
        ("PASSWORD=pass-secret python deploy.py", "pass-secret"),
        ('python deploy.py --header "Authorization: Bearer bearer-secret"', "bearer-secret"),
    ],
)
def test_redact_command_hides_sensitive_values(command, secret):
    redacted = _redact_command(command)

    assert secret not in redacted
    assert "***" in redacted


def test_guard_confirmation_shows_redacted_command_and_matching_reason(monkeypatch):
    prompts = []

    def reject(prompt):
        prompts.append(prompt)
        return False

    monkeypatch.setattr("tools.terminal_tools.confirm", reject)

    with pytest.raises(UserRejectedCommandError):
        _guard_command("python deploy.py --api-key sk-secret")

    assert len(prompts) == 1
    assert "匹配危险模式:" in prompts[0]
    assert "待执行命令：python deploy.py --api-key ***" in prompts[0]
    assert "sk-secret" not in prompts[0]


class _FakeResult:
    returncode = 0
    stdout = "ok"
    stderr = None


def test_run_shell_safe(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult())
    r = run_shell.invoke({"command": "echo hi"})
    assert r["success"] is True
    assert r["stdout"] == "ok"


def test_run_shell_deny(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeResult())
    r = run_shell.invoke({"command": "rm -rf /"})
    assert r["success"] is False
