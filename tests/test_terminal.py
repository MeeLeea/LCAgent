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
