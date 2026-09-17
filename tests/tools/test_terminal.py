import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.config import (
    DEFAULT_TIMEOUT,
    SOFT_TIMEOUT_MARGIN,
    TOOL_TIMEOUTS,
    soft_timeout_for,
)
from tools.terminal_tools import (
    _GRACE_PERIOD,
    UserRejectedCommandError,
    _classify_timeout,
    _guard_command,
    _redact_command,
    _truncate,
    run_cmd,
    run_python,
    run_shell,
)


def test_truncate_short():
    assert _truncate("abc", 10) == "abc"


def test_truncate_long():
    out = _truncate("a" * 10, 5)
    assert out.startswith("a" * 5)
    assert "截断" in out


def test_guard_deny():
    # rm -rf tools 目录是询问级路径，会请求确认
    # 模拟用户拒绝确认
    import tools.terminal_tools
    original_confirm = tools.terminal_tools.confirm
    
    def reject_confirm(prompt):
        return False
    
    tools.terminal_tools.confirm = reject_confirm
    try:
        with pytest.raises(UserRejectedCommandError):
            _guard_command("rm -rf tools")
    finally:
        tools.terminal_tools.confirm = original_confirm


def test_guard_deny_protected_path():
    # rm -rf agent/ 是保护级路径，应直接拒绝（不需要确认）
    r = _guard_command("rm -rf agent")
    assert r is not None and r["success"] is False
    assert "保护级路径" in r.get("error", "")


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


class _FakePopen:
    """模拟 subprocess.Popen，配合 _run_with_timeout 测试。

    timeout_once=True 时第一次 communicate 抛 TimeoutExpired（模拟超时），
    _run_with_timeout 会发 ctrl+c 后再次 communicate，第二次返回正常输出。
    """

    def __init__(self, returncode=0, stdout="ok", stderr="", timeout_once=False):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._timeout_once = timeout_once
        self._call_count = 0
        self.pid = 12345

    def communicate(self, timeout=None):
        self._call_count += 1
        if self._timeout_once and self._call_count == 1:
            raise subprocess.TimeoutExpired(cmd="test", timeout=timeout or 0)
        return self._stdout, self._stderr

    def send_signal(self, sig):
        pass

    def kill(self):
        pass


def test_run_shell_safe(monkeypatch):
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: _FakePopen(returncode=0, stdout="ok")
    )
    r = run_shell.invoke({"command": "echo hi"})
    assert r["success"] is True
    assert r["stdout"] == "ok"


def test_run_shell_deny(monkeypatch):
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: _FakePopen(returncode=0, stdout="ok")
    )
    # rm -rf tools 目录是询问级路径，会请求确认
    # 模拟用户拒绝确认
    monkeypatch.setattr("tools.terminal_tools.confirm", lambda prompt: False)

    # 用户拒绝时应该抛出异常
    with pytest.raises(UserRejectedCommandError):
        run_shell.invoke({"command": "rm -rf tools"})


def test_run_shell_timeout(monkeypatch):
    """超时返回 error_type=timeout + timeout_reason + partial_stdout。"""
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *a, **k: _FakePopen(timeout_once=True, stdout="partial output"),
    )
    r = run_shell.invoke({"command": "ping localhost", "timeout": 1})
    assert r["success"] is False
    assert r["error_type"] == "timeout"
    assert "timeout_reason" in r
    assert r["partial_stdout"] == "partial output"


def test_classify_timeout_interactive():
    """交互式命令（python 未加 -c 等非交互标志）。"""
    assert _classify_timeout("python", "", None) == "interactive"


def test_classify_timeout_network():
    """网络命令未带超时标志。"""
    assert _classify_timeout("curl http://example.com", "", None) == "network"


def test_classify_timeout_network_with_timeout_flag():
    """网络命令自带 --max-time 不算网络阻塞。"""
    assert _classify_timeout("curl --max-time 10 http://example.com", "", None) != "network"


def test_classify_timeout_io_block():
    """tail -f 等 IO 持续监听。"""
    assert _classify_timeout("tail -f log.txt", "", None) == "io_block"


def test_classify_timeout_command_error():
    """有输出且 returncode 异常。"""
    assert _classify_timeout("ls", "error output", 1) == "command_error"


def test_classify_timeout_dead_loop():
    """无明显特征，兜底为死循环。"""
    assert _classify_timeout("while true; do :; done", "", None) == "dead_loop"


# --- 软/硬超时分层不变量 ---------------------------------------------------

_TERMINAL_TOOLS = [run_shell, run_python, run_cmd]


def _tool_timeout_default(tool) -> float:
    """从 @tool 暴露给 LLM 的 args schema 读取 timeout 参数默认值。"""
    return float(tool.args["timeout"]["default"])


def test_soft_timeout_margin_exceeds_ctrl_c_grace_period():
    """软/硬超时余量必须大于 ctrl+c grace period，保证软中断流程先跑完。"""
    assert SOFT_TIMEOUT_MARGIN > _GRACE_PERIOD


@pytest.mark.parametrize("tool", _TERMINAL_TOOLS, ids=lambda t: t.name)
def test_terminal_tool_soft_timeout_strictly_less_than_hard(tool):
    """每个终端工具的内层软超时必须严格小于外层硬超时，否则外层先触发丢富结果。"""
    hard = TOOL_TIMEOUTS.get(tool.name, DEFAULT_TIMEOUT)
    soft = _tool_timeout_default(tool)

    assert soft < hard


@pytest.mark.parametrize("tool", _TERMINAL_TOOLS, ids=lambda t: t.name)
def test_terminal_tool_default_equals_derived_soft_timeout(tool):
    """函数签名默认值必须由 soft_timeout_for 派生，避免手写魔法数漂移。"""
    assert _tool_timeout_default(tool) == soft_timeout_for(tool.name)


@pytest.mark.parametrize("name", sorted(TOOL_TIMEOUTS))
def test_soft_timeout_derivation_less_than_hard_for_every_configured_tool(name):
    """TOOL_TIMEOUTS 中每个工具派生出的软超时都必须小于其硬超时。"""
    assert soft_timeout_for(name) < TOOL_TIMEOUTS[name]


def test_run_shell_default_is_590():
    """run_shell 硬超时 600s - 余量 10s = 590s。"""
    assert _tool_timeout_default(run_shell) == 590.0


def test_run_python_and_run_cmd_default_are_50():
    """未在 TOOL_TIMEOUTS 覆盖的工具回退 DEFAULT_TIMEOUT 60s - 余量 10s = 50s。"""
    assert _tool_timeout_default(run_python) == 50.0
    assert _tool_timeout_default(run_cmd) == 50.0


def test_soft_timeout_for_unknown_tool_falls_back_to_default_minus_margin():
    """未知工具名回退 DEFAULT_TIMEOUT - SOFT_TIMEOUT_MARGIN。"""
    assert soft_timeout_for("__unknown_tool__") == DEFAULT_TIMEOUT - SOFT_TIMEOUT_MARGIN
