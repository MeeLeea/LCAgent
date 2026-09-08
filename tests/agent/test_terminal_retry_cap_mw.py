"""TerminalRetryCapMW 超时重试上限中间件测试。

验证方案 B 的 cap 逻辑：
- 历史 exec 工具超时 < MAX_TIMEOUT_RETRIES 时放行（主模型自行反思重试）
- 达上限时拦截，返回失败 ToolMessage(status="error")，阻止无限重试
- 非 exec 工具不受 cap 影响
- state 为 None / 非常规格式时不崩溃
"""
import asyncio
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain.messages import ToolMessage

from agent.terminal_retry_cap_mw import (
    MAX_TIMEOUT_RETRIES,
    TerminalRetryCapMW,
    is_timeout_content,
)


def _make_timeout_tool_msg(name: str, call_id: str) -> ToolMessage:
    """构造一个超时结果 ToolMessage（L1 新格式 error_type:timeout）。"""
    return ToolMessage(
        content='{"error_type": "timeout", "error": "命令超时（60秒）：死循环或计算密集"}',
        tool_call_id=call_id,
        name=name,
    )


def _make_request(tool_name: str, state: object) -> SimpleNamespace:
    """构造最小化 request（只含 tool_call 和 state，够中间件使用）。"""
    return SimpleNamespace(
        tool_call={"name": tool_name, "args": {}, "id": "current_call"},
        state=state,
    )


# ============ is_timeout_content 识别测试 ============

def test_is_timeout_content_new_format():
    """L1 新格式 error_type:timeout。"""
    assert is_timeout_content('{"error_type": "timeout", "error": "命令超时"}')


def test_is_timeout_content_wrapper_format():
    """tool_wrapper 外层格式 error:tool_timeout。"""
    assert is_timeout_content('{"error": "tool_timeout", "tool": "run_shell"}')


def test_is_timeout_content_legacy_text():
    """旧文案格式。"""
    assert is_timeout_content("Python 脚本执行超时（60秒）")
    assert is_timeout_content("命令超时（60秒）")


def test_is_timeout_content_non_timeout():
    """非超时结果不应误判。"""
    assert not is_timeout_content('{"success": true, "stdout": "ok"}')
    assert not is_timeout_content("脚本执行失败 (exit 1): error")
    assert not is_timeout_content("")
    assert not is_timeout_content(None)


# ============ cap 放行测试 ============

def test_cap_allows_when_no_prior_timeout():
    """无历史超时，放行。"""
    mw = TerminalRetryCapMW()
    request = _make_request("run_shell", {"messages": []})
    called = []

    def handler(req):
        called.append(req)
        return ToolMessage(content="ok", tool_call_id="current_call", name="run_shell")

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "ok"


def test_cap_allows_below_limit():
    """历史超时 < MAX_TIMEOUT_RETRIES，放行。"""
    mw = TerminalRetryCapMW()
    state = {
        "messages": [
            _make_timeout_tool_msg("run_shell", "c1"),
            _make_timeout_tool_msg("run_shell", "c2"),
        ]
    }
    assert len(state["messages"]) < MAX_TIMEOUT_RETRIES
    request = _make_request("run_shell", state)
    called = []

    def handler(req):
        called.append(req)
        return ToolMessage(content="ok", tool_call_id="current_call", name="run_shell")

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "ok"


# ============ cap 拦截测试 ============

def test_cap_blocks_at_limit():
    """历史超时 >= MAX_TIMEOUT_RETRIES，拦截返回失败 ToolMessage。"""
    mw = TerminalRetryCapMW()
    state = {
        "messages": [
            _make_timeout_tool_msg("run_shell", "c1"),
            _make_timeout_tool_msg("run_shell", "c2"),
            _make_timeout_tool_msg("run_shell", "c3"),
        ]
    }
    request = _make_request("run_shell", state)
    called = []

    def handler(req):
        called.append(req)  # 不应被调用
        return ToolMessage(content="should not reach", tool_call_id="x", name="run_shell")

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 0
    assert "超时重试已达上限" in result.content
    assert result.status == "error"
    assert result.name == "run_shell"


def test_cap_async_blocks_at_limit():
    """异步版同样拦截。"""
    mw = TerminalRetryCapMW()
    state = {
        "messages": [
            _make_timeout_tool_msg("run_shell", f"c{i}")
            for i in range(MAX_TIMEOUT_RETRIES)
        ]
    }
    request = _make_request("run_shell", state)
    called = []

    async def handler(req):
        called.append(req)
        return ToolMessage(content="x", tool_call_id="x", name="run_shell")

    result = asyncio.run(mw.awrap_tool_call(request, handler))
    assert len(called) == 0
    assert "超时重试已达上限" in result.content


# ============ 非 exec 工具不拦截 ============

def test_cap_ignores_non_exec_tools():
    """非 exec 工具（如 read_file）不受 cap 影响，直接放行。"""
    mw = TerminalRetryCapMW()
    state = {
        "messages": [
            _make_timeout_tool_msg("run_shell", f"c{i}")
            for i in range(MAX_TIMEOUT_RETRIES + 5)
        ]
    }
    request = _make_request("read_file", state)
    called = []

    def handler(req):
        called.append(req)
        return ToolMessage(content="file content", tool_call_id="current_call", name="read_file")

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "file content"


# ============ 边界：state 格式容错 ============

def test_cap_handles_none_state():
    """state 为 None 不崩溃，视为 0 次超时，放行。"""
    mw = TerminalRetryCapMW()
    request = SimpleNamespace(
        tool_call={"name": "run_shell", "args": {}, "id": "x"},
        state=None,
    )
    called = []

    def handler(req):
        called.append(req)
        return ToolMessage(content="ok", tool_call_id="x", name="run_shell")

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "ok"


def test_cap_counts_only_exec_timeouts():
    """只统计 exec 工具的超时，非 exec 工具的 error 不计入。"""
    mw = TerminalRetryCapMW()
    state = {
        "messages": [
            _make_timeout_tool_msg("run_shell", "c1"),
            _make_timeout_tool_msg("read_file", "c2"),  # 非 exec，不计入
            _make_timeout_tool_msg("run_python", "c3"),
        ]
    }
    # 实际 exec 超时 = 2（run_shell + run_python），< 3，应放行
    request = _make_request("run_shell", state)
    called = []

    def handler(req):
        called.append(req)
        return ToolMessage(content="ok", tool_call_id="x", name="run_shell")

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "ok"
