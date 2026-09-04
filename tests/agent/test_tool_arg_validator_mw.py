"""ToolArgValidatorMW 工具参数校验中间件测试。

验证 B 方案的拦截逻辑：
- 互斥参数同时存在时拦截，返回 error ToolMessage（含修正提示）
- 仅一个或都不存在时放行
- None 值视为"未指定"，不触发互斥
- 非目标工具不受影响
- 自定义规则可注入

运行：
  pytest tests/agent/test_tool_arg_validator_mw.py -v
"""
from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain.messages import ToolMessage

from agent.tool_arg_validator_mw import ArgRule, MutexRule, ToolArgValidatorMW


def _make_request(tool_name: str, args: dict) -> SimpleNamespace:
    """构造最小化 request（只含 tool_call，够中间件使用）。"""
    return SimpleNamespace(
        tool_call={"name": tool_name, "args": args, "id": "call-1"},
    )


def _make_handler(called: list):
    """构造一个记录调用的 handler，返回成功 ToolMessage。"""
    def handler(req):
        called.append(req)
        return ToolMessage(content="ok", tool_call_id="call-1", name="read_file")
    return handler


# ============ MutexRule.check 单元测试 ============

def test_mutex_rule_blocks_both_present():
    """两个互斥参数同时存在 → 返回错误信息。"""
    rule = MutexRule(
        tool_name="read_file",
        params=("head", "tail"),
        message="冲突提示",
    )
    assert rule.check({"head": 50, "tail": 100}) == "冲突提示"


def test_mutex_rule_allows_only_head():
    """仅 head 存在 → 通过。"""
    rule = MutexRule(
        tool_name="read_file",
        params=("head", "tail"),
        message="冲突提示",
    )
    assert rule.check({"head": 50, "tail": None}) is None
    assert rule.check({"head": 50}) is None


def test_mutex_rule_allows_only_tail():
    """仅 tail 存在 → 通过。"""
    rule = MutexRule(
        tool_name="read_file",
        params=("head", "tail"),
        message="冲突提示",
    )
    assert rule.check({"head": None, "tail": 100}) is None
    assert rule.check({"tail": 100}) is None


def test_mutex_rule_allows_neither():
    """都不存在 → 通过。"""
    rule = MutexRule(
        tool_name="read_file",
        params=("head", "tail"),
        message="冲突提示",
    )
    assert rule.check({}) is None
    assert rule.check({"head": None, "tail": None}) is None


# ============ 中间件同步拦截测试 ============

def test_sync_blocks_head_tail_conflict():
    """同步：head+tail 同时存在 → 拦截，返回 error ToolMessage。"""
    mw = ToolArgValidatorMW()
    request = _make_request("read_file", {"head": 50, "tail": 100})
    called: list = []
    result = mw.wrap_tool_call(request, _make_handler(called))

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "互斥" in result.content
    assert "head" in result.content
    assert "tail" in result.content
    assert len(called) == 0  # handler 不应被调用


def test_sync_allows_head_only():
    """同步：仅 head → 放行，调用 handler。"""
    mw = ToolArgValidatorMW()
    request = _make_request("read_file", {"head": 50, "path": "x.py"})
    called: list = []
    result = mw.wrap_tool_call(request, _make_handler(called))

    assert len(called) == 1
    assert result.content == "ok"


def test_sync_allows_tail_only():
    """同步：仅 tail → 放行。"""
    mw = ToolArgValidatorMW()
    request = _make_request("read_file", {"tail": 100, "path": "x.py"})
    called: list = []
    result = mw.wrap_tool_call(request, _make_handler(called))

    assert len(called) == 1


def test_sync_allows_neither():
    """同步：head/tail 都不传 → 放行。"""
    mw = ToolArgValidatorMW()
    request = _make_request("read_file", {"path": "x.py"})
    called: list = []
    result = mw.wrap_tool_call(request, _make_handler(called))

    assert len(called) == 1


def test_sync_allows_none_values():
    """同步：head=None, tail=None → 视为未指定，放行。"""
    mw = ToolArgValidatorMW()
    request = _make_request("read_file", {"head": None, "tail": None, "path": "x.py"})
    called: list = []
    result = mw.wrap_tool_call(request, _make_handler(called))

    assert len(called) == 1


def test_sync_allows_non_target_tool():
    """同步：非 read_file 工具 → 不校验，直接放行。"""
    mw = ToolArgValidatorMW()
    request = _make_request("write_file", {"head": 50, "tail": 100, "path": "x.py"})
    called: list = []
    result = mw.wrap_tool_call(request, _make_handler(called))

    assert len(called) == 1


def test_sync_error_message_has_correction_hint():
    """同步：错误信息含修正提示（利于 LLM 自愈）。"""
    mw = ToolArgValidatorMW()
    request = _make_request("read_file", {"head": 50, "tail": 100})
    called: list = []
    result = mw.wrap_tool_call(request, _make_handler(called))

    assert "移除" in result.content  # 含修正指示


# ============ 中间件异步拦截测试 ============

def test_async_blocks_head_tail_conflict():
    """异步：head+tail 同时存在 → 拦截。"""
    mw = ToolArgValidatorMW()
    request = _make_request("read_file", {"head": 50, "tail": 100})
    called: list = []

    async def handler(req):
        called.append(req)
        return ToolMessage(content="ok", tool_call_id="call-1", name="read_file")

    result = asyncio.run(mw.awrap_tool_call(request, handler))

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "互斥" in result.content
    assert len(called) == 0


def test_async_allows_head_only():
    """异步：仅 head → 放行。"""
    mw = ToolArgValidatorMW()
    request = _make_request("read_file", {"head": 50, "path": "x.py"})
    called: list = []

    async def handler(req):
        called.append(req)
        return ToolMessage(content="ok", tool_call_id="call-1", name="read_file")

    result = asyncio.run(mw.awrap_tool_call(request, handler))

    assert len(called) == 1


# ============ read_text_file 工具测试 ============

def test_sync_blocks_read_text_file_conflict():
    """read_text_file 也有 head/tail 互斥规则。"""
    mw = ToolArgValidatorMW()
    request = _make_request("read_text_file", {"head": 50, "tail": 100})
    called: list = []
    result = mw.wrap_tool_call(request, _make_handler(called))

    assert result.status == "error"
    assert "互斥" in result.content
    assert len(called) == 0


# ============ 自定义规则注入测试 ============

def test_custom_rule_injection():
    """自定义规则可通过构造函数注入。"""
    custom_rule = MutexRule(
        tool_name="write_file",
        params=("mode", "append"),
        message="write_file 的 mode 和 append 互斥",
    )
    mw = ToolArgValidatorMW(rules=[custom_rule])
    request = _make_request("write_file", {"mode": "w", "append": True})
    called: list = []
    result = mw.wrap_tool_call(request, _make_handler(called))

    assert result.status == "error"
    assert "mode" in result.content
    assert "append" in result.content
    assert len(called) == 0


def test_custom_rule_does_not_affect_default():
    """自定义规则不影响默认规则覆盖的工具。"""
    custom_rule = MutexRule(
        tool_name="search",
        params=("regex", "glob"),
        message="regex 和 glob 互斥",
    )
    mw = ToolArgValidatorMW(rules=[custom_rule])
    # read_file 不在自定义规则列表中 → 不拦截
    request = _make_request("read_file", {"head": 50, "tail": 100})
    called: list = []
    result = mw.wrap_tool_call(request, _make_handler(called))

    # 自定义规则不包含 read_file → 放行
    assert len(called) == 1
