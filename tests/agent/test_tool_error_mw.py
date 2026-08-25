"""ToolExecutionErrorMW 测试 - 验证工具异常转换为 LLM 可读的错误 ToolMessage。

覆盖：
- 异常类型名出现在错误内容中（langchain 官方建议"点名异常类型"）
- 从 runtime config 动态读取 workspace 并附加路径提示
- 未绑定 workspace 时省略路径提示
- 附加反思指令
- awrap_tool_call 捕获异常 → 返回 ToolMessage(status="error")，ReAct 循环继续
- 工具正常执行时不干预（透传结果）

运行：
  pytest tests/agent/test_tool_error_mw.py -v
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain.messages import ToolMessage

from agent.tool_error_mw import ToolExecutionErrorMW


@dataclass
class FakeRuntime:
    """模拟 ToolRuntime，只保留中间件用到的 config 字段。"""

    config: dict[str, Any]


@dataclass
class FakeToolCallRequest:
    """模拟 ToolCallRequest，对齐 langchain 结构（runtime.config 存 workspace）。"""

    tool_call: dict[str, Any]
    runtime: FakeRuntime
    tool: Any = None


def _make_request(tool_name: str, args: dict, config: dict | None = None):
    """构造 FakeToolCallRequest。"""
    return FakeToolCallRequest(
        tool_call={"name": tool_name, "args": args, "id": "call-1"},
        runtime=FakeRuntime(config=config or {}),
    )


def _make_config(workspace: str | None) -> dict:
    """构造含 workspace_path 的 config。"""
    if workspace is None:
        return {"configurable": {}}
    return {"configurable": {"workspace_path": workspace}}


# --------------------------------------------------------------------------- #
# 错误消息格式化
# --------------------------------------------------------------------------- #
class TestErrorFormatting:
    """错误消息内容断言。"""

    def test_mentions_exception_type_and_message(self):
        """内容点名异常类型 + 原始错误信息。"""
        mw = ToolExecutionErrorMW()
        request = _make_request("read_text_file", {"path": "x.md"})

        content = asyncio.run(mw._aon_error(FileNotFoundError("no such file"), request))

        assert "FileNotFoundError" in content
        assert "no such file" in content
        assert "read_text_file" in content

    def test_includes_workspace_hint_when_bound(self, tmp_path):
        """绑定 workspace 时附加工作空间根目录 + 相对路径提示。"""
        mw = ToolExecutionErrorMW()
        ws = str(tmp_path / "document")
        request = _make_request(
            "read_text_file", {"path": "document/x.md"}, _make_config(ws),
        )

        content = asyncio.run(mw._aon_error(OSError("path error"), request))

        assert ws in content
        assert "相对路径" in content
        assert "重复拼接" in content

    def test_omits_workspace_hint_when_unbound(self):
        """未绑定 workspace（旧会话）时省略路径提示，但保留反思指令。"""
        mw = ToolExecutionErrorMW()
        request = _make_request("read_text_file", {"path": "x.md"}, _make_config(None))

        content = asyncio.run(mw._aon_error(RuntimeError("boom"), request))

        assert "RuntimeError" in content
        assert "工作空间根目录" not in content
        assert "反思" in content

    def test_includes_reflection_directive(self):
        """所有错误内容都包含反思指令。"""
        mw = ToolExecutionErrorMW()
        request = _make_request("calculate", {"expression": "1+1"})

        content = asyncio.run(mw._aon_error(ValueError("bad expr"), request))

        assert "反思失败原因" in content
        assert "修正后重试" in content

    def test_empty_exception_message_fallback(self):
        """异常消息为空时使用兜底文案。"""
        mw = ToolExecutionErrorMW()
        request = _make_request("read_file", {"path": "x"})

        content = asyncio.run(mw._aon_error(ValueError(""), request))

        assert "无详细信息" in content


# --------------------------------------------------------------------------- #
# awrap_tool_call 行为
# --------------------------------------------------------------------------- #
class TestAwrapToolCall:
    """中间件对工具调用链路的拦截行为。"""

    def test_error_converted_to_tool_message_with_error_status(self):
        """工具抛异常 → 返回 ToolMessage(status="error")，不向外抛。"""
        mw = ToolExecutionErrorMW()
        request = _make_request(
            "read_text_file", {"path": "x.md"}, _make_config(r"D:\work\document"),
        )

        async def failing_handler(req) -> ToolMessage:
            raise FileNotFoundError("parent directory does not exist")

        result = asyncio.run(mw.awrap_tool_call(request, failing_handler))

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "call-1"
        assert "FileNotFoundError" in result.content
        assert "反思失败原因" in result.content

    def test_success_passthrough(self):
        """工具正常执行时透传结果，不干预。"""
        mw = ToolExecutionErrorMW()
        request = _make_request("calculate", {"expression": "1+1"})

        async def ok_handler(req) -> ToolMessage:
            return ToolMessage(content="2", tool_call_id="call-1", name="calculate")

        result = asyncio.run(mw.awrap_tool_call(request, ok_handler))

        assert result.content == "2"
        assert getattr(result, "status", "") != "error"

    def test_all_exceptions_are_handled(self):
        """任意异常类型都转为 ToolMessage（用户确认：所有工具错误都转成 toolMessage）。"""
        mw = ToolExecutionErrorMW()
        request = _make_request("run_shell", {"command": "ls"})

        for exc in (
            PermissionError("denied"),
            TimeoutError("timeout"),
            KeyError("missing"),
            Exception("generic"),
        ):
            async def failing_handler(req, e=exc) -> ToolMessage:
                raise e

            result = asyncio.run(mw.awrap_tool_call(request, failing_handler))

            assert isinstance(result, ToolMessage)
            assert result.status == "error"
            assert type(exc).__name__ in result.content
