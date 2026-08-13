"""WorkspaceSecurityMiddleware 测试 - 验证文件/执行类工具的 workspace 隔离。

覆盖：
- 文件工具相对路径解析为 workspace 内绝对路径
- 路径逃逸（../）被拦截
- 执行类工具 cwd 强制对齐 workspace
- 执行类工具 file_path 逃逸被拦截
- 无 workspace 绑定时直接放行（兼容旧会话）
- 非文件/执行类工具直接放行

运行：
  pytest tests/test_workspace_middleware.py -v
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from agent.workspace_middleware import WorkspaceSecurityMiddleware


@dataclass
class FakeRuntime:
    """模拟 ToolRuntime，只保留中间件用到的 config 字段。"""

    config: dict[str, Any]


@dataclass
class FakeToolCallRequest:
    """模拟 ToolCallRequest，使 override() 返回含真实 args 的对象。

    绕开 MagicMock 的链式属性返回 MagicMock 问题：
    override() 返回新的 FakeToolCallRequest，其 tool_call 指向真实 dict。

    结构对齐 langchain 1.3 ToolCallRequest：config 在 runtime.config 而非顶层。
    """

    tool_call: dict[str, Any]
    runtime: FakeRuntime

    def override(self, **kwargs: Any) -> FakeToolCallRequest:
        """返回新的 request，tool_call 用 kwargs 覆盖。"""
        new_tc = dict(self.tool_call)
        if "tool_call" in kwargs:
            new_tc = dict(kwargs["tool_call"])
        return FakeToolCallRequest(tool_call=new_tc, runtime=self.runtime)


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
    return {"configurable": {"workspace_path": os.path.realpath(workspace)}}


# --------------------------------------------------------------------------- #
# 文件类工具：路径解析
# --------------------------------------------------------------------------- #
class TestFilesystemToolResolution:
    """文件工具的 workspace 路径解析。"""

    def test_read_file_relative_path_resolved(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "test.txt").write_text("hello", encoding="utf-8")

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request("read_file", {"path": "test.txt"}, _make_config(str(ws)))

        mw.wrap_tool_call(request, handler)

        # handler 被调用，且 path 被解析为绝对路径
        assert handler.called
        new_args = handler.call_args[0][0].tool_call["args"]
        assert new_args["path"] == os.path.realpath(str(ws / "test.txt"))

    def test_write_file_relative_path_resolved(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "write_file", {"path": "out.txt", "content": "data"},
            _make_config(str(ws)),
        )

        mw.wrap_tool_call(request, handler)

        new_args = handler.call_args[0][0].tool_call["args"]
        assert new_args["path"] == os.path.realpath(str(ws / "out.txt"))

    def test_list_directory_default_dot_resolved(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "list_directory", {"path": "."}, _make_config(str(ws)),
        )

        mw.wrap_tool_call(request, handler)

        new_args = handler.call_args[0][0].tool_call["args"]
        assert new_args["path"] == os.path.realpath(str(ws))

    def test_move_file_both_paths_resolved(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "move_file",
            {"source_path": "a.txt", "destination_path": "b.txt"},
            _make_config(str(ws)),
        )

        mw.wrap_tool_call(request, handler)

        new_args = handler.call_args[0][0].tool_call["args"]
        assert new_args["source_path"] == os.path.realpath(str(ws / "a.txt"))
        assert new_args["destination_path"] == os.path.realpath(str(ws / "b.txt"))


# --------------------------------------------------------------------------- #
# 文件类工具：逃逸拦截
# --------------------------------------------------------------------------- #
class TestFilesystemToolEscapeBlocked:
    """路径逃逸被拦截。"""

    def test_read_file_dotdot_escape_blocked(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "read_file", {"path": "../../etc/passwd"}, _make_config(str(ws)),
        )

        result = mw.wrap_tool_call(request, handler)

        # handler 不应被调用（逃逸被拦截）
        assert not handler.called
        # 返回错误 ToolMessage
        assert "逃逸" in result.content

    def test_read_file_absolute_escape_blocked(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("secret", encoding="utf-8")

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "read_file", {"path": str(outside)}, _make_config(str(ws)),
        )

        result = mw.wrap_tool_call(request, handler)

        assert not handler.called
        assert "逃逸" in result.content

    def test_move_file_destination_escape_blocked(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "move_file",
            {"source_path": "a.txt", "destination_path": "../../outside.txt"},
            _make_config(str(ws)),
        )

        result = mw.wrap_tool_call(request, handler)

        assert not handler.called
        assert "逃逸" in result.content


# --------------------------------------------------------------------------- #
# 执行类工具：cwd 强制对齐 + 路径校验
# --------------------------------------------------------------------------- #
class TestExecToolCwdAlignment:
    """执行类工具 cwd 强制对齐 workspace。"""

    def test_run_shell_cwd_forced_to_workspace(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "run_shell",
            {"command": "ls", "cwd": "/some/other/dir"},
            _make_config(str(ws)),
        )

        mw.wrap_tool_call(request, handler)

        new_args = handler.call_args[0][0].tool_call["args"]
        # cwd 被强制覆盖为 workspace
        assert new_args["cwd"] == os.path.realpath(str(ws))

    def test_run_shell_cwd_default_when_not_provided(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "run_shell", {"command": "ls"}, _make_config(str(ws)),
        )

        mw.wrap_tool_call(request, handler)

        new_args = handler.call_args[0][0].tool_call["args"]
        assert new_args["cwd"] == os.path.realpath(str(ws))

    def test_run_python_file_path_escape_blocked(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "run_python",
            {"file_path": "../../malicious.py", "cwd": str(ws)},
            _make_config(str(ws)),
        )

        result = mw.wrap_tool_call(request, handler)

        assert not handler.called
        assert "逃逸" in result.content

    def test_run_python_file_path_in_workspace_allowed(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()
        (ws / "script.py").write_text("print(1)", encoding="utf-8")

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "run_python",
            {"file_path": "script.py"},
            _make_config(str(ws)),
        )

        mw.wrap_tool_call(request, handler)

        new_args = handler.call_args[0][0].tool_call["args"]
        assert new_args["file_path"] == os.path.realpath(str(ws / "script.py"))
        assert new_args["cwd"] == os.path.realpath(str(ws))


# --------------------------------------------------------------------------- #
# 无 workspace 绑定 + 非目标工具：直接放行
# --------------------------------------------------------------------------- #
class TestPassthroughCases:
    """无 workspace 绑定或非目标工具时直接放行。"""

    def test_no_workspace_passes_through(self, tmp_path):
        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "read_file", {"path": "test.txt"}, _make_config(None),
        )

        mw.wrap_tool_call(request, handler)

        # handler 被调用，args 不变
        assert handler.called
        assert handler.call_args[0][0].tool_call["args"]["path"] == "test.txt"

    def test_non_target_tool_passes_through(self, tmp_path):
        ws = tmp_path / "workspace"
        ws.mkdir()

        mw = WorkspaceSecurityMiddleware()
        handler = MagicMock(return_value="ok")
        request = _make_request(
            "calculate", {"expression": "1+1"}, _make_config(str(ws)),
        )

        mw.wrap_tool_call(request, handler)

        assert handler.called
        assert handler.call_args[0][0].tool_call["args"]["expression"] == "1+1"
