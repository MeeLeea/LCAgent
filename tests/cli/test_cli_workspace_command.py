"""workspace 命令的 CLI 层测试。

覆盖子命令分支：
- workspace（无参数=查看）
- workspace <path>（设置，含校验失败）
- workspace:clear / workspace clear（清除，含已绑定/未绑定）
- workspace:help / workspace help（帮助）
- dispatcher 路由验证（workspace 命令不被 chat_mode 兜底吞掉）
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cli.commands.dispatcher import dispatch_command
from cli.commands.types import CommandContext
from cli.commands.workspace import workspace_command

# ============ Fake 层 ============


@dataclass
class FakeSession:
    """模拟 SessionRegistry 的 workspace 相关接口。

    aset_workspace 模拟真实校验：路径必须是已存在的目录，否则抛 ValueError。
    """

    current_session_id: str = "thread-test"
    _workspace: str | None = None
    calls: list[tuple[str, Any]] = field(default_factory=list)

    async def aget_workspace(self, session_id: str | None = None) -> str | None:
        self.calls.append(("aget_workspace", session_id))
        return self._workspace

    async def aset_workspace(
        self, workspace_path: str, session_id: str | None = None
    ) -> str:
        self.calls.append(("aset_workspace", (workspace_path, session_id)))
        # 模拟真实校验：必须是已存在的目录
        if not workspace_path or not workspace_path.strip():
            raise ValueError("工作空间路径不能为空")
        abs_path = os.path.abspath(workspace_path)
        if not os.path.isdir(abs_path):
            raise ValueError(f"工作空间路径不存在或不是目录: {abs_path}")
        real_path = os.path.realpath(abs_path)
        self._workspace = real_path
        return real_path

    async def aclear_workspace(self, session_id: str | None = None) -> bool:
        self.calls.append(("aclear_workspace", session_id))
        existed = self._workspace is not None
        self._workspace = None
        return existed


@dataclass
class FakeAgent:
    session: FakeSession = field(default_factory=FakeSession)


# ============ 辅助 ============


def _make_context(agent: FakeAgent) -> CommandContext:
    printed: list[str] = []
    return CommandContext(
        agent=agent,  # type: ignore[arg-type]
        base_dir=".",
        config_file="config/llm_config.json",
        mcp_config_file="config/mcp_servers.json",
        print_fn=printed.append,
        input_fn=lambda prompt="": "",
        select_menu=lambda *args, **kwargs: None,
        create_llm=lambda provider: None,
        list_providers=dict,
        run_structured_until_completion=lambda agent, task: None,
        chat_until_completion=lambda agent, task: None,
        safety_backend=None,  # type: ignore[arg-type]
    ), printed


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


# ============ 直接调用 workspace_command 的测试 ============


class TestShowWorkspace:
    """workspace（无参数）→ 查看。"""

    def test_show_unbound(self) -> None:
        agent = FakeAgent()
        ctx, printed = _make_context(agent)
        _run(workspace_command(ctx, "workspace"))
        assert ("aget_workspace", "thread-test") in agent.session.calls
        assert any("未绑定" in p for p in printed)
        assert any("workspace <path>" in p for p in printed)

    def test_show_bound(self, tmp_path: Path) -> None:
        agent = FakeAgent()
        agent.session._workspace = str(tmp_path)
        ctx, printed = _make_context(agent)
        _run(workspace_command(ctx, "workspace"))
        assert any("thread-test" in p for p in printed)
        assert any(str(tmp_path) in p for p in printed)


class TestSetWorkspace:
    """workspace <path> → 设置。"""

    def test_set_valid_path(self, tmp_path: Path) -> None:
        agent = FakeAgent()
        ctx, printed = _make_context(agent)
        _run(workspace_command(ctx, f"workspace {tmp_path}"))
        assert ("aset_workspace", (str(tmp_path), "thread-test")) in agent.session.calls
        assert any("已为会话" in p for p in printed)
        assert any("限制在该目录内" in p for p in printed)

    def test_set_nonexistent_path(self) -> None:
        agent = FakeAgent()
        ctx, printed = _make_context(agent)
        _run(workspace_command(ctx, "workspace /nonexistent/path/xyz"))
        assert ("aset_workspace", ("/nonexistent/path/xyz", "thread-test")) in agent.session.calls
        assert any("绑定失败" in p for p in printed)
        # 失败后不应打印成功消息
        assert not any("已为会话" in p for p in printed)

    def test_set_empty_path_falls_through_to_show(self) -> None:
        """workspace 后无参数 → show 分支。"""
        agent = FakeAgent()
        ctx, _printed = _make_context(agent)
        _run(workspace_command(ctx, "workspace"))
        # 应调用 aget_workspace 而非 aset_workspace
        assert any(c[0] == "aget_workspace" for c in agent.session.calls)
        assert not any(c[0] == "aset_workspace" for c in agent.session.calls)


class TestClearWorkspace:
    """workspace:clear / workspace clear → 清除。"""

    def test_clear_bound(self, tmp_path: Path) -> None:
        agent = FakeAgent()
        agent.session._workspace = str(tmp_path)
        ctx, printed = _make_context(agent)
        _run(workspace_command(ctx, "workspace:clear"))
        assert ("aclear_workspace", "thread-test") in agent.session.calls
        assert any("已清除" in p for p in printed)

    def test_clear_unbound(self) -> None:
        agent = FakeAgent()
        ctx, printed = _make_context(agent)
        _run(workspace_command(ctx, "workspace:clear"))
        assert any("原本未绑定" in p for p in printed)

    def test_clear_space_form(self) -> None:
        """workspace clear（空格形式）等价于 workspace:clear。"""
        agent = FakeAgent()
        ctx, _ = _make_context(agent)
        _run(workspace_command(ctx, "workspace clear"))
        assert any(c[0] == "aclear_workspace" for c in agent.session.calls)


class TestHelpWorkspace:
    """workspace:help / workspace help → 帮助。"""

    def test_help_colon_form(self) -> None:
        agent = FakeAgent()
        ctx, printed = _make_context(agent)
        _run(workspace_command(ctx, "workspace:help"))
        assert any("工作空间(workspace)命令" in p for p in printed)
        assert any("workspace:clear" in p for p in printed)

    def test_help_space_form(self) -> None:
        agent = FakeAgent()
        ctx, printed = _make_context(agent)
        _run(workspace_command(ctx, "workspace help"))
        assert any("工作空间(workspace)命令" in p for p in printed)


# ============ dispatcher 路由测试 ============


class TestDispatcherRouting:
    """验证 dispatcher 正确路由 workspace 命令，不被 chat_mode 兜底。"""

    def test_dispatch_workspace_routes_to_handler(self) -> None:
        agent = FakeAgent()
        ctx, _printed = _make_context(agent)
        result = _run(dispatch_command(ctx, "workspace"))
        assert result.handled is True
        assert any(c[0] == "aget_workspace" for c in agent.session.calls)

    def test_dispatch_workspace_with_path_routes_to_handler(self, tmp_path: Path) -> None:
        agent = FakeAgent()
        ctx, _printed = _make_context(agent)
        result = _run(dispatch_command(ctx, f"workspace {tmp_path}"))
        assert result.handled is True
        assert any(c[0] == "aset_workspace" for c in agent.session.calls)

    def test_dispatch_workspace_clear_routes_to_handler(self) -> None:
        agent = FakeAgent()
        ctx, _ = _make_context(agent)
        result = _run(dispatch_command(ctx, "workspace:clear"))
        assert result.handled is True
        assert any(c[0] == "aclear_workspace" for c in agent.session.calls)

    def test_dispatch_workspace_help_routes_to_handler(self) -> None:
        agent = FakeAgent()
        ctx, printed = _make_context(agent)
        result = _run(dispatch_command(ctx, "workspace:help"))
        assert result.handled is True
        assert any("工作空间(workspace)命令" in p for p in printed)
