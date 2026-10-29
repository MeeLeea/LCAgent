"""workflow 命令的 workspace 继承测试。

背景：workspace 按 session_id 存储在 ``session_workspaces`` 表。``run_workflow``
在 CLI 场景会生成新的工作流 thread_id，该 sid 默认没有任何 workspace 绑定，
导致 ``WorkspaceSecurityMW`` 读到 ``workspace_path=None`` 而放行工具调用。
本文件验证新生成的 workflow 会话会从当前会话继承 workspace。

覆盖用例：
- 当前会话已绑定 workspace → 新 workflow sid 上收到 aset_workspace(ws, new_sid)
- 当前会话未绑定 → 不调用 aset_workspace
- 当前会话本身是 workflow 会话 → 复用该 sid 且不做任何 workspace 读写
- aset_workspace 抛 ValueError → 不中断工作流，仍执行到 done
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from cli.commands.types import CommandContext
from cli.commands.workflow import run_workflow

# ============ Fake 层 ============


@dataclass
class FakeSession:
    """模拟 SessionRegistry 中 workflow 相关的接口。"""

    current_session_id: str = "thread-test"
    workspace: str | None = None
    workflow_session_ids: set[str] = field(default_factory=set)
    generated_session_id: str = "workflow-simple-thread-abc123"
    # 为 True 时 aset_workspace 抛 ValueError（模拟路径已删除/不再是目录）
    raise_on_set: bool = False
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def is_workflow_session(self, session_id: str) -> bool:
        self.calls.append(("is_workflow_session", session_id))
        return session_id in self.workflow_session_ids

    def generate_session_id(self, workflow_name: str | None = None) -> str:
        self.calls.append(("generate_session_id", workflow_name))
        return self.generated_session_id

    async def aget_workspace(self, session_id: str | None = None) -> str | None:
        self.calls.append(("aget_workspace", session_id))
        return self.workspace

    async def aset_workspace(
        self, workspace_path: str, session_id: str | None = None
    ) -> str:
        self.calls.append(("aset_workspace", (workspace_path, session_id)))
        if self.raise_on_set:
            raise ValueError(
                f"工作空间路径不存在或不是目录: {workspace_path}"
            )
        return workspace_path

    def workspace_calls(self) -> list[tuple[str, Any]]:
        """只返回 workspace 相关调用（aget_workspace / aset_workspace）。"""
        return [c for c in self.calls if c[0] in ("aget_workspace", "aset_workspace")]


@dataclass
class FakeAgent:
    session: FakeSession = field(default_factory=FakeSession)


@dataclass
class FakeWorkflowSM:
    """模拟绑定 WorkflowAdapter 的 SessionManager 门面（仅实现 arun_stream）。"""

    calls: list[tuple[str, str | None]] = field(default_factory=list)

    async def arun_stream(
        self, task: str, thread_id: str | None = None
    ) -> AsyncIterator[dict[str, str]]:
        self.calls.append(("arun_stream", thread_id))
        yield {"type": "workflow_node", "node": "manager_plan", "status": "running"}
        yield {"type": "done", "content": "OK"}


# ============ 辅助 ============


def _make_context(
    agent: FakeAgent, sm: FakeWorkflowSM
) -> tuple[CommandContext, list[str]]:
    """构造命令上下文，并返回打印缓冲以便断言用户可见输出。"""
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
        workflow_sm=sm,
    ), printed


# ============ 测试 ============


class TestWorkflowWorkspaceInheritance:
    """新生成的工作流会话应继承当前会话的 workspace。"""

    def test_inherits_bound_workspace(self) -> None:
        agent = FakeAgent(session=FakeSession(workspace=r"D:\work\LCAgent"))
        sm = FakeWorkflowSM()
        ctx, _printed = _make_context(agent, sm)

        result = asyncio.run(run_workflow(ctx, "simple", "分析项目结构"))

        assert result == {"final_answer": "OK"}
        assert (
            "aset_workspace",
            (r"D:\work\LCAgent", "workflow-simple-thread-abc123"),
        ) in agent.session.calls
        # 工作流用新生成的 sid 执行
        assert sm.calls == [("arun_stream", "workflow-simple-thread-abc123")]

    def test_unbound_workspace_skips_set(self) -> None:
        agent = FakeAgent(session=FakeSession(workspace=None))
        sm = FakeWorkflowSM()
        ctx, _printed = _make_context(agent, sm)

        result = asyncio.run(run_workflow(ctx, "simple", "分析项目结构"))

        assert result == {"final_answer": "OK"}
        assert ("aget_workspace", "thread-test") in agent.session.calls
        assert not any(c[0] == "aset_workspace" for c in agent.session.calls)

    def test_workflow_session_reuses_id_without_workspace_io(self) -> None:
        agent = FakeAgent(
            session=FakeSession(
                current_session_id="workflow-simple-thread-existing",
                workflow_session_ids={"workflow-simple-thread-existing"},
                workspace=r"D:\work\LCAgent",
            )
        )
        sm = FakeWorkflowSM()
        ctx, _printed = _make_context(agent, sm)

        result = asyncio.run(run_workflow(ctx, "simple", "分析项目结构"))

        assert result == {"final_answer": "OK"}
        # 复用已有 workflow sid
        assert sm.calls == [("arun_stream", "workflow-simple-thread-existing")]
        # 不做任何 workspace 读写
        assert agent.session.workspace_calls() == []

    def test_set_workspace_value_error_does_not_abort(self) -> None:
        agent = FakeAgent(
            session=FakeSession(workspace=r"D:\work\deleted-dir", raise_on_set=True)
        )
        sm = FakeWorkflowSM()
        ctx, printed = _make_context(agent, sm)

        result = asyncio.run(run_workflow(ctx, "simple", "分析项目结构"))

        # 继承失败仍降级执行到 done
        assert result == {"final_answer": "OK"}
        assert any(c[0] == "aset_workspace" for c in agent.session.calls)
        assert sm.calls == [("arun_stream", "workflow-simple-thread-abc123")]
        # 用户可见提示：不能只写日志，否则用户误以为已继承
        assert any("继承工作空间失败" in p for p in printed)
