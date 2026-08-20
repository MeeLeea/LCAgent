"""工作流 workspace 隔离测试 - 验证 workspace_path 从运行器透传到 Worker 工具调用。

覆盖：
- worker_exec 节点把 LangGraph 注入的 config 透传给 Worker.execute_task
- arun_compiled_workflow 把 workspace_path 注入 config.configurable
- TeamAgent 工具模式构造最小 config(仅 workspace_path,不转发 callbacks)
- TeamAgent._create_tool_agent 挂载 WorkspaceSecurityMW
- cli run_workflow 经 workflow_sm 执行(workspace 注入由 WorkflowAdapter 承载)

运行：
  pytest tests/graph/test_workflow_workspace.py -v
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

from agent.workspace_mw import WorkspaceSecurityMW
from graph.simple import arun_compiled_workflow, worker_exec_node
from team.base import TeamAgent


# --------------------------------------------------------------------------- #
# 节点级:config 透传
# --------------------------------------------------------------------------- #
@dataclass
class CapturingWorker:
    """捕获 execute_task 收到的 config 的伪 Worker。"""

    response: str = "done"
    received_config: dict[str, Any] | None = None

    async def aexecute_task(self, plan: str, injector=None, config: dict[str, Any] | None = None) -> str:
        """异步版 execute_task(节点现直接 await 调用)"""
        self.received_config = config
        return self.response


def test_worker_exec_node_passes_workspace_config():
    """simple 的 worker_exec 节点:config(含 workspace_path)透传给 execute_task。"""
    worker = CapturingWorker()
    state = {"plan": "计划: 步骤1"}
    config = {"configurable": {"thread_id": "t1", "workspace_path": "C:/ws"}}

    result = asyncio.run(worker_exec_node(state, worker, config=config))

    assert result["worker_result"] == "done"
    assert worker.received_config == config


def test_worker_exec_node_config_default_none():
    """未传 config 时 execute_task 收到 None(向后兼容)。"""
    worker = CapturingWorker()
    state = {"plan": "计划: 步骤1"}

    result = asyncio.run(worker_exec_node(state, worker))

    assert result["worker_result"] == "done"
    assert worker.received_config is None


def test_real_langgraph_injects_config_into_worker_exec_node():
    """真实 LangGraph 运行:config(含 workspace_path)由框架按节点签名注入。

    回归保护:若节点 config 注解在 ``from __future__ import annotations`` 下
    写成字符串不匹配的形态(如 'RunnableConfig | None'),LangGraph 会跳过注入,
    execute_task 收到 None,本测试失败。
    """
    from functools import partial
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class WfState(TypedDict):
        plan: str
        worker_result: str

    worker = CapturingWorker()
    graph = StateGraph(WfState)
    graph.add_node(
        "worker_exec", partial(worker_exec_node, worker=worker, injector=None)
    )
    graph.add_edge(START, "worker_exec")
    graph.add_edge("worker_exec", END)
    compiled = graph.compile()

    asyncio.run(
        compiled.ainvoke(
            {"plan": "计划: 步骤1"},
            {"configurable": {"thread_id": "t1", "workspace_path": "C:/ws"}},
        )
    )

    assert worker.received_config is not None, "LangGraph 未注入 config,注解形态不匹配"
    assert worker.received_config["configurable"]["workspace_path"] == "C:/ws"


# --------------------------------------------------------------------------- #
# 运行器级:workspace_path 注入 config.configurable
# --------------------------------------------------------------------------- #
class FakeGraph:
    """捕获 ainvoke config 的伪图(aget_state 缺失会被跨轮次记忆逻辑静默降级)。"""

    def __init__(self) -> None:
        self.captured_config: dict[str, Any] | None = None

    async def ainvoke(self, initial_state: dict, config: dict | None = None) -> dict:
        self.captured_config = config
        return {"final_answer": "ok"}


def test_arun_compiled_workflow_injects_workspace_path():
    """workspace_path 注入 config.configurable,与 thread_id 并列。"""
    graph = FakeGraph()

    asyncio.run(arun_compiled_workflow(graph, "任务", thread_id="wf-t", workspace_path="C:/ws"))

    configurable = graph.captured_config["configurable"]
    assert configurable["thread_id"] == "wf-t"
    assert configurable["workspace_path"] == "C:/ws"


def test_arun_compiled_workflow_no_workspace_by_default():
    """未传 workspace_path 时 config.configurable 不含该键(兼容旧场景)。"""
    graph = FakeGraph()

    asyncio.run(arun_compiled_workflow(graph, "任务", thread_id="wf-t"))

    configurable = graph.captured_config["configurable"]
    assert "workspace_path" not in configurable
    assert configurable["thread_id"] == "wf-t"


# --------------------------------------------------------------------------- #
# TeamAgent 级:工具模式中间件挂载
# --------------------------------------------------------------------------- #
def test_create_tool_agent_mounts_workspace_middleware(monkeypatch):
    """工具模式的 create_agent 挂载 WorkspaceSecurityMW。"""
    captured: dict[str, list[Any]] = {}

    def fake_create_agent(*args: Any, **kwargs: Any) -> MagicMock:
        captured["middleware"] = kwargs.get("middleware", [])
        return MagicMock()

    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)

    agent = object.__new__(TeamAgent)
    agent.llm = MagicMock()
    agent.llm.get_chat_model.return_value = MagicMock()
    agent.tools = [MagicMock()]
    agent.system_prompt = ""

    agent._create_tool_agent()

    assert any(
        isinstance(m, WorkspaceSecurityMW) for m in captured["middleware"]
    ), "工具 agent 必须挂载 WorkspaceSecurityMW"


# --------------------------------------------------------------------------- #
# CLI 级:run_workflow 经 workflow_sm 执行(workspace 注入由 WorkflowAdapter 承载)
# --------------------------------------------------------------------------- #
def test_run_workflow_via_workflow_sm():
    """run_workflow 经 workflow_sm.arun_stream 执行:专属会话复用 thread_id + 事件转发。"""
    import cli.commands.workflow as wf_module
    from cli.commands.types import CommandContext

    class FakeSession:
        current_session_id = "workflow-simple-thread-abc"

        def generate_session_id(self, workflow_name=None) -> str:
            return "workflow-simple-thread-abc"

        def is_workflow_session(self, session_id: str) -> bool:
            return True

    class FakeAgentCore:
        def __init__(self) -> None:
            self.session = FakeSession()

    class FakeWorkflowSM:
        """模拟 workflow 门面:捕获调用并产出节点/done 事件。"""

        def __init__(self):
            self.calls: list[tuple[str, str | None]] = []

        async def arun_stream(self, task, thread_id=None):
            self.calls.append((task, thread_id))
            yield {"type": "workflow_node", "node": "worker", "status": "running"}
            yield {"type": "workflow_node", "node": "worker", "status": "done"}
            yield {"type": "done", "content": "最终答案"}

    forwarded: list[dict] = []
    sm = FakeWorkflowSM()
    ctx = CommandContext(
        agent=FakeAgentCore(),
        base_dir="",
        config_file="",
        mcp_config_file="",
        print_fn=lambda s: None,
        input_fn=lambda p: "",
        select_menu=lambda *a: None,
        create_llm=lambda p: None,
        list_providers=dict,
        run_structured_until_completion=lambda a, t: "",
        chat_until_completion=lambda a, t: "",
        safety_backend=None,
        mcp_backend=None,
        workflow_event_cb=lambda ev: forwarded.append(ev),
        workflow_sm=sm,
    )

    result = asyncio.run(wf_module.run_workflow(ctx, "simple", "测试任务"))

    assert result["final_answer"] == "最终答案"
    # 专属 workflow 会话:复用 current_session_id(持久化绑定)
    assert sm.calls == [("测试任务", "workflow-simple-thread-abc")]
    # 节点事件转发给前端(SSE)
    assert {"type": "workflow_node", "node": "worker", "status": "running"} in forwarded
    assert {"type": "workflow_node", "node": "worker", "status": "done"} in forwarded
    # 整体状态复位:运行中 → 完成
    statuses = [ev.get("status") for ev in forwarded if ev.get("type") == "workflow_status"]
    assert statuses == ["running", "done"]


def test_run_workflow_requires_workflow_sm():
    """workflow_sm 未注入 CommandContext 时 run_workflow 抛错(快速失败)。"""
    import cli.commands.workflow as wf_module
    from cli.commands.types import CommandContext

    ctx = CommandContext(
        agent=MagicMock(),
        base_dir="",
        config_file="",
        mcp_config_file="",
        print_fn=lambda s: None,
        input_fn=lambda p: "",
        select_menu=lambda *a: None,
        create_llm=lambda p: None,
        list_providers=dict,
        run_structured_until_completion=lambda a, t: "",
        chat_until_completion=lambda a, t: "",
        safety_backend=None,
        mcp_backend=None,
        workflow_sm=None,
    )

    try:
        asyncio.run(wf_module.run_workflow(ctx, "simple", "测试任务"))
    except RuntimeError as error:
        assert "workflow_sm" in str(error)
    else:
        raise AssertionError("应抛出 RuntimeError")