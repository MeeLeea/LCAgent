"""workflow 命令的 HITL 分支测试(交互 / 非交互入口)

覆盖:
- ``_collect_interrupt_choice``:user_confirmation(items 分组)返回
  ``{"answers": {item_id: {"choice_id": ...}}}``;human_choice 保持
  ``{"choice_id": ...}``;取消返回 ``{"cancelled": True}``
- ``run_workflow`` 非交互入口(API):遇到 interrupt 只转发事件并退出,
  **不**在后台替用户作答(不调 aresume_stream),pending interrupt 留给
  /api/chat/resume
- ``run_workflow`` 交互入口(CLI):正常收集答案并 resume 续跑

背景:HTTP 入口曾用 ``select_menu=lambda: choices[0]`` 秒答用户问题,
且答出的 ``{"choice_id": ...}`` 与 user_confirmation 的 ``{"answers": ...}``
契约不符,导致工作流中断形同虚设。
"""
from __future__ import annotations

import asyncio
from typing import Any

from cli.commands.types import CommandContext
from cli.commands.workflow import _collect_interrupt_choice, run_workflow


class _FakeSession:
    """假 SessionRegistry:固定为工作流会话"""

    def __init__(self) -> None:
        self.current_session_id = "workflow-rtl_graph-thread-t1"

    def is_workflow_session(self, session_id: str) -> bool:
        return True

    async def aget_workspace(self, session_id: str | None = None) -> str | None:
        return None

    async def aset_workspace(self, path: str, session_id: str | None = None) -> str:
        return path


class _FakeWorkflowSM:
    """假 workflow 门面:首轮流 yield 给定事件,记录 resume 调用"""

    def __init__(self, first_events: list[dict[str, Any]], resume_events: list[dict[str, Any]] | None = None) -> None:
        self._first_events = first_events
        self._resume_events = resume_events or [{"type": "done", "content": "最终答案"}]
        self.arun_calls: list[tuple[str, str]] = []
        self.aresume_calls: list[dict[str, Any]] = []

    async def arun_stream(self, task: str, thread_id: str):
        self.arun_calls.append((task, thread_id))
        for ev in self._first_events:
            yield ev

    async def aresume_stream(self, payload: dict[str, Any], thread_id: str):
        self.aresume_calls.append(payload)
        for ev in self._resume_events:
            yield ev


class _FakeAgent:
    """假 AgentCore:workflow 命令只读取 session"""

    def __init__(self) -> None:
        self.session = _FakeSession()


def _make_context(
    sm: _FakeWorkflowSM,
    *,
    interactive: bool = True,
    selection: str | None = "a",
    events: list[dict[str, str]] | None = None,
) -> CommandContext:
    """构造最小 CommandContext(只填 workflow 命令用到的字段)"""

    def _print(value: str = "") -> None:
        pass

    def _select_menu(title: str, choices: object = None) -> str | None:
        return selection

    def _event_cb(event: dict[str, str]) -> None:
        if events is not None:
            events.append(event)

    return CommandContext(
        agent=_FakeAgent(),  # type: ignore[arg-type]  # 只用到 session
        print_fn=_print,
        input_fn=lambda prompt="": "",
        select_menu=_select_menu,
        create_llm=lambda provider: None,  # type: ignore[arg-type,return-value]
        list_providers=dict,
        run_structured_until_completion=None,  # type: ignore[arg-type]
        chat_until_completion=None,  # type: ignore[arg-type]
        safety_backend=None,  # type: ignore[arg-type]
        base_dir=".",
        config_file="agent/agent_config.json",
        mcp_config_file="config/mcp_servers.json",
        workflow_sm=sm,
        workflow_event_cb=_event_cb,
        interactive=interactive,
    )


_ITEMS_EVENT: dict[str, Any] = {
    "type": "interrupt",
    "kind": "user_confirmation",
    "prompt": "请确认架构方案",
    "items": [
        {"id": "q1", "question": "模块划分?", "choices": [{"id": "a", "label": "按功能"}, {"id": "b", "label": "按层次"}]},
        {"id": "q2", "question": "时钟频率?", "choices": [{"id": "a", "label": "100MHz"}, {"id": "b", "label": "200MHz"}]},
    ],
    "choices": [
        {"id": "a", "label": "按功能", "item_id": "q1"},
        {"id": "b", "label": "按层次", "item_id": "q1"},
    ],
}


# ── _collect_interrupt_choice ────────────────────────────────────


def test_collect_items_returns_answers_contract() -> None:
    """items 分组中断 → 按 user_confirmation 契约返回 answers 映射

    Given:user_confirmation 事件(items 含 2 个问题),用户依次选 a / b
    When:_collect_interrupt_choice 收集
    Then:返回 {"answers": {"q1": {"choice_id": "a"}, "q2": {"choice_id": "b"}}}
    """
    sm = _FakeWorkflowSM([])
    context = _make_context(sm, selection="a")

    # 逐个问题返回不同选择:用可变的菜单替身模拟用户操作
    answers_iter = iter(["a", "b"])

    def _select_menu(title: str, choices: list[tuple[str, str]] | None = None) -> str | None:
        return next(answers_iter)

    context.select_menu = _select_menu

    payload = _collect_interrupt_choice(_ITEMS_EVENT, context)

    assert payload == {"answers": {"q1": {"choice_id": "a"}, "q2": {"choice_id": "b"}}}


def test_collect_single_choice_keeps_choice_id() -> None:
    """human_choice 单选中断 → 保持 {"choice_id": ...} 契约"""
    context = _make_context(_FakeWorkflowSM([]), selection="approve")
    event = {
        "type": "interrupt",
        "kind": "human_choice",
        "prompt": "确认执行?",
        "choices": [{"id": "approve", "label": "确认"}, {"id": "deny", "label": "拒绝"}],
    }

    payload = _collect_interrupt_choice(event, context)

    assert payload == {"choice_id": "approve"}


def test_collect_cancel_returns_cancelled() -> None:
    """用户 Esc(select_menu 返回 None)→ {"cancelled": True}"""
    context = _make_context(_FakeWorkflowSM([]), selection=None)

    payload = _collect_interrupt_choice(_ITEMS_EVENT, context)

    assert payload == {"cancelled": True}


# ── run_workflow 交互 / 非交互 ───────────────────────────────────


def test_non_interactive_pauses_without_answering() -> None:
    """非交互入口遇到中断不替用户作答

    Given:workflow_sm 首轮流产出 interrupt 事件
    When:run_workflow(interactive=False)
    Then:中断事件已转发;未调用 aresume_stream;返回值无最终答案
    """
    sm = _FakeWorkflowSM([_ITEMS_EVENT])
    events: list[dict[str, str]] = []
    context = _make_context(sm, interactive=False, events=events)

    result = asyncio.run(run_workflow(context, "rtl_graph", "设计一个 FIFO"))

    assert sm.arun_calls == [("设计一个 FIFO", "workflow-rtl_graph-thread-t1")]
    assert sm.aresume_calls == []
    assert result["final_answer"] == ""
    # 中断事件(含 items)必须已推给前端,否则前端拿不到中断卡片
    assert any(ev.get("type") == "interrupt" for ev in events)


def test_interactive_collects_and_resumes() -> None:
    """交互入口(CLI)收集答案后经 aresume_stream 续跑至 done

    Given:workflow_sm 首轮流产出 interrupt 事件,resume 流产出 done
    When:run_workflow(interactive=True)
    Then:aresume_stream 收到 answers 契约负载;返回最终答案
    """
    sm = _FakeWorkflowSM([_ITEMS_EVENT])
    context = _make_context(sm, interactive=True, selection="a")

    result = asyncio.run(run_workflow(context, "rtl_graph", "设计一个 FIFO"))

    assert sm.aresume_calls == [
        {"answers": {"q1": {"choice_id": "a"}, "q2": {"choice_id": "a"}}}
    ]
    assert result["final_answer"] == "最终答案"
