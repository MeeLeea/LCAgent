"""run_team_turn_with_interrupt 中断转发语义测试(节点重入回归守护)

覆盖:
- 全新任务:只调一次 arun_structured,completed 结果原样返回
- 全新任务 + 内层残留陈旧 pending:先用取消载荷清场,再执行新任务
- 真实 LangGraph 图 + MemorySaver:外层 resume 时**不重跑** arun_structured,
  直接把内层 pending 交给 interrupt() → aresume_structured 注入答案
- 多轮 interrupt:重入时按位置重放已消费轮次,第二轮拿到的是**新答案**而非过期答案
- 只有 arun_structured 的旧式 Agent(无 aget_pending_interrupts)保持兼容

回归背景:LangGraph resume 会从节点首行重跑节点函数,历史实现首行无条件调
arun_structured,导致向内层 agent_executor 追加新 HumanMessage 而非
Command(resume=...),内层 AIMessage.tool_calls 永远拿不到 ToolMessage,
LLM 请求被网关 400 拒绝(tool_call_ids did not have response messages)。

参考:
- tests/team/test_team_interrupt.py(Interrupt 构造与 Fake executor 模式)
- team/interrupt.py 模块 docstring(位置重放 / 跳过 arun 重放 / 陈旧清场)
"""
from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt

from agent.turn_types import AgentTurnResult

# 先解析 graph 工作流注册表(registry 在导入期会反向 import graph.rtl_graph →
# team.architect),再导入 team.interrupt,避免"单独运行本文件"时触发
# team/__init__ → architect → graph.common → rtl_graph 的循环导入。
import graph.common  # noqa: F401  # isort:skip
from team.interrupt import run_team_turn_with_interrupt

# 陈旧中断清场所用的取消载荷(与 team/interrupt.py 的 _STALE_CANCEL_PAYLOAD 对齐)
_CANCEL_PAYLOAD: dict[str, Any] = {"cancelled": True, "answers": {}}


def _interrupt_value(question: str = "确认执行?") -> dict[str, Any]:
    """构造 user_confirmation 形态的 interrupt value"""
    return {
        "kind": "user_confirmation",
        "prompt": "请确认以下配置",
        "items": [{"id": "q1", "question": question}],
        "choices": [{"id": "a", "label": "选项 A", "item_id": "q1"}],
    }


class _ScriptedAgent:
    """按脚本返回 AgentTurnResult 的假 TeamAgent

    只实现中断转发所需的三个方法,行为对齐 TeamAgent:
    - arun_structured:消费脚本头部结果,interrupted 时同步 pending 状态
    - aresume_structured:消费脚本头部结果,按结果更新 pending 状态
    - aget_pending_interrupts:返回当前 pending 列表
    """

    def __init__(self, script: list[AgentTurnResult] | None = None) -> None:
        self.name = "fake"
        self._script: list[AgentTurnResult] = list(script or [])
        self._pending: list[Interrupt] = []
        self.arun_calls: list[str] = []
        self.resume_calls: list[Any] = []

    def _next_result(self) -> AgentTurnResult:
        if self._script:
            return self._script.pop(0)
        return AgentTurnResult.completed("默认完成")

    async def aget_pending_interrupts(self, config: dict | None = None) -> list[Interrupt]:
        return list(self._pending)

    async def arun_structured(
        self, task: str, config: dict | None = None
    ) -> AgentTurnResult:
        self.arun_calls.append(task)
        result = self._next_result()
        self._pending = list(result.interrupts)
        return result

    async def aresume_structured(
        self, payload: Any, config: dict | None = None
    ) -> AgentTurnResult:
        self.resume_calls.append(payload)
        result = self._next_result()
        self._pending = list(result.interrupts)
        return result


class _LegacyAgent:
    """只有 arun_structured 的旧式 Agent(无 aget_pending_interrupts)"""

    def __init__(self) -> None:
        self.name = "legacy"
        self.arun_calls: list[str] = []

    async def arun_structured(
        self, task: str, config: dict | None = None
    ) -> AgentTurnResult:
        self.arun_calls.append(task)
        return AgentTurnResult.completed("旧式完成")


class _ForwardState(TypedDict):
    """最小工作流 State:仅记录节点返回值"""

    result: str


def _build_graph(agent: Any) -> Any:
    """构建单节点工作流图,节点内调用 run_team_turn_with_interrupt"""

    async def architect_plan(state: _ForwardState) -> dict[str, str]:
        output = await run_team_turn_with_interrupt(agent, "architect 任务")
        return {"result": output}

    builder = StateGraph(_ForwardState)
    builder.add_node("architect_plan", architect_plan)
    builder.add_edge(START, "architect_plan")
    builder.add_edge("architect_plan", END)
    return builder.compile(checkpointer=MemorySaver())


def _run(coro: Any) -> Any:
    """在独立事件循环内执行协程(图与 checkpointer 均在同一 loop 内创建/使用)"""
    return asyncio.run(coro)


# ── 全新任务:只调一次 arun_structured ─────────────────────────────


def test_fresh_turn_calls_arun_once() -> None:
    """全新任务(无 graph 上下文)→ 调一次 arun_structured,completed 原样返回

    Given:假 Agent 脚本为 completed("结果"),无内层挂起中断
    When:run_team_turn_with_interrupt 执行
    Then:返回 "结果";arun 调 1 次;从未 resume
    """
    agent = _ScriptedAgent([AgentTurnResult.completed("结果")])

    output = _run(run_team_turn_with_interrupt(agent, "任务"))

    assert output == "结果"
    assert agent.arun_calls == ["任务"]
    assert agent.resume_calls == []


def test_fresh_turn_flushes_stale_pending() -> None:
    """全新任务 + 内层残留陈旧 pending → 先取消清场再执行新任务

    Given:假 Agent 内层有 1 个残留 pending,清场 resume 后返回 completed,
        随后 arun 返回 completed("新结果")
    When:run_team_turn_with_interrupt 执行(非 resume 运行)
    Then:aresume 先收到取消载荷;arun 仍被调用;最终返回 "新结果"
    """
    agent = _ScriptedAgent(
        [
            AgentTurnResult.completed("清场完成"),
            AgentTurnResult.completed("新结果"),
        ]
    )
    stale = Interrupt(value=_interrupt_value("上一轮遗留问题"), id="stale-1")
    agent._pending = [stale]

    output = _run(run_team_turn_with_interrupt(agent, "新任务"))

    assert output == "新结果"
    assert agent.resume_calls == [_CANCEL_PAYLOAD]
    assert agent.arun_calls == ["新任务"]


def test_legacy_agent_without_pending_api() -> None:
    """旧式 Agent(无 aget_pending_interrupts)→ 保持原有执行路径

    Given:只有 arun_structured 的假 Agent
    When:run_team_turn_with_interrupt 执行
    Then:返回 completed 结果,不抛 AttributeError
    """
    agent = _LegacyAgent()

    output = _run(run_team_turn_with_interrupt(agent, "任务"))

    assert output == "旧式完成"
    assert agent.arun_calls == ["任务"]


# ── 真实图重入:不重放 arun_structured ────────────────────────────


def test_graph_resume_skips_arun_replay() -> None:
    """外层 resume 时不重跑 arun_structured,直接转发已有 pending

    Given:真实 StateGraph + MemorySaver,首轮节点 arun 返回 interrupted(i-1)
    When:用 Command(resume=答案) 恢复外层图
    Then:节点重入后 arun 仍只被调用 1 次;aresume 收到原答案;图返回 "已恢复"
    """
    first = Interrupt(value=_interrupt_value("需要确认"), id="i-1")
    agent = _ScriptedAgent(
        [
            AgentTurnResult.interrupted([first]),
            AgentTurnResult.completed("已恢复"),
        ]
    )
    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "thread-resume-once"}}
    answer = {"answers": {"q1": {"choice_id": "a"}}}

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        first_state = await graph.ainvoke({"result": ""}, config)
        final_state = await graph.ainvoke(Command(resume=answer), config)
        return first_state, final_state

    # 同一事件循环内跑完两次 invoke,避免跨 loop 复用 checkpointer 连接
    first_state, final_state = _run(scenario())

    assert "__interrupt__" in first_state
    assert agent.arun_calls == ["architect 任务"]
    assert final_state["result"] == "已恢复"
    # 关键回归断言:节点重入没有重跑 arun_structured(旧实现在此会调 2 次)
    assert agent.arun_calls == ["architect 任务"]
    assert agent.resume_calls == [answer]


def test_graph_multi_round_uses_fresh_answer() -> None:
    """多轮中断:重入按位置重放已消费轮次,第二轮拿到新答案而非过期答案

    Given:真实图,首轮 arun 中断 i-1;第一轮 resume 后内层再次中断 i-2
    When:第二轮用新答案 resume
    Then:aresume 依次收到 [v1, v2](而非 [v1, v1]);arun 只调 1 次;图返回 "最终完成"
    """
    first = Interrupt(value=_interrupt_value("第一轮确认"), id="i-1")
    second = Interrupt(value=_interrupt_value("第二轮确认"), id="i-2")
    agent = _ScriptedAgent(
        [
            AgentTurnResult.interrupted([first]),
            AgentTurnResult.interrupted([second]),
            AgentTurnResult.completed("最终完成"),
        ]
    )
    graph = _build_graph(agent)
    config = {"configurable": {"thread_id": "thread-multi-round"}}
    answer_1 = {"answers": {"q1": {"choice_id": "a"}}}
    answer_2 = {"answers": {"q1": {"choice_id": "b"}}}

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        await graph.ainvoke({"result": ""}, config)
        second_state = await graph.ainvoke(Command(resume=answer_1), config)
        final_state = await graph.ainvoke(Command(resume=answer_2), config)
        return second_state, final_state

    # 同一事件循环内跑完三次 invoke,避免跨 loop 复用 checkpointer 连接
    second_interrupt_state, final_state = _run(scenario())

    # 第一轮 resume 后又一次中断:暴露的是第二轮的问题
    assert "__interrupt__" in second_interrupt_state
    assert second_interrupt_state["__interrupt__"][0].value == second.value

    assert final_state["result"] == "最终完成"
    assert agent.arun_calls == ["architect 任务"]
    # 关键回归断言:第二轮必须拿到 answer_2(缺少位置重放时会复用 answer_1)
    assert agent.resume_calls == [answer_1, answer_2]
