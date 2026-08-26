"""批量用户确认工具 — 把架构评审的"待确认输入清单"升级为 LangGraph interrupt/resume 语义。

与 ``cli.human_input.ask_human`` 的职责边界:
    - ``ask_human``: 单问题单选(CLI 菜单语义), interrupt value kind="human_choice",
      resume payload = {"choice_id": ...} / {"cancelled": True}
    - ``request_user_confirmation``: 多问题批量确认(架构评审语义),
      interrupt value kind="user_confirmation",
      resume payload = {"answers": {item_id: {"choice_id": selected_id}}}

执行链路(经 ``graph/common.run_team_turn_with_interrupt`` 透传):
    1. LLM 在 ReAct 循环中调用本工具 → ``_request_user_confirmation`` 内调
       ``langgraph.types.interrupt({"kind": "user_confirmation", "items": items})``
    2. 内层 agent_executor(checkpointer 已注入)捕获 pending interrupt,
       ``TeamAgent.arun_structured`` 流后经 ``_aget_pending_interrupts`` 检测到 →
       返回 ``AgentTurnResult.interrupted``
    3. ``run_team_turn_with_interrupt`` 收到 interrupted → 调外层 ``interrupt()``
       暂停外层 graph, resume 时把用户返回值经 ``aresume_structured`` 注入内层
    4. 外层 graph 暂停 → ``graph.ainvoke`` 返回含 ``__interrupt__`` 的 dict →
       ``workflow_adapter._arun_input_events`` 经 ``build_interrupt_event`` 转成
       前端 SSE ``{type: "interrupt", prompt, choices}``
    5. 前端渲染确认 UI, 用户回答后 POST resume → ``aresume_events`` 用
       ``Command(resume=payload)`` 恢复外层 graph → ``interrupt()`` 返回
       resume_value → 本工具 ``json.dumps`` 成 str 返回给 LLM 下轮 ReAct
"""
from __future__ import annotations

import json

import langgraph.types
from typing_extensions import TypedDict

from cli.human_input import CallableStructuredTool


class ConfirmationChoice(TypedDict):
    """单个待确认问题的可选项(与 ``ask_human`` 的 ``Choice`` 同构)。"""

    id: str
    label: str


class ConfirmationItem(TypedDict):
    """一个待确认问题: 含 id/问题文本/可选项列表。"""

    id: str
    question: str
    choices: list[ConfirmationChoice]


def _request_user_confirmation(items: list[ConfirmationItem]) -> str:
    """批量向用户征询多个架构决策点的确认。

    暂停 graph 执行, 把待确认清单发给前端。resume 后返回用户的回答映射,
    JSON 字符串形式(兼容 LangChain ReAct 工具约定: 工具返回值注入
    ``ToolMessage.content`` 供 LLM 下轮消费)。

    Args:
        items: 待确认问题列表, 每项含 ``id``/``question``/``choices``。
            ``id`` 需在全列表内唯一, 供 resume payload 按 id 映射答案。

    Returns:
        JSON 字符串, 结构为 ``{"answers": {item_id: {"choice_id": selected_id}}}``。
        与 ``ask_human`` 的 ``{"choice_id": ...}`` 同构, 每个问题的答案即一个
        单选结果, 顶层 ``answers`` 键避免与其他字段(如 ``cancelled``)冲突。

    Note:
        ``interrupt()`` 的返回值即前端 resume 时回传的 payload。无 checkpointer
        时 ``interrupt()`` 会抛错, 故本工具仅在 TeamAgent 工具模式(checkpointer
        已注入)下可用。
    """
    answer = langgraph.types.interrupt(
        {
            "kind": "user_confirmation",
            "items": items,
        }
    )
    # answer = {"answers": {item_id: {"choice_id": selected_id}}}
    # 防御: resume payload 异常时兜底为空 answers, 不让 LLM 下轮读到非法结构
    if not isinstance(answer, dict):
        answer = {"answers": {}}
    return json.dumps(answer, ensure_ascii=False)


request_user_confirmation: CallableStructuredTool = (
    CallableStructuredTool.from_function(
        func=_request_user_confirmation,
        name="request_user_confirmation",
        description=(
            "批量向用户征询多个架构决策点的确认。每个 item 含 id/question/choices, "
            "用户为每项选一个 choice。仅在需要用户拍板的系统级/微架构级决策时调用"
            "(如 FPGA 实现路径、握手协议、舍入饱和、复位策略、阵列规模等)。"
            "resume 后返回 JSON: {\"answers\": {item_id: {\"choice_id\": \"xxx\"}}}, "
            "据此在后续 ReAct 轮次中按用户选择推进设计。"
        ),
    )
)


__all__ = [
    "ConfirmationChoice",
    "ConfirmationItem",
    "request_user_confirmation",
]
