"""测试 ManagerAgent.summarize_context_node(真实节点方法,不联网)。

绕过 ``ManagerAgent.__init__``(object.__new__,不构造 LLMClient),
预设工作流模板缓存,实例级替换 ``arun_structured`` 为假实现;
helper ``run_team_turn_with_interrupt`` 走真实链路(interrupt 循环逻辑
由 tests/graph/test_workflow_interrupt_integration.py 等覆盖)。

覆盖:
- raw_context 为空时短路返回空串,不调用 LLM
- raw_context 非空时 prompt = 模板 + raw,返回 context_summary + messages
"""
from __future__ import annotations

import asyncio

import pytest

from agent.turn_types import AgentTurnResult
from team.manager.manager import ManagerAgent

# 与 team/manager/AGENT.md 的 ## workflow:summarize_context 小节特征一致
_SUMMARIZE_TEMPLATE = "你是一个工作流上下文提炼助手。"


def _make_manager(summary: str = "摘要") -> tuple[ManagerAgent, list[tuple[str, str]]]:
    """构造不联网的 ManagerAgent:预设模板缓存 + 实例级 fake arun_structured。"""
    m = object.__new__(ManagerAgent)
    m._workflow_templates = {"summarize_context": _SUMMARIZE_TEMPLATE}
    calls: list[tuple[str, str]] = []

    async def _fake_arun(prompt: str, config=None) -> AgentTurnResult:
        calls.append(("arun_structured", prompt))
        return AgentTurnResult.completed(summary)

    m.arun_structured = _fake_arun
    return m, calls


def test_summarize_node_empty_raw_short_circuits():
    """raw_context 为空:短路返回空串,不调 LLM(零 arun_structured 调用)。"""
    m, calls = _make_manager()

    result = asyncio.run(m.summarize_context_node({"raw_context": ""}))

    assert result["context_summary"] == ""
    assert result["messages"][0].content == ""
    assert calls == []


def test_summarize_node_non_empty_raw_calls_llm():
    """raw_context 非空:prompt 含 summarize_context 模板前缀 + raw 全文。"""
    m, calls = _make_manager("摘要: 用户偏好中文")

    result = asyncio.run(m.summarize_context_node({"raw_context": "历史对话"}))

    assert result["context_summary"] == "摘要: 用户偏好中文"
    assert result["messages"][0].content == "摘要: 用户偏好中文"
    assert len(calls) == 1
    assert calls[0][0] == "arun_structured"
    prompt = calls[0][1]
    assert _SUMMARIZE_TEMPLATE in prompt
    assert "历史对话" in prompt
    # 模板在前、raw 在后(与节点拼装顺序一致)
    assert prompt.index(_SUMMARIZE_TEMPLATE) < prompt.index("历史对话")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
