"""llm.message_utils.final_answer_is_empty 的单元测试。

该函数用于流式层守卫：LLM 最终 AIMessage 为空（无内容且无 tool_calls，例如
``finish_reason="length"`` 截断）时返回中文原因字符串，调用方据此发 ERROR 而非 DONE。
"""
from langchain_core.messages import AIMessage, HumanMessage

from llm.message_utils import final_answer_is_empty


def test_none_message_returns_none():
    """msg 为 None 时视为无异常（调用方尚无最终消息），返回 None。"""
    assert final_answer_is_empty(None) is None


def test_message_with_content_returns_none():
    """有可 strip 内容的 AIMessage 是正常回答，返回 None。"""
    assert final_answer_is_empty(AIMessage(content="hello")) is None


def test_message_with_tool_calls_returns_none():
    """带 tool_calls 的 AIMessage 是工具调用中间态，即使 content 为空也不算空回答。"""
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call-1"}],
    )
    assert final_answer_is_empty(msg) is None


def test_whitespace_content_returns_reason():
    """纯空白内容不可 strip，视为空回答，返回非空原因字符串。"""
    reason = final_answer_is_empty(AIMessage(content="  "))

    assert reason is not None
    assert isinstance(reason, str)


def test_empty_content_with_length_finish_reason_mentions_length():
    """content 为空且 finish_reason=length 时返回原因，并提示 length。"""
    msg = AIMessage(content="", response_metadata={"finish_reason": "length"})

    reason = final_answer_is_empty(msg)

    assert reason is not None
    assert "length" in reason.lower()


def test_empty_content_without_metadata_returns_reason():
    """content 为空且无 response_metadata 时仍返回非空原因字符串。"""
    reason = final_answer_is_empty(AIMessage(content="", response_metadata={}))

    assert reason is not None
    assert isinstance(reason, str)


def test_non_ai_message_with_content_returns_none():
    """非 AI 消息（如 HumanMessage）只要有内容就不算空回答。"""
    assert final_answer_is_empty(HumanMessage(content="用户消息")) is None
