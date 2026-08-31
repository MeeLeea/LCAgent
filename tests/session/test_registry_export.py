"""SessionRegistry.aexport_session 的单元测试。

聚焦 content 提取对齐:改用 stringify_content 后,多模态 content(list of dict)
应提取 text 字段而非 str(dict),与 api/server.py 的 export 端点行为一致。
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from session import SessionRegistry, SessionStore


class _ExportStubRegistry(SessionRegistry):
    """绕过 checkpointer,直接喂固定消息,仅测 aexport_session 格式化逻辑。"""

    def __init__(self, messages: list) -> None:
        super().__init__(MemorySaver(), SessionStore(), process_type=None)
        self._messages = messages

    async def aget_messages(self, session_id: str | None = None) -> list:
        return self._messages


def test_export_multimodal_content_extracts_text_field() -> None:
    # Given: 用户消息内容为多模态块列表(含 text 字段)。
    messages = [HumanMessage(content=[{"type": "text", "text": "你好世界"}])]
    reg = _ExportStubRegistry(messages)
    # When: 导出为 text 格式。
    text = asyncio.run(reg.aexport_session("mm"))
    # Then: 提取出 text 字段内容,而非整个 dict 的 str 表示。
    assert "你好世界" in text
    assert "{'type'" not in text  # 旧 _message_text 会产出 str(dict)


def test_export_markdown_format_multimodal() -> None:
    # Given: 多模态消息 + 回答(两条,才会出现 markdown 分隔符)。
    messages = [
        HumanMessage(content=[{"type": "text", "text": "结构化提问"}]),
        AIMessage(content="结构化回答"),
    ]
    reg = _ExportStubRegistry(messages)
    # When: 导出为 markdown 格式。
    text = asyncio.run(reg.aexport_session("mm", fmt="markdown"))
    # Then: 使用 markdown 角色标记,且提取 text 字段;多消息间用 --- 分隔。
    assert "**用户**" in text
    assert "**助手**" in text
    assert "结构化提问" in text
    assert "---" in text  # markdown 多消息分隔符


def test_export_role_mapping_covers_all_types() -> None:
    # Given: 四种角色消息齐全。
    from langchain_core.messages import ToolMessage

    messages = [
        SystemMessage(content="系统指令"),
        HumanMessage(content="用户提问"),
        AIMessage(content="助手回答"),
        ToolMessage(content="工具结果", tool_call_id="c1"),
    ]
    reg = _ExportStubRegistry(messages)
    # When: 导出。
    text = asyncio.run(reg.aexport_session("all"))
    # Then: 四种中文角色标记齐全。
    assert "【系统】" in text
    assert "【用户】" in text
    assert "【助手】" in text
    assert "【工具】" in text


def test_export_skips_empty_content() -> None:
    # Given: 含空内容消息。
    messages = [
        HumanMessage(content=""),
        HumanMessage(content="有效内容"),
    ]
    reg = _ExportStubRegistry(messages)
    # When: 导出。
    text = asyncio.run(reg.aexport_session("empty"))
    # Then: 跳过空消息,只保留有效内容。
    assert "有效内容" in text
    # 空消息不产生块(blocks 用 sep 连接,空消息被 continue 跳过)
    assert text.count("【用户】") == 1


def test_export_string_content_unchanged() -> None:
    """回归:纯字符串 content 行为不变(stringify_content 对 str 直接返回)。"""
    messages = [HumanMessage(content="纯文本提问")]
    reg = _ExportStubRegistry(messages)
    text = asyncio.run(reg.aexport_session("str"))
    assert "纯文本提问" in text
    assert "【用户】" in text
