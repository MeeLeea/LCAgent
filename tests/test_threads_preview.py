"""会话菜单预览的单元测试。

覆盖 _messages_preview 用第一条用户消息生成会话标题的行为,
全部离线,不发起任何 LLM 调用。
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from cli.commands.threads import _messages_preview


def test_preview_returns_first_user_message() -> None:
    # Given: 会话以用户消息开头。
    messages = [
        HumanMessage(content="帮我搜索Python教程"),
        AIMessage(content="好的"),
    ]
    # When: 生成预览。
    preview = _messages_preview(messages)
    # Then: 直接取第一条用户消息。
    assert preview == "帮我搜索Python教程"


def test_preview_truncates_long_first_message() -> None:
    # Given: 第一条用户消息超过默认 15 字。
    messages = [HumanMessage(content="这是一个非常长的用户提问消息用于测试截断行为")]
    # When: 生成预览。
    preview = _messages_preview(messages)
    # Then: 截断并加省略号。
    assert preview == "这是一个非常长的用户提问消息用..."
    assert preview.endswith("...")


def test_preview_normalizes_newlines_to_spaces() -> None:
    # Given: 用户消息含换行。
    messages = [HumanMessage(content="第一行\n第二行\r\n第三行")]
    # When: 生成预览。
    preview = _messages_preview(messages)
    # Then: 换行被归一化为空格。
    assert preview == "第一行 第二行 第三行"


def test_preview_skips_empty_and_falls_back_to_next_user_message() -> None:
    # Given: 开头是空内容/非用户消息,再往后才有有效用户消息。
    messages = [
        HumanMessage(content="   \n  "),
        SystemMessage(content="system"),
        HumanMessage(content="实际的问题"),
    ]
    # When: 生成预览。
    preview = _messages_preview(messages)
    # Then: 取第一条非空用户消息。
    assert preview == "实际的问题"


def test_preview_empty_when_no_user_message() -> None:
    # Given: 会话只有 assistant/system 消息。
    messages = [
        SystemMessage(content="system"),
        AIMessage(content="assistant"),
    ]
    # When: 生成预览。
    preview = _messages_preview(messages)
    # Then: 返回空串,由调用方回退到 thread_id。
    assert preview == ""


def test_preview_handles_non_string_content() -> None:
    # Given: 用户消息内容是消息块列表(非纯字符串)。
    messages = [HumanMessage(content=[{"type": "text", "text": "结构化提问"}])]
    # When: 生成预览。
    preview = _messages_preview(messages)
    # Then: 不抛异常,返回可展示的文本。
    assert "结构化提问" in preview
