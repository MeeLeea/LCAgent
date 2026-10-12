"""llm.message_utils 的单元测试（extract_llm_error / final_answer_is_empty）。"""

from llm.message_utils import extract_llm_error, final_answer_is_empty


def test_extract_429_json_message():
    """429 + {'message': ...} 结构：提取真实文案并附带 HTTP 状态码。"""
    raw = "Error code: 429 - {'error': {'code': '1305', 'message': '该模型当前访问量过大，请您稍后再试'}}"
    result = extract_llm_error(RuntimeError(raw))
    assert "[HTTP 429]" in result
    assert "访问量过大" in result


def test_extract_503_plain_text():
    """纯文本 503 网关提示：给出服务暂时不可用、请稍后重试的友好文案。"""
    result = extract_llm_error(RuntimeError("Service temporarily unavailable"))
    assert "暂时不可用" in result
    assert "稍后重试" in result


def test_extract_429_plain_text():
    """纯文本 429 限流提示：提示访问量过大/限流。"""
    result = extract_llm_error(RuntimeError("Too Many Requests"))
    assert "429" in result
    assert "限流" in result


def test_extract_502_with_status_prefix():
    """502 + {'message': ...}：优先返回真实文案，并带状态码前缀。"""
    raw = "Error code: 502 - {'error': {'message': 'Bad Gateway'}}"
    result = extract_llm_error(RuntimeError(raw))
    assert "[HTTP 502]" in result
    assert "Bad Gateway" in result


def test_extract_401_unauthorized():
    """401 / Unauthorized：提示检查 API Key。"""
    result = extract_llm_error(RuntimeError("Error code: 401 - Unauthorized"))
    assert "鉴权" in result
    assert "API Key" in result


def test_extract_unknown_error_fallback():
    """未知错误：兜底返回原始信息，不吞掉细节。"""
    result = extract_llm_error(RuntimeError("some weird error"))
    assert result == "执行出错: some weird error"


def test_extract_empty_exception_fallback():
    """空错误信息：使用异常类型名兜底。"""
    result = extract_llm_error(RuntimeError())
    assert result == "执行出错: RuntimeError"


# ============ final_answer_is_empty ============


def _ai(content: str = "", *, finish_reason: str | None = None, tool_calls: list | None = None):
    """构造测试用 AIMessage。"""
    from langchain_core.messages import AIMessage

    kwargs: dict = {}
    if finish_reason is not None:
        kwargs["response_metadata"] = {"finish_reason": finish_reason}
    return AIMessage(content=content, tool_calls=tool_calls or [], **kwargs)


def test_final_answer_none_message():
    """无消息：非终态，返回 None。"""
    assert final_answer_is_empty(None) is None


def test_final_answer_with_tool_calls():
    """带 tool_calls：仍在工具循环中，返回 None。"""
    msg = _ai("", tool_calls=[{"name": "t", "args": {}, "id": "c1", "type": "tool_call"}])
    assert final_answer_is_empty(msg) is None


def test_final_answer_with_content():
    """有内容：正常回答，返回 None。"""
    assert final_answer_is_empty(_ai("正常回答", finish_reason="stop")) is None


def test_final_answer_empty_length_mentions_max_tokens():
    """空内容 + finish_reason=length：提示提高 max_tokens（推理 token 耗尽预算）。"""
    reason = final_answer_is_empty(_ai("", finish_reason="length"))
    assert reason is not None
    assert "finish_reason=length" in reason
    assert "max_tokens" in reason


def test_final_answer_empty_network_error_mentions_retry_not_max_tokens():
    """空内容 + finish_reason=network_error：提示网络重试，不误导提 max_tokens。"""
    reason = final_answer_is_empty(_ai("", finish_reason="network_error"))
    assert reason is not None
    assert "finish_reason=network_error" in reason
    assert "max_tokens" not in reason
    assert "重试" in reason


def test_final_answer_empty_unknown_finish_reason():
    """空内容 + 无 response_metadata：通用兜底文案，不提 max_tokens。"""
    reason = final_answer_is_empty(_ai(""))
    assert reason is not None
    assert "finish_reason=unknown" in reason
    assert "max_tokens" not in reason
