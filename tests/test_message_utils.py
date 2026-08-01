"""agent.message_utils.extract_llm_error 的单元测试。"""
from agent.message_utils import extract_llm_error


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
