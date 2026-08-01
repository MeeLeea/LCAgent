"""llm_client.should_retry / 瞬时错误自动重试 的单元测试。"""
import pytest

import llm_client
from llm_client import RETRY_ATTEMPTS, _make_retryer, should_retry


class _RetryableStatusError(Exception):
    """模拟带 status_code 属性的可重试 HTTP 异常(openai.RateLimitError 等)。"""

    def __init__(self, status_code: int):
        super().__init__(f"http error {status_code}")
        self.status_code = status_code


# ============ should_retry 判定矩阵 ============

@pytest.mark.parametrize("exc", [
    ConnectionError("connection refused"),
    TimeoutError("timed out"),
    _RetryableStatusError(429),
    _RetryableStatusError(503),
    Exception("Error code: 500 - internal error"),
    Exception("HTTP 502 Bad Gateway"),
    Exception("Service temporarily unavailable"),
    Exception("Too Many Requests"),
])
def test_should_retry_returns_true(exc):
    assert should_retry(exc) is True


@pytest.mark.parametrize("exc", [
    ValueError("bad request"),
    Exception("Error code: 400 - invalid param"),
    _RetryableStatusError(401),
    _RetryableStatusError(404),
    Exception("authentication failed"),
    Exception("some unrelated error"),
])
def test_should_retry_returns_false(exc):
    assert should_retry(exc) is False


# ============ 重试行为 ============

def _fast_retryer(monkeypatch):
    """用极小退避时间构建生产环境使用的重试器，保证测试快速。"""
    monkeypatch.setattr(llm_client, "RETRY_BASE_DELAY", 0.01)
    monkeypatch.setattr(llm_client, "RETRY_MAX_DELAY", 0.02)
    return _make_retryer()


def test_chat_retry_success_after_transient_errors(monkeypatch):
    """先失败两次(瞬时错误)再成功：自动重试并返回最终结果。"""
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls < RETRY_ATTEMPTS:
                raise RuntimeError("Service temporarily unavailable")
            return type("Resp", (), {"content": "ok"})()

    fake = FakeClient()
    retryer = _fast_retryer(monkeypatch)
    result = retryer(fake.invoke, [])
    assert result.content == "ok"
    assert fake.calls == RETRY_ATTEMPTS


def test_chat_retry_exhausted_reraises(monkeypatch):
    """始终返回瞬时错误：重试耗尽后原样抛出原始异常。"""
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise RuntimeError("Service temporarily unavailable")

    fake = FakeClient()
    retryer = _fast_retryer(monkeypatch)
    with pytest.raises(RuntimeError, match="Service temporarily unavailable"):
        retryer(fake.invoke, [])
    assert fake.calls == RETRY_ATTEMPTS


def test_chat_retry_not_triggered_for_fatal_error(monkeypatch):
    """非瞬时错误(400)：不重试，直接抛出。"""
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise ValueError("Error code: 400 - bad request")

    fake = FakeClient()
    retryer = _fast_retryer(monkeypatch)
    with pytest.raises(ValueError, match="400"):
        retryer(fake.invoke, [])
    assert fake.calls == 1
