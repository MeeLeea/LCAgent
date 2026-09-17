"""StreamStallRetryMixin：流式 chunk 超时透明重试 + 无重复输出边界 的单元测试。

全部离线：通过 monkeypatch 替换底层 `ChatOpenAI._astream`，不发起任何网络请求。
"""
import asyncio

import pytest
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, StreamChunkTimeoutError

from llm import llm_client
from llm.llm_client import (
    STREAM_RETRY_ATTEMPTS,
    CloudmistChatOpenAI,
    StreamStallRetryMixin,
    should_retry,
)

_END = object()  # 行为列表哨兵：底层流正常结束


class _FakeChunk:
    """最小 chunk 替身：与底层 `ChatGenerationChunk.text` 语义一致。"""

    def __init__(self, text: str) -> None:
        self.text = text


def _install_fake_astream(monkeypatch, scripts):
    """用脚本化的假 `_astream` 替换底层 `ChatOpenAI._astream`。

    Args:
        monkeypatch: pytest monkeypatch fixture
        scripts: 每次调用的行为列表，元素为 _FakeChunk(产出) / 异常(抛出) / _END(正常结束)。
            未以 _END 结束的行为视为流中断，产出后抛 StreamChunkTimeoutError；
            调用次数超出 scripts 长度时复用最后一个行为。

    Returns:
        记录调用次数的 dict
    """
    calls = {"n": 0}

    async def fake_astream(self, messages, stop=None, run_manager=None, **kwargs):
        idx = calls["n"]
        calls["n"] += 1
        behavior = scripts[min(idx, len(scripts) - 1)]
        for item in behavior:
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
        raise StreamChunkTimeoutError(
            120.0, model_name="fake", chunks_received=len(behavior)
        )

    monkeypatch.setattr(ChatOpenAI, "_astream", fake_astream)
    return calls


def _model():
    """构造一个不联网的 CloudmistChatOpenAI（仅本地构造，不发起请求）。"""
    return CloudmistChatOpenAI(
        model="test-model", api_key="sk-test", base_url="http://localhost"
    )


async def _collect(model):
    return [chunk.text async for chunk in model._astream([HumanMessage("hi")])]


# ============ 1. 透明重试：零可见内容时重启 ============

def test_retries_before_any_visible_content(monkeypatch):
    """首轮只产出空 text chunk 后中断：重启并交付第二轮内容，且不重复。"""
    monkeypatch.setattr(llm_client, "RETRY_MAX_DELAY", 0.01)
    calls = _install_fake_astream(
        monkeypatch,
        [
            [_FakeChunk(""), _FakeChunk("")],
            [_FakeChunk("你好，世界"), _END],
        ],
    )
    texts = asyncio.run(_collect(_model()))
    # 首轮空(reasoning)chunk 亦被透传(空 text 无害)，非空内容应且仅出现一次
    assert texts.count("你好，世界") == 1
    assert calls["n"] == 2


# ============ 2. 硬边界：已交付可见内容后不再重启 ============

def test_hard_stop_after_visible_content(monkeypatch):
    """首轮已交付非空内容后中断：异常上抛、不重启、不重复内容。"""
    monkeypatch.setattr(llm_client, "RETRY_MAX_DELAY", 0.01)
    timeout = StreamChunkTimeoutError(120.0, model_name="fake", chunks_received=1)
    calls = _install_fake_astream(monkeypatch, [[_FakeChunk("可见内容"), timeout]])

    async def _scenario():
        received = []
        with pytest.raises(StreamChunkTimeoutError):
            async for chunk in _model()._astream([HumanMessage("hi")]):
                received.append(chunk.text)
        return received

    received = asyncio.run(_scenario())
    assert received == ["可见内容"]
    assert calls["n"] == 1


# ============ 3. 重试上限：耗尽后原样上抛 ============

def test_stops_after_retry_cap(monkeypatch):
    """始终在产出内容前中断：恰好尝试 STREAM_RETRY_ATTEMPTS + 1 次后上抛。"""
    monkeypatch.setattr(llm_client, "RETRY_MAX_DELAY", 0.01)
    timeout = StreamChunkTimeoutError(120.0, model_name="fake", chunks_received=0)
    calls = _install_fake_astream(monkeypatch, [[timeout]])

    async def _scenario():
        with pytest.raises(StreamChunkTimeoutError):
            await _collect(_model())

    asyncio.run(_scenario())
    assert calls["n"] == STREAM_RETRY_ATTEMPTS + 1


# ============ 4. should_retry 排除项：不与图级重试叠加 ============

def test_should_retry_carves_out_stream_chunk_timeout():
    """StreamChunkTimeoutError 由图级重试让位给流层；普通 TimeoutError 仍可重试。"""
    assert should_retry(StreamChunkTimeoutError(120.0)) is False
    assert should_retry(TimeoutError()) is True


# ============ 5. MRO 与网关 workaround 回归守护 ============

def test_mixin_precedes_base_and_gateway_workaround_survives():
    """mixin 的 `_astream` 位于底层之上，且 max_completion_tokens 改名逻辑不受影响。"""
    model = _model()
    mro = type(model).__mro__
    assert StreamStallRetryMixin in mro
    assert mro.index(StreamStallRetryMixin) < mro.index(ChatOpenAI)
    assert type(model)._astream is StreamStallRetryMixin._astream

    payload = model._get_request_payload([HumanMessage("hi")], max_tokens=128)
    assert "max_tokens" in payload
    assert "max_completion_tokens" not in payload
