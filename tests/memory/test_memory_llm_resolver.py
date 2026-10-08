"""agent.memory_llm.build_memory_llm_resolver 单元测试。

覆盖：
- ``thread_id`` 为空（``None`` / ``""``）→ 返回默认 getter 结果
- 会话有 provider 配置 → 按 ``(provider, model, temperature, max_tokens)``
  构造 ``LLMClient``，且不触碰默认 getter
- LRU 缓存命中 / 不同配置不同对象 / 超 ``max_size`` 淘汰最旧
- 全部回退路径返回默认 getter：config 为 None、provider 为空、
  ``apeek_session_config`` 抛错、``LLMClient`` 构造抛错
- ``max_size < 1`` → ``ValueError``

全部离线：monkeypatch ``agent.memory_llm.LLMClient`` 为记录构造参数的假类，
用假 session 提供异步 ``apeek_session_config``。
"""
from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest

from agent import memory_llm
from session.config import SessionConfig


class _FakeLLMClient:
    """记录构造 kwargs 的 LLMClient 替身（不触网、不建模）。"""

    instances: ClassVar[list[_FakeLLMClient]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        type(self).instances.append(self)


class _RaisingLLMClient:
    """构造即失败的 LLMClient 替身，用于验证构造异常回退。"""

    def __init__(self, **kwargs: Any) -> None:
        raise RuntimeError("LLMClient 构造失败")


class _DefaultGetter:
    """默认 getter：返回稳定哨兵对象并记录调用次数。"""

    def __init__(self) -> None:
        self.calls = 0
        self.value = object()

    def __call__(self) -> Any:
        self.calls += 1
        return self.value


class _FakeSession:
    """提供异步 ``apeek_session_config`` 的假会话。"""

    def __init__(
        self, config: SessionConfig | None = None, *, raises: bool = False
    ) -> None:
        self.config = config
        self._raises = raises
        self.calls: list[str] = []

    async def apeek_session_config(self, session_id: str) -> SessionConfig | None:
        self.calls.append(session_id)
        if self._raises:
            raise RuntimeError("读取会话配置失败")
        return self.config


@pytest.fixture(autouse=True)
def _patch_llm_client(monkeypatch: pytest.MonkeyPatch) -> type[_FakeLLMClient]:
    """每个用例前把 LLMClient 换成假类并清空实例记录。"""
    _FakeLLMClient.instances = []
    monkeypatch.setattr(memory_llm, "LLMClient", _FakeLLMClient)
    return _FakeLLMClient


# ════════════════════════════════════════════════════════════════════════
#  thread_id 为空 → 默认 getter
# ════════════════════════════════════════════════════════════════════════


def test_falsy_thread_id_returns_default_without_peeking():
    default = _DefaultGetter()
    session = _FakeSession(config=SessionConfig(provider="p1"))
    resolver = memory_llm.build_memory_llm_resolver(session, default)

    result_none = asyncio.run(resolver(None))
    result_empty = asyncio.run(resolver(""))

    assert result_none is default.value
    assert result_empty is default.value
    assert default.calls == 2
    # 空 thread_id 不应查询会话配置
    assert session.calls == []


# ════════════════════════════════════════════════════════════════════════
#  会话有配置 → 按 provider 构造 / 复用 LLMClient
# ════════════════════════════════════════════════════════════════════════


def test_session_config_constructs_llm_for_provider():
    default = _DefaultGetter()
    config = SessionConfig(provider="p1", model="m1", temperature=0.5, max_tokens=128)
    session = _FakeSession(config=config)
    resolver = memory_llm.build_memory_llm_resolver(session, default)

    llm = asyncio.run(resolver("t1"))

    assert isinstance(llm, _FakeLLMClient)
    assert llm.kwargs == {
        "provider": "p1",
        "model": "m1",
        "temperature": 0.5,
        "max_tokens": 128,
    }
    # 会话配置按 thread_id 读取，且未使用默认 getter
    assert session.calls == ["t1"]
    assert default.calls == 0


def test_cache_reuse_and_distinct_configs():
    default = _DefaultGetter()
    session = _FakeSession(config=SessionConfig(provider="p1", model="m1"))
    resolver = memory_llm.build_memory_llm_resolver(session, default)

    async def run():
        first = await resolver("t1")
        second = await resolver("t1")
        session.config = SessionConfig(provider="p2", model="m2")
        third = await resolver("t2")
        return first, second, third

    first, second, third = asyncio.run(run())

    # 相同配置复用同一对象；不同配置构造新对象
    assert first is second
    assert third is not first
    assert len(_FakeLLMClient.instances) == 2


def test_lru_eviction_when_exceeding_max_size():
    default = _DefaultGetter()
    config_a = SessionConfig(provider="p1", model="m1")
    config_b = SessionConfig(provider="p2", model="m2")
    session = _FakeSession(config=config_a)
    resolver = memory_llm.build_memory_llm_resolver(session, default, max_size=1)

    async def run():
        first_a = await resolver("t1")
        session.config = config_b
        first_b = await resolver("t2")
        # config_a 已被 max_size=1 挤出缓存，重新解析应构造新对象
        session.config = config_a
        second_a = await resolver("t3")
        return first_a, first_b, second_a

    first_a, first_b, second_a = asyncio.run(run())

    assert first_b is not first_a
    assert second_a is not first_a
    assert len(_FakeLLMClient.instances) == 3


# ════════════════════════════════════════════════════════════════════════
#  回退路径 → 默认 getter
# ════════════════════════════════════════════════════════════════════════


def test_none_config_falls_back_to_default():
    default = _DefaultGetter()
    session = _FakeSession(config=None)
    resolver = memory_llm.build_memory_llm_resolver(session, default)

    assert asyncio.run(resolver("t1")) is default.value
    assert default.calls == 1
    assert session.calls == ["t1"]


def test_falsy_provider_falls_back_to_default():
    default = _DefaultGetter()
    session = _FakeSession(config=SessionConfig(provider=""))
    resolver = memory_llm.build_memory_llm_resolver(session, default)

    assert asyncio.run(resolver("t1")) is default.value
    assert default.calls == 1
    assert _FakeLLMClient.instances == []


def test_apeek_raises_falls_back_to_default():
    default = _DefaultGetter()
    session = _FakeSession(config=SessionConfig(provider="p1"), raises=True)
    resolver = memory_llm.build_memory_llm_resolver(session, default)

    assert asyncio.run(resolver("t1")) is default.value
    assert default.calls == 1


def test_construction_failure_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(memory_llm, "LLMClient", _RaisingLLMClient)
    default = _DefaultGetter()
    session = _FakeSession(config=SessionConfig(provider="p1"))
    resolver = memory_llm.build_memory_llm_resolver(session, default)

    assert asyncio.run(resolver("t1")) is default.value
    assert default.calls == 1


# ════════════════════════════════════════════════════════════════════════
#  非法 max_size
# ════════════════════════════════════════════════════════════════════════


def test_max_size_below_one_raises_value_error():
    default = _DefaultGetter()
    session = _FakeSession()

    with pytest.raises(ValueError):
        memory_llm.build_memory_llm_resolver(session, default, max_size=0)
