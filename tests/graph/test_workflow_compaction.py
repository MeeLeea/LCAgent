"""
测试 _build_compaction_middleware 压缩中间件构造工具

覆盖：
- agent 无 llm / 无 get_chat_model / 构造失败时禁用（返回 None）
- agent 有 llm.get_chat_model 时正确构造中间件（阈值来自配置）

节点级 _compaction_wrapper / wrap_node_with_compaction 已移除，
压缩统一由 LCAgentCompactionMiddleware.before_model 中间件触发。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from graph.common import _build_compaction_middleware


def test_build_middleware_none_without_llm():
    """agent 无 llm 属性时返回 None(压缩禁用)。"""
    agent = SimpleNamespace(name="fake", response="x")
    assert _build_compaction_middleware(agent) is None


def test_build_middleware_none_when_get_chat_model_missing():
    """agent.llm 无 get_chat_model 时返回 None。"""
    agent = SimpleNamespace(name="fake", llm=SimpleNamespace())
    assert _build_compaction_middleware(agent) is None


def test_build_middleware_constructs_with_llm():
    """agent 有 llm.get_chat_model 时构造中间件(阈值来自配置)。"""
    agent = SimpleNamespace(name="real", llm=SimpleNamespace(get_chat_model=lambda: MagicMock()))
    mw = _build_compaction_middleware(agent)
    assert mw is not None
    assert mw.config.max_messages == 50  # 默认配置


def test_build_middleware_disables_on_construction_error():
    """构造失败(如 get_chat_model 抛异常)时返回 None,不向外抛。"""
    agent = SimpleNamespace(name="broken", llm=SimpleNamespace(get_chat_model=lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    assert _build_compaction_middleware(agent) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
