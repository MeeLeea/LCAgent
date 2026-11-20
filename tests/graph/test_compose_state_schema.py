"""compose_state_schema 单元测试。

用极简的角色桩类（仅含 ``state_fields`` ClassVar）验证动态 State 组合器的
合并、冲突检测与 reducer 保留，不依赖真实 team 角色。
"""
from __future__ import annotations

from typing import ClassVar, get_args

import pytest
from langgraph.graph.message import add_messages

from graph.common.schema import _BASE_CHANNELS, compose_state_schema


class _ManagerLike:
    """桩：模拟 ManagerAgent 声明 context_summary / plan。"""

    state_fields: ClassVar[dict[str, type]] = {"context_summary": str, "plan": str}


class _WorkerLike:
    """桩：模拟 WorkerAgent 声明 worker_result。"""

    state_fields: ClassVar[dict[str, type]] = {"worker_result": str}


class _DupA:
    """桩：与 _DupB 声明同名通道，用于冲突检测。"""

    state_fields: ClassVar[dict[str, type]] = {"shared_field": str}


class _DupB:
    """桩：与 _DupA 声明同名通道，用于冲突检测。"""

    state_fields: ClassVar[dict[str, type]] = {"shared_field": str}


class _TaskCollider:
    """桩：声明底座已有的 task 通道，用于底座 vs 角色冲突检测。"""

    state_fields: ClassVar[dict[str, type]] = {"task": str}


def test_compose_includes_base_channels() -> None:
    """底座通道 task/raw_context/messages/active_skills 必须存在。"""
    schema = compose_state_schema({})
    annotations = schema.__annotations__
    for channel in ("task", "raw_context", "messages", "active_skills"):
        assert channel in annotations


def test_compose_includes_agent_fields() -> None:
    """角色声明的字段应被合并进 schema。"""
    schema = compose_state_schema({"manager": _ManagerLike(), "worker": _WorkerLike()})
    annotations = schema.__annotations__
    assert annotations["context_summary"] is str
    assert annotations["plan"] is str
    assert annotations["worker_result"] is str


def test_compose_no_summary_channel() -> None:
    """T3 守护：底座无 summary；T2 守护：底座无 context_summary。"""
    schema = compose_state_schema({"worker": _WorkerLike()})
    annotations = schema.__annotations__
    assert "summary" not in annotations
    assert "context_summary" not in annotations


def test_compose_context_summary_from_manager() -> None:
    """Manager 声明 context_summary 时应出现（T2：该通道归 Manager）。"""
    schema = compose_state_schema({"manager": _ManagerLike()})
    assert schema.__annotations__["context_summary"] is str


def test_compose_extra_channels() -> None:
    """extra_channels 应被合并进 schema（工作流级编排通道）。"""
    schema = compose_state_schema(
        {"worker": _WorkerLike()},
        extra_channels={"round": int, "sim_status": str},
    )
    annotations = schema.__annotations__
    assert annotations["round"] is int
    assert annotations["sim_status"] is str


def test_compose_duplicate_channel_raises() -> None:
    """两个角色声明同名通道 → ValueError。"""
    with pytest.raises(ValueError, match="state 通道冲突"):
        compose_state_schema({"a": _DupA(), "b": _DupB()})


def test_messages_reducer_preserved() -> None:
    """messages 通道必须保留 Annotated[..., add_messages] reducer。"""
    schema = compose_state_schema({})
    annotation = schema.__annotations__["messages"]
    assert get_args(annotation)  # Annotated 带元数据
    assert add_messages in getattr(annotation, "__metadata__", ())
    # 与底座定义保持一致（未在组合过程中被复制丢元数据）
    assert annotation == _BASE_CHANNELS["messages"]


def test_base_and_agent_collision_raises() -> None:
    """角色声明底座已有通道（task）→ ValueError。"""
    with pytest.raises(ValueError, match="state 通道冲突"):
        compose_state_schema({"collider": _TaskCollider()})


def test_extra_channel_collision_raises() -> None:
    """extra_channels 与既有通道冲突 → ValueError（错误信息不含角色）。"""
    with pytest.raises(ValueError, match="state 通道冲突"):
        compose_state_schema({}, extra_channels={"task": str})
