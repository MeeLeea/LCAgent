"""Agent 回合结果类型 - AgentTurnResult（单次 LangGraph turn 的返回封装）。

从 agent_core.py 抽离，避免 turn_runners / streaming 等 mixin 反向导入
agent_core 造成循环依赖：所有引用方统一从本模块导入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from langgraph.types import Interrupt


@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    """Typed result for one LangGraph turn."""

    status: Literal["completed", "interrupted", "cancelled"]
    output: str | None = None
    interrupts: list[Interrupt] = field(default_factory=list)

    @classmethod
    def completed(cls, output: str) -> AgentTurnResult:
        return cls(status="completed", output=output, interrupts=[])

    @classmethod
    def interrupted(cls, interrupts: list[Interrupt]) -> AgentTurnResult:
        return cls(status="interrupted", output=None, interrupts=interrupts)

    @classmethod
    def cancelled(cls, output: str) -> AgentTurnResult:
        return cls(status="cancelled", output=output, interrupts=[])

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_interrupted(self) -> bool:
        return self.status == "interrupted"


__all__ = ["AgentTurnResult"]