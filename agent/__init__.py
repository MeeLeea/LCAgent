"""Agent模块

延迟导入 AgentCore 以避免与 session 包的循环导入：
session/__init__ → manager → utils.events → agent/__init__ → agent_core → session
事件模型已迁移至 utils.events，`from utils.events import AgentEvent` 不触发 agent_core。
压缩中间件（compaction）同样已迁移至 utils.compaction，agent_core 经 `from
utils.compaction import ...` 导入，不依赖 agent 包内部。
"""
__all__ = ['AgentCore']


def __getattr__(name: str):
    """延迟导入 AgentCore，避免顶层导入触发 session 循环依赖。"""
    if name == "AgentCore":
        from .agent_core import AgentCore
        return AgentCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
