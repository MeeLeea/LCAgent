"""Agent模块

延迟导入 AgentCore 以避免与 session 包的循环导入：
session/__init__ → manager → agent.events → agent/__init__ → agent_core → session
改为 __getattr__ 延迟加载，使 `from agent.events import AgentEvent` 不触发 agent_core。
"""
__all__ = ['AgentCore']


def __getattr__(name: str):
    """延迟导入 AgentCore，避免顶层导入触发 session 循环依赖。"""
    if name == "AgentCore":
        from .agent_core import AgentCore
        return AgentCore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
