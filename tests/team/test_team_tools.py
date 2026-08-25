"""TeamAgent 工具模式测试 - 验证超时包裹与中间件链挂载

覆盖团队 Agent 复用 agent/ 模块的两项能力:
- ToolExecutionErrorMW: 工具异常 → ToolMessage(status="error") + 反思指令,
  使 ReAct 循环内 LLM 能读到报错并修正重试
- wrap_tools_with_timeout: 工具超时保护(防卡死),超时返回 JSON 错误不抛异常

两个模块各自已有独立单测(tests/agent/test_tool_error_mw.py /
tests/tools/test_tool_wrapper.py),本文件仅验证 TeamAgent._create_tool_agent
正确接入它们。
"""
from langchain_core.tools import tool

from team.base import TeamAgent


@tool
def _echo(message: str) -> str:
    """回显输入文本"""
    return message


class _FakeLLM:
    """替换真实 LLMClient,避免测试依赖 API 密钥与网络"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_chat_model(self):
        return None

    def chat(self, messages) -> str:
        return ""


def _monkeypatch_llm(monkeypatch) -> None:
    """把 team.base 中的 LLMClient 替换为假实现"""
    monkeypatch.setattr("team.base.LLMClient", _FakeLLM)


def _monkeypatch_create_agent(monkeypatch, captured: dict) -> None:
    """拦截 langchain.agents.create_agent,捕获构造参数并返回占位对象"""
    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return object()
    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)


def test_tool_timeout_parameter_normalization(monkeypatch):
    """tool_timeout 参数: 正值透传,0/None 归一为 None(走默认超时策略)"""
    _monkeypatch_llm(monkeypatch)
    _monkeypatch_create_agent(monkeypatch, {})

    agent = TeamAgent(name="test", tools=[_echo], tool_timeout=30)
    assert agent.tool_timeout == 30

    agent_default = TeamAgent(name="test", tools=[_echo])
    assert agent_default.tool_timeout is None

    agent_zero = TeamAgent(name="test", tools=[_echo], tool_timeout=0)
    assert agent_zero.tool_timeout is None


def test_tool_agent_middleware_chain(monkeypatch):
    """工具模式 executor 挂载 错误纠错 + 工作空间安全 两个中间件"""
    _monkeypatch_llm(monkeypatch)
    captured: dict = {}
    _monkeypatch_create_agent(monkeypatch, captured)

    TeamAgent(name="test", tools=[_echo], system_prompt="sys")

    from agent.tool_error_mw import ToolExecutionErrorMW
    from agent.workspace_mw import WorkspaceSecurityMW

    mw_types = [type(m) for m in captured["middleware"]]
    assert ToolExecutionErrorMW in mw_types
    assert WorkspaceSecurityMW in mw_types
    assert captured["system_prompt"] == "sys"


def test_tool_agent_tools_wrapped_with_timeout(monkeypatch):
    """工具模式传入的 tools 被超时包裹(带 _timeout_wrapped 标记)"""
    _monkeypatch_llm(monkeypatch)
    captured: dict = {}
    _monkeypatch_create_agent(monkeypatch, captured)

    TeamAgent(name="test", tools=[_echo], tool_timeout=5)

    wrapped = captured["tools"]
    assert len(wrapped) == 1
    assert getattr(wrapped[0], "_timeout_wrapped", False) is True