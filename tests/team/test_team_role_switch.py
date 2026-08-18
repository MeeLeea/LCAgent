"""测试团队角色切换：rebuild_agent_from_team_dir 单一入口。

覆盖三件事：
1. _locate_team_agent_dir 扫描 team/ 精确定位角色目录，未命中抛 KeyError。
2. arebuild_from_team_dir 在仅提示词变化与 provider/model 变化两种场景下，
   都调用 _arebuild_agent_executor 重建 executor（system_prompt 已改为静态字符串，
   旧的仅更新提示词路径已移除，两条路径统一走 _arebuild_agent_executor）。
3. arebuild_from_team_dir 在 provider/model 变化时重建 LLMClient + executor。
"""
import asyncio

import pytest
from langchain_core.messages import SystemMessage

from agent import role_sw
from agent.agent_core import AgentCore
from agent.role_sw import _locate_team_agent_dir

# ============ 辅助：最小化 AgentCore 与 FakeLLM ============


class FakeLLM:
    """最小化 LLM mock，提供切换判断所需属性"""

    def __init__(self, provider="zhipu", model="glm-4-flash"):
        self.provider = provider
        self.model = model
        self.config_file = "config/llm_config.json"
        self.temperature = 0.7
        self.max_tokens = 2048


def _make_minimal_core(llm=None):
    """创建一个最小化的 AgentCore 实例（绕过 __init__）"""
    core = object.__new__(AgentCore)
    core.name = "test"
    core.verbose = False
    core.max_iterations = 25
    core.agent_core_prompt = "base prompt"
    core.active_skills = set()
    core.auto_match_skills = False
    core._state_lock = asyncio.Lock()
    core._system_message = SystemMessage(content="base prompt")
    core.llm = llm or FakeLLM()
    core._closed = False
    return core


# ============ 测试：目录定位 ============


def test_locate_team_agent_dir_finds_manager():
    # When: 定位内置 manager 角色
    path = _locate_team_agent_dir("manager")

    # Then: 返回的目录包含必需文件
    import os

    assert os.path.isdir(path)
    assert path.endswith("manager")
    assert os.path.isfile(os.path.join(path, "agent_config.json"))
    assert os.path.isfile(os.path.join(path, "AGENT.md"))


def test_locate_team_agent_dir_raises_on_missing():
    # When/Then: 未知角色抛 KeyError，错误信息含可用角色
    with pytest.raises(KeyError) as exc:
        _locate_team_agent_dir("nonexistent_role_xyz")
    assert "nonexistent_role_xyz" in str(exc.value)


# ============ 测试：_arebuild_agent_executor 在两种场景下都被调用 ============


def test_rebuild_calls_agent_executor_for_prompt_only_and_llm_change(monkeypatch):
    """仅提示词变化与 LLM 变化两种场景都应调用 _arebuild_agent_executor。

    新架构下 system_prompt 已改为静态字符串，prompt-only 变化也需重建 executor；
    旧的仅更新提示词路径已移除，两条路径统一走 _arebuild_agent_executor。
    """
    # --- 场景 1：仅提示词变化（provider/model 不变）---
    # Given: 主 agent 当前 provider=zhipu（与 manager 配置一致）
    core = _make_minimal_core(FakeLLM(provider="zhipu", model="glm-4-flash"))

    rebuild_calls_prompt_only = 0

    async def counting_rebuild_prompt_only(task=""):
        nonlocal rebuild_calls_prompt_only
        rebuild_calls_prompt_only += 1

    core._arebuild_agent_executor = counting_rebuild_prompt_only

    # When: 切换到 manager 角色（model=null → 沿用当前 model，provider 相同）
    asyncio.run(core.arebuild_from_team_dir("manager"))

    # Then: 仅提示词变化也重建了 executor
    assert rebuild_calls_prompt_only == 1
    # 角色提示词已切换（manager 提示词包含"任务规划者"）
    assert "任务规划者" in core.agent_core_prompt
    assert core.name == "manager"

    # --- 场景 2：LLM 变化（provider 不同）---
    # Given: 主 agent 当前 provider=qwen，切换到 provider=zhipu 的 manager 角色
    core2 = _make_minimal_core(FakeLLM(provider="qwen", model="qwen-max"))

    rebuild_calls_llm_change = 0

    async def counting_rebuild_llm_change(task=""):
        nonlocal rebuild_calls_llm_change
        rebuild_calls_llm_change += 1

    def fake_llm_ctor(**kwargs):
        return FakeLLM(provider=kwargs["provider"], model=kwargs.get("model"))

    core2._arebuild_agent_executor = counting_rebuild_llm_change
    # 拦截 LLMClient 构造，避免真实 API key 依赖
    monkeypatch.setattr(role_sw, "LLMClient", fake_llm_ctor)

    # When: 切换到 manager（provider=zhipu ≠ 当前 qwen → 触发 LLM 重建）
    asyncio.run(core2.arebuild_from_team_dir("manager"))

    # Then: LLM 变化也重建了 executor
    assert rebuild_calls_llm_change == 1
    assert core2.llm.provider == "zhipu"


# ============ 测试：provider 变化触发重建 ============


def test_rebuild_switches_llm_when_provider_changes(monkeypatch):
    # Given: 主 agent 当前 provider=qwen，切换到 provider=zhipu 的 manager 角色
    core = _make_minimal_core(FakeLLM(provider="qwen", model="qwen-max"))

    rebuild_calls = 0
    constructed = {}

    async def counting_rebuild(task=""):
        nonlocal rebuild_calls
        rebuild_calls += 1

    def fake_llm_ctor(**kwargs):
        constructed.update(kwargs)
        return FakeLLM(provider=kwargs["provider"], model=kwargs.get("model"))

    core._arebuild_agent_executor = counting_rebuild
    # 拦截 LLMClient 构造，避免真实 API key 依赖(现由 role_sw 模块调用)
    monkeypatch.setattr(role_sw, "LLMClient", fake_llm_ctor)

    # When: 切换到 manager（provider=zhipu ≠ 当前 qwen → 触发 LLM 重建）
    asyncio.run(core.arebuild_from_team_dir("manager"))

    # Then: 重建了 executor，且用新 provider 构造了 LLM
    assert rebuild_calls == 1
    assert constructed["provider"] == "zhipu"
    assert core.llm.provider == "zhipu"
