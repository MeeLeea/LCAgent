"""测试团队角色切换：rebuild_agent_from_team_dir 单一入口。

覆盖三件事：
1. _locate_team_agent_dir 扫描 team/ 精确定位角色目录，未命中抛 KeyError。
2. arebuild_from_team_dir 在仅提示词变化与 provider/model 变化两种场景下，
   都调用 _arebuild_agent_executor 重建 executor（system_prompt 已改为静态字符串，
   旧的仅更新提示词路径已移除，两条路径统一走 _arebuild_agent_executor）。
3. arebuild_from_team_dir 在 provider/model 变化时重建 LLMClient + executor。

断言约定：LLM 相关的预期值从 team/<角色>/agent_config.json 动态读取，
不硬编码具体 provider/model 名——验证的是"切换后 LLM 与角色配置一致"
这一行为，而非绑定某个模型（避免默认 provider 调整导致测试失配）。
"""
import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import SystemMessage

from agent import role_sw
from agent.agent_core import AgentCore
from agent.role_sw import _locate_team_agent_dir

# 项目根目录（tests/team/ 上两级）
_ROOT = Path(__file__).resolve().parents[2]

# ============ 辅助：最小化 AgentCore 与 FakeLLM ============


def _read_role_llm_config(role: str) -> dict:
    """读取 team/<role>/agent_config.json 声明的 LLM 配置（provider/model/采样参数，可缺省）"""
    config_path = _ROOT / "team" / role / "agent_config.json"
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "provider": data.get("provider"),
        "model": data.get("model"),
        "temperature": data.get("temperature"),
        "max_tokens": data.get("max_tokens"),
    }


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
    断言只关心切换行为（executor 重建次数 + 角色 name），不绑定具体 LLM。
    """
    # --- 场景 1：仅提示词变化（provider/model 不变）---
    # Given: 主 agent 当前 provider/model 与 manager 角色配置一致（保证走 prompt-only 分支）
    manager_llm = _read_role_llm_config("manager")
    core = _make_minimal_core(
        FakeLLM(provider=manager_llm["provider"], model=manager_llm["model"])
    )

    rebuild_calls_prompt_only = 0

    async def counting_rebuild_prompt_only(task=""):
        nonlocal rebuild_calls_prompt_only
        rebuild_calls_prompt_only += 1

    core._arebuild_agent_executor = counting_rebuild_prompt_only

    # When: 切换到 manager 角色（provider/model 与当前一致 → 仅重建 executor）
    asyncio.run(core.arebuild_from_team_dir("manager"))

    # Then: 仅提示词变化也重建了 executor
    assert rebuild_calls_prompt_only == 1
    # 角色已切换：name 与提示词对应 manager
    assert core.name == "manager"
    assert "任务规划者" in core.agent_core_prompt

    # --- 场景 2：LLM 变化（provider 不同）---
    # Given: 主 agent 当前 provider=qwen，与 manager 角色配置不同
    core2 = _make_minimal_core(FakeLLM(provider="qwen", model="qwen-max"))

    rebuild_calls_llm_change = 0
    constructed_llm_change = {}

    async def counting_rebuild_llm_change(task=""):
        nonlocal rebuild_calls_llm_change
        rebuild_calls_llm_change += 1

    def fake_llm_ctor_change(**kwargs):
        constructed_llm_change.update(kwargs)
        return FakeLLM(provider=kwargs["provider"], model=kwargs.get("model"))

    core2._arebuild_agent_executor = counting_rebuild_llm_change
    # 拦截 LLMClient 构造，避免真实 API key 依赖
    monkeypatch.setattr(role_sw, "LLMClient", fake_llm_ctor_change)

    # When: 切换到 manager（provider 与当前不同 → 触发 LLM 重建）
    asyncio.run(core2.arebuild_from_team_dir("manager"))

    # Then: LLM 变化也重建了 executor，且 LLMClient 被按角色配置重建
    assert rebuild_calls_llm_change == 1
    assert core2.name == "manager"
    # LLM 确实被重建（LLMClient 以角色配置参数被构造），不绑定具体值
    assert constructed_llm_change["provider"] == manager_llm["provider"]
    assert core2.llm.provider == constructed_llm_change["provider"]


# ============ 测试：provider 变化触发重建 ============


def test_rebuild_switches_llm_when_provider_changes(monkeypatch):
    # Given: 主 agent 当前 provider=qwen，与 manager 角色配置不同
    manager_llm = _read_role_llm_config("manager")
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

    # When: 切换到 manager（provider 与当前不同 → 触发 LLM 重建）
    asyncio.run(core.arebuild_from_team_dir("manager"))

    # Then: 重建了 executor，LLMClient 按角色配置重建（不绑定具体 provider 名）
    assert rebuild_calls == 1
    assert core.name == "manager"
    assert constructed["provider"] == manager_llm["provider"]
    assert core.llm.provider == constructed["provider"]


def test_rebuild_applies_role_sampling_params(monkeypatch):
    """角色切换重建 LLM 时，应用角色级 agent_config.json 的 temperature/max_tokens

    验证:切换后 LLMClient 以角色配置的采样参数构造(而非保留当前 LLM 的值)。
    """
    worker_llm = _read_role_llm_config("worker")
    assert worker_llm["temperature"] is not None, "worker 角色应配置采样参数"

    # Given: 主 agent 当前 provider 与 worker 角色不同 → 触发 LLM 重建
    core = _make_minimal_core(FakeLLM(provider="qwen", model="qwen-max"))
    constructed = {}

    async def noop_rebuild(task=""):
        pass

    def fake_llm_ctor(**kwargs):
        constructed.update(kwargs)
        return FakeLLM(provider=kwargs["provider"], model=kwargs.get("model"))

    core._arebuild_agent_executor = noop_rebuild
    monkeypatch.setattr(role_sw, "LLMClient", fake_llm_ctor)

    # When: 切换到 worker 角色
    asyncio.run(core.arebuild_from_team_dir("worker"))

    # Then: 重建的 LLMClient 携带角色级采样参数
    assert constructed["provider"] == worker_llm["provider"]
    assert constructed["temperature"] == worker_llm["temperature"]
    assert constructed["max_tokens"] == worker_llm["max_tokens"]
