"""测试 Graph 重建消除：技能变化不触发 Graph 重建，工具变化才重建。

核心机制：create_agent 接收可变 SystemMessage 对象作为 system_prompt，
model_node 闭包捕获该对象引用。修改 .content 即可动态更新提示词，
无需重新编译 LangGraph。
"""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from agent.agent_core import AgentCore
from agent.compaction import CompactionConfig, LCAgentCompactionMiddleware


# ============ 辅助：创建最小化 AgentCore 测试实例 ============


def _make_minimal_core():
    """创建一个最小化的 AgentCore 实例（绕过 __init__）"""
    core = object.__new__(AgentCore)
    core.name = "test"
    core.verbose = False
    core.max_iterations = 25
    core.agent_core_prompt = "base prompt"
    core.active_skills = set()
    core.auto_match_skills = False
    core.execution_history = []
    core._recorded_tool_call_ids = set()
    core._state_lock = asyncio.Lock()
    core.compaction_config = CompactionConfig(max_messages=999, keep_recent=10)
    core._system_message = SystemMessage(content="base prompt")
    return core


class FakeModel:
    """最小化的 LLM 模型 mock"""

    async def ainvoke(self, prompt):
        return SimpleNamespace(text="fake summary")

    def invoke(self, prompt):
        return SimpleNamespace(text="fake summary")


class FakeExecutor:
    """记录 invoke 调用的假 executor"""

    def __init__(self):
        self.invoke_calls = 0

    def invoke(self, value, config):
        self.invoke_calls += 1
        return {"messages": [AIMessage(content="done")]}

    def get_state(self, config):
        return SimpleNamespace(values={"messages": [], "summary": ""})

    async def aupdate_state(self, config, values):
        pass


class FakeSkillManager:
    """假技能管理器"""

    def __init__(self):
        self._skills = {"git-commit": "# Git Commit Skill\nHelp with git commits.",
                        "code-review": "# Code Review Skill\nReview code quality."}

    def get_skill(self, name):
        return self._skills.get(name)

    def match_skills(self, task):
        matches = []
        for name in self._skills:
            if name.split("-")[0] in task.lower():
                matches.append(name)
        return matches

    def render_block(self, names):
        return "\n".join(self._skills[n] for n in names if n in self._skills)

    def list_skills(self):
        return [{"name": n, "description": d[:50]} for n, d in self._skills.items()]


class FakeToolObj:
    """轻量假工具（不继承 BaseTool，避免 pydantic 开销）"""

    def __init__(self, name):
        self.name = name


# ============ 测试：_update_system_prompt 不触发 Graph 重建 ============


def test_update_system_prompt_does_not_call_create_agent_executor():
    # Given: 一个已初始化的 AgentCore
    core = _make_minimal_core()
    core.agent_executor = FakeExecutor()
    core._compute_skill_block = lambda task: ""
    create_calls = 0

    def counting_create(skill_block=""):
        nonlocal create_calls
        create_calls += 1
        return FakeExecutor()

    core._create_agent_executor = counting_create

    # When: 调用 _update_system_prompt
    core._update_system_prompt("some task")

    # Then: _create_agent_executor 没有被调用（没有 Graph 重建）
    assert create_calls == 0
    # 但系统提示词内容确实更新了
    assert core._system_message.content == "base prompt"


def test_update_system_prompt_with_skill_block():
    # Given: 带技能管理器的 AgentCore
    core = _make_minimal_core()
    core.agent_executor = FakeExecutor()
    core.skill_manager = FakeSkillManager()
    core.active_skills = {"git-commit"}

    def compute_skill_block(task):
        names = set(core.active_skills)
        if core.auto_match_skills and task:
            names.update(core.skill_manager.match_skills(task))
        if not names:
            return ""
        return core.skill_manager.render_block(sorted(names))

    core._compute_skill_block = compute_skill_block

    # When: 更新系统提示词
    core._update_system_prompt("commit my changes")

    # Then: 提示词包含技能内容
    assert "Git Commit Skill" in core._system_message.content
    assert "base prompt" in core._system_message.content


def test_arun_structured_uses_update_not_rebuild():
    # Given: AgentCore 带技能管理器
    core = _make_minimal_core()
    executor = FakeExecutor()
    core.agent_executor = executor
    core.skill_manager = FakeSkillManager()
    core.active_skills = set()
    core.auto_match_skills = True

    class FakeMemory:
        def get_config(self):
            return {"configurable": {"thread_id": "thread-1"}}

        def add(self, role, content, metadata=None):
            pass

    core.memory = FakeMemory()

    def real_compute(task):
        names = set(core.active_skills)
        if core.auto_match_skills and task:
            names.update(core.skill_manager.match_skills(task))
        if not names:
            return ""
        return core.skill_manager.render_block(sorted(names))

    core._compute_skill_block = real_compute

    rebuild_count = 0

    async def counting_rebuild(task=""):
        nonlocal rebuild_count
        rebuild_count += 1

    core._arebuild_agent_executor = counting_rebuild

    # When: 执行任务（"git" 关键词匹配 git-commit 技能）
    turn = asyncio.run(core.arun_structured("git commit my code"))

    # Then: Graph 没有被重建
    assert rebuild_count == 0
    # 但系统提示词包含了匹配到的技能
    assert "Git Commit Skill" in core._system_message.content
    # 执行器被调用了一次
    assert executor.invoke_calls == 1


def test_aload_skill_does_not_rebuild_graph():
    # Given: AgentCore 带技能管理器
    core = _make_minimal_core()
    core.agent_executor = FakeExecutor()
    core.skill_manager = FakeSkillManager()
    core.auto_match_skills = False

    rebuild_count = 0

    async def counting_rebuild(task=""):
        nonlocal rebuild_count
        rebuild_count += 1

    core._arebuild_agent_executor = counting_rebuild

    # When: 加载技能
    result = asyncio.run(core.aload_skill("git-commit"))

    # Then: 技能加载成功，但没有重建 Graph
    assert result is True
    assert rebuild_count == 0
    assert "git-commit" in core.active_skills
    # 系统提示词更新了
    assert "Git Commit Skill" in core._system_message.content


def test_aclear_skills_does_not_rebuild_graph():
    # Given: AgentCore 已加载技能
    core = _make_minimal_core()
    core.agent_executor = FakeExecutor()
    core.skill_manager = FakeSkillManager()
    core.active_skills = {"git-commit", "code-review"}
    core.auto_match_skills = False

    rebuild_count = 0

    async def counting_rebuild(task=""):
        nonlocal rebuild_count
        rebuild_count += 1

    core._arebuild_agent_executor = counting_rebuild

    # When: 清空技能
    asyncio.run(core.aclear_skills())

    # Then: 技能被清空，但没有重建 Graph
    assert rebuild_count == 0
    assert len(core.active_skills) == 0
    # 系统提示词恢复为基础提示词
    assert core._system_message.content == "base prompt"


# ============ 测试：工具列表变化才触发 Graph 重建 ============


def test_areload_mcp_tools_rebuilds_when_tools_change():
    # Given: AgentCore 带有工具
    core = _make_minimal_core()
    core.local_tools = [FakeToolObj("tool_a")]
    core.mcp_tools = []
    core.tools = list(core.local_tools)
    core.agent_executor = FakeExecutor()
    core._tools_signature = frozenset(["tool_a"])
    core._compaction_middleware = LCAgentCompactionMiddleware(FakeModel(), core.compaction_config)

    rebuild_count = 0

    def counting_create(skill_block=""):
        nonlocal rebuild_count
        rebuild_count += 1
        return FakeExecutor()

    core._create_agent_executor = counting_create

    # 模拟 MCP 工具加载：新增了工具
    async def fake_load():
        core.mcp_tools = [FakeToolObj("tool_b")]
        return 1

    core._async_load_mcp_tools = fake_load

    # When: 重新加载 MCP 工具（工具列表变化）
    count = asyncio.run(core.areload_mcp_tools())

    # Then: Graph 被重建
    assert rebuild_count == 1
    assert count == 1


def test_areload_mcp_tools_skips_rebuild_when_tools_unchanged():
    # Given: AgentCore 带有工具
    core = _make_minimal_core()
    core.local_tools = [FakeToolObj("tool_a")]
    core.mcp_tools = []
    core.tools = list(core.local_tools)
    core.agent_executor = FakeExecutor()
    core._tools_signature = frozenset(["tool_a"])
    core._compaction_middleware = LCAgentCompactionMiddleware(FakeModel(), core.compaction_config)

    rebuild_count = 0

    def counting_create(skill_block=""):
        nonlocal rebuild_count
        rebuild_count += 1
        return FakeExecutor()

    core._create_agent_executor = counting_create

    # 模拟 MCP 工具加载：工具列表不变
    async def fake_load():
        core.mcp_tools = []
        return 0

    core._async_load_mcp_tools = fake_load

    # When: 重新加载 MCP 工具（工具列表不变）
    count = asyncio.run(core.areload_mcp_tools())

    # Then: Graph 没有被重建
    assert rebuild_count == 0
    assert count == 0


# ============ 测试：_system_message 可变性 ============


def test_system_message_content_is_mutable_after_compilation():
    # Given: 创建一个 SystemMessage 并验证其 content 可变
    sys_msg = SystemMessage(content="original prompt")

    # When: 修改 content
    sys_msg.content = "updated prompt with skills"

    # Then: 同一对象的 content 已更新
    assert sys_msg.content == "updated prompt with skills"


def test_achat_structured_updates_prompt_without_rebuild():
    # Given: AgentCore 带技能管理器
    core = _make_minimal_core()
    executor = FakeExecutor()
    core.agent_executor = executor
    core.skill_manager = FakeSkillManager()
    core.active_skills = set()
    core.auto_match_skills = True

    class FakeMemory:
        def get_config(self):
            return {"configurable": {"thread_id": "thread-chat-1"}}

        def add(self, role, content, metadata=None):
            pass

    core.memory = FakeMemory()

    def real_compute(task):
        names = set(core.active_skills)
        if core.auto_match_skills and task:
            names.update(core.skill_manager.match_skills(task))
        if not names:
            return ""
        return core.skill_manager.render_block(sorted(names))

    core._compute_skill_block = real_compute

    rebuild_count = 0

    async def counting_rebuild(task=""):
        nonlocal rebuild_count
        rebuild_count += 1

    core._arebuild_agent_executor = counting_rebuild

    # When: 对话（"code" 关键词匹配 code-review 技能）
    turn = asyncio.run(core.achat_structured("review my code"))

    # Then: Graph 没有被重建，但提示词更新了
    assert rebuild_count == 0
    assert "Code Review Skill" in core._system_message.content
    assert executor.invoke_calls == 1
