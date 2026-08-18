"""测试 Graph 重建消除：技能变化不触发 Graph 重建，工具变化才重建。

核心机制：AgentCore 无状态化——system_prompt 为静态字符串，
技能注入由 SkillInjectionMiddleware 在 model 调用时从 LCAgentState.active_skills
读取（随 checkpoint per-thread 隔离）。技能变化只写入 state，不重建 Graph；
仅工具列表/LLM 变化时才调用 _arebuild_agent_executor 重新编译 LangGraph。
"""
import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, SystemMessage

from agent.agent_core import AgentCore
from utils.compaction import CompactionConfig, LCAgentCompactionMiddleware

# ============ 辅助：创建最小化 AgentCore 测试实例 ============


def _make_minimal_core():
    """创建一个最小化的 AgentCore 实例（绕过 __init__）"""
    core = object.__new__(AgentCore)
    core.name = "test"
    core.verbose = False
    core.max_iterations = 25
    core.agent_core_prompt = "base prompt"
    core.auto_match_skills = False
    core._state_lock = asyncio.Lock()
    core.compaction_config = CompactionConfig(max_messages=999, keep_recent=10)
    return core


class FakeModel:
    """最小化的 LLM 模型 mock"""

    async def ainvoke(self, prompt):
        return SimpleNamespace(text="fake summary")

    def invoke(self, prompt):
        return SimpleNamespace(text="fake summary")


class FakeExecutor:
    """记录调用并模拟 LangGraph executor 的 invoke / state API"""

    def __init__(self):
        self.invoke_calls = 0
        self.get_state_calls = 0
        self.update_state_calls = 0
        self._state_values = {"active_skills": [], "messages": [], "summary": ""}

    def invoke(self, value, config):
        self.invoke_calls += 1
        return {"messages": [AIMessage(content="done")]}

    async def ainvoke(self, value, config):
        self.invoke_calls += 1
        return {"messages": [AIMessage(content="done")]}

    def get_state(self, config):
        self.get_state_calls += 1
        return SimpleNamespace(values=dict(self._state_values))

    async def aget_state(self, config):
        self.get_state_calls += 1
        return SimpleNamespace(values=dict(self._state_values))

    async def aupdate_state(self, config, values):
        self.update_state_calls += 1
        self._state_values.update(values)


class FakeMemory:
    """假记忆：提供 get_config / aadd，模拟 AgentMemory 的最小接口"""

    def __init__(self, thread_id="thread-1"):
        self.thread_id = thread_id

    def get_config(self, thread_id=None):
        tid = thread_id or self.thread_id
        return {"configurable": {"thread_id": tid}}

    def add(self, role, content, metadata=None):
        pass

    async def aadd(self, role, content, metadata=None):
        self.add(role, content, metadata)


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


# ============ 测试：技能变化不触发 Graph 重建 ============


def test_arun_structured_uses_update_not_rebuild():
    # Given: AgentCore 带技能管理器
    core = _make_minimal_core()
    executor = FakeExecutor()
    core.agent_executor = executor
    core.skill_manager = FakeSkillManager()
    core.memory = FakeMemory()
    core.auto_match_skills = True

    rebuild_count = 0

    async def counting_rebuild(task=""):
        nonlocal rebuild_count
        rebuild_count += 1

    core._arebuild_agent_executor = counting_rebuild

    # When: 执行任务（技能注入由 SkillInjectionMiddleware 自动完成，无需重建 Graph）
    asyncio.run(core.arun_structured("git commit my code"))

    # Then: Graph 没有被重建，但 executor 被调用了一次
    assert rebuild_count == 0
    assert executor.invoke_calls == 1


def test_aload_skill_does_not_rebuild_graph():
    # Given: AgentCore 带 mock executor（提供 aget_state / aupdate_state）与 memory
    core = _make_minimal_core()
    executor = FakeExecutor()
    core.agent_executor = executor
    core.skill_manager = FakeSkillManager()
    core.memory = FakeMemory()
    core.auto_match_skills = False

    rebuild_count = 0

    async def counting_rebuild(task=""):
        nonlocal rebuild_count
        rebuild_count += 1

    core._arebuild_agent_executor = counting_rebuild

    # When: 加载技能
    result = asyncio.run(core.aload_skill("git-commit"))

    # Then: 技能加载成功，但没有重建 Graph；技能通过 aupdate_state 写入 state
    assert result is True
    assert rebuild_count == 0
    assert executor.update_state_calls == 1
    assert "git-commit" in executor._state_values["active_skills"]


def test_aclear_skills_does_not_rebuild_graph():
    # Given: AgentCore 已在 state 中加载技能
    core = _make_minimal_core()
    executor = FakeExecutor()
    executor._state_values["active_skills"] = ["git-commit", "code-review"]
    core.agent_executor = executor
    core.skill_manager = FakeSkillManager()
    core.memory = FakeMemory()
    core.auto_match_skills = False

    rebuild_count = 0

    async def counting_rebuild(task=""):
        nonlocal rebuild_count
        rebuild_count += 1

    core._arebuild_agent_executor = counting_rebuild

    # When: 清空技能
    asyncio.run(core.aclear_skills())

    # Then: 没有重建 Graph；技能通过 aupdate_state 清空为空列表
    assert rebuild_count == 0
    assert executor.update_state_calls == 1
    assert executor._state_values["active_skills"] == []


def test_achat_structured_updates_prompt_without_rebuild():
    # Given: AgentCore 带技能管理器
    core = _make_minimal_core()
    executor = FakeExecutor()
    core.agent_executor = executor
    core.skill_manager = FakeSkillManager()
    core.memory = FakeMemory()
    core.auto_match_skills = True

    rebuild_count = 0

    async def counting_rebuild(task=""):
        nonlocal rebuild_count
        rebuild_count += 1

    core._arebuild_agent_executor = counting_rebuild

    # When: 对话（技能注入由 SkillInjectionMiddleware 自动完成，无需重建 Graph）
    asyncio.run(core.achat_structured("review my code"))

    # Then: Graph 没有被重建，但 executor 被调用了一次
    assert rebuild_count == 0
    assert executor.invoke_calls == 1


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


# ============ 测试：SystemMessage 可变性 ============


def test_system_message_content_is_mutable_after_compilation():
    # Given: 创建一个 SystemMessage 并验证其 content 可变
    sys_msg = SystemMessage(content="original prompt")

    # When: 修改 content
    sys_msg.content = "updated prompt with skills"

    # Then: 同一对象的 content 已更新
    assert sys_msg.content == "updated prompt with skills"
