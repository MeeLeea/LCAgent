"""
测试多 Agent 工作流功能
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from agent.turn_types import AgentTurnResult
from cli.commands.workflow import workflow_command
from graph.simple import (
    arun_simple_workflow,
    build_simple_workflow,
    manager_plan_node,
    summarize_context,
    terminator_final_node,
    worker_exec_node,
)
from team.base import TeamAgent

# 默认模板(与各角色类的 default_templates 一致,供 FakeAgent 兜底)
DEFAULT_TEMPLATES: dict[str, str] = {
    "manager_plan": "请为以下任务制定详细的执行计划:\n\n{task}\n\n记忆上下文摘要:\n{context_summary}",
    "summarize_context": "你是一个工作流上下文提炼助手。",
    "worker_exec": "请执行以下计划:\n\n{plan}",
    "terminator_final": (
        "原始任务: {task}\n\n执行计划: {plan}\n\n执行结果: {worker_result}\n\n"
        "记忆上下文摘要:\n{context_summary}\n\n请汇总以上信息,为用户提供清晰的最终答案。"
    ),
}


@dataclass
class FakeAgent:
    """模拟 TeamAgent,不联网;节点改调 run_team_turn_with_interrupt 后,
    统一经 ``arun_structured`` 入口记录调用并返回 ``AgentTurnResult.completed``。

    summarize 节点(summarize_context 节点)的 prompt 含 ``summarize_context``
    模板前缀(节点把模板拼到 raw 前部),据此识别后返回 ``summary_response``,
    与原 ``asummarize_context`` 返回 summary_response 的语义对齐;其余节点
    回退 ``response``。
    """

    # summarize_context 节点 prompt 的特征字符串(节点模板固定,稳定可识别)
    _SUMMARIZE_MARKER: str = "你是一个工作流上下文提炼助手"

    name: str = "test-agent"
    response: str = "fake response"
    summary_response: str = "记忆摘要: 用户偏好中文"
    prompt_file: str | None = None
    default_templates: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TEMPLATES))
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def ainvoke(self, task: str, config=None) -> str:
        """异步版 invoke:记录调用并返回模拟结果(兼容 TeamAgent.ainvoke 既有调用方)"""
        self.calls.append(("invoke", task))
        return self.response

    def _is_summarize_node(self, prompt: str) -> bool:
        """识别是否为 summarize_context 节点的 prompt(含模板特征)"""
        return self._SUMMARIZE_MARKER in prompt

    def _next(self, prompt: str) -> str:
        """按节点类型返回模拟响应:summarize 节点返回 summary_response,其余返回 response"""
        if self._is_summarize_node(prompt):
            return self.summary_response
        return self.response

    def get_template(self, name: str) -> str:
        """懒加载模板:优先读 AGENT.md 小节,缺失回退默认模板"""
        content = TeamAgent._read_prompt_file(self.prompt_file)
        if content is not None:
            _, templates = TeamAgent.parse_prompt_sections(content)
            if name in templates:
                return templates[name]
        return self.default_templates.get(name, "")

    def render_template(self, template: str, **kwargs) -> str:
        """占位符替换"""
        return TeamAgent.render_template(template, **kwargs)

    def inject_into_prompt(self, prompt: str, task: str, active_names=()) -> str:
        """技能注入占位:测试不验证技能块,原样返回 prompt(已含渲染内容)"""
        return prompt

    async def arun_structured(self, prompt: str, config=None) -> AgentTurnResult:
        """节点经 run_team_turn_with_interrupt 调用的唯一入口;
        记录 (arun_structured, prompt) 并返回 completed(self._next(prompt))
        """
        self.calls.append(("arun_structured", prompt))
        return AgentTurnResult.completed(self._next(prompt))


# ==================== 测试工作流节点 ====================

def test_summarize_context_node():
    """测试 summarize 节点:Manager 提炼记忆上下文摘要"""
    manager = FakeAgent(name="manager", summary_response="记忆摘要: 项目背景A")
    state = {"raw_context": "用户: 之前聊过项目A"}

    result = asyncio.run(summarize_context(state, manager))

    assert result["context_summary"] == "记忆摘要: 项目背景A"
    assert len(manager.calls) == 1
    # 节点改调 run_team_turn_with_interrupt → arun_structured
    assert manager.calls[0][0] == "arun_structured"
    # prompt 含 summarize_context 模板前缀 + raw_context
    assert "用户: 之前聊过项目A" in manager.calls[0][1]


def test_manager_plan_node():
    """测试 Manager 节点:拆解任务"""
    manager = FakeAgent(name="manager", response="计划: 步骤1、步骤2")
    state = {"task": "帮我分析项目结构"}
    
    result = asyncio.run(manager_plan_node(state, manager))
    
    assert result["plan"] == "计划: 步骤1、步骤2"
    assert len(manager.calls) == 1
    assert manager.calls[0][0] == "arun_structured"
    assert "请为以下任务制定详细的执行计划" in manager.calls[0][1]


def test_worker_exec_node():
    """测试 Worker 节点:执行计划"""
    worker = FakeAgent(name="worker", response="已完成步骤1和步骤2")
    state = {"plan": "计划: 步骤1、步骤2"}
    
    result = asyncio.run(worker_exec_node(state, worker))
    
    assert result["worker_result"] == "已完成步骤1和步骤2"
    assert len(worker.calls) == 1
    assert worker.calls[0][0] == "arun_structured"
    assert "请执行以下计划" in worker.calls[0][1]


def test_terminator_final_node():
    """测试 Terminator 节点:汇总结果"""
    terminator = FakeAgent(name="terminator", response="最终答案: 项目包含3个模块")
    state = {
        "task": "帮我分析项目结构",
        "plan": "计划: 步骤1、步骤2",
        "worker_result": "已完成步骤1和步骤2"
    }
    
    result = asyncio.run(terminator_final_node(state, terminator))
    
    assert result["final_answer"] == "最终答案: 项目包含3个模块"
    assert len(terminator.calls) == 1
    assert terminator.calls[0][0] == "arun_structured"
    prompt = terminator.calls[0][1]
    assert "原始任务" in prompt
    assert "执行计划" in prompt
    assert "执行结果" in prompt


def test_manager_plan_node_with_summary():
    """测试 Manager 节点:有记忆摘要时注入摘要块"""
    manager = FakeAgent(name="manager", response="计划")
    state = {"task": "任务", "context_summary": "用户偏好中文"}

    result = asyncio.run(manager_plan_node(state, manager))

    assert result["plan"] == "计划"
    prompt = manager.calls[0][1]
    assert "记忆上下文摘要:" in prompt
    assert "用户偏好中文" in prompt


def test_terminator_final_node_with_summary():
    """测试 Terminator 节点:有记忆摘要时注入摘要块"""
    terminator = FakeAgent(name="terminator", response="最终答案")
    state = {
        "task": "任务",
        "plan": "计划",
        "worker_result": "结果",
        "context_summary": "用户偏好中文",
    }

    result = asyncio.run(terminator_final_node(state, terminator))

    assert result["final_answer"] == "最终答案"
    prompt = terminator.calls[0][1]
    assert "记忆上下文摘要:" in prompt
    assert "用户偏好中文" in prompt


def test_terminator_final_node_no_summary():
    """测试 Terminator 节点:无记忆摘要时摘要段头仍在但内容为空"""
    terminator = FakeAgent(name="terminator", response="最终答案")
    state = {
        "task": "任务",
        "plan": "计划",
        "worker_result": "结果",
    }

    asyncio.run(terminator_final_node(state, terminator))

    prompt = terminator.calls[0][1]
    assert "记忆上下文摘要:" in prompt
    assert "用户偏好" not in prompt


# ==================== 测试 AGENT.md 提示词模板 ====================

def test_parse_prompt_sections():
    """测试按 ## workflow: 小节拆分系统提示词与工作流模板"""
    content = (
        "# Agent 核心提示词\n\n"
        "你是经理。\n\n"
        "## 重要规则\n"
        "1. 规则A\n\n"
        "## workflow:manager_plan\n"
        "请制定计划:\n"
        "{task}\n\n"
        "## workflow:summarize_context\n"
        "你是一个提炼助手。"
    )
    system, templates = TeamAgent.parse_prompt_sections(content)
    assert "你是经理" in system
    assert "## 重要规则" in system
    assert "请制定计划" not in system
    assert templates["manager_plan"] == "请制定计划:\n{task}"
    assert templates["summarize_context"] == "你是一个提炼助手。"


def test_render_template():
    """测试占位符安全替换,JSON 花括号字面量不报错"""
    template = '任务: {task}\n数据: {"key": 1}\n{plan}'
    rendered = TeamAgent.render_template(template, task="分析项目", plan="执行A")
    assert "任务: 分析项目" in rendered
    assert '数据: {"key": 1}' in rendered
    assert "执行A" in rendered


def test_role_agent_md_strips_workflow_sections():
    """测试角色 AGENT.md 的 ## workflow:* 小节从系统提示词剥离且模板可读"""
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    manager_md = os.path.join(base_dir, "team", "manager", "AGENT.md")

    with open(manager_md, encoding="utf-8") as f:
        content = f.read()
    system, templates = TeamAgent.parse_prompt_sections(content)

    assert "workflow:" not in system
    assert "manager_plan" in templates
    assert "summarize_context" in templates


# ==================== 测试工作流图构建 ====================

def test_build_simple_workflow():
    """测试构建监督者工作流图"""
    manager = FakeAgent(name="manager")
    worker = FakeAgent(name="worker")
    terminator = FakeAgent(name="terminator")
    
    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})
    
    # 验证图已编译
    assert graph is not None
    
    # 验证节点存在
    nodes = graph.get_graph().nodes
    node_names = [n.id for n in nodes.values()]
    assert "summarize" in node_names
    assert "manager_plan" in node_names
    assert "worker_exec" in node_names
    assert "terminator_final" in node_names


def test_workflow_edges():
    """测试工作流边的正确连接"""
    manager = FakeAgent(name="manager")
    worker = FakeAgent(name="worker")
    terminator = FakeAgent(name="terminator")
    
    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})
    graph_obj = graph.get_graph()
    
    # 验证边: START → summarize → manager_plan → worker_exec → terminator_final → END
    edges = [(e.source, e.target) for e in graph_obj.edges]
    
    assert ("__start__", "summarize") in edges
    assert ("summarize", "manager_plan") in edges
    assert ("manager_plan", "worker_exec") in edges
    assert ("worker_exec", "terminator_final") in edges
    assert ("terminator_final", "__end__") in edges


def test_build_simple_workflow_with_prompt_file(tmp_path):
    """测试从各角色 AGENT.md 加载工作流节点模板并注入记忆摘要"""
    manager_md = tmp_path / "manager.md"
    manager_md.write_text(
        "## workflow:manager_plan\n请按文件计划: {task}\n\n记忆块: {context_summary}",
        encoding="utf-8",
    )
    worker_md = tmp_path / "worker.md"
    worker_md.write_text("## workflow:worker_exec\n按文件执行: {plan}", encoding="utf-8")
    terminator_md = tmp_path / "terminator.md"
    terminator_md.write_text(
        "## workflow:terminator_final\n文件汇总: {task} / {worker_result}\n记忆块: {context_summary}",
        encoding="utf-8",
    )

    manager = FakeAgent(name="manager", response="计划X", summary_response="摘要S", prompt_file=str(manager_md))
    worker = FakeAgent(name="worker", response="结果X", prompt_file=str(worker_md))
    terminator = FakeAgent(name="terminator", response="答案X", prompt_file=str(terminator_md))

    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})
    asyncio.run(graph.ainvoke({
        "task": "任务T",
        "raw_context": "原始上下文",  # 非空 → summarize 调 helper 返回 summary_response
        "context_summary": "摘要S",
        "plan": "",
        "worker_result": "",
        "final_answer": "",
    }))

    # summarize 节点(raw 非空)调 arun_structured 返回 summary_response="摘要S";
    # manager_plan 节点经 arun_structured 注入 context_summary="摘要S"
    assert "按文件计划: 任务T" in manager.calls[1][1]
    assert "记忆块: 摘要S" in manager.calls[1][1]
    assert "按文件执行: 计划X" in worker.calls[0][1]
    assert "文件汇总: 任务T / 结果X" in terminator.calls[0][1]
    assert "记忆块: 摘要S" in terminator.calls[0][1]


def test_run_simple_workflow():
    """测试运行工作流并验证状态流转"""
    manager = FakeAgent(name="manager", response="计划: A、B、C")
    worker = FakeAgent(name="worker", response="执行结果: 完成A、B、C")
    terminator = FakeAgent(name="terminator", response="最终答案: 全部完成")
    
    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})
    result = asyncio.run(arun_simple_workflow(graph, "测试任务"))
    
    # 验证最终状态
    assert result["task"] == "测试任务"
    assert result["plan"] == "计划: A、B、C"
    assert result["worker_result"] == "执行结果: 完成A、B、C"
    assert result["final_answer"] == "最终答案: 全部完成"
    
    # raw_context 默认空串 → summarize 节点短路不调 helper;
    # manager 只剩 manager_plan 一次调用(经 arun_structured)
    assert len(manager.calls) == 1
    assert len(worker.calls) == 1
    assert len(terminator.calls) == 1


# ==================== 测试 main.py 工作流接口 ====================

def test_workflow_registry():
    """测试 WORKFLOWS 注册表"""
    from graph.registry import WORKFLOWS

    assert "simple" in WORKFLOWS
    spec = WORKFLOWS["simple"]
    # register_workflow 注册的是规格字典,不是裸函数
    assert isinstance(spec, dict)
    assert callable(spec["builder"])
    assert spec["runner"] is not None


def test_build_workflow_unknown_name():
    """测试未知工作流名称抛出 KeyError"""
    from graph.registry import build_workflow
    
    with pytest.raises(KeyError) as exc_info:
        build_workflow("unknown_workflow")
    
    assert "未知工作流: unknown_workflow" in str(exc_info.value)
    assert "可用工作流:" in str(exc_info.value)
    assert "simple" in str(exc_info.value)


# ==================== 测试 Agent 注册装饰器 ====================

def test_register_agent_decorator():
    """测试 register_agent 装饰器:注册条目 + 原样返回类"""
    from graph.registry import AGENT_REGISTRY, register_agent
    from team.base import TeamAgent
    
    @register_agent("fake_role", "team/fake/agent_config.json", tools=None)
    class FakeRoleAgent(TeamAgent):
        pass
    
    try:
        assert "fake_role" in AGENT_REGISTRY
        spec = AGENT_REGISTRY["fake_role"]
        assert spec["agent_class"] is FakeRoleAgent
        assert spec["config_file"] == "team/fake/agent_config.json"
        assert spec["tools"] is None
        # 装饰器应原样返回被装饰的类
        assert FakeRoleAgent.__name__ == "FakeRoleAgent"
    finally:
        # 避免污染全局注册表影响其他测试
        AGENT_REGISTRY.pop("fake_role", None)


def test_register_agent_tools_passthrough():
    """测试装饰器的 tools 参数原样透传到注册表"""
    from graph.registry import AGENT_REGISTRY, register_agent
    from team.base import TeamAgent
    
    fake_tools = ["tool_a", "tool_b"]
    
    @register_agent("fake_worker", "team/fake_worker/agent_config.json", tools=fake_tools)
    class FakeWorkerAgent(TeamAgent):
        pass
    
    try:
        assert AGENT_REGISTRY["fake_worker"]["tools"] == fake_tools
    finally:
        AGENT_REGISTRY.pop("fake_worker", None)


def test_register_agent_mcp_tools_passthrough():
    """测试装饰器的 mcp_tools 参数原样透传到注册表(默认 None,显式声明时存列表)"""
    from graph.registry import AGENT_REGISTRY, register_agent
    from team.base import TeamAgent

    # 显式声明 mcp_tools
    @register_agent(
        "fake_mcp_role",
        "team/fake_mcp/agent_config.json",
        tools=None,
        mcp_tools=["write_file"],
    )
    class FakeMcpAgent(TeamAgent):
        pass

    try:
        spec = AGENT_REGISTRY["fake_mcp_role"]
        assert spec["mcp_tools"] == ["write_file"]
        # 未声明 mcp_tools 时默认 None(向后兼容)
    finally:
        AGENT_REGISTRY.pop("fake_mcp_role", None)

    # 不传 mcp_tools 时默认 None
    @register_agent("fake_no_mcp", "team/fake_no_mcp/agent_config.json")
    class FakeNoMcpAgent(TeamAgent):
        pass

    try:
        assert AGENT_REGISTRY["fake_no_mcp"]["mcp_tools"] is None
    finally:
        AGENT_REGISTRY.pop("fake_no_mcp", None)


def test_builtin_agents_registered():
    """测试内置角色(manager/worker/terminator/architect)已通过装饰器注册"""
    import team  # noqa: F401 - 触发各 agent 模块加载,完成注册
    from graph.registry import AGENT_REGISTRY
    from team.base import TeamAgent
    
    for role in ("manager", "worker", "terminator", "architect"):
        assert role in AGENT_REGISTRY
        spec = AGENT_REGISTRY[role]
        assert issubclass(spec["agent_class"], TeamAgent)
        assert spec["config_file"].startswith(f"team/{role}/")
    
    # Worker 注入全部本地工具,Manager/Terminator 纯文本模式
    assert AGENT_REGISTRY["worker"]["tools"] is not None
    assert len(AGENT_REGISTRY["worker"]["tools"]) > 0
    assert AGENT_REGISTRY["manager"]["tools"] is None
    assert AGENT_REGISTRY["terminator"]["tools"] is None

    # Architect 声明 mcp_tools=["write_file"],本地 tools 为 None
    # (MCP 工具由 build_workflow 装配期同步拉取,见 test_build_workflow_mcp_tools_injection)
    assert AGENT_REGISTRY["architect"]["tools"] is None
    assert AGENT_REGISTRY["architect"]["mcp_tools"] == ["write_file"]

    # RTL Designer / Verification 同样声明 mcp_tools=["write_file"],
    # 让规格文档/RTL 源码/验证报告/验证计划可写入 workspace
    assert AGENT_REGISTRY["rtl_designer"]["tools"] is None
    assert AGENT_REGISTRY["rtl_designer"]["mcp_tools"] == ["write_file"]
    assert AGENT_REGISTRY["rtl_verification"]["tools"] is None
    assert AGENT_REGISTRY["rtl_verification"]["mcp_tools"] == ["write_file"]


def test_build_workflow_mcp_tools_injection():
    """测试 build_workflow 装配期同步拉取声明的 MCP 工具并合并到 tools

    通过 patch load_mcp_tools_by_name_sync 返回伪造工具,验证:
    - 声明 mcp_tools 的角色,工具被合并进 build_team_agent 的 tools 参数
    - 未声明 mcp_tools 的角色,tools 保持原样(本地 tools 或 None)
    - MCP 加载失败(返回空)时,角色降级为纯文本模式(tools=None)
    """
    from graph.registry import AGENT_REGISTRY, register_agent, build_workflow
    from team.base import TeamAgent
    from dataclasses import dataclass

    @dataclass
    class FakeTool:
        """最小工具桩:只携带 name,供按名筛选测试。"""
        name: str

    fake_write_file = FakeTool(name="write_file")

    captured: dict = {}

    # 用 fake role 测试,避免污染真实 architect 注册项
    @register_agent(
        "fake_mcp_inject_role",
        "team/architect/agent_config.json",  # 复用 architect 配置避免新建文件
        tools=None,
        mcp_tools=["write_file"],
    )
    class FakeMcpInjectAgent(TeamAgent):
        pass

    # 另注册一个无 mcp_tools 的角色作对照
    @register_agent(
        "fake_no_mcp_inject_role",
        "team/manager/agent_config.json",
        tools=None,
    )
    class FakeNoMcpInjectAgent(TeamAgent):
        pass

    original_build = None
    try:
        # registry._build 内部用 `from team import build_team_agent` 取符号,
        # 每次调用都重新从 team 包命名空间解析,因此 patch team.build_team_agent 即可拦截
        import team as team_mod

        original_build = team_mod.build_team_agent

        def _spy_build(agent_class, config_file, base_dir, tools=None, **kwargs):
            captured[agent_class.__name__] = tools
            # 返回一个最小实例,避免真实 LLM/MCP 初始化
            inst = object.__new__(agent_class)
            inst.tools = tools or []
            inst.agent_executor = None
            return inst

        team_mod.build_team_agent = _spy_build

        # 还需 patch 工作流 builder,避免真实图编译
        from graph import registry as reg_mod

        original_get_spec = reg_mod._get_workflow_spec

        def _fake_get_spec(name):
            return {
                "builder": lambda agents, checkpointer=None: type(
                    "FakeGraph", (), {"get_graph": lambda self: type(
                        "G", (), {"nodes": []}
                    )()}
                )(),
                "runner": None,
                "roles": ["fake_mcp_inject_role", "fake_no_mcp_inject_role"],
                "description": "test",
            }

        reg_mod._get_workflow_spec = _fake_get_spec
        reg_mod.WORKFLOWS["__test_mcp_inject__"] = {
            "builder": _fake_get_spec("__test_mcp_inject__")["builder"],
            "runner": None,
            "roles": ["fake_mcp_inject_role", "fake_no_mcp_inject_role"],
            "description": "test",
        }

        # Case 1: MCP 工具加载成功
        with patch(
            "tools.mcp_loader.load_mcp_tools_by_name_sync",
            return_value=[fake_write_file],
        ):
            build_workflow("__test_mcp_inject__", checkpointer=None)

        # 声明 mcp_tools 的角色: tools 含 write_file
        assert captured.get("FakeMcpInjectAgent") is not None
        assert len(captured["FakeMcpInjectAgent"]) == 1
        assert captured["FakeMcpInjectAgent"][0].name == "write_file"
        # 未声明 mcp_tools 的角色: tools 为 None(本地 tools 也为 None)
        assert captured.get("FakeNoMcpInjectAgent") is None

        # Case 2: MCP 加载失败(返回空)→ 降级纯文本模式(tools=None)
        captured.clear()
        with patch(
            "tools.mcp_loader.load_mcp_tools_by_name_sync",
            return_value=[],
        ):
            build_workflow("__test_mcp_inject__", checkpointer=None)

        # 声明 mcp_tools 但加载失败: tools 为 None(本地 tools 也为空 → or None)
        assert captured.get("FakeMcpInjectAgent") is None

    finally:
        if original_build is not None:
            team_mod.build_team_agent = original_build
        if original_get_spec is not None:
            reg_mod._get_workflow_spec = original_get_spec
        AGENT_REGISTRY.pop("fake_mcp_inject_role", None)
        AGENT_REGISTRY.pop("fake_no_mcp_inject_role", None)
        reg_mod.WORKFLOWS.pop("__test_mcp_inject__", None)


# ==================== 测试 CLI 命令 ====================

@dataclass
class FakeContext:
    """模拟 CommandContext"""
    output: list[str] = field(default_factory=list)
    
    def print(self, text: str) -> None:
        self.output.append(text)


def test_workflow_list_command():
    """测试 workflow 命令列出可用工作流"""
    context = FakeContext()
    outcome = asyncio.run(workflow_command(context, "workflow"))
    
    assert outcome.handled
    output_text = "\n".join(context.output)
    assert "可用工作流" in output_text
    assert "simple" in output_text
    assert "用法: workflow:<name> <task>" in output_text


def test_workflow_execute_command_missing_task():
    """测试缺少任务描述时的错误提示"""
    context = FakeContext()
    outcome = asyncio.run(workflow_command(context, "workflow:simple"))
    
    assert outcome.handled
    output_text = "\n".join(context.output)
    assert "错误: 缺少任务描述" in output_text


def test_workflow_execute_command_unknown_workflow():
    """测试未知工作流的错误提示"""
    context = FakeContext()
    outcome = asyncio.run(workflow_command(context, "workflow:unknown 测试任务"))
    
    assert outcome.handled
    output_text = "\n".join(context.output)
    assert "错误: 未知工作流 'unknown'" in output_text
    assert "可用工作流:" in output_text
    assert "simple" in output_text


def test_workflow_command_parse():
    """测试命令解析:workflow:simple 测试任务"""
    context = FakeContext()
    
    # Mock run_workflow 避免实际执行
    import cli.commands.workflow as wf_module
    original_run = None
    
    async def mock_run(context, name: str, task: str):
        return {"final_answer": f"Mock 结果: {name} / {task}"}
    
    # 暂存原函数(如果存在)
    if hasattr(wf_module, "run_workflow"):
        original_run = wf_module.run_workflow
    
    # 临时替换
    wf_module.run_workflow = mock_run
    
    try:
        outcome = asyncio.run(workflow_command(context, "workflow:simple 帮我分析项目"))
        
        assert outcome.handled
        output_text = "\n".join(context.output)
        assert "工作流执行完成" in output_text
        assert "Mock 结果: simple / 帮我分析项目" in output_text
    finally:
        # 恢复
        if original_run:
            wf_module.run_workflow = original_run


# ==================== 测试 thread_id 隔离 ====================

def test_workflow_thread_isolation():
    """测试多次运行使用独立 thread_id"""
    manager = FakeAgent(name="manager", response="计划1")
    worker = FakeAgent(name="worker", response="结果1")
    terminator = FakeAgent(name="terminator", response="答案1")
    
    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})
    
    # 第一次运行
    result1 = asyncio.run(arun_simple_workflow(graph, "任务1"))
    
    # 第二次运行(重置 agent 调用记录)
    manager.calls.clear()
    worker.calls.clear()
    terminator.calls.clear()
    manager.response = "计划2"
    worker.response = "结果2"
    terminator.response = "答案2"
    
    result2 = asyncio.run(arun_simple_workflow(graph, "任务2"))
    
    # 验证两次运行结果独立
    assert result1["task"] == "任务1"
    assert result1["final_answer"] == "答案1"
    
    assert result2["task"] == "任务2"
    assert result2["final_answer"] == "答案2"
    
    # 验证两次都调用了 agent(raw_context 默认空 → summarize 短路,
    # manager 只剩 manager_plan 一次调用)
    assert len(manager.calls) == 1
    assert len(worker.calls) == 1
    assert len(terminator.calls) == 1


# ==================== 测试记忆注入与写回 ====================

def test_run_simple_workflow_with_memory():
    """测试带记忆运行时:摘要注入 plan/final 节点,worker 不注入"""
    manager = FakeAgent(name="manager", response="计划: A、B、C", summary_response="记忆摘要: 用户偏好中文")
    worker = FakeAgent(name="worker", response="执行结果: 完成")
    terminator = FakeAgent(name="terminator", response="最终答案: 完成")
    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})

    result = asyncio.run(arun_simple_workflow(graph, "测试任务", raw_context="用户: 之前聊过偏好中文"))

    # manager 两次调用:summarize + plan(均经 arun_structured)
    assert manager.calls[0][0] == "arun_structured"
    assert "用户: 之前聊过偏好中文" in manager.calls[0][1]
    assert manager.calls[1][0] == "arun_structured"
    assert "记忆摘要: 用户偏好中文" in manager.calls[1][1]
    # worker 只拿 plan,不注入摘要
    assert worker.calls[0][0] == "arun_structured"
    assert "请执行以下计划" in worker.calls[0][1]
    assert "记忆摘要" not in worker.calls[0][1]
    # terminator 注入摘要
    assert "记忆摘要: 用户偏好中文" in terminator.calls[0][1]
    # 状态透传
    assert result["context_summary"] == "记忆摘要: 用户偏好中文"
    assert result["final_answer"] == "最终答案: 完成"


class FakeMemory:
    """模拟 AgentMemory 的记忆读取接口"""
    def __init__(self, short_term=None, long_term=None):
        self._short = short_term or []
        self._long = long_term or []

    def get_short_term(self, limit=None):
        return self._short

    def get_long_term(self, limit=5):
        return self._long


class FakeExecutor:
    """记录 aupdate_state 调用的假 executor。

    使用异步 ``aupdate_state`` 以匹配真实 LangGraph CompiledGraph 接口:
    会话层 checkpointer 为 ``AsyncSqliteSaver``,同步 ``update_state``
    在主线程会抛 ``Synchronous calls to AsyncSqliteSaver ...``。
    """
    def __init__(self):
        self.updated = []

    async def aupdate_state(self, config, values):
        self.updated.append((config, values))


class FakeSession:
    """模拟 SessionRegistry 的短期记忆接口"""
    current_session_id = "t1"
    checkpointer = None  # 测试无持久化

    def generate_session_id(self, workflow_name=None):
        return f"test-workflow-{workflow_name}-thread-xxxx"

    def is_workflow_session(self, session_id):
        return False

    async def aget_short_term(self, session_id=None):
        return []


class FakeAgentCore:
    """模拟 AgentCore 的 executor + memory"""
    def __init__(self):
        self.agent_executor = FakeExecutor()
        self.memory = FakeMemory()
        self.session = FakeSession()

    def _invoke_config(self, thread_id=None):
        return {"configurable": {"thread_id": "t1"}}


class WriteBackContext:
    """携带 agent 的假上下文(供写回测试)"""
    def __init__(self, agent=None):
        self.agent = agent or FakeAgentCore()
        self.output: list[str] = []
        # 与 CommandContext 对齐:工作流事件回调默认为空
        self.workflow_event_cb = None
        # workflow 门面(绑定 WorkflowAdapter);None 时 run_workflow 抛错
        self.workflow_sm = None

    def print(self, text: str) -> None:
        self.output.append(text)


def test_run_workflow_via_workflow_sm():
    """run_workflow 经 workflow_sm.arun_stream 执行:节点事件打印 + done 收集 final_answer。"""
    import cli.commands.workflow as wf_module

    class FakeWorkflowSM:
        """模拟 SessionManager(workflow 门面):捕获调用并产出 SSE 事件流。"""

        def __init__(self):
            self.calls: list[tuple[str, str | None]] = []
            self.events = [
                {"type": "workflow_node", "node": "manager", "status": "running"},
                {"type": "workflow_node", "node": "manager", "status": "done"},
                {"type": "done", "content": "最终答案"},
            ]

        async def arun_stream(self, task, thread_id=None):
            self.calls.append((task, thread_id))
            for ev in self.events:
                yield ev

    sm = FakeWorkflowSM()
    ctx = WriteBackContext(FakeAgentCore())
    ctx.workflow_sm = sm

    result = asyncio.run(wf_module.run_workflow(ctx, "simple", "测试任务"))

    assert result["final_answer"] == "最终答案"
    # 非 workflow 会话 → generate_session_id(name)
    assert sm.calls == [("测试任务", "test-workflow-simple-thread-xxxx")]
    assert "▸ 节点开始: manager" in ctx.output
    assert "✓ 节点完成: manager" in ctx.output


def test_run_workflow_reuses_workflow_session():
    """当前会话已是 workflow 专属会话时复用其 thread_id(持久化绑定)。"""
    import cli.commands.workflow as wf_module

    class FakeWorkflowSM:
        def __init__(self):
            self.calls: list[tuple[str, str | None]] = []

        async def arun_stream(self, task, thread_id=None):
            self.calls.append((task, thread_id))
            yield {"type": "done", "content": "答案"}

    sm = FakeWorkflowSM()
    core = FakeAgentCore()
    core.session.is_workflow_session = lambda sid: True
    ctx = WriteBackContext(core)
    ctx.workflow_sm = sm

    result = asyncio.run(wf_module.run_workflow(ctx, "simple", "测试任务"))

    assert result["final_answer"] == "答案"
    assert sm.calls == [("测试任务", "t1")]  # 复用 FakeSession.current_session_id


def test_run_workflow_error_event():
    """workflow_sm 产出 error 事件时打印错误并转发给事件回调。"""
    import cli.commands.workflow as wf_module

    class FakeWorkflowSM:
        async def arun_stream(self, task, thread_id=None):
            yield {"type": "error", "content": "图执行失败"}

    forwarded: list[dict] = []
    sm = FakeWorkflowSM()
    ctx = WriteBackContext(FakeAgentCore())
    ctx.workflow_sm = sm
    ctx.workflow_event_cb = lambda ev: forwarded.append(ev)

    result = asyncio.run(wf_module.run_workflow(ctx, "simple", "测试任务"))

    assert result["final_answer"] == ""
    assert "工作流执行错误: 图执行失败" in "".join(ctx.output)
    assert any(ev.get("type") == "error" for ev in forwarded)
    # 整体状态复位:运行中 → 完成(error 事件不带 status,按类型过滤)
    assert [ev.get("status") for ev in forwarded if ev.get("type") == "workflow_status"] == [
        "running",
        "done",
    ]


# ==================== 测试运行进度跟踪 ====================

def test_run_simple_workflow_node_callbacks():
    """测试节点进度回调:4 个业务节点按链路顺序各触发 start/end 且 start 先于 end"""
    manager = FakeAgent(name="manager", response="计划: A")
    worker = FakeAgent(name="worker", response="结果: 完成")
    terminator = FakeAgent(name="terminator", response="答案: 完成")
    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})

    events: list[tuple[str, str]] = []
    asyncio.run(arun_simple_workflow(
        graph,
        "测试任务",
        on_node_start=lambda e: events.append(("start", e.node)),
        on_node_end=lambda e: events.append(("end", e.node)),
    ))

    expected = ["summarize", "manager_plan", "worker_exec", "terminator_final"]
    # start/end 各 4 次,且顺序与执行链路一致
    assert [e for e in events if e[0] == "start"] == [("start", n) for n in expected]
    assert [e for e in events if e[0] == "end"] == [("end", n) for n in expected]
    # 每个节点 start 严格先于其 end
    for node in expected:
        assert events.index(("start", node)) < events.index(("end", node))


def test_run_simple_workflow_no_callbacks_still_works():
    """测试不传回调时工作流正常运行(向后兼容)"""
    manager = FakeAgent(name="manager", response="计划")
    worker = FakeAgent(name="worker", response="结果")
    terminator = FakeAgent(name="terminator", response="答案")
    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})

    result = asyncio.run(arun_simple_workflow(graph, "测试任务"))
    assert result["final_answer"] == "答案"


def test_run_workflow_emits_workflow_events():
    """测试 run_workflow:节点/整体状态通过 workflow_event_cb 转发结构化事件"""
    import cli.commands.workflow as wf_module
    from cli.commands.types import CommandContext

    events: list[dict[str, str]] = []

    class FakeWorkflowSM:
        """模拟 workflow 门面:产出 4 个业务节点的 running/done 事件 + done。"""

        async def arun_stream(self, task, thread_id=None):
            for node in ("summarize", "manager_plan", "worker_exec", "terminator_final"):
                yield {"type": "workflow_node", "node": node, "status": "running"}
                yield {"type": "workflow_node", "node": node, "status": "done"}
            yield {"type": "done", "content": "答案"}

    core = FakeAgentCore()
    ctx = CommandContext(
        agent=core,
        base_dir=".",
        config_file="",
        mcp_config_file="",
        print_fn=lambda t: None,
        input_fn=lambda p="": "",
        select_menu=lambda *a, **k: "",
        create_llm=lambda p: None,
        list_providers=dict,
        run_structured_until_completion=lambda a, t: "",
        chat_until_completion=lambda a, t: "",
        safety_backend=object(),
        workflow_event_cb=events.append,
        workflow_sm=FakeWorkflowSM(),
    )

    asyncio.run(wf_module.run_workflow(ctx, "simple", "测试任务"))

    # 整体状态:先 running 后 done
    statuses = [e["status"] for e in events if e["type"] == "workflow_status"]
    assert statuses == ["running", "done"]
    # 4 个节点各产生 running + done,且每个节点 running 先于 done
    node_events = [e for e in events if e["type"] == "workflow_node"]
    assert len(node_events) == 8
    for node in ("summarize", "manager_plan", "worker_exec", "terminator_final"):
        running_idx = node_events.index({"type": "workflow_node", "node": node, "status": "running"})
        done_idx = node_events.index({"type": "workflow_node", "node": node, "status": "done"})
        assert running_idx < done_idx


# ==================== 测试 TeamAgent 异步能力(ainvoke/astream) ====================


class FakeChunk:
    """模拟 langchain AIMessageChunk:仅含 content。"""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatModel:
    """模拟 langchain chat model:记录 astream 调用与收到的 config。"""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.received_config: dict | None = None

    async def astream(self, messages, config=None):
        self.received_config = config
        for content in self.chunks:
            yield FakeChunk(content)


class FakeLLMClient:
    """模拟 LLMClient(纯文本模式):get_chat_model 返回 FakeChatModel。"""

    def __init__(self, chunks: list[str]) -> None:
        self.model = FakeChatModel(chunks)

    def get_chat_model(self):
        return self.model


class FakeAgentExecutor:
    """模拟 create_agent 产物:astream_events 产出 on_chat_model_stream。"""

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.received_config: dict | None = None

    async def astream_events(self, inputs, config=None, version=None):
        self.received_config = config
        for content in self.chunks:
            yield {"event": "on_chat_model_stream", "data": {"chunk": FakeChunk(content)}}
        yield {"event": "on_chain_end", "data": {}}


def _make_text_agent(chunks: list[str]) -> TeamAgent:
    """构造纯文本模式 TeamAgent 替身(跳过 __init__,避免真实 LLMClient)。"""
    agent = TeamAgent.__new__(TeamAgent)
    agent.name = "test"
    agent.system_prompt = "系统提示"
    agent.llm = FakeLLMClient(chunks)
    agent.agent_executor = None
    agent.verbose = False
    return agent


def test_team_agent_astream_pure_text_yields_chunks():
    """纯文本模式 astream:逐块产出 LLM 流式输出,空块被过滤。"""
    agent = _make_text_agent(["你", "", "好"])

    async def _run():
        return [c async for c in agent.astream("任务")]

    assert asyncio.run(_run()) == ["你", "好"]


def test_team_agent_ainvoke_aggregates_stream():
    """ainvoke = astream 聚合:返回拼接结果。"""
    agent = _make_text_agent(["你", "好"])

    async def _run():
        return await agent.ainvoke("任务")

    assert asyncio.run(_run()) == "你好"


def test_team_agent_astream_passes_callbacks():
    """astream 透传外层 callbacks 到 chat model(TOKEN 流式通道)。"""
    agent = _make_text_agent(["a"])
    sentinel = object()
    config = {"callbacks": [sentinel]}

    async def _run():
        _ = [c async for c in agent.astream("任务", config)]

    asyncio.run(_run())
    assert agent.llm.model.received_config == {"callbacks": [sentinel]}


def test_team_agent_astream_with_tools_yields_token_chunks():
    """工具模式 astream:经 astream_events 过滤 on_chat_model_stream 产出 token。"""
    agent = _make_text_agent([])
    agent.agent_executor = FakeAgentExecutor(["工", "具"])
    agent.max_iterations = 10

    async def _run():
        return [c async for c in agent.astream("任务")]

    assert asyncio.run(_run()) == ["工", "具"]


def test_team_agent_astream_with_tools_workspace_and_callbacks():
    """工具模式 astream:提取 workspace_path 并透传 callbacks,thread_id 不透传。"""
    agent = _make_text_agent([])
    executor = FakeAgentExecutor(["x"])
    agent.agent_executor = executor
    agent.max_iterations = 10
    config = {
        "configurable": {"workspace_path": "C:/ws", "thread_id": "t1"},
        "callbacks": [object()],
    }

    async def _run():
        _ = [c async for c in agent.astream("任务", config)]

    asyncio.run(_run())
    assert executor.received_config["recursion_limit"] == 10
    assert executor.received_config["configurable"] == {"workspace_path": "C:/ws"}
    assert "callbacks" in executor.received_config
    assert executor.received_config["configurable"].get("thread_id") is None


def test_team_agent_astream_error_yields_error_message():
    """LLM 流式失败时 yield 错误信息(与同步版行为一致)。"""

    class BoomModel(FakeChatModel):
        async def astream(self, messages, config=None):
            if False:  # 保持 async generator 语义(遍历时才执行函数体)
                yield
            raise RuntimeError("模型挂了")

    agent = _make_text_agent([])
    agent.llm.model = BoomModel([])
    agent.verbose = False

    async def _run():
        return [c async for c in agent.astream("任务")]

    assert asyncio.run(_run()) == ["任务执行失败: 模型挂了"]


# ==================== 测试 SkillInjector 技能注入 ====================

def test_skill_injector_no_skills_dir(tmp_path):
    """技能目录不存在时注入器正常降级(不抛异常)"""
    from skmng.injector import SkillInjector

    injector = SkillInjector(skills_dir=str(tmp_path / "nonexistent"), auto_match=True)
    block = injector.build_skill_block("随便什么任务")
    assert block == ""

    prompt = injector.inject_into_prompt("原提示词", "任务")
    assert prompt == "原提示词"


def test_skill_injector_injects_matched_skill(tmp_path):
    """任务匹配到技能时注入技能指引块"""
    from skmng.injector import SkillInjector

    skill_dir = tmp_path / "skills"
    skill_md = skill_dir / "git" / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text(
        "---\nname: git-commit\ndescription: commit git 提交 版本控制\n---\n# git 提交指引\n按规范提交",
        encoding="utf-8",
    )

    injector = SkillInjector(skills_dir=str(skill_dir), auto_match=True)
    # 任务包含 git/提交 关键词,应命中 git-commit 技能
    block = injector.build_skill_block("请帮我提交代码")
    assert "git-commit" in block

    prompt = injector.inject_into_prompt("原提示词", "请帮我提交代码")
    assert prompt.startswith("原提示词")
    assert "git-commit" in prompt


def test_skill_injector_auto_match_disabled(tmp_path):
    """auto_match=False 时不自动注入技能块"""
    from skmng.injector import SkillInjector

    skill_dir = tmp_path / "skills"
    skill_md = skill_dir / "git" / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text(
        "---\nname: git-commit\ndescription: commit git\n---\n# git 提交指引",
        encoding="utf-8",
    )

    injector = SkillInjector(skills_dir=str(skill_dir), auto_match=False)
    block = injector.build_skill_block("请帮我提交代码")
    assert block == ""

    prompt = injector.inject_into_prompt("原提示词", "请帮我提交代码")
    assert prompt == "原提示词"


def test_skill_injector_skip_when_already_injected(tmp_path):
    """已注入过技能块时不重复注入"""
    from skmng.injector import SkillInjector

    skill_dir = tmp_path / "skills"
    skill_md = skill_dir / "git" / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text(
        "---\nname: git-commit\ndescription: commit git\n---\n# git 提交指引",
        encoding="utf-8",
    )

    injector = SkillInjector(skills_dir=str(skill_dir), auto_match=True)
    prompt = "原提示词\n\n【已加载的技能指引(请在处理任务时遵循)】\nxxx"
    result = injector.inject_into_prompt(prompt, "请帮我提交代码")
    assert result == prompt


# ==================== 测试跨轮次记忆压缩 ====================

def test_workflow_cross_round_compression():
    """同一 thread_id + checkpointer 多轮运行时,第二轮注入上一轮工作流记录"""
    from langgraph.checkpoint.memory import MemorySaver

    manager = FakeAgent(name="manager", response="计划: A、B、C", summary_response="摘要")
    worker = FakeAgent(name="worker", response="执行结果: 完成")
    terminator = FakeAgent(name="terminator", response="最终答案: 全部完成")

    checkpointer = MemorySaver()
    graph = build_simple_workflow(
        {"manager": manager, "worker": worker, "terminator": terminator},
        checkpointer=checkpointer,
    )

    # 第一轮:无历史,不注入
    result1 = asyncio.run(arun_simple_workflow(graph, "第一轮任务", thread_id="wf-t1"))
    assert result1["final_answer"] == "最终答案: 全部完成"
    # summarize 节点第一轮 raw_context 为空 → 短路不调 helper;
    # manager 只剩 manager_plan 一次调用(经 arun_structured)
    assert len(manager.calls) == 1
    assert manager.calls[0][0] == "arun_structured"

    # 第二轮:同一 thread_id,上一轮状态应注入 raw_context
    manager.calls.clear()
    worker.calls.clear()
    terminator.calls.clear()
    manager.response = "计划2"
    worker.response = "结果2"
    terminator.response = "答案2"

    result2 = asyncio.run(arun_simple_workflow(graph, "第二轮任务", thread_id="wf-t1"))
    assert result2["final_answer"] == "答案2"

    # summarize 节点收到上一轮工作流记录(经 arun_structured)
    assert manager.calls[0][0] == "arun_structured"
    assert "【上一轮工作流记录】" in manager.calls[0][1]
    assert "第一轮任务" in manager.calls[0][1]
    assert "最终答案: 全部完成" in manager.calls[0][1]


def test_workflow_cross_round_compression_truncation():
    """跨轮次记录超长时截断,不撑爆上下文"""
    from langgraph.checkpoint.memory import MemorySaver

    manager = FakeAgent(name="manager", response="计划", summary_response="摘要")
    worker = FakeAgent(name="worker", response="结果")
    terminator = FakeAgent(name="terminator", response="答案")

    checkpointer = MemorySaver()
    graph = build_simple_workflow(
        {"manager": manager, "worker": worker, "terminator": terminator},
        checkpointer=checkpointer,
    )

    asyncio.run(arun_simple_workflow(graph, "长内容任务 " + "X" * 500, thread_id="wf-t2"))

    manager.calls.clear()
    result2 = asyncio.run(
        arun_simple_workflow(graph, "第二轮", thread_id="wf-t2", max_history_chars=100)
    )
    assert result2["final_answer"] == "答案"
    summary_text = manager.calls[0][1]
    assert "已截断" in summary_text
    assert len(summary_text) < 300


def test_workflow_cross_round_no_checkpointer():
    """无 checkpointer 时跨轮次压缩静默降级,不影响运行"""
    manager = FakeAgent(name="manager", response="计划", summary_response="摘要")
    worker = FakeAgent(name="worker", response="结果")
    terminator = FakeAgent(name="terminator", response="答案")

    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})

    result = asyncio.run(arun_simple_workflow(graph, "任务", thread_id="wf-t3"))
    assert result["final_answer"] == "答案"
    # 无 checkpointer → 无历史,summarize 节点 raw 为空 → 短路不调 helper;
    # manager.calls[0] 是 manager_plan 节点调用,prompt 含 manager_plan 模板内容
    assert manager.calls[0][0] == "arun_structured"
    assert "请为以下任务制定详细的执行计划" in manager.calls[0][1]


# ==================== 节点产出提取（NODE_END content） ====================


def _make_tracking_handler():
    """构造 NodeTrackingHandler：收集 NODE_END 事件。"""
    from langchain_core.messages import AIMessage

    from graph.common import NodeTrackingHandler
    from utils.events import EventType

    return NodeTrackingHandler, AIMessage, EventType


def test_node_end_carries_node_output():
    """on_chain_end 从 output.messages 提取节点产出 → NODE_END.content"""
    NodeTrackingHandler, AIMessage, EventType = _make_tracking_handler()
    events: list = []
    handler = NodeTrackingHandler({"node_a"}, on_node_end=events.append)
    handler.on_chain_start({}, {}, run_id="r1", metadata={"langgraph_node": "node_a"})
    handler.on_chain_end(
        {"messages": [AIMessage(content="节点产出内容")]}, run_id="r1"
    )
    assert len(events) == 1
    assert events[0].event_type == EventType.NODE_END
    assert events[0].node == "node_a"
    assert events[0].content == "节点产出内容"


def test_node_end_content_empty_without_messages():
    """output 无 messages 通道时 content 为空串（静默降级，不阻塞事件流）"""
    NodeTrackingHandler, _, _ = _make_tracking_handler()
    events: list = []
    handler = NodeTrackingHandler({"node_a"}, on_node_end=events.append)
    handler.on_chain_start({}, {}, run_id="r1", metadata={"langgraph_node": "node_a"})
    handler.on_chain_end({}, run_id="r1")
    assert len(events) == 1
    assert events[0].content == ""


def test_node_end_joins_text_content_blocks():
    """content blocks（list[dict]）仅拼接 text 类型块，忽略 tool_use 等"""
    NodeTrackingHandler, AIMessage, _ = _make_tracking_handler()
    events: list = []
    handler = NodeTrackingHandler({"node_a"}, on_node_end=events.append)
    handler.on_chain_start({}, {}, run_id="r1", metadata={"langgraph_node": "node_a"})
    handler.on_chain_end(
        {
            "messages": [
                AIMessage(
                    content=[
                        {"type": "text", "text": "块A"},
                        {"type": "tool_use", "text": "忽略"},
                        {"type": "text", "text": "块B"},
                    ]
                )
            ]
        },
        run_id="r1",
    )
    assert events[0].content == "块A块B"


def test_node_end_ignores_non_aimessage_output():
    """messages 最后一条非 AIMessage（如 HumanMessage）不算节点产出"""
    NodeTrackingHandler, _, _ = _make_tracking_handler()
    from langchain_core.messages import HumanMessage

    events: list = []
    handler = NodeTrackingHandler({"node_a"}, on_node_end=events.append)
    handler.on_chain_start({}, {}, run_id="r1", metadata={"langgraph_node": "node_a"})
    handler.on_chain_end({"messages": [HumanMessage(content="用户输入")]}, run_id="r1")
    assert events[0].content == ""


def test_node_end_output_not_dict_is_empty():
    """output 非 dict（None/标量）时 content 为空串"""
    NodeTrackingHandler, _, _ = _make_tracking_handler()
    events: list = []
    handler = NodeTrackingHandler({"node_a"}, on_node_end=events.append)
    handler.on_chain_start({}, {}, run_id="r1", metadata={"langgraph_node": "node_a"})
    handler.on_chain_end(None, run_id="r1")
    assert events[0].content == ""
