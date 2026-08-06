"""
测试多 Agent 工作流功能
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from cli.commands.workflow import workflow_command
from graph.simple import (
    build_simple_workflow,
    manager_plan_node,
    run_simple_workflow,
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
    """模拟 TeamAgent,不联网"""
    name: str = "test-agent"
    response: str = "fake response"
    summary_response: str = "记忆摘要: 用户偏好中文"
    prompt_file: str | None = None
    default_templates: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TEMPLATES))
    calls: list[tuple[str, str]] = field(default_factory=list)
    
    def invoke(self, task: str) -> str:
        """记录调用并返回模拟结果"""
        self.calls.append(("invoke", task))
        return self.response

    def summarize_context(self, memory_text: str) -> str:
        """记录记忆提炼调用并返回模拟摘要"""
        self.calls.append(("summarize", memory_text))
        return self.summary_response

    def get_template(self, name: str) -> str:
        """懒加载模板:优先读 AGENT.md 小节,缺失回退默认模板"""
        return TeamAgent.load_workflow_template(
            self.prompt_file, name, self.default_templates.get(name, "")
        )

    def render_template(self, template: str, **kwargs) -> str:
        """占位符替换"""
        return TeamAgent.render_template(template, **kwargs)


# ==================== 测试工作流节点 ====================

def test_summarize_context_node():
    """测试 summarize 节点:Manager 提炼记忆上下文摘要"""
    manager = FakeAgent(name="manager", summary_response="记忆摘要: 项目背景A")
    state = {"raw_context": "用户: 之前聊过项目A"}

    result = summarize_context(state, manager)

    assert result["context_summary"] == "记忆摘要: 项目背景A"
    assert len(manager.calls) == 1
    assert manager.calls[0][0] == "summarize"
    assert manager.calls[0][1] == "用户: 之前聊过项目A"


def test_manager_plan_node():
    """测试 Manager 节点:拆解任务"""
    manager = FakeAgent(name="manager", response="计划: 步骤1、步骤2")
    state = {"task": "帮我分析项目结构"}
    
    result = manager_plan_node(state, manager)
    
    assert result["plan"] == "计划: 步骤1、步骤2"
    assert len(manager.calls) == 1
    assert manager.calls[0][0] == "invoke"
    assert "请为以下任务制定详细的执行计划" in manager.calls[0][1]


def test_worker_exec_node():
    """测试 Worker 节点:执行计划"""
    worker = FakeAgent(name="worker", response="已完成步骤1和步骤2")
    state = {"plan": "计划: 步骤1、步骤2"}
    
    result = worker_exec_node(state, worker)
    
    assert result["worker_result"] == "已完成步骤1和步骤2"
    assert len(worker.calls) == 1
    assert worker.calls[0][0] == "invoke"
    assert "请执行以下计划" in worker.calls[0][1]


def test_terminator_final_node():
    """测试 Terminator 节点:汇总结果"""
    terminator = FakeAgent(name="terminator", response="最终答案: 项目包含3个模块")
    state = {
        "task": "帮我分析项目结构",
        "plan": "计划: 步骤1、步骤2",
        "worker_result": "已完成步骤1和步骤2"
    }
    
    result = terminator_final_node(state, terminator)
    
    assert result["final_answer"] == "最终答案: 项目包含3个模块"
    assert len(terminator.calls) == 1
    assert terminator.calls[0][0] == "invoke"
    prompt = terminator.calls[0][1]
    assert "原始任务" in prompt
    assert "执行计划" in prompt
    assert "执行结果" in prompt


def test_manager_plan_node_with_summary():
    """测试 Manager 节点:有记忆摘要时注入摘要块"""
    manager = FakeAgent(name="manager", response="计划")
    state = {"task": "任务", "context_summary": "用户偏好中文"}

    result = manager_plan_node(state, manager)

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

    result = terminator_final_node(state, terminator)

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

    terminator_final_node(state, terminator)

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


def test_load_workflow_template(tmp_path):
    """测试从 AGENT.md 加载工作流模板,小节缺失/文件缺失回退默认"""
    prompt_file = tmp_path / "AGENT.md"
    prompt_file.write_text("## workflow:worker_exec\n\n请执行计划:\n{plan}", encoding="utf-8")

    assert TeamAgent.load_workflow_template(str(prompt_file), "worker_exec", "默认") == "请执行计划:\n{plan}"
    assert TeamAgent.load_workflow_template(str(prompt_file), "manager_plan", "默认计划") == "默认计划"
    assert TeamAgent.load_workflow_template(None, "worker_exec", "默认") == "默认"


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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    graph.invoke({
        "task": "任务T",
        "raw_context": "",
        "context_summary": "摘要S",
        "plan": "",
        "worker_result": "",
        "final_answer": "",
    })

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
    result = run_simple_workflow(graph, "测试任务")
    
    # 验证最终状态
    assert result["task"] == "测试任务"
    assert result["plan"] == "计划: A、B、C"
    assert result["worker_result"] == "执行结果: 完成A、B、C"
    assert result["final_answer"] == "最终答案: 全部完成"
    
    # 验证三个 Agent 都被调用(manager 承担 summarize + plan 两次调用)
    assert len(manager.calls) == 2
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


def test_builtin_agents_registered():
    """测试三个内置角色(manager/worker/terminator)已通过装饰器注册"""
    import team  # noqa: F401 - 触发各 agent 模块加载,完成注册
    from graph.registry import AGENT_REGISTRY
    from team.base import TeamAgent
    
    for role in ("manager", "worker", "terminator"):
        assert role in AGENT_REGISTRY
        spec = AGENT_REGISTRY[role]
        assert issubclass(spec["agent_class"], TeamAgent)
        assert spec["config_file"].startswith(f"team/{role}/")
    
    # Worker 注入全部本地工具,Manager/Terminator 纯文本模式
    assert AGENT_REGISTRY["worker"]["tools"] is not None
    assert len(AGENT_REGISTRY["worker"]["tools"]) > 0
    assert AGENT_REGISTRY["manager"]["tools"] is None
    assert AGENT_REGISTRY["terminator"]["tools"] is None


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
    result1 = run_simple_workflow(graph, "任务1")
    
    # 第二次运行(重置 agent 调用记录)
    manager.calls.clear()
    worker.calls.clear()
    terminator.calls.clear()
    manager.response = "计划2"
    worker.response = "结果2"
    terminator.response = "答案2"
    
    result2 = run_simple_workflow(graph, "任务2")
    
    # 验证两次运行结果独立
    assert result1["task"] == "任务1"
    assert result1["final_answer"] == "答案1"
    
    assert result2["task"] == "任务2"
    assert result2["final_answer"] == "答案2"
    
    # 验证两次都调用了 agent(manager 为 summarize + plan 两次调用)
    assert len(manager.calls) == 2
    assert len(worker.calls) == 1
    assert len(terminator.calls) == 1


# ==================== 测试记忆注入与写回 ====================

def test_run_simple_workflow_with_memory():
    """测试带记忆运行时:摘要注入 plan/final 节点,worker 不注入"""
    manager = FakeAgent(name="manager", response="计划: A、B、C", summary_response="记忆摘要: 用户偏好中文")
    worker = FakeAgent(name="worker", response="执行结果: 完成")
    terminator = FakeAgent(name="terminator", response="最终答案: 完成")
    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})

    result = run_simple_workflow(graph, "测试任务", raw_context="用户: 之前聊过偏好中文")

    # manager 两次调用:summarize + plan
    assert manager.calls[0] == ("summarize", "用户: 之前聊过偏好中文")
    assert manager.calls[1][0] == "invoke"
    assert "记忆摘要: 用户偏好中文" in manager.calls[1][1]
    # worker 只拿 plan,不注入摘要
    assert worker.calls[0][0] == "invoke"
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


def test_build_memory_context():
    """测试记忆文本提取:短期+长期拼装"""
    from cli.commands.workflow import build_memory_context
    memory = FakeMemory(
        short_term=[
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好!"},
        ],
        long_term=[{"role": "system", "content": "用户偏好中文"}],
    )
    text = build_memory_context(memory)
    assert "【当前会话】" in text
    assert "user: 你好" in text
    assert "assistant: 你好!" in text
    assert "【长期记忆】" in text
    assert "system: 用户偏好中文" in text


def test_build_memory_context_empty():
    """测试无记忆时返回空串"""
    from cli.commands.workflow import build_memory_context
    assert build_memory_context(FakeMemory()) == ""


def test_build_memory_context_truncation(monkeypatch):
    """测试记忆文本超长截断"""
    import cli.commands.workflow as wf_module
    from cli.commands.workflow import build_memory_context
    monkeypatch.setattr(wf_module, "MAX_RAW_CONTEXT_CHARS", 20)
    memory = FakeMemory(short_term=[{"role": "user", "content": "这是一段很长很长的内容" * 10}])
    text = build_memory_context(memory)
    assert "已截断" in text


class FakeExecutor:
    """记录 update_state 调用的假 executor"""
    def __init__(self):
        self.updated = []

    def update_state(self, config, values):
        self.updated.append((config, values))


class FakeAgentCore:
    """模拟 AgentCore 的 executor + memory"""
    def __init__(self):
        self.agent_executor = FakeExecutor()
        self.memory = FakeMemory()


class WriteBackContext:
    """携带 agent 的假上下文(供写回测试)"""
    def __init__(self, agent=None):
        self.agent = agent or FakeAgentCore()
        self.output: list[str] = []
        # 与 CommandContext 对齐:工作流事件回调默认为空
        self.workflow_event_cb = None

    def print(self, text: str) -> None:
        self.output.append(text)


def test_record_workflow_result():
    """测试写回:任务与最终答案写入当前会话 checkpoint"""
    from cli.commands.workflow import _record_workflow_result
    core = FakeAgentCore()
    core.memory.get_config = lambda: {"configurable": {"thread_id": "t1"}}
    ctx = WriteBackContext(core)

    _record_workflow_result(ctx, "simple", "测试任务", {"final_answer": "最终答案"})

    assert len(core.agent_executor.updated) == 1
    config, values = core.agent_executor.updated[0]
    assert config == {"configurable": {"thread_id": "t1"}}
    msgs = values["messages"]
    assert msgs[0].content == "workflow:simple 测试任务"
    assert msgs[1].content == "最终答案"


def test_record_workflow_result_empty_answer():
    """测试无最终答案时不写回"""
    from cli.commands.workflow import _record_workflow_result
    core = FakeAgentCore()
    ctx = WriteBackContext(core)

    _record_workflow_result(ctx, "simple", "测试任务", {"final_answer": ""})

    assert core.agent_executor.updated == []


def test_record_workflow_result_no_executor():
    """测试没有 executor 时不写回也不报错"""
    from cli.commands.workflow import _record_workflow_result
    core = FakeAgentCore()
    core.agent_executor = None
    ctx = WriteBackContext(core)

    _record_workflow_result(ctx, "simple", "测试任务", {"final_answer": "答案"})
    # 不应抛异常


def test_run_workflow_injects_memory(monkeypatch):
    """测试 run_workflow:提取记忆传入图并写回会话"""
    import cli.commands.workflow as wf_module

    class Mem:
        def get_short_term(self, limit=None):
            return [{"role": "user", "content": "之前聊过X"}]

        async def aget_short_term(self, limit=None):
            return [{"role": "user", "content": "之前聊过X"}]

        def get_long_term(self, limit=5):
            return []

        def get_config(self):
            return {"configurable": {"thread_id": "t1"}}

    core = FakeAgentCore()
    core.memory = Mem()
    ctx = WriteBackContext(core)

    captured = {}

    def fake_build(name):
        return ("graph", {})

    def fake_run(graph, task, raw_context="", on_node_start=None, on_node_end=None):
        captured["raw_context"] = raw_context
        return {"final_answer": "答案"}

    monkeypatch.setattr("graph.registry.build_workflow", fake_build)
    # 注册表捕获的是原始函数对象,monkeypatch 模块属性无法穿透,需直接 patch 注册表 runner
    from graph.registry import WORKFLOWS

    monkeypatch.setitem(WORKFLOWS["simple"], "runner", fake_run)

    result = asyncio.run(wf_module.run_workflow(ctx, "simple", "测试任务"))

    assert result["final_answer"] == "答案"
    assert "user: 之前聊过X" in captured["raw_context"]
    assert len(core.agent_executor.updated) == 1


# ==================== 测试运行进度跟踪 ====================

def test_run_simple_workflow_node_callbacks():
    """测试节点进度回调:4 个业务节点按链路顺序各触发 start/end 且 start 先于 end"""
    manager = FakeAgent(name="manager", response="计划: A")
    worker = FakeAgent(name="worker", response="结果: 完成")
    terminator = FakeAgent(name="terminator", response="答案: 完成")
    graph = build_simple_workflow({"manager": manager, "worker": worker, "terminator": terminator})

    events: list[tuple[str, str]] = []
    run_simple_workflow(
        graph,
        "测试任务",
        on_node_start=lambda n: events.append(("start", n)),
        on_node_end=lambda n: events.append(("end", n)),
    )

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

    result = run_simple_workflow(graph, "测试任务")
    assert result["final_answer"] == "答案"


def test_run_workflow_emits_workflow_events(monkeypatch):
    """测试 run_workflow:节点/整体状态通过 workflow_event_cb 转发结构化事件"""
    import cli.commands.workflow as wf_module
    from cli.commands.types import CommandContext

    events: list[dict[str, str]] = []

    def fake_build(name):
        return ("graph", {"manager": object(), "worker": object(), "terminator": object()})

    def fake_run(graph, task, raw_context="", on_node_start=None, on_node_end=None):
        # 模拟 4 个业务节点依次执行:每个节点 start → end
        for node in ("summarize", "manager_plan", "worker_exec", "terminator_final"):
            if on_node_start:
                on_node_start(node)
            if on_node_end:
                on_node_end(node)
        return {"final_answer": "答案"}

    monkeypatch.setattr("graph.registry.build_workflow", fake_build)
    # 注册表捕获的是原始函数对象,monkeypatch 模块属性无法穿透,需直接 patch 注册表 runner
    from graph.registry import WORKFLOWS

    monkeypatch.setitem(WORKFLOWS["simple"], "runner", fake_run)

    class Mem:
        def get_short_term(self, limit=None):
            return []

        async def aget_short_term(self, limit=None):
            return []

        def get_long_term(self, limit=5):
            return []

        def get_config(self):
            return {"configurable": {"thread_id": "t1"}}

    core = FakeAgentCore()
    core.memory = Mem()
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
