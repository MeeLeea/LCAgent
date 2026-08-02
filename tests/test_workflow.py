"""
测试多 Agent 工作流功能
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cli.commands.workflow import workflow_command
from graph.simple import (
    build_simple_workflow,
    manager_plan_node,
    run_simple_workflow,
    terminator_final_node,
    worker_exec_node,
)


@dataclass
class FakeAgent:
    """模拟 TeamAgent,不联网"""
    name: str = "test-agent"
    response: str = "fake response"
    calls: list[tuple[str, str]] = field(default_factory=list)
    
    def invoke(self, task: str) -> str:
        """记录调用并返回模拟结果"""
        self.calls.append(("invoke", task))
        return self.response


# ==================== 测试工作流节点 ====================

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
    
    # 验证边: START → manager_plan → worker_exec → terminator_final → END
    edges = [(e.source, e.target) for e in graph_obj.edges]
    
    assert ("__start__", "manager_plan") in edges
    assert ("manager_plan", "worker_exec") in edges
    assert ("worker_exec", "terminator_final") in edges
    assert ("terminator_final", "__end__") in edges


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
    
    # 验证三个 Agent 都被调用
    assert len(manager.calls) == 1
    assert len(worker.calls) == 1
    assert len(terminator.calls) == 1


# ==================== 测试 main.py 工作流接口 ====================

def test_workflow_registry():
    """测试 WORKFLOWS 注册表"""
    from graph.registry import WORKFLOWS
    
    assert "simple" in WORKFLOWS
    assert callable(WORKFLOWS["simple"])


def test_build_workflow_unknown_name():
    """测试未知工作流名称抛出 KeyError"""
    from graph.registry import build_workflow
    
    with pytest.raises(KeyError) as exc_info:
        build_workflow("unknown_workflow")
    
    assert "未知工作流: unknown_workflow" in str(exc_info.value)
    assert "可用工作流: simple" in str(exc_info.value)


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
    outcome = workflow_command(context, "workflow")
    
    assert outcome.handled
    output_text = "\n".join(context.output)
    assert "可用工作流" in output_text
    assert "simple" in output_text
    assert "用法: workflow:<name> <task>" in output_text


def test_workflow_execute_command_missing_task():
    """测试缺少任务描述时的错误提示"""
    context = FakeContext()
    outcome = workflow_command(context, "workflow:simple")
    
    assert outcome.handled
    output_text = "\n".join(context.output)
    assert "错误: 缺少任务描述" in output_text


def test_workflow_execute_command_unknown_workflow():
    """测试未知工作流的错误提示"""
    context = FakeContext()
    outcome = workflow_command(context, "workflow:unknown 测试任务")
    
    assert outcome.handled
    output_text = "\n".join(context.output)
    assert "错误: 未知工作流 'unknown'" in output_text
    assert "可用工作流: simple" in output_text


def test_workflow_command_parse():
    """测试命令解析:workflow:simple 测试任务"""
    context = FakeContext()
    
    # Mock run_workflow 避免实际执行
    import cli.commands.workflow as wf_module
    original_run = None
    
    def mock_run(context, name: str, task: str):
        return {"final_answer": f"Mock 结果: {name} / {task}"}
    
    # 暂存原函数(如果存在)
    if hasattr(wf_module, "run_workflow"):
        original_run = wf_module.run_workflow
    
    # 临时替换
    wf_module.run_workflow = mock_run
    
    try:
        outcome = workflow_command(context, "workflow:simple 帮我分析项目")
        
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
    
    # 验证两次都调用了 agent
    assert len(manager.calls) == 1
    assert len(worker.calls) == 1
    assert len(terminator.calls) == 1
