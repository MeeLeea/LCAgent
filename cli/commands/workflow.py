"""
Workflow 命令处理 - 多 Agent 工作流执行
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage

from .types import HANDLED, CommandContext, CommandOutcome

logger = logging.getLogger(__name__)

# 注入工作流的原始记忆文本上限,防止超长上下文撑爆 summarize 的 LLM 调用
MAX_RAW_CONTEXT_CHARS = 6000


async def abuild_memory_context(agent) -> str:
    """从 Agent 提取当前会话短期记忆与长期记忆，拼装为文本并截断。

    短期记忆从 SessionRegistry 读取（checkpoint 消息），长期记忆从 ThreadMemoryStore 读取。

    Args:
        agent: AgentCore 实例（需有 session 和 long_term_memory 属性）

    Returns:
        记忆文本;无记忆时返回空串
    """
    blocks = []

    short_term = await agent.session.aget_short_term() or []
    if short_term:
        lines = [f"{item.get('role', 'user')}: {item.get('content', '')}" for item in short_term]
        blocks.append("【当前会话】\n" + "\n".join(lines))

    facts = await agent.long_term_memory.query_facts(agent.session.current_session_id)
    if facts:
        lines = [f"[{f.category}] {f.content}" for f in facts]
        blocks.append("【长期记忆】\n" + "\n".join(lines))

    text = "\n\n".join(blocks).strip()
    if len(text) > MAX_RAW_CONTEXT_CHARS:
        text = text[:MAX_RAW_CONTEXT_CHARS] + "\n...(记忆文本过长,已截断)"
    return text


def _record_workflow_result(context: CommandContext, name: str, task: str, result: dict) -> None:
    """把工作流任务与最终答案写回当前会话记忆,形成记忆闭环"""
    final_answer = result.get("final_answer", "")
    if not final_answer:
        return
    executor = getattr(context.agent, "agent_executor", None)
    if executor is None:
        return
    try:
        executor.update_state(
            context.agent._invoke_config(),
            {"messages": [
                HumanMessage(content=f"workflow:{name} {task}"),
                AIMessage(content=final_answer),
            ]},
        )
    except Exception as error:
        # 写回失败不影响工作流主流程,记录后跳过
        logger.warning("工作流记忆写回失败: %s", error)


async def run_workflow(context: CommandContext, name: str, task: str) -> dict:
    """
    运行指定工作流 - 异步版本

    Args:
        context: 命令上下文
        name: 工作流名称
        task: 用户任务

    Returns:
        工作流执行结果字典(包含 final_answer)
    """
    # 函数内延迟导入,避免循环依赖
    from graph.registry import build_workflow, get_workflow_runner
    from graph.simple import run_simple_workflow

    context.print(f"\n构建工作流: {name}")

    graph, agents = build_workflow(name)

    # 从构建结果动态获取角色名,避免写死
    role_names = "、".join(agents.keys())
    context.print(f"初始化团队 Agent({role_names})...")
    context.print(f"工作流 {name} 构建完成")
    context.print(f"\n执行任务: {task}")
    context.print("-" * 50)

    # 节点进度跟踪:打印节点状态 + 转发结构化事件(供 SSE 等实时通道更新前端节点高亮)
    def _emit(event: dict[str, str]) -> None:
        if context.workflow_event_cb:
            context.workflow_event_cb(event)

    def _on_node_start(node: str) -> None:
        context.print(f"▸ 节点开始: {node}")
        _emit({"type": "workflow_node", "node": node, "status": "running"})

    def _on_node_end(node: str) -> None:
        context.print(f"✓ 节点完成: {node}")
        _emit({"type": "workflow_node", "node": node, "status": "done"})

    _emit({"type": "workflow_status", "status": "running"})
    try:
        # 获取工作流专用运行器,缺失则回退到通用运行器
        runner = get_workflow_runner(name) or run_simple_workflow
        result = runner(
            graph,
            task,
            raw_context=await abuild_memory_context(context.agent),
            on_node_start=_on_node_start,
            on_node_end=_on_node_end,
        )
    finally:
        # 无论成功失败都复位整体状态,避免前端 UI 停留在"运行中"
        _emit({"type": "workflow_status", "status": "done"})
    _record_workflow_result(context, name, task, result)
    return result


async def workflow_command(context: CommandContext, user_input: str) -> CommandOutcome:
    """
    处理 workflow 相关命令 - 异步版本
    
    支持命令:
        workflow              - 列出可用工作流
        workflow:<name> <task> - 运行指定工作流
    """
    # 导入放在函数内避免循环依赖
    from graph.registry import list_workflows
    
    if user_input.strip() == "workflow":
        # 列出可用工作流(含描述)
        context.print("\n可用工作流:")
        for wf_name, desc in list_workflows():
            if desc:
                context.print(f"  - {wf_name}: {desc}")
            else:
                context.print(f"  - {wf_name}")
        context.print("\n用法: workflow:<name> <task>")
        context.print("示例: workflow:simple 帮我分析一下项目结构")
        return HANDLED
    
    if user_input.startswith("workflow:"):
        # 解析命令: workflow:<name> <task>
        rest = user_input[9:].strip()  # 去掉 "workflow:"
        
        if not rest:
            context.print("\n用法: workflow:<name> <task>")
            context.print("示例: workflow:simple 帮我分析一下项目结构")
            return HANDLED
        
        # 分割工作流名称和任务
        parts = rest.split(None, 1)
        if len(parts) < 2:
            context.print("\n错误: 缺少任务描述")
            context.print("用法: workflow:<name> <task>")
            return HANDLED
        
        workflow_name = parts[0]
        task = parts[1]
        
        # 检查工作流是否存在
        from graph.registry import WORKFLOWS
        if workflow_name not in WORKFLOWS:
            available = ", ".join(name for name, _ in list_workflows())
            context.print(f"\n错误: 未知工作流 '{workflow_name}'")
            context.print(f"可用工作流: {available}")
            return HANDLED
        
        # 执行工作流
        try:
            result = await run_workflow(context, workflow_name, task)
            
            # 输出最终答案
            context.print("\n" + "=" * 50)
            context.print("工作流执行完成")
            context.print("=" * 50)
            context.print(f"\n{result.get('final_answer', '无结果')}")
            
        except Exception as e:
            context.print(f"\n工作流执行失败: {e}")
        
        return HANDLED
    
    # 不应该到这里,但保险起见返回未处理
    return CommandOutcome(handled=False, should_break=False)
