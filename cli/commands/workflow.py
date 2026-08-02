"""
Workflow 命令处理 - 多 Agent 工作流执行
"""
from __future__ import annotations

from .types import HANDLED, CommandContext, CommandOutcome


def run_workflow(context: CommandContext, name: str, task: str) -> dict:
    """
    运行指定工作流
    
    Args:
        context: 命令上下文
        name: 工作流名称
        task: 用户任务
        
    Returns:
        工作流执行结果字典(包含 final_answer)
    """
    # 函数内延迟导入,避免循环依赖
    from graph.registry import build_workflow
    from graph.simple import run_simple_workflow
    
    context.print(f"\n构建工作流: {name}")
    
    graph, agents = build_workflow(name)
    
    # 从构建结果动态获取角色名,避免写死
    role_names = "、".join(agents.keys())
    context.print(f"初始化团队 Agent({role_names})...")
    context.print(f"工作流 {name} 构建完成")
    context.print(f"\n执行任务: {task}")
    context.print("-" * 50)
    
    result = run_simple_workflow(graph, task)
    return result


def workflow_command(context: CommandContext, user_input: str) -> CommandOutcome:
    """
    处理 workflow 相关命令
    
    支持命令:
        workflow              - 列出可用工作流
        workflow:<name> <task> - 运行指定工作流
    """
    # 导入放在函数内避免循环依赖
    from graph.registry import WORKFLOWS
    
    if user_input.strip() == "workflow":
        # 列出可用工作流
        context.print("\n可用工作流:")
        for name in WORKFLOWS.keys():
            context.print(f"  - {name}")
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
            available = ", ".join(WORKFLOWS.keys())
            context.print(f"\n错误: 未知工作流 '{workflow_name}'")
            context.print(f"可用工作流: {available}")
            return HANDLED
        
        # 执行工作流
        try:
            result = run_workflow(context, workflow_name, task)
            
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
