# -*- coding: utf-8 -*-
"""
任务执行桥接 - 接收数据库任务，调用 AgentCore.run() 或工作流执行

职责（逻辑 B 的执行端）：
    scheduler 拿到到期任务 → executor 判断任务类型 → 调用 Agent 或工作流 → 返回结果

设计要点：
    - 通过 agent_factory（Callable[[], AgentCore]）解耦 Agent 的创建方式，
      调度器不需要知道 LLM / MCP / 技能等初始化细节
    - 执行是同步阻塞的（AgentCore.run 本身是同步的），调度器在线程池中调用
    - 捕获所有异常，返回 (success, output/error) 元组，绝不向调度器抛异常
    - 支持 workflow: 前缀: task_text 以 "workflow:" 开头时路由到多 Agent 工作流
"""
from typing import Any, Callable, Dict, Tuple
import asyncio
import logging

logger = logging.getLogger(__name__)

AgentFactory = Callable[[], Any]  # 返回具备 .run(task_text) -> str 接口的对象

# 工作流任务前缀
_WORKFLOW_PREFIX = "workflow:"


def _is_workflow_task(task_text: str) -> bool:
    """判断 task_text 是否为工作流任务（以 'workflow:' 开头）"""
    return task_text.strip().lower().startswith(_WORKFLOW_PREFIX)


def _parse_workflow_task(task_text: str) -> tuple[str, str]:
    """
    解析工作流任务文本

    格式: workflow:<name> <task>
    例如: workflow:systemc_cmodel 为同步FIFO编写C-Model

    Returns:
        (workflow_name, task) 元组
    """
    rest = task_text.strip()[len(_WORKFLOW_PREFIX):].strip()
    parts = rest.split(None, 1)
    workflow_name = parts[0] if parts else ""
    task = parts[1].strip() if len(parts) > 1 else ""
    return workflow_name, task


def _execute_workflow_task(task_id: Any, task_text: str) -> Tuple[bool, str]:
    """
    执行工作流任务（延迟导入 graph 模块避免循环依赖）

    Args:
        task_id: 任务 ID（用于日志）
        task_text: 原始任务文本（含 workflow: 前缀）

    Returns:
        (success, output) 元组
    """
    workflow_name, task = _parse_workflow_task(task_text)

    if not workflow_name:
        return False, "工作流任务格式错误：缺少工作流名称（格式: workflow:<name> <task>）"

    if not task:
        return False, f"工作流任务格式错误：缺少任务描述（格式: workflow:{workflow_name} <task>）"

    logger.info("任务 #%d → 工作流: %s", task_id, workflow_name)
    logger.info("  任务内容: %s", task[:80])

    try:
        from graph.registry import run_workflow_by_name, list_workflows
    except ImportError as exc:
        return False, f"无法导入工作流模块: {exc}"

    # 检查工作流是否存在
    available = [name for name, _ in list_workflows()]
    if workflow_name not in available:
        return False, f"未知工作流: {workflow_name}。可用: {', '.join(available)}"

    # 节点进度打印（供调度器日志查看执行过程）
    def _on_node_start(node: str) -> None:
        logger.info("  ▸ 节点开始: %s", node)

    def _on_node_end(node: str) -> None:
        logger.info("  ✓ 节点完成: %s", node)

    try:
        result = run_workflow_by_name(
            workflow_name,
            task,
            on_node_start=_on_node_start,
            on_node_end=_on_node_end,
        )
        final_answer = result.get("final_answer", "")
        if not final_answer:
            return False, "工作流执行完成但未返回结果"
        logger.info("  工作流 %s 执行完成", workflow_name)
        return True, final_answer

    except KeyError as exc:
        return False, f"工作流执行失败（角色/工作流未注册）: {exc}"
    except Exception as exc:
        return False, f"工作流执行异常: {exc}"


def execute_task(task: Dict[str, Any], agent_factory: AgentFactory) -> Tuple[bool, str]:
    """
    执行单条定时任务。

    根据 task_text 自动判断执行路径：
    - 以 "workflow:" 开头 → 多 Agent 工作流执行
    - 其他 → AgentCore.run() 执行

    Args:
        task:          TaskStore 返回的任务字典（含 task_text 等字段）
        agent_factory: 返回 Agent 实例的可调用对象（每次执行调用一次）

    Returns:
        (success, output):
            success=True  → output 为执行结果文本
            success=False → output 为错误信息
    """
    task_id = task.get("id", "?")
    task_text = task.get("task_text", "")
    task_type = task.get("task_type", "one_time")

    if not task_text:
        return False, "任务内容为空，无法执行"

    logger.info("开始执行任务 #%d (%s): %s", task_id, task_type, task_text[:80])

    # 工作流任务路由
    if _is_workflow_task(task_text):
        return _execute_workflow_task(task_id, task_text)

    # 普通 Agent 任务
    try:
        agent = agent_factory()
    except Exception as exc:
        return False, f"创建 Agent 实例失败: {exc}"

    try:
        # AgentCore.run() 返回执行结果字符串
        result = asyncio.run(agent.arun(task_text))
        output = str(result) if result else "(Agent 未返回内容)"
        logger.info("任务 #%d 执行完成", task_id)
        return True, output
    except Exception as exc:
        return False, f"Agent 执行异常: {exc}"
