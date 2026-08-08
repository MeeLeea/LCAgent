"""
任务执行桥接 - 接收数据库任务，调用 SessionManager.arun() 或工作流执行

职责（逻辑 B 的执行端）：
    scheduler 拿到到期任务 → executor 判断任务类型 → 调用 Session 或工作流 → 返回结果

设计要点：
    - 通过 agent_factory（Callable[[], AgentCore]）解耦 Agent 的创建方式，
      调度器不需要知道 LLM / MCP / 技能等初始化细节
    - 执行通过 SessionManager.arun()（异步），调度器用 asyncio.run 在线程池中调用
    - 捕获所有异常，返回 (success, output/error) 元组，绝不向调度器抛异常
    - 支持 workflow: 前缀: task_text 以 "workflow:" 开头时路由到多 Agent 工作流
"""
import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

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


def _execute_workflow_task(task_id: Any, task_text: str) -> tuple[bool, str]:
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
        from graph.registry import list_workflows, run_workflow_by_name
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


class _AgentCreateError(Exception):
    """Agent 创建阶段失败的标记异常（与执行异常区分，返回不同错误提示）。"""


async def _run_agent_task(task_text: str, agent_factory: AgentFactory) -> str:
    """在单个事件循环内创建 Agent、通过 SessionManager 执行任务并释放资源。

    关键约束：AsyncSqliteSaver 绑定创建它的事件循环，跨 loop 使用会挂起。
    因此 factory 构造（await 若返回协程）、session_manager.arun 执行、
    aclose 关闭必须在同一个 loop 内完成。本函数由 execute_task 用 asyncio.run 包裹执行。

    Args:
        task_text: 任务描述文本
        agent_factory: 返回 Agent 实例的可调用对象；返回协程时自动 await

    Returns:
        执行结果字符串（可能为空）

    Raises:
        _AgentCreateError: factory 构造失败（含 async factory 抛异常）
    """
    try:
        agent = agent_factory()
        if inspect.isawaitable(agent):
            agent = await agent
    except Exception as exc:
        raise _AgentCreateError(str(exc)) from exc

    try:
        result = await agent.session_manager.arun(task_text)
        return str(result) if result else ""
    finally:
        # 优先通过 SessionManager 关闭（刷新记忆 buffer + 释放 Agent 资源），
        # 回退到 agent.aclose（兼容未初始化 SessionManager 的场景）
        sm = getattr(agent, "_session_manager", None)
        if sm is not None:
            try:
                aclose = getattr(sm, "aclose", None)
                if callable(aclose):
                    close_result = aclose()
                    if inspect.isawaitable(close_result):
                        await close_result
            except Exception as exc:
                logger.warning("SessionManager 关闭失败: %s", exc, exc_info=True)
        else:
            aclose = getattr(agent, "aclose", None)
            if callable(aclose):
                try:
                    close_result = aclose()
                    if inspect.isawaitable(close_result):
                        await close_result
                except Exception as exc:
                    logger.warning("Agent 关闭失败: %s", exc, exc_info=True)
        # 关闭 MemoryContext（释放 SQLite 连接等底层资源）
        mem_ctx = getattr(agent, "_memory_context", None)
        if mem_ctx is not None:
            try:
                await mem_ctx.aclose()
            except Exception as exc:
                logger.warning("MemoryContext 关闭失败: %s", exc, exc_info=True)


def execute_task(task: dict[str, Any], agent_factory: AgentFactory) -> tuple[bool, str]:
    """执行单条定时任务。

    根据 task_text 自动判断执行路径：
    - 以 "workflow:" 开头 → 多 Agent 工作流执行
    - 其他 → SessionManager.arun() 执行

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
        result = asyncio.run(_run_agent_task(task_text, agent_factory))
        output = str(result) if result else "(Agent 未返回内容)"
        logger.info("任务 #%d 执行完成", task_id)
        return True, output
    except _AgentCreateError as exc:
        return False, f"创建 Agent 实例失败: {exc}"
    except Exception as exc:
        return False, f"Agent 执行异常: {exc}"
