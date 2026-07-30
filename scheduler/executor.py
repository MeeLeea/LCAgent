# -*- coding: utf-8 -*-
"""
任务执行桥接 - 接收数据库任务，调用 AgentCore.run() 执行

职责（逻辑 B 的执行端）：
    scheduler 拿到到期任务 → executor 调用 Agent → 返回结果

设计要点：
    - 通过 agent_factory（Callable[[], AgentCore]）解耦 Agent 的创建方式，
      调度器不需要知道 LLM / MCP / 技能等初始化细节
    - 执行是同步阻塞的（AgentCore.run 本身是同步的），调度器在线程池中调用
    - 捕获所有异常，返回 (success, output/error) 元组，绝不向调度器抛异常
"""
from typing import Any, Callable, Dict, Tuple

AgentFactory = Callable[[], Any]  # 返回具备 .run(task_text) -> str 接口的对象


def execute_task(task: Dict[str, Any], agent_factory: AgentFactory) -> Tuple[bool, str]:
    """
    执行单条定时任务。

    Args:
        task:          TaskStore 返回的任务字典（含 task_text 等字段）
        agent_factory: 返回 Agent 实例的可调用对象（每次执行调用一次）

    Returns:
        (success, output):
            success=True  → output 为 Agent 执行结果文本
            success=False → output 为错误信息
    """
    task_id = task.get("id", "?")
    task_text = task.get("task_text", "")
    task_type = task.get("task_type", "one_time")

    if not task_text:
        return False, "任务内容为空，无法执行"

    print(f"\n[Scheduler] 开始执行任务 #{task_id} ({task_type}): {task_text[:80]}")

    try:
        agent = agent_factory()
    except Exception as exc:
        return False, f"创建 Agent 实例失败: {exc}"

    try:
        # AgentCore.run() 返回执行结果字符串
        result = agent.run(task_text)
        output = str(result) if result else "(Agent 未返回内容)"
        print(f"[Scheduler] 任务 #{task_id} 执行完成")
        return True, output
    except Exception as exc:
        return False, f"Agent 执行异常: {exc}"