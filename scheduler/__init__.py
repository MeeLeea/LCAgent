"""
定时任务调度子系统 - Agent 理解任务，外部调度盯时间，两者分离

架构:
    逻辑 A（对话阶段）: 用户消息 → Agent 解析意图 → schedule_task 入库 → 回复"已登记"
    逻辑 B（后台调度）: SchedulerEngine 轮询 DB 到期任务 / cron 触发 → AgentCore.run() 执行

模块:
    store       - SQLite 持久化层（CRUD + 原子抢占）
    executor    - 桥接：存储任务 → 调用 AgentCore.run()
    engine      - APScheduler 调度引擎（轮询一次性 + cron 周期）
    run         - 独立调度器进程入口
"""

from .engine import SchedulerEngine
from .executor import execute_task
from .store import TaskStore

__all__ = [
    "SchedulerEngine",
    "TaskStore",
    "execute_task",
]