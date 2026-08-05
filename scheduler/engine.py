# -*- coding: utf-8 -*-
"""
调度引擎 - 基于 APScheduler

职责（逻辑 B 的调度端）：
    1. 一次性任务：每隔 N 秒轮询数据库，拾取 pending + 到期的任务 → 执行
    2. 周期任务：直接向 APScheduler 注册 CronTrigger job（不靠轮询比对时间）
    3. 启动时从数据库同步所有 pending 周期任务，重建 cron job（程序重启不丢失）

设计要点：
    - Agent 绝不阻塞等待时间；所有时间判断由 APScheduler 在独立线程完成
    - 任务执行前先 claim_task 原子抢占，防止多调度器实例重复执行
    - 周期任务的 cron job ID = f"periodic_{task_id}"，便于取消/重建
    - 支持 BackgroundScheduler（嵌入主进程后台线程）和 BlockingScheduler（独立进程）
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from typing import Any, Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .executor import execute_task
from .store import TaskStore

logger = logging.getLogger(__name__)


# APScheduler 的 BaseScheduler 类型（两个子类的公共基类）
try:
    from apscheduler.schedulers.base import BaseScheduler  # noqa: F401
except Exception:  # pragma: no cover
    BaseScheduler = None


# 默认轮询间隔（秒）
DEFAULT_POLL_INTERVAL = 30

# 默认并发执行线程数
DEFAULT_MAX_WORKERS = 5


class SchedulerEngine:
    """
    定时任务调度引擎。

    用法::

        engine = SchedulerEngine(task_store, agent_factory)
        engine.start()          # 启动后台调度（非阻塞）
        engine.register_periodic_task(task_dict)  # 运行时注册周期任务
        ...
        engine.stop()

    或作为独立进程阻塞运行::

        engine = SchedulerEngine(task_store, agent_factory, blocking=True)
        engine.start()          # 阻塞直到 Ctrl+C
    """

    def __init__(
        self,
        task_store: TaskStore,
        agent_factory: Callable[[], Any],
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        blocking: bool = False,
        timezone: Optional[str] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ):
        """
        Args:
            task_store:     TaskStore 实例（数据库访问）
            agent_factory:  返回 Agent 实例的可调用对象
            poll_interval:  一次性任务轮询间隔（秒）
            blocking:       True=BlockingScheduler（独立进程），False=BackgroundScheduler（后台线程）
            timezone:       时区字符串（如 "Asia/Shanghai"），None 则用系统默认
            max_workers:    任务并发执行的最大线程数，同一轮到期的多个任务会并发跑
        """
        self.task_store = task_store
        self.agent_factory = agent_factory
        self.poll_interval = poll_interval
        self.blocking = blocking
        self.max_workers = max_workers

        trigger_kwargs = {"timezone": timezone} if timezone else {}
        self._trigger_kwargs = trigger_kwargs

        if blocking:
            self._scheduler = BlockingScheduler(timezone=timezone)
        else:
            self._scheduler = BackgroundScheduler(timezone=timezone)

        self._started = False
        self._lock = threading.Lock()
        # 已注册的周期任务 job_id 集合（避免重复注册）
        self._registered_periodic: set[str] = set()
        # 任务执行线程池：多个到期任务并发执行，避免长任务阻塞后续任务
        self._executor: Optional[ThreadPoolExecutor] = None
        # 跟踪已提交的 future，stop 时可等待或取消
        self._pending_futures: set[Future] = set()

    # ---- 生命周期 ----

    def start(self):
        """启动调度器：创建线程池 + 注册轮询 job + 同步周期任务。"""
        with self._lock:
            if self._started:
                logger.warning("引擎已在运行，跳过重复启动")
                return

            # 0. 创建任务执行线程池
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="task-worker",
            )
            logger.info("任务执行线程池已创建（max_workers=%d）", self.max_workers)

            # 1. 一次性任务轮询 job
            self._scheduler.add_job(
                self._poll_and_execute,
                trigger=IntervalTrigger(seconds=self.poll_interval, **self._trigger_kwargs),
                id="_poll_one_time_tasks",
                replace_existing=True,
                max_instances=1,  # 防止上一轮还没跑完就启动下一轮
                coalesce=True,
            )
            logger.info("已注册一次性任务轮询（间隔 %ds）", self.poll_interval)

            # 2. 同步数据库中的周期任务
            self.sync_periodic_tasks()

            # 3. 启动调度器
            self._scheduler.start()
            self._started = True
            logger.info("调度引擎已启动")

            if self.blocking:
                logger.info("阻塞模式运行中，按 Ctrl+C 停止...")

    def stop(self):
        """停止调度器并关闭线程池。"""
        with self._lock:
            if not self._started:
                return
            self._scheduler.shutdown(wait=False)
            self._started = False
            self._registered_periodic.clear()
            # 关闭线程池：取消尚未开始的任务，等待正在执行的任务完成
            if self._executor is not None:
                for fut in list(self._pending_futures):
                    fut.cancel()
                self._executor.shutdown(wait=True)
                self._executor = None
                self._pending_futures.clear()
            logger.info("调度引擎已停止")

    @property
    def running(self) -> bool:
        return self._started

    # ---- 周期任务管理 ----

    def sync_periodic_tasks(self):
        """
        从数据库同步所有 pending 周期任务到 APScheduler。

        在 start() 时调用，确保程序重启后周期任务不丢失。
        幂等：已注册的 job 不会重复注册。
        """
        periodic_tasks = self.task_store.list_periodic_tasks(status="pending")
        registered = 0
        for task in periodic_tasks:
            if self.register_periodic_task(task):
                registered += 1
        if registered:
            logger.info("从数据库同步了 %d 个周期任务", registered)

    def register_periodic_task(self, task: dict) -> bool:
        """
        向 APScheduler 注册一条周期任务的 cron job。

        Args:
            task: TaskStore 返回的任务字典（须含 id, cron_expr, task_text）

        Returns:
            True=注册成功，False=注册失败（cron 表达式无效等）
        """
        task_id = task.get("id")
        cron_expr = task.get("cron_expr")
        if task_id is None or not cron_expr:
            return False

        job_id = f"periodic_{task_id}"
        if job_id in self._registered_periodic:
            return False  # 已注册

        try:
            trigger = CronTrigger.from_crontab(cron_expr, **self._trigger_kwargs)
        except Exception as exc:
            logger.error("任务 #%d 的 cron 表达式无效 [%s]: %s", task_id, cron_expr, exc)
            return False

        self._scheduler.add_job(
            self._execute_periodic,
            trigger=trigger,
            args=[task_id],
            id=job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self._registered_periodic.add(job_id)
        logger.info("已注册周期任务 #%d (cron: %s)", task_id, cron_expr)
        return True

    def unregister_periodic_task(self, task_id: int):
        """从 APScheduler 移除一条周期任务的 cron job（取消任务时调用）。"""
        job_id = f"periodic_{task_id}"
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass  # job 不存在或调度器未启动，忽略
        self._registered_periodic.discard(job_id)

    # ---- 执行逻辑 ----

    def _submit_task(self, task: dict, is_periodic: bool = False) -> Optional[Future]:
        """
        将任务提交到线程池并发执行。

        Returns:
            Future 对象，或 None（线程池未就绪）
        """
        if self._executor is None:
            # 线程池未初始化（理论上不会发生），降级为同步执行
            self._run_one(task, is_periodic=is_periodic)
            return None

        future = self._executor.submit(self._run_one, task, is_periodic=is_periodic)
        self._pending_futures.add(future)
        # 完成后自动从 _pending_futures 移除，避免集合无限增长
        future.add_done_callback(self._pending_futures.discard)
        return future

    def _poll_and_execute(self):
        """
        轮询一次性任务：查询到期任务 → 抢占 → 提交线程池并发执行。

        由 APScheduler 的 IntervalTrigger 定时调用，运行在调度器线程中。
        本方法只负责查询和抢占，实际执行在 worker 线程中并发进行。
        """
        try:
            due_tasks = self.task_store.get_due_tasks()
        except Exception as exc:
            logger.error("轮询查询失败: %s", exc, exc_info=True)
            return

        if not due_tasks:
            return

        logger.info("发现 %d 个到期的一次性任务，提交线程池并发执行", len(due_tasks))

        for task in due_tasks:
            task_id = task["id"]
            # 原子抢占：pending → running，防止并发重复执行
            if not self.task_store.claim_task(task_id):
                continue
            self._submit_task(task, is_periodic=False)

    def _execute_periodic(self, task_id: int):
        """
        执行周期任务（由 APScheduler CronTrigger 触发）。

        周期任务不需要 claim（不会被并发拾取），但用 max_instances=1 防止同一任务重叠。
        多个不同周期任务同时触发时，各自提交到线程池并发执行。
        """
        task = self.task_store.get_task(task_id)
        if task is None:
            logger.warning("周期任务 #%d 不存在，移除 cron job", task_id)
            self.unregister_periodic_task(task_id)
            return

        if task["status"] == "cancelled":
            self.unregister_periodic_task(task_id)
            return

        # 周期任务直接提交线程池执行（不修改 status，因为它每次都要重复跑）
        self._submit_task(task, is_periodic=True)

    def _run_one(self, task: dict, is_periodic: bool = False):
        """执行单条任务并更新数据库状态。"""
        task_id = task["id"]

        success, output = execute_task(task, self.agent_factory)

        if is_periodic:
            # 周期任务不改状态（保持 pending 以便下次触发），仅记录结果
            # 用 result 字段临时存最近一次执行结果
            self.task_store._update_status(task_id, "pending", result=output[:2000])
            if not success:
                logger.error("周期任务 #%d 执行失败: %s", task_id, output[:200])
        else:
            if success:
                self.task_store.mark_done(task_id, output[:2000])
                logger.info("一次性任务 #%d 已完成", task_id)
            else:
                self.task_store.mark_failed(task_id, output[:2000])
                logger.error("一次性任务 #%d 执行失败（将自动重试或标记 failed）", task_id)