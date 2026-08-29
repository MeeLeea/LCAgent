"""
定时任务调度子系统测试

覆盖:
    - TaskStore: CRUD / 原子抢占 / 到期查询 / 重试逻辑
    - TaskExecutor: 成功 / 失败 / 空任务（agent_factory 用 mock）
    - SchedulerEngine: 周期注册 / 同步 / 轮询执行（agent_factory 用 mock）
    - Tool: schedule_task / list / cancel（@tool 直接调用 .invoke）
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from scheduler.engine import SchedulerEngine
from scheduler.executor import execute_task
from scheduler.store import TaskStore
from tools import scheduler_tool as scheduler_tool_mod

# ================================================================ Fixtures ===

@pytest.fixture
def store(tmp_path):
    """每个测试用独立的临时 SQLite 文件。"""
    db_path = str(tmp_path / "test_tasks.sqlite")
    return TaskStore(db_path)


@pytest.fixture
def configured_tool(store):
    """为 @tool 函数注入测试用 TaskStore。"""
    scheduler_tool_mod.configure(task_store=store, engine=None)
    yield store
    # 重置单例，避免污染其他测试
    scheduler_tool_mod._store = None
    scheduler_tool_mod._engine = None


def _parse_json_result(result_str: str) -> dict:
    """把 @tool 返回的 JSON 字符串解析为 dict。"""
    return json.loads(result_str)


# ================================================================ TaskStore ===

class TestTaskStore:

    def test_create_one_time_task(self, store):
        task_id = store.create_task(
            task_type="one_time",
            task_text="生成报告",
            execute_time="2026-07-29T15:00:00",
        )
        assert task_id > 0
        task = store.get_task(task_id)
        assert task["task_type"] == "one_time"
        assert task["task_text"] == "生成报告"
        assert task["status"] == "pending"
        assert task["execute_time"] == "2026-07-29T15:00:00"
        assert task["cron_expr"] is None

    def test_create_periodic_task(self, store):
        task_id = store.create_task(
            task_type="periodic",
            task_text="发送日报",
            cron_expr="0 9 * * *",
        )
        task = store.get_task(task_id)
        assert task["task_type"] == "periodic"
        assert task["cron_expr"] == "0 9 * * *"
        assert task["execute_time"] is None

    def test_create_invalid_type(self, store):
        with pytest.raises(ValueError, match="task_type"):
            store.create_task(task_type="invalid", task_text="test", execute_time="2026-01-01T00:00:00")

    def test_create_one_time_without_execute_time(self, store):
        with pytest.raises(ValueError, match="execute_time"):
            store.create_task(task_type="one_time", task_text="test")

    def test_create_periodic_without_cron(self, store):
        with pytest.raises(ValueError, match="cron_expr"):
            store.create_task(task_type="periodic", task_text="test")

    def test_create_empty_task_text(self, store):
        with pytest.raises(ValueError, match="task_text"):
            store.create_task(task_type="one_time", task_text="  ", execute_time="2026-01-01T00:00:00")

    def test_claim_task_success(self, store):
        task_id = store.create_task(
            task_type="one_time", task_text="test",
            execute_time="2020-01-01T00:00:00",
        )
        assert store.claim_task(task_id) is True
        task = store.get_task(task_id)
        assert task["status"] == "running"
        assert task["executed_at"] is not None

    def test_claim_task_already_claimed(self, store):
        task_id = store.create_task(
            task_type="one_time", task_text="test",
            execute_time="2020-01-01T00:00:00",
        )
        assert store.claim_task(task_id) is True
        # 第二次抢占应失败
        assert store.claim_task(task_id) is False

    def test_mark_done(self, store):
        task_id = store.create_task(
            task_type="one_time", task_text="test",
            execute_time="2020-01-01T00:00:00",
        )
        store.claim_task(task_id)
        store.mark_done(task_id, "执行成功")
        task = store.get_task(task_id)
        assert task["status"] == "done"
        assert task["result"] == "执行成功"

    def test_mark_failed_with_retry(self, store):
        task_id = store.create_task(
            task_type="one_time", task_text="test",
            execute_time="2020-01-01T00:00:00",
            max_retries=3,
        )
        store.claim_task(task_id)
        store.mark_failed(task_id, "网络错误")
        # 还有重试次数 → 回退 pending
        task = store.get_task(task_id)
        assert task["status"] == "pending"
        assert task["retry_count"] == 1

    def test_mark_failed_exhaust_retries(self, store):
        task_id = store.create_task(
            task_type="one_time", task_text="test",
            execute_time="2020-01-01T00:00:00",
            max_retries=2,
        )
        store.claim_task(task_id)
        store.mark_failed(task_id, "错误1")  # retry_count → 1 < 2 → pending
        assert store.get_task(task_id)["status"] == "pending"

        store.claim_task(task_id)
        store.mark_failed(task_id, "错误2")  # retry_count → 2 >= 2 → failed
        task = store.get_task(task_id)
        assert task["status"] == "failed"
        assert task["retry_count"] == 2

    def test_cancel_task(self, store):
        task_id = store.create_task(
            task_type="one_time", task_text="test",
            execute_time="2026-01-01T00:00:00",
        )
        assert store.cancel_task(task_id) is True
        assert store.get_task(task_id)["status"] == "cancelled"

    def test_cancel_non_pending_task(self, store):
        task_id = store.create_task(
            task_type="one_time", task_text="test",
            execute_time="2020-01-01T00:00:00",
        )
        store.claim_task(task_id)  # running
        assert store.cancel_task(task_id) is False

    def test_get_due_tasks(self, store):
        past = "2020-01-01T00:00:00"
        future = "2099-12-31T23:59:59"
        store.create_task("one_time", "过期任务", execute_time=past)
        store.create_task("one_time", "未来任务", execute_time=future)
        store.create_task("periodic", "周期任务", cron_expr="0 9 * * *")

        due = store.get_due_tasks()
        assert len(due) == 1
        assert due[0]["task_text"] == "过期任务"

    def test_get_due_tasks_respects_status(self, store):
        task_id = store.create_task("one_time", "test", execute_time="2020-01-01T00:00:00")
        store.claim_task(task_id)  # running
        due = store.get_due_tasks()
        assert len(due) == 0  # running 不算到期

    def test_list_periodic_tasks(self, store):
        store.create_task("periodic", "日报", cron_expr="0 9 * * *")
        store.create_task("periodic", "周报", cron_expr="0 9 * * 1")
        store.create_task("one_time", "一次性", execute_time="2026-01-01T00:00:00")

        periodic = store.list_periodic_tasks()
        assert len(periodic) == 2

    def test_list_tasks_with_status_filter(self, store):
        id1 = store.create_task("one_time", "t1", execute_time="2026-01-01T00:00:00")
        store.create_task("one_time", "t2", execute_time="2026-01-01T00:00:00")
        store.claim_task(id1)

        all_tasks = store.list_tasks()
        assert len(all_tasks) == 2
        running = store.list_tasks(status="running")
        assert len(running) == 1
        pending = store.list_tasks(status="pending")
        assert len(pending) == 1


# ================================================================ TaskExecutor ===

class TestTaskExecutor:

    def test_success(self):
        mock_agent = MagicMock()
        mock_agent.session_manager.arun = AsyncMock(return_value="任务执行完成")
        factory = lambda: mock_agent

        task = {"id": 1, "task_type": "one_time", "task_text": "生成报告"}
        success, output = execute_task(task, factory)
        assert success is True
        assert output == "任务执行完成"
        mock_agent.session_manager.arun.assert_called_once_with("生成报告")

    def test_agent_raises(self):
        mock_agent = MagicMock()
        mock_agent.session_manager.arun = AsyncMock(side_effect=RuntimeError("Agent 崩溃"))
        factory = lambda: mock_agent

        task = {"id": 2, "task_type": "one_time", "task_text": "test"}
        success, output = execute_task(task, factory)
        assert success is False
        assert "Agent 执行异常" in output

    def test_factory_raises(self):
        factory = lambda: (_ for _ in ()).throw(RuntimeError("LLM 未配置"))
        task = {"id": 3, "task_type": "one_time", "task_text": "test"}
        success, output = execute_task(task, factory)
        assert success is False
        assert "创建 Agent 实例失败" in output

    def test_empty_task_text(self):
        task = {"id": 4, "task_type": "one_time", "task_text": ""}
        success, output = execute_task(task, lambda: MagicMock())
        assert success is False
        assert "任务内容为空" in output


# ================================================================ SchedulerEngine ===

class TestSchedulerEngine:

    def test_register_periodic_task(self, store):
        mock_agent = MagicMock()
        mock_agent.session_manager.arun = AsyncMock(return_value="ok")
        engine = SchedulerEngine(
            task_store=store,
            agent_factory=lambda: mock_agent,
            poll_interval=3600,  # 长间隔避免轮询干扰
        )
        engine.start()

        task_id = store.create_task("periodic", "日报", cron_expr="0 9 * * *")
        task = store.get_task(task_id)
        assert engine.register_periodic_task(task) is True
        # 重复注册返回 False
        assert engine.register_periodic_task(task) is False

        engine.stop()

    def test_register_invalid_cron(self, store):
        mock_agent = MagicMock()
        engine = SchedulerEngine(
            task_store=store,
            agent_factory=lambda: mock_agent,
            poll_interval=3600,
        )
        engine.start()

        task_id = store.create_task("periodic", "test", cron_expr="invalid cron")
        task = store.get_task(task_id)
        assert engine.register_periodic_task(task) is False

        engine.stop()

    def test_sync_periodic_tasks(self, store):
        mock_agent = MagicMock()
        mock_agent.session_manager.arun = AsyncMock(return_value="ok")
        # 先入库两条周期任务（引擎未启动）
        store.create_task("periodic", "日报", cron_expr="0 9 * * *")
        store.create_task("periodic", "周报", cron_expr="0 9 * * 1")

        engine = SchedulerEngine(
            task_store=store,
            agent_factory=lambda: mock_agent,
            poll_interval=3600,
        )
        engine.start()  # start 内部会调 sync_periodic_tasks
        assert len(engine._registered_periodic) == 2
        engine.stop()

    def test_poll_and_execute_due_task(self, store):
        mock_agent = MagicMock()
        mock_agent.session_manager.arun = AsyncMock(return_value="报告已生成")
        engine = SchedulerEngine(
            task_store=store,
            agent_factory=lambda: mock_agent,
            poll_interval=3600,
        )
        engine.start()

        # 插入一条已过期的一次性任务
        task_id = store.create_task("one_time", "生成报告", execute_time="2020-01-01T00:00:00")
        # 手动触发轮询（不等 APScheduler 定时）
        engine._poll_and_execute()

        # 任务在线程池中异步执行，等待完成
        import time as _time
        _time.sleep(0.3)

        task = store.get_task(task_id)
        assert task["status"] == "done"
        assert task["result"] == "报告已生成"

        engine.stop()

    def test_poll_skips_non_due_tasks(self, store):
        mock_agent = MagicMock()
        engine = SchedulerEngine(
            task_store=store,
            agent_factory=lambda: mock_agent,
            poll_interval=3600,
        )
        engine.start()

        store.create_task("one_time", "未来任务", execute_time="2099-12-31T23:59:59")
        engine._poll_and_execute()
        mock_agent.session_manager.arun.assert_not_called()

        engine.stop()

    def test_unregister_periodic_task(self, store):
        mock_agent = MagicMock()
        engine = SchedulerEngine(
            task_store=store,
            agent_factory=lambda: mock_agent,
            poll_interval=3600,
        )
        engine.start()

        task_id = store.create_task("periodic", "日报", cron_expr="0 9 * * *")
        task = store.get_task(task_id)
        engine.register_periodic_task(task)
        assert f"periodic_{task_id}" in engine._registered_periodic

        engine.unregister_periodic_task(task_id)
        assert f"periodic_{task_id}" not in engine._registered_periodic

        engine.stop()

    def test_unregister_missing_periodic_task_does_not_raise(self, store, caplog):
        mock_agent = MagicMock()
        engine = SchedulerEngine(
            task_store=store,
            agent_factory=lambda: mock_agent,
            poll_interval=3600,
        )
        engine.start()

        with caplog.at_level("DEBUG"):
            engine.unregister_periodic_task(999999)

        assert "移除周期任务 job 失败，按缺省处理忽略" in caplog.text

        engine.stop()

    def test_concurrent_execution_of_multiple_due_tasks(self, store):
        """同一轮到期的多个任务在线程池中并发执行，而非串行。"""
        import threading as _threading

        # 用 Barrier 确保两个任务真正并行：都到达 barrier 后才放行
        barrier = _threading.Barrier(2, timeout=10)
        started = _threading.Event()
        done_order = []

        def slow_agent_run(task_text):
            started.set()
            # 等待另一个任务也到达这里，证明两者在并行执行
            barrier.wait()
            done_order.append(task_text)
            return f"done: {task_text}"

        mock_agent = MagicMock()
        mock_agent.session_manager.arun = AsyncMock(side_effect=slow_agent_run)

        engine = SchedulerEngine(
            task_store=store,
            agent_factory=lambda: mock_agent,
            poll_interval=3600,
            max_workers=5,
        )
        engine.start()

        # 插入两条同时到期的一次性任务
        store.create_task("one_time", "任务A", execute_time="2020-01-01T00:00:00")
        store.create_task("one_time", "任务B", execute_time="2020-01-01T00:00:00")

        engine._poll_and_execute()

        # 等待两个任务都完成（barrier 通过后才会有 2 个结果）
        assert started.wait(timeout=5), "至少有一个任务开始执行"
        # barrier.wait 内部会阻塞直到两个线程都到达；超时则抛 BrokenBarrierError
        # 给线程池一点时间写回数据库
        import time as _time
        _time.sleep(0.3)

        tasks = store.list_tasks(status="done")
        assert len(tasks) == 2
        texts = {t["task_text"] for t in tasks}
        assert texts == {"任务A", "任务B"}

        engine.stop()

    def test_max_workers_limits_concurrency(self, store):
        """max_workers=1 时任务退化为串行执行：第二个任务在第一个执行期间不会启动。"""
        import threading as _threading
        import time as _time

        active = _threading.Event()
        overlap_detected = _threading.Event()

        def serial_agent_run(task_text):
            # 若上一个任务仍在执行（active 未清除），说明并发了
            if active.is_set():
                overlap_detected.set()
            active.set()
            _time.sleep(0.3)
            active.clear()
            return f"done: {task_text}"

        mock_agent = MagicMock()
        mock_agent.session_manager.arun = AsyncMock(side_effect=serial_agent_run)

        engine = SchedulerEngine(
            task_store=store,
            agent_factory=lambda: mock_agent,
            poll_interval=3600,
            max_workers=1,
        )
        engine.start()

        store.create_task("one_time", "任务1", execute_time="2020-01-01T00:00:00")
        store.create_task("one_time", "任务2", execute_time="2020-01-01T00:00:00")

        engine._poll_and_execute()
        _time.sleep(1.0)

        assert not overlap_detected.is_set(), "max_workers=1 时不应有并发执行"

        engine.stop()


# ================================================================ Tool ===

class TestScheduleTaskTool:

    def test_schedule_one_time(self, configured_tool):
        result = _parse_json_result(
            scheduler_tool_mod.schedule_task.invoke({
                "task_text": "生成报告",
                "task_type": "one_time",
                "execute_time": "2026-07-29T15:00:00",
                "cron_expr": "",
            })
        )
        assert result["success"] is True
        assert result["task_id"] > 0
        assert result["task_type"] == "one_time"
        assert "已登记" in result["message"]

    def test_schedule_periodic(self, configured_tool):
        result = _parse_json_result(
            scheduler_tool_mod.schedule_task.invoke({
                "task_text": "发送日报",
                "task_type": "periodic",
                "execute_time": "",
                "cron_expr": "0 9 * * *",
            })
        )
        assert result["success"] is True
        assert result["cron_expr"] == "0 9 * * *"

    def test_schedule_invalid_type(self, configured_tool):
        result = _parse_json_result(
            scheduler_tool_mod.schedule_task.invoke({
                "task_text": "test",
                "task_type": "invalid",
                "execute_time": "",
                "cron_expr": "",
            })
        )
        assert result["success"] is False
        assert "task_type" in result["error"]

    def test_schedule_one_time_missing_time(self, configured_tool):
        result = _parse_json_result(
            scheduler_tool_mod.schedule_task.invoke({
                "task_text": "test",
                "task_type": "one_time",
                "execute_time": "",
                "cron_expr": "",
            })
        )
        assert result["success"] is False
        assert "execute_time" in result["error"]

    def test_schedule_periodic_missing_cron(self, configured_tool):
        result = _parse_json_result(
            scheduler_tool_mod.schedule_task.invoke({
                "task_text": "test",
                "task_type": "periodic",
                "execute_time": "",
                "cron_expr": "",
            })
        )
        assert result["success"] is False
        assert "cron_expr" in result["error"]


class TestListScheduledTasksTool:

    def test_list_all(self, configured_tool):
        configured_tool.create_task("one_time", "t1", execute_time="2026-01-01T00:00:00")
        configured_tool.create_task("periodic", "t2", cron_expr="0 9 * * *")

        result = _parse_json_result(
            scheduler_tool_mod.list_scheduled_tasks.invoke({"status": ""})
        )
        assert result["success"] is True
        assert result["count"] == 2

    def test_list_by_status(self, configured_tool):
        task_id = configured_tool.create_task("one_time", "t1", execute_time="2026-01-01T00:00:00")
        configured_tool.cancel_task(task_id)

        result = _parse_json_result(
            scheduler_tool_mod.list_scheduled_tasks.invoke({"status": "cancelled"})
        )
        assert result["count"] == 1
        assert result["tasks"][0]["status"] == "cancelled"


class TestCancelScheduledTaskTool:

    def test_cancel_pending(self, configured_tool):
        task_id = configured_tool.create_task("one_time", "t1", execute_time="2026-01-01T00:00:00")

        result = _parse_json_result(
            scheduler_tool_mod.cancel_scheduled_task.invoke({"task_id": task_id})
        )
        assert result["success"] is True
        assert configured_tool.get_task(task_id)["status"] == "cancelled"

    def test_cancel_nonexistent(self, configured_tool):
        result = _parse_json_result(
            scheduler_tool_mod.cancel_scheduled_task.invoke({"task_id": 9999})
        )
        assert result["success"] is False
        assert "不存在" in result["error"]

    def test_cancel_already_running(self, configured_tool):
        task_id = configured_tool.create_task("one_time", "t1", execute_time="2020-01-01T00:00:00")
        configured_tool.claim_task(task_id)  # → running

        result = _parse_json_result(
            scheduler_tool_mod.cancel_scheduled_task.invoke({"task_id": task_id})
        )
        assert result["success"] is False
        assert "running" in result["error"]
