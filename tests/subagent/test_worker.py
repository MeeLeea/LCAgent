"""Planner worker（一次性 one-shot 运行时）与提示词解析的离线单元测试。

设计约束：
    - 全程不触网、不调用真实 LLM、不拉起真实子进程，全部依赖注入假对象；
    - 未安装 ``pytest-asyncio``，故统一用 ``asyncio.run`` 在同步测试函数中驱动协程；
    - 所有任务数据写入 ``tmp_path`` 下的 :class:`PlannerStore`，测试间互不影响。

覆盖范围：
    - load_prompt_sections：真实 AGENT.md 解析、占位符、缺失/空/无 workflow 报错
    - run_task：happy path、缺失任务、已持锁跳过、模型异常、空输出、
      调用中取消、认领前取消、超时
    - _await_with_cancel：正常完成、取消、超时、无悬挂任务
"""
from __future__ import annotations

import asyncio
import dataclasses
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from subagent.planner.config import DEFAULT_AGENT_PROMPT_FILE, PlannerConfig
from subagent.planner.store import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    PlannerStore,
)
from subagent.planner.worker import (
    PlannerCancelled,
    _await_with_cancel,
    load_prompt_sections,
    run_task,
)

# 模型调用处理器：入参为（调用序号, 消息列表），返回一条 AIMessage 或抛异常。
ModelHandler = Callable[[int, list[BaseMessage]], Awaitable[AIMessage]]


class FakeModel:
    """可编排的假聊天模型：按调用序号返回消息、抛异常或执行副作用。"""

    def __init__(self, handler: ModelHandler | None = None) -> None:
        self.calls: int = 0
        self.handler = handler
        self.bind_kwargs: dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> FakeModel:
        """记录绑定的 max_tokens 并返回自身，模拟可链式绑定的模型。"""
        self.bind_kwargs = dict(kwargs)
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        """按序号执行处理器；未提供处理器时返回占位文本。"""
        self.calls += 1
        if self.handler is not None:
            return await self.handler(self.calls, messages)
        return AIMessage(content=f"方案输出-{self.calls}")


@pytest.fixture
def store(tmp_path: Path) -> PlannerStore:
    """每个测试使用独立的临时任务根目录。"""
    return PlannerStore(str(tmp_path))


def _run(
    store: PlannerStore,
    task_id: str,
    model: FakeModel,
    *,
    config: PlannerConfig | None = None,
) -> int:
    """以同步方式驱动一次 ``run_task`` 并返回进程返回码。"""
    return asyncio.run(
        run_task(task_id, store=store, config=config or PlannerConfig(), model=model)
    )


# ================================================================ 提示词解析 ===


def test_load_prompt_sections_real_agent_md() -> None:
    """真实 AGENT.md：系统提示与模板均非空，且只有一个 workflow 小节。"""
    system_prompt, template = load_prompt_sections(DEFAULT_AGENT_PROMPT_FILE)

    assert system_prompt.strip()
    assert template.strip()
    assert template.rstrip().endswith("<!-- END OF PLAN -->")

    text = Path(DEFAULT_AGENT_PROMPT_FILE).read_text(encoding="utf-8")
    workflow_lines = [line for line in text.splitlines() if line.strip().startswith("## workflow:")]
    assert len(workflow_lines) == 1


def test_load_prompt_sections_placeholders() -> None:
    """真实模板中的占位符恰好为 {requirement} 与 {context}。"""
    _system_prompt, template = load_prompt_sections(DEFAULT_AGENT_PROMPT_FILE)
    placeholders = set(re.findall(r"\{[a-zA-Z_]+\}", template))
    assert placeholders == {"{requirement}", "{context}"}


def test_load_prompt_sections_invalid_files(tmp_path: Path) -> None:
    """缺失、空文件、无 workflow 小节、workflow 模板为空均抛 ValueError。"""
    with pytest.raises(ValueError):
        load_prompt_sections(str(tmp_path / "missing.md"))

    empty_path = tmp_path / "empty.md"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompt_sections(str(empty_path))

    no_workflow = tmp_path / "no_workflow.md"
    no_workflow.write_text("# 标题\n\n## 职责\n内容\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompt_sections(str(no_workflow))

    empty_template = tmp_path / "empty_template.md"
    empty_template.write_text("# 标题\n\n## workflow:design\n\n## 下一节\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompt_sections(str(empty_template))


# ================================================================ run_task ===


def test_run_task_happy_path(store: PlannerStore) -> None:
    """正常路径：返回 0、done、has_result、plan.md 与 prompt.md 落盘、锁被移除。"""
    task_id = store.create("实现一个功能", "上下文")

    return_code = _run(store, task_id, FakeModel())

    assert return_code == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_DONE
    assert task["has_result"] is True
    plan = store.read_text(task_id, "plan.md")
    assert plan and plan.strip()
    assert store.read_text(task_id, "prompt.md") is not None
    assert not (Path(store.task_dir(task_id)) / "worker.lock").exists()


def test_run_task_missing_task_returns_two(store: PlannerStore) -> None:
    """任务不存在时返回 2，且不创建任何任务文件。"""
    return_code = _run(store, "no-such-task", FakeModel())

    assert return_code == 2
    assert store.get("no-such-task") is None
    assert not Path(store.task_dir("no-such-task")).exists()


def test_run_task_skips_when_lock_exists(store: PlannerStore) -> None:
    """worker.lock 已存在时认领失败，返回 0 且任务保持 queued。"""
    task_id = store.create("需求")
    (Path(store.task_dir(task_id)) / "worker.lock").write_text("held", encoding="utf-8")

    return_code = _run(store, task_id, FakeModel())

    assert return_code == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_QUEUED


def test_run_task_model_raises_marks_failed(store: PlannerStore) -> None:
    """模型抛异常：返回 1、failed、有 error、无 plan.md、锁被移除。"""
    task_id = store.create("需求")

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        raise RuntimeError("模型崩溃")

    return_code = _run(store, task_id, FakeModel(handler))

    assert return_code == 1
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_FAILED
    assert task["error"]
    assert store.read_text(task_id, "plan.md") is None
    assert not (Path(store.task_dir(task_id)) / "worker.lock").exists()


def test_run_task_empty_content_marks_failed(store: PlannerStore) -> None:
    """模型返回空白内容时失败，错误信息包含“空”。"""
    task_id = store.create("需求")

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="   ")

    return_code = _run(store, task_id, FakeModel(handler))

    assert return_code == 1
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_FAILED
    assert "空" in (task["error"] or "")


def test_run_task_cancel_during_call(store: PlannerStore) -> None:
    """LLM 调用期间写入取消标记：返回 0、cancelled、不写 plan.md。"""
    task_id = store.create("需求")

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        store.request_cancel(task_id)
        await asyncio.sleep(10)  # 保持挂起，直到 worker 轮询到取消标记
        return AIMessage(content="不应到达")

    return_code = _run(store, task_id, FakeModel(handler))

    assert return_code == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_CANCELLED
    assert store.read_text(task_id, "plan.md") is None


def test_run_task_cancel_before_claim(store: PlannerStore) -> None:
    """认领前已存在 cancel.flag：直接 cancelled，返回 0，不建锁、不写 plan.md。"""
    task_id = store.create("需求")
    (Path(store.task_dir(task_id)) / "cancel.flag").write_text("", encoding="utf-8")

    return_code = _run(store, task_id, FakeModel())

    assert return_code == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_CANCELLED
    assert not (Path(store.task_dir(task_id)) / "worker.lock").exists()
    assert store.read_text(task_id, "plan.md") is None


def test_run_task_timeout_marks_failed(store: PlannerStore) -> None:
    """单次调用超时：返回 1、failed、错误信息包含“超时”。"""
    task_id = store.create("需求")

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        await asyncio.sleep(0.5)
        return AIMessage(content="迟到输出")

    config = dataclasses.replace(PlannerConfig(), timeout_seconds=0.05)
    return_code = _run(store, task_id, FakeModel(handler), config=config)

    assert return_code == 1
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_FAILED
    assert "超时" in (task["error"] or "")


# ================================================================ _await_with_cancel ===


def _open_tasks_excluding_current() -> int:
    """返回当前事件循环中除本协程外尚未完成的任务数量。"""
    current = asyncio.current_task()
    return sum(1 for task in asyncio.all_tasks() if task is not current and not task.done())


def test_await_with_cancel_completes_normally() -> None:
    """协程正常完成时返回结果，且不留下悬挂任务。"""

    async def scenario() -> Any:
        async def coro() -> int:
            return 42

        result = await _await_with_cancel(
            coro(), timeout=1.0, cancel_check=lambda: False, poll_seconds=0.01
        )
        assert _open_tasks_excluding_current() == 0
        return result

    assert asyncio.run(scenario()) == 42


def test_await_with_cancel_raises_on_cancel() -> None:
    """cancel_check 为真时抛 PlannerCancelled，且不留下悬挂任务。"""

    async def scenario() -> None:
        async def coro() -> int:
            await asyncio.sleep(10)
            return 1

        with pytest.raises(PlannerCancelled):
            await _await_with_cancel(
                coro(), timeout=1.0, cancel_check=lambda: True, poll_seconds=0.01
            )
        assert _open_tasks_excluding_current() == 0

    asyncio.run(scenario())


def test_await_with_cancel_raises_on_timeout() -> None:
    """超过 deadline 时抛 TimeoutError，且不留下悬挂任务。"""

    async def scenario() -> None:
        async def coro() -> int:
            await asyncio.sleep(10)
            return 1

        with pytest.raises(TimeoutError):
            await _await_with_cancel(
                coro(), timeout=0.05, cancel_check=lambda: False, poll_seconds=0.01
            )
        assert _open_tasks_excluding_current() == 0

    asyncio.run(scenario())
