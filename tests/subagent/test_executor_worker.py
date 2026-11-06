"""Executor worker（ReAct 多步工具循环运行时）的离线单元测试。

设计约束：
    - 全程不触网、不调用真实 LLM、不拉起真实子进程，全部依赖注入假对象；
    - 未安装 ``pytest-asyncio``，故统一用 ``asyncio.run`` 在同步测试函数中驱动协程；
    - 所有任务数据写入 ``tmp_path`` 下的 :class:`ExecutorStore`，测试间互不影响。

FakeModel 契约：
    - ``bind_tools(tools)`` 记录绑定工具并返回自身（worker 必须用 bind_tools）；
    - ``ainvoke(messages)`` 按调用序号执行 handler，返回 AIMessage。

覆盖范围：
    - load_prompt_sections：真实 AGENT.md 解析、占位符集合、无效文件报错
    - run_task：happy path（含工具真实写盘）、bind_tools 绑定、task.json 中
      workspace 缺失、工作区被占用、模型异常、空最终输出、步数耗尽、
      小于 MIN_STEP_SECONDS 的预算、调用中取消、认领前取消
    - _await_with_cancel：正常完成、取消、超时且无悬挂任务
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from subagent.common.config import EXECUTOR_AGENT_PROMPT_FILE, ExecutorConfig
from subagent.executor.store import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    ExecutorStore,
    acquire_workspace_lock,
    release_workspace_lock,
    resolve_workspace_at_start,
)
from subagent.executor.worker import (
    RESULT_MARKER,
    ExecutorCancelled,
    _await_with_cancel,
    load_prompt_sections,
    run_task,
)
from tests.subagent._fakes import (
    ExecutorFakeModel,
    open_tasks_excluding_current,
    tool_call_message,
)

# 复用共享假模型（默认回复模拟带结束标记的最终答复）与工具消息构造器
FakeModel = ExecutorFakeModel
_tool_call_message = tool_call_message


@pytest.fixture
def store(tmp_path: Path) -> ExecutorStore:
    """每个测试使用独立的临时任务根目录。"""
    return ExecutorStore(str(tmp_path))


def _write_prompt_file(tmp_path: Path) -> str:
    """写一个带占位符的最小 executor 提示词文件，返回路径。"""
    path = tmp_path / "AGENT_test.md"
    path.write_text(
        "你是任务执行者，必须使用工具在工作区内完成任务。\n\n"
        "## workflow:execute\n"
        "任务：{task}\n"
        "计划：{plan}\n"
        "上下文：{context}\n"
        "工作区：{workspace}\n"
        f"最终答复必须以 {RESULT_MARKER} 结尾。\n",
        encoding="utf-8",
    )
    return str(path)


def _config(prompt_file: str, **overrides: Any) -> ExecutorConfig:
    """构造专用于测试的 ExecutorConfig（超时预算必须 >= MIN_STEP_SECONDS）。"""
    defaults: dict[str, Any] = {"agent_prompt_file": prompt_file, "timeout_seconds": 120.0}
    defaults.update(overrides)
    return ExecutorConfig(**defaults)


def _prepare(store: ExecutorStore, plan: str = "计划正文") -> str:
    """创建任务并用隔离 workspace 预填 task.json，返回 task_id。"""
    task_id = store.create("把内容写进文件", plan=plan, context="上下文")
    resolve_workspace_at_start(store, task_id, None)
    return task_id


def _run(
    store: ExecutorStore,
    task_id: str,
    model: FakeModel,
    config: ExecutorConfig,
) -> int:
    """以同步方式驱动一次 ``run_task`` 并返回进程返回码。"""
    return asyncio.run(run_task(task_id, store=store, config=config, model=model))


def _workspace_startswith_error(observation: str) -> bool:
    """判断工具观测是否为错误提示。"""
    return observation.startswith("[错误]")


# ================================================================ 提示词解析 ===


def test_load_prompt_sections_real_agent_md() -> None:
    """真实 AGENT.md：系统提示与模板均非空，且只有一个 workflow 小节。"""
    system_prompt, template = load_prompt_sections(EXECUTOR_AGENT_PROMPT_FILE)

    assert system_prompt.strip()
    assert template.strip()

    text = Path(EXECUTOR_AGENT_PROMPT_FILE).read_text(encoding="utf-8")
    workflow_lines = [
        line for line in text.splitlines() if line.strip().startswith("## workflow:")
    ]
    assert len(workflow_lines) == 1


def test_load_prompt_sections_placeholders(tmp_path: Path) -> None:
    """真实模板的占位符恰为 {task}/{plan}/{context}/{workspace}。"""
    _system_prompt, template = load_prompt_sections(EXECUTOR_AGENT_PROMPT_FILE)

    import re

    placeholders = set(re.findall(r"\{[a-zA-Z_]+\}", template))
    assert placeholders == {"{task}", "{plan}", "{context}", "{workspace}"}


def test_load_prompt_sections_invalid_files(tmp_path: Path) -> None:
    """缺失、空文件、无 workflow 小节、模板为空均抛 ValueError。"""
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
    empty_template.write_text(
        "# 标题\n\n## workflow:execute\n\n## 下一节\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_prompt_sections(str(empty_template))


# ================================================================ run_task ===


def test_run_task_happy_path_with_tool_writes(store: ExecutorStore, tmp_path: Path) -> None:
    """happy path：工具真实写盘、result.md 带 marker、done、锁被移除。"""
    task_id = _prepare(store)
    prompt_file = _write_prompt_file(tmp_path)

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        if call == 1:
            return _tool_call_message(
                "write_file",
                {"path": "hello.txt", "content": "你好，工作区"},
                call,
            )
        return AIMessage(content=f"已写入 hello.txt。\n\n{RESULT_MARKER}")

    model = FakeModel(handler)
    return_code = _run(store, task_id, model, _config(prompt_file))

    assert return_code == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_DONE
    assert task["has_result"] is True
    assert task["marker_present"] is True
    assert task["steps_used"] == 2
    assert model.calls == 2

    # 提示词审计产物落盘
    prompt_text = store.read_text(task_id, "prompt.md")
    assert prompt_text is not None
    assert "把内容写进文件" in prompt_text

    # 工具真实写盘到隔离 workspace
    workspace = Path(task["workspace"])
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "你好，工作区"

    # 最终答复写入 result.md
    assert (store.read_text(task_id, "result.md") or "").endswith(RESULT_MARKER)

    # 任务锁与工作区锁均被释放
    assert not (Path(store.task_dir(task_id)) / "worker.lock").exists()
    assert ".lcagent-executor.lock" not in [p.name for p in workspace.iterdir()]


def test_run_task_binds_tools_via_bind_tools(store: ExecutorStore, tmp_path: Path) -> None:
    """必须用 bind_tools 绑定（Oracle 结论），且绑定的是 3 个受限工具。"""
    task_id = _prepare(store)
    prompt_file = _write_prompt_file(tmp_path)
    model = FakeModel()

    return_code = _run(store, task_id, model, _config(prompt_file))

    assert return_code == 0
    assert model.bound_tools is not None
    assert {t.name for t in model.bound_tools} == {"read_file", "write_file", "list_dir"}


def test_run_task_missing_workspace_fails(store: ExecutorStore, tmp_path: Path) -> None:
    """task.json 缺少有效 workspace 时判 failed，不进入循环。"""
    task_id = store.create("任务")
    prompt_file = _write_prompt_file(tmp_path)

    return_code = _run(store, task_id, FakeModel(), _config(prompt_file))

    assert return_code == 1
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_FAILED
    assert "workspace" in (task["error"] or "")
    assert not (Path(store.task_dir(task_id)) / "worker.lock").exists()


def test_run_task_workspace_busy_fails(store: ExecutorStore, tmp_path: Path) -> None:
    """工作区已被其他任务占用时判 failed，且不破坏占用方的锁。"""
    task_id = _prepare(store)
    workspace = store.get(task_id)["workspace"]  # type: ignore[index]
    prompt_file = _write_prompt_file(tmp_path)

    assert acquire_workspace_lock(workspace) is True
    try:
        return_code = _run(store, task_id, FakeModel(), _config(prompt_file))
        assert return_code == 1
        task = store.get(task_id)
        assert task is not None
        assert task["status"] == STATUS_FAILED
        assert "工作区正被其他 executor 任务占用" in (task["error"] or "")
    finally:
        release_workspace_lock(workspace)


def test_run_task_model_raises_marks_failed(store: ExecutorStore, tmp_path: Path) -> None:
    """模型抛异常：返回 1、failed、有 error、无 result.md、锁被移除。"""
    task_id = _prepare(store)
    prompt_file = _write_prompt_file(tmp_path)

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        raise RuntimeError("模型崩溃")

    return_code = _run(store, task_id, FakeModel(handler), _config(prompt_file))

    assert return_code == 1
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_FAILED
    assert "模型崩溃" in (task["error"] or "")
    assert store.read_text(task_id, "result.md") is None
    assert not (Path(store.task_dir(task_id)) / "worker.lock").exists()


def test_run_task_empty_final_answer_fails(store: ExecutorStore, tmp_path: Path) -> None:
    """最终答复为空白：failed，错误包含“空”。"""
    task_id = _prepare(store)
    prompt_file = _write_prompt_file(tmp_path)

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="   ")

    return_code = _run(store, task_id, FakeModel(handler), _config(prompt_file))

    assert return_code == 1
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_FAILED
    assert "空" in (task["error"] or "")


def test_run_task_max_steps_exhausted_fails(store: ExecutorStore, tmp_path: Path) -> None:
    """max_steps 耗尽仍返回 tool_calls：判 failed，steps_used 达到上限。"""
    task_id = _prepare(store)
    prompt_file = _write_prompt_file(tmp_path)

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        return _tool_call_message("list_dir", {"path": "."}, call)

    config = _config(prompt_file, max_steps=2)
    return_code = _run(store, task_id, FakeModel(handler), config)

    assert return_code == 1
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_FAILED
    assert "步数耗尽" in (task["error"] or "")
    assert task["steps_used"] == 2


def test_run_task_budget_below_min_step_fails(store: ExecutorStore, tmp_path: Path) -> None:
    """总预算小于 MIN_STEP_SECONDS：直接判 failed，不发起任何 LLM 调用。"""
    task_id = _prepare(store)
    prompt_file = _write_prompt_file(tmp_path)
    model = FakeModel()

    config = _config(prompt_file, timeout_seconds=0.5)
    return_code = _run(store, task_id, model, config)

    assert return_code == 1
    assert model.calls == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_FAILED
    assert "剩余预算不足" in (task["error"] or "")


def test_run_task_unknown_tool_gets_paired_error(
    store: ExecutorStore, tmp_path: Path
) -> None:
    """未知工具调用也得配对 error ToolMessage，循环不中断并可收敛。"""
    task_id = _prepare(store)
    prompt_file = _write_prompt_file(tmp_path)

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        if call == 1:
            return _tool_call_message("no_such_tool", {}, call)
        # 最后一条消息应是未知工具的 error ToolMessage
        last = messages[-1]
        assert type(last).__name__ == "ToolMessage"
        assert "[错误] 未知工具" in str(last.content)
        return AIMessage(content=f"已恢复并完成。\n\n{RESULT_MARKER}")

    return_code = _run(store, task_id, FakeModel(handler), _config(prompt_file))

    assert return_code == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_DONE


def test_run_task_tool_observation_truncated(
    store: ExecutorStore, tmp_path: Path
) -> None:
    """超长工具观测被截断到 max_observation_chars，最终答复仍可达成。"""
    task_id = _prepare(store)
    prompt_file = _write_prompt_file(tmp_path)
    big_content = "x" * 5000

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        if call == 1:
            return _tool_call_message(
                "write_file", {"path": "big.txt", "content": big_content}, call
            )
        return AIMessage(content=f"完成。\n\n{RESULT_MARKER}")

    config = _config(prompt_file, max_observation_chars=100)
    return_code = _run(store, task_id, FakeModel(handler), config)

    assert return_code == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_DONE
    assert Path(task["workspace"], "big.txt").read_text(encoding="utf-8") == big_content


def test_run_task_cancel_during_call(store: ExecutorStore, tmp_path: Path) -> None:
    """LLM 调用期间写入取消标记：返回 0、cancelled、不写 result.md。"""
    task_id = _prepare(store)
    prompt_file = _write_prompt_file(tmp_path)

    async def handler(call: int, messages: list[BaseMessage]) -> AIMessage:
        store.request_cancel(task_id)
        await asyncio.sleep(10)  # 保持挂起，直到 worker 轮询到取消标记（2s 轮询）
        return AIMessage(content="不应到达")

    return_code = _run(store, task_id, FakeModel(handler), _config(prompt_file))

    assert return_code == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_CANCELLED
    assert store.read_text(task_id, "result.md") is None


def test_run_task_cancel_before_claim(store: ExecutorStore, tmp_path: Path) -> None:
    """认领前已存在 cancel.flag：直接 cancelled，不建锁、不写 result.md。"""
    task_id = store.create("任务")
    resolve_workspace_at_start(store, task_id, None)
    (Path(store.task_dir(task_id)) / "cancel.flag").write_text("", encoding="utf-8")
    prompt_file = _write_prompt_file(tmp_path)
    model = FakeModel()

    return_code = _run(store, task_id, model, _config(prompt_file))

    assert return_code == 0
    assert model.calls == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_CANCELLED
    assert not (Path(store.task_dir(task_id)) / "worker.lock").exists()
    assert store.read_text(task_id, "result.md") is None


def test_run_task_missing_task_returns_two(store: ExecutorStore, tmp_path: Path) -> None:
    """任务不存在时返回 2，且不创建任何任务文件。"""
    return_code = _run(
        store, "no-such-task", FakeModel(), _config(_write_prompt_file(tmp_path))
    )
    assert return_code == 2
    assert store.get("no-such-task") is None


def test_run_task_skips_when_lock_exists(store: ExecutorStore, tmp_path: Path) -> None:
    """worker.lock 已存在时认领失败，返回 0 且任务保持 queued。"""
    task_id = _prepare(store)
    (Path(store.task_dir(task_id)) / "worker.lock").write_text("held", encoding="utf-8")
    prompt_file = _write_prompt_file(tmp_path)

    return_code = _run(store, task_id, FakeModel(), _config(prompt_file))

    assert return_code == 0
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_QUEUED


# ================================================================ _await_with_cancel ===


# 复用共享辅助（防任务泄漏断言）
_open_tasks_excluding_current = open_tasks_excluding_current


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
    """cancel_check 为真时抛 ExecutorCancelled，且不留下悬挂任务。"""

    async def scenario() -> None:
        async def coro() -> int:
            await asyncio.sleep(10)
            return 1

        with pytest.raises(ExecutorCancelled):
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
