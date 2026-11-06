"""Executor 子程序的多步（多轮工具循环）运行时。

与 planner 一次性 one-shot 的差异：
- worker 持一个**受限工具集**（read_file / write_file / list_dir，全部绑定到
  workspace），按 ReAct 循环多步执行：每步 = 一次 LLM 调用 + 本轮 tool_calls
  全部执行完毕；无 tool_calls 时视为最终答复。
- 终态保证与 planner 相同：``try/except BaseException/finally`` 包裹，正常落
  ``done``、取消落 ``cancelled``、异常落 ``failed`` 并写 ``error``，绝不留
  ``running`` 僵尸（``steps_used`` 在每步后更新，供 reap 失败后查看进度）。
- 取消语义：``cancel.flag`` 在**每步开始、每次 LLM 调用中、写 result.md 前**
  检查；取消不写 ``result.md``，且**不回滚**工作区已写入的文件（工具 docstring
  与消息均已声明该语义）。

循环纪律（防孤儿 tool_call / 网关异常）：
- 用 ``model.bind_tools(tools)`` 绑定工具（``bind(tools=...)`` 不会序列化 schema）；
- 每个工具调用必须恰好得到一条同 ``tool_call_id`` 的 ``ToolMessage``——工具抛
  异常也在循环内 catch 成 ``status="error"``，杜绝孤儿 tool_call 触发网关 400；
- 单条观测截断到 ``max_observation_chars``（防上下文爆炸）；
- 空 AIMessage / ``finish_reason=length`` 一律判 failed，不做同预算重试；
- ``max_steps`` 耗尽或剩余 deadline 不足 ``EXECUTOR_MIN_STEP_SECONDS`` 时判
  failed，不发起注定失败的调用。

进程入口：``python -m subagent.executor.worker --task-id <id>``。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import traceback
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from llm.llm_client import LLMClient
from llm.message_utils import final_answer_is_empty, stringify_content
from subagent.common.awaitx import await_with_cancel
from subagent.common.config import resolve_llm_settings
from subagent.common.fsx import truncate_text
from subagent.common.paths import normalize_path
from subagent.common.prompt import parse_workflow_section, render_placeholders
from subagent.config import EXECUTOR_MIN_STEP_SECONDS, ExecutorConfig, load_executor_config
from subagent.executor.store import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    ExecutorStore,
    acquire_workspace_lock,
    release_workspace_lock,
)
from subagent.executor.tools import build_executor_tools

if TYPE_CHECKING:
    from utils.logging_config import TraceContext

logger = logging.getLogger(__name__)

# 审计产物：本次实际投喂的提示词（非记忆，仅用于事后排查）
PROMPT_FILE: str = "prompt.md"
# 最终结果文档（store.read_result 固定读取此名）
FINAL_RESULT_FILE: str = "result.md"
# 结果标记（AGENT.md 模板要求的尾标记；缺失不判失败，仅记录 marker_present）
RESULT_MARKER: str = "<!-- END OF RESULT -->"

__all__ = ["ExecutorCancelled", "load_prompt_sections", "main", "run_task"]


class ExecutorCancelled(Exception):
    """用户请求取消时在等待中途抛出，用于快速跳出。"""


def _now_iso() -> str:
    """返回带本地时区信息的 ISO 时间戳（秒级精度）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_prompt_sections(prompt_file: str) -> tuple[str, str]:
    """解析提示词文件，返回 ``(系统提示词, 执行模板)``。

    契约与 planner 相同（公共实现薄包装，注入 executor 缺失文案）：文件被唯一
    一个 ``## workflow:`` 小节切成两半——其上为系统提示词，其下为该小节模板正文。

    Raises: ValueError（文件缺失/为空、缺 ``## workflow:`` 小节、或任一侧为空）。
    """
    return parse_workflow_section(
        prompt_file, missing_file_label="执行者提示词文件缺失或为空"
    )


async def _await_with_cancel(
    awaitable: Awaitable[Any], *, timeout: float,
    cancel_check: Callable[[], bool], poll_seconds: float,
) -> Any:
    """在可取消、可超时的前提下等待协程完成（公共实现薄包装，注入 executor 文案）。

    Raises: ExecutorCancelled（命中取消）/ TimeoutError（超过 ``timeout``）。
    """
    return await await_with_cancel(
        awaitable,
        timeout=timeout,
        cancel_check=cancel_check,
        poll_seconds=poll_seconds,
        cancel_exception=ExecutorCancelled,
        timeout_message=f"模型调用超时（{timeout}s）",
    )


def _mark_terminal(
    store: ExecutorStore, task_id: str, status: str, error: str | None, *,
    has_result: bool = False, marker_present: bool = False,
) -> None:
    """尽力写入终态；写失败只记日志，不掩盖原始异常。"""
    now = _now_iso()
    try:
        store.update(
            task_id,
            status=status,
            finished_at=now,
            has_result=has_result,
            marker_present=marker_present,
            error=error,
        )
    except BaseException as exc:
        logger.error("写入任务状态失败 (%s): %s", task_id, exc)


def _to_tool_message(
    call: dict[str, Any],
    observation: str,
    *,
    ok: bool,
) -> ToolMessage:
    """把工具调用结果包装为一条与 ``tool_call_id`` 严格配对的 ToolMessage。"""
    call_id = str(call.get("id") or "unknown")
    return ToolMessage(
        content=observation,
        tool_call_id=call_id,
        status="error" if not ok else "success",
    )


def _execute_tool_calls(
    tool_calls: list[dict[str, Any]],
    registry: dict[str, Callable[..., Any]],
    *,
    max_observation_chars: int,
) -> list[ToolMessage]:
    """执行一批工具调用；**每个调用都必须产出一条配对的 ToolMessage**。

    工具抛异常时不让它逃出循环，而是包装为 ``status="error"`` 的观测文本，
    否则后续消息序列会出现孤儿 tool_call_id（OpenAI 兼容网关直接 400）。
    """
    messages: list[ToolMessage] = []
    for call in tool_calls:
        name = str(call.get("name") or "")
        if name not in registry:
            messages.append(_to_tool_message(
                call,
                f"[错误] 未知工具: {name}。可用工具: {', '.join(sorted(registry))}",
                ok=False,
            ))
            continue
        tool = registry[name]
        try:
            observation = str(tool.invoke(call.get("args") or {}))
        except Exception as exc:
            observation = f"[错误] 工具执行异常（{type(exc).__name__}）: {exc}"
        messages.append(_to_tool_message(
            call,
            truncate_text(observation, max_observation_chars),
            ok=not observation.startswith("[错误]"),
        ))
    return messages


async def _arun_loop(
    *, bound: BaseChatModel, messages: list[BaseMessage],
    registry: dict[str, Callable[..., Any]], store: ExecutorStore,
    task_id: str, config: ExecutorConfig, deadline: float,
) -> str:
    """执行 ReAct 工具循环，返回最终答复文本（供写 result.md）。

    Raises:
        ExecutorCancelled: 命中取消标记。
        RuntimeError: 空输出 / 步数耗尽 / deadline 不足。
        TimeoutError: 模型调用超过剩余预算。
    """
    cancel_check = lambda: store.cancel_requested(task_id)
    for step in range(1, config.max_steps + 1):
        store.update(task_id, steps_used=step)

        # 取消检查点 1：每步开始
        if cancel_check():
            raise ExecutorCancelled("用户取消")

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"总预算耗尽（timeout_seconds={config.timeout_seconds}s）")
        if remaining < EXECUTOR_MIN_STEP_SECONDS:
            raise RuntimeError(
                f"剩余预算不足（{remaining:.0f}s < {EXECUTOR_MIN_STEP_SECONDS:.0f}s），"
                "不再发起可能注定超时的新一步"
            )

        response = await _await_with_cancel(
            bound.ainvoke(messages),
            timeout=remaining,
            cancel_check=cancel_check,
            poll_seconds=2.0,
        )

        tool_calls = list(response.tool_calls or [])
        invalid_calls = list(response.invalid_tool_calls or [])
        content = stringify_content(response.content).strip()

        # 最终答复：无工具调用且有实质内容
        if not tool_calls and not invalid_calls:
            empty_reason = final_answer_is_empty(response)
            if empty_reason is not None:
                raise RuntimeError(f"LLM 输出为空: {empty_reason}")
            if not content:
                raise RuntimeError("LLM 输出为空")
            return content

        messages.append(response)

        # 非法工具调用（schema 校验失败）也必须配对回一条 error ToolMessage
        for bad in invalid_calls:
            call_id = str(bad.get("id") or "unknown")
            messages.append(ToolMessage(
                content=f"[错误] 工具调用参数非法: {bad.get('error')}",
                tool_call_id=call_id,
                status="error",
            ))

        # 取消检查点 2：执行工具前（本轮调用在 LLM 调用内已被轮询覆盖）
        if cancel_check():
            raise ExecutorCancelled("用户取消")

        messages.extend(_execute_tool_calls(
            tool_calls,
            registry,
            max_observation_chars=config.max_observation_chars,
        ))

    raise RuntimeError(f"步数耗尽（max_steps={config.max_steps}），任务未收敛")


async def run_task(
    task_id: str, *, store: ExecutorStore | None = None,
    config: ExecutorConfig | None = None, model: BaseChatModel | None = None,
) -> int:
    """执行一次多步任务执行（ReAct 工具循环）。

    Returns: 返回码 0=完成/取消，1=失败，2=任务不存在。
    """
    store = store or ExecutorStore()
    config = config or load_executor_config()

    task = store.get(task_id)
    if task is None:
        logger.error("任务不存在: %s", task_id)
        return 2

    # 认领前的取消短路：仅对未终结任务生效，避免改写终态任务。
    status = task.get("status")
    if store.cancel_requested(task_id) and status in (STATUS_QUEUED, STATUS_RUNNING):
        _mark_terminal(store, task_id, STATUS_CANCELLED, "用户取消")
        return 0

    if not store.claim(task_id, os.getpid()):
        logger.warning("任务已被其他进程认领或不可认领: %s", task_id)
        return 0

    # 工作区并发互斥：同一 workspace 同时只允许一个 executor worker
    locked_workspace: str | None = None
    try:
        # 先判空再归一化：normalize_path("") 会返回当前工作目录（abspath("")==cwd），
        # 必须在归一化前拦截，否则缺 workspace 的任务会在项目根静默执行
        workspace_raw = str(task.get("workspace") or "").strip()
        if not workspace_raw:
            raise ValueError("任务缺少有效 workspace: task.json 中 workspace 为空")
        workspace = normalize_path(workspace_raw)
        if not os.path.isdir(workspace):
            raise ValueError(f"任务缺少有效 workspace: {workspace_raw!r}")
        if not acquire_workspace_lock(workspace):
            raise RuntimeError(
                f"工作区正被其他 executor 任务占用: {workspace}"
            )
        locked_workspace = workspace

        system_prompt, template = load_prompt_sections(config.agent_prompt_file)

        if model is None:
            provider, model_name = resolve_llm_settings(config)
            store.update(task_id, provider=provider, model=model_name)
            client = LLMClient(
                provider=provider, model=model_name,
                temperature=config.temperature, max_tokens=config.max_tokens,
            )
            model = client.get_chat_model()

        task = store.get(task_id) or task
        user_prompt = render_placeholders(template, {
            "{task}": str(task.get("task") or ""),
            "{plan}": str(task.get("plan") or "") or "（无）",
            "{context}": str(task.get("context") or "") or "（无）",
            "{workspace}": workspace,
        })

        # 审计产物：把实际投喂的提示词落盘（不是记忆）
        store.write_text(task_id, PROMPT_FILE, f"{system_prompt}\n\n---\n\n{user_prompt}")

        tools = build_executor_tools(workspace, config)
        registry = {tool.name: tool for tool in tools}
        bound = model.bind_tools(tools)

        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        deadline = time.monotonic() + config.timeout_seconds

        final_text = await _arun_loop(
            bound=bound, messages=messages, registry=registry,
            store=store, task_id=task_id, config=config, deadline=deadline,
        )

        # 取消检查点 3：写 result.md 之前（收尾不应被竞态取消截断）
        if store.cancel_requested(task_id):
            raise ExecutorCancelled("用户取消")

        marker_present = RESULT_MARKER in final_text
        store.write_text(task_id, FINAL_RESULT_FILE, final_text)
        store.update(
            task_id,
            status=STATUS_DONE,
            finished_at=_now_iso(),
            has_result=True,
            marker_present=marker_present,
            error=None,
        )
        logger.info("任务完成: %s", task_id)
        return 0
    except ExecutorCancelled:
        # 取消不写 result.md；工作区已写入的文件不回滚（语义已在消息/docstring 声明）
        _mark_terminal(store, task_id, STATUS_CANCELLED, "用户取消")
        return 0
    except BaseException as exc:
        logger.error("executor 任务失败: %s\n%s", task_id, traceback.format_exc())
        _mark_terminal(store, task_id, STATUS_FAILED, f"{type(exc).__name__}: {exc}")
        return 1
    finally:
        if locked_workspace is not None:
            release_workspace_lock(locked_workspace)
        store.release(task_id)


def _make_trace_context(task_id: str) -> TraceContext | None:
    """尝试构造 trace 上下文；结构化日志不可用时返回 ``None``（惰性可选导入）。"""
    try:
        from utils.logging_config import TraceContext
    except ImportError:
        return None
    return TraceContext(trace_id=task_id)


def _setup_logging_quietly() -> None:
    """初始化结构化日志；不可用时回退到基础配置并记警告。"""
    try:
        from utils.logging_config import setup_logging

        setup_logging(level=logging.INFO)
    except Exception as exc:
        logging.basicConfig(level=logging.INFO)
        logger.warning("结构化日志初始化失败，回退基础配置: %s", exc)


def main(argv: list[str] | None = None) -> int:
    """命令行入口：解析 ``--task-id`` 并在事件循环中执行任务（返回码见 run_task）。"""
    _setup_logging_quietly()

    parser = argparse.ArgumentParser(description="Executor 任务执行子程序 worker（多步工具循环）")
    parser.add_argument("--task-id", required=True, help="任务 ID")
    args = parser.parse_args(argv)
    task_id: str = args.task_id

    trace_context = _make_trace_context(task_id)
    if trace_context is None:
        return asyncio.run(run_task(task_id))
    with trace_context:
        return asyncio.run(run_task(task_id))


if __name__ == "__main__":
    raise SystemExit(main())
