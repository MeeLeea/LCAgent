"""Planner 子程序的一次性（one-shot）对话运行时。

- 一次性唤醒：读取任务 → 拼 system+user 提示 → 调用一次 LLM → 落 ``plan.md`` 后退出，
  无阶段循环、无逐阶段落盘。
- 不做记忆/checkpoint：任务一次唤醒即完结、进程用完即退，无跨唤醒延续需求，引入记忆或
  checkpoint 只增负担无收益，故刻意不做。
- 终态保证：``try/except BaseException/finally`` 包裹，正常落 ``done``、取消落
  ``cancelled``、异常落 ``failed`` 并写 ``error``，绝不留 ``running`` 僵尸。
- 取消语义：取消经 ``cancel.flag`` 表达，LLM 调用期间按 ``poll_seconds`` 轮询，命中抛
  :class:`PlannerCancelled` 落 ``cancelled``；一次性设计下取消不写 ``plan.md``。

提示词解析契约（:func:`load_prompt_sections`）：文件被唯一一个 ``## workflow:`` 小节切成
两半——其上为系统提示词，其下为该小节模板正文（从该行下一行起直到下一个 ``## `` 二级
标题（不含）；模板内 ``###`` 不截断）。提示词与模板任一为空即抛 ``ValueError``。

进程入口：``python -m subagent.planner.worker --task-id <id>``。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import traceback
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from llm.llm_client import LLMClient
from llm.message_utils import final_answer_is_empty, stringify_content

from .config import PlannerConfig, resolve_llm_settings
from .store import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    PlannerStore,
)

if TYPE_CHECKING:
    from utils.logging_config import TraceContext

logger = logging.getLogger(__name__)

# 审计产物：本次实际拼装的提示词（非记忆，仅用于事后排查）
PROMPT_FILE: str = "prompt.md"
# 最终方案文档（store.read_result 固定读取此名）
FINAL_PLAN_FILE: str = "plan.md"

__all__ = ["PlannerCancelled", "load_prompt_sections", "main", "run_task"]


class PlannerCancelled(Exception):
    """用户请求取消时在 LLM 调用等待中途抛出，用于快速跳出。"""


def _now_iso() -> str:
    """返回带本地时区信息的 ISO 时间戳（秒级精度）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_prompt_sections(prompt_file: str) -> tuple[str, str]:
    """解析提示词文件，返回 ``(系统提示词, 设计模板)``（契约见模块文档）。

    Raises: ValueError（文件缺失/为空、缺 ``## workflow:`` 小节、或任一侧为空）。
    """
    try:
        with open(prompt_file, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        raise ValueError(f"规划者提示词文件缺失或为空: {prompt_file}") from None
    if not text.strip():
        raise ValueError(f"规划者提示词文件缺失或为空: {prompt_file}")

    lines = text.splitlines()
    workflow_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip().startswith("## workflow:"):
            workflow_index = index
            break
    if workflow_index is None:
        raise ValueError(f"提示词文件缺少 ## workflow: 小节: {prompt_file}")

    system_prompt = "\n".join(lines[:workflow_index]).strip()
    template_lines: list[str] = []
    for line in lines[workflow_index + 1:]:
        if line.strip().startswith("## "):
            break
        template_lines.append(line)
    design_template = "\n".join(template_lines).strip()

    if not system_prompt:
        raise ValueError(f"提示词文件在 ## workflow: 之前缺少系统提示词: {prompt_file}")
    if not design_template:
        raise ValueError(f"提示词文件的 ## workflow: 小节缺少模板内容: {prompt_file}")
    return system_prompt, design_template


async def _await_with_cancel(
    awaitable: Awaitable[Any], *, timeout: float,
    cancel_check: Callable[[], bool], poll_seconds: float,
) -> Any:
    """在可取消、可超时的前提下等待协程完成，不泄漏挂起任务。

    Raises: PlannerCancelled（命中取消）/ TimeoutError（超过 ``timeout``）。
    """
    work = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    effective_poll = poll_seconds if poll_seconds > 0 else 0.5
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"LLM 调用超时（{timeout}s）")
            if cancel_check():
                raise PlannerCancelled("用户取消")
            done, _pending = await asyncio.wait({work}, timeout=min(effective_poll, remaining))
            if work in done:
                return work.result()
    finally:
        if not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)


def _mark_terminal(
    store: PlannerStore, task_id: str, status: str, error: str | None, *, has_result: bool = False
) -> None:
    """尽力写入终态；写失败只记日志，不掩盖原始异常。"""
    now = _now_iso()
    try:
        store.update(task_id, status=status, finished_at=now, has_result=has_result, error=error)
    except BaseException as exc:
        logger.error("写入任务状态失败 (%s): %s", task_id, exc)


async def run_task(
    task_id: str, *, store: PlannerStore | None = None,
    config: PlannerConfig | None = None, model: BaseChatModel | None = None,
) -> int:
    """执行一次一次性方案设计任务。

    Returns: 返回码 0=完成/取消/跳过，1=失败，2=任务不存在。
    """
    store = store or PlannerStore()

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

    try:
        config = config or PlannerConfig.load()
        system_prompt, template = load_prompt_sections(config.agent_prompt_file)

        if model is None:
            provider, model_name = resolve_llm_settings(config)
            store.update(task_id, provider=provider, model=model_name)
            client = LLMClient(
                provider=provider, model=model_name,
                temperature=config.temperature, max_tokens=config.max_tokens,
            )
            model = client.get_chat_model()

        requirement = str(task.get("requirement") or "")
        context = str(task.get("context") or "")
        user_prompt = template.replace("{requirement}", requirement)
        user_prompt = user_prompt.replace("{context}", context or "（无）")

        # 审计产物：把实际投喂的提示词落盘，便于事后复盘（不是记忆）。
        store.write_text(task_id, PROMPT_FILE, f"{system_prompt}\n\n---\n\n{user_prompt}")

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        bind = getattr(model, "bind", None)
        bound = bind(max_tokens=config.max_tokens) if callable(bind) else model

        response = await _await_with_cancel(
            bound.ainvoke(messages), timeout=config.timeout_seconds,
            cancel_check=lambda: store.cancel_requested(task_id), poll_seconds=2.0,
        )

        empty_reason = final_answer_is_empty(response)
        if empty_reason is not None:
            raise RuntimeError(f"LLM 输出为空: {empty_reason}")

        text = stringify_content(response.content).strip()
        if not text:
            raise RuntimeError("LLM 输出为空")

        store.write_text(task_id, FINAL_PLAN_FILE, text)
        now = _now_iso()
        store.update(task_id, status=STATUS_DONE, finished_at=now, has_result=True, error=None)
        logger.info("任务完成: %s", task_id)
        return 0
    except PlannerCancelled:
        # 一次性设计下没有部分方案：取消即不写 plan.md。
        _mark_terminal(store, task_id, STATUS_CANCELLED, "用户取消")
        return 0
    except BaseException as exc:
        logger.error("planner 任务失败: %s\n%s", task_id, traceback.format_exc())
        _mark_terminal(store, task_id, STATUS_FAILED, f"{type(exc).__name__}: {exc}")
        return 1
    finally:
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

    parser = argparse.ArgumentParser(description="Planner 方案设计子程序 worker（一次性）")
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
