"""结构化日志配置

提供 trace_id / thread_id 上下文注入和统一日志格式。

使用 contextvars 实现异步安全的 trace_id 传递：
- set_trace_context() 设置当前请求/会话的 trace_id 和 thread_id
- LogFormatter 自动在每条日志中附加这些上下文
- 未设置时显示 "-" 占位

用法:
    from utils.logging_config import setup_logging, TraceContext

    # 程序入口调用一次
    setup_logging(level=logging.INFO)

    # 在请求/会话入口注入上下文
    with TraceContext(trace_id="req-123", thread_id="thread-abc"):
        logger.info("processing")  # 自动附带 trace_id 和 thread_id
"""
from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from typing_extensions import Self

# ── 上下文变量（asyncio 安全） ──────────────────────────────────

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="-",
)
_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "thread_id", default="-",
)


# ── Formatter ──────────────────────────────────────────────────

class StructuredFormatter(logging.Formatter):
    """结构化日志格式器

    输出格式:
      2026-08-05 14:30:00 [INFO ] [agent.agent_core] [trace:abc123] [thread:t-1] 消息内容
    """

    def format(self, record: logging.LogRecord) -> str:
        # 注入上下文变量到 record，供 fmt 字符串引用
        record.trace_id = _trace_id.get()
        record.thread_id = _thread_id.get()
        return super().format(record)


# ── 公开 API ───────────────────────────────────────────────────

def generate_trace_id() -> str:
    """生成新的 trace_id（8 位 hex）"""
    return uuid.uuid4().hex[:8]


# ── 运行时日志级别 ─────────────────────────────────────────────

# 支持的级别名称 → logging 级别常量（大写规范化后查表）
LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def set_log_level(level: str | int) -> int:
    """运行时调整全局（root logger）日志级别

    无需重启即可在交互式 CLI 中切换日志粒度。仅改变级别，
    不触碰已有 handler / formatter 配置（区别于 setup_logging 的重建）。

    Args:
        level: 级别名称（不区分大小写，如 "debug"/"INFO"）或
               logging 整型常量（如 logging.DEBUG）。

    Returns:
        实际生效的 logging 整型级别。

    Raises:
        ValueError: 传入未知的级别名称。
        TypeError:  传入既非 str 也非 int 的类型。
    """
    if isinstance(level, str):
        name = level.strip().upper()
        if name not in LOG_LEVELS:
            valid = ", ".join(LOG_LEVELS)
            raise ValueError(f"未知日志级别 {level!r}，可选: {valid}")
        resolved = LOG_LEVELS[name]
    elif isinstance(level, int):
        resolved = level
    else:
        raise TypeError(f"level 必须为 str 或 int，收到 {type(level).__name__}")

    logging.getLogger().setLevel(resolved)
    return resolved


def get_log_level_name() -> str:
    """获取当前 root logger 的日志级别名称（如 "INFO"）"""
    return logging.getLevelName(logging.getLogger().level)


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    """初始化全局日志配置

    在程序入口（main.py / scheduler/run.py / remote 飞书入口）调用一次。
    幂等：重复调用会清除旧 handler 再重新配置。

    Args:
        level: 全局日志级别（默认 INFO）
        log_file: 日志文件路径（None=仅控制台输出到 stderr）
    """
    formatter = StructuredFormatter(
        fmt="%(asctime)s [%(levelname)-5s] [%(name)s] "
            "[trace:%(trace_id)s] [thread:%(thread_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # 清除已有 handlers（幂等，避免重复配置）
    root.handlers.clear()

    # 控制台 handler → stderr（不影响 stdout 上的工具输出和 CLI 交互）
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    # 文件 handler（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


class TraceContext:
    """上下文管理器：设置 trace_id/thread_id 并在退出时自动恢复

    支持 async with 和 with 两种用法（contextvars.ContextVar.reset
    在同步/异步上下文中均安全）。

    用法:
        with TraceContext(trace_id="req-123", thread_id="thread-abc"):
            logger.info("processing")

        # 也可以只设一个
        with TraceContext(thread_id="thread-xyz"):
            ...
    """

    def __init__(
        self,
        trace_id: str | None = None,
        thread_id: str | None = None,
        *,
        auto_generate_trace: bool = False,
    ):
        """
        Args:
            trace_id: 指定 trace_id（None 且 auto_generate_trace=True 时自动生成）
            thread_id: 指定 thread_id
            auto_generate_trace: trace_id 为 None 时是否自动生成
        """
        if trace_id is None and auto_generate_trace:
            trace_id = generate_trace_id()
        self.trace_id = trace_id
        self.thread_id = thread_id
        self._token_trace: contextvars.Token | None = None
        self._token_thread: contextvars.Token | None = None

    def __enter__(self) -> Self:
        if self.trace_id is not None:
            self._token_trace = _trace_id.set(self.trace_id)
        if self.thread_id is not None:
            self._token_thread = _thread_id.set(self.thread_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token_trace is not None:
            _trace_id.reset(self._token_trace)
        if self._token_thread is not None:
            _thread_id.reset(self._token_thread)

    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


__all__ = [
    "LOG_LEVELS",
    "StructuredFormatter",
    "TraceContext",
    "generate_trace_id",
    "get_log_level_name",
    "set_log_level",
    "setup_logging",
]
