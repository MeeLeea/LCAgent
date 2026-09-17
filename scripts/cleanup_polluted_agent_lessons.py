"""一次性清理脚本：删除 agent 级长期记忆中被污染的"经验教训"。

背景
----
历史 bug：危险命令确认（工具内部 ``interrupt()`` 抛 ``GraphInterrupt``）在事件流层
被误映射为 ``[工具执行失败]`` TOOL_RESULT，记忆流水线的确定性 lesson 路径
（同类失败 ≥2 次）又把错误原文**逐字**写入跨会话共享的 agent namespace，
导致完整命令脚本 + 反思指令后缀以"经验教训"的形式沉淀并注入所有会话的
SystemMessage。代码层已修复（事件流排除控制流信号 + lesson 内容改走 LLM 蒸馏），
本脚本负责清理存量污染数据。

判定标准
--------
agent 级 fact 同时满足以下条件即视为污染条目：
- ``category == "lesson"``
- content 含以下任一污染标记：``[工具执行失败]`` / ``请反思失败原因`` /
  ``GraphInterrupt`` / ``dangerous_command``

实现要点
--------
- 复用项目自身的 LangGraph ``AsyncSqliteStore``（构造方式与 ``memory/agent_memory.py``
  完全一致：独立 ``aiosqlite`` 连接 + WAL + ``busy_timeout=10000``），
  不直接对 ``store`` 表执行 raw SQL。
- **默认 dry-run（只读）**，只有显式 ``--apply`` 才会删除；删除不可逆。
- 只作用于 agent 级 namespace ``(agent_key, "global_facts")``（默认 ``global``），
  不触碰 thread 级 facts 与 checkpoint。

用法
----
    # 1. 预演（只读，第一步必须执行，确认待删条数与内容符合预期）
    uv run python scripts/cleanup_polluted_agent_lessons.py --dry-run

    # 2. 确认后真正删除（不可逆）
    uv run python scripts/cleanup_polluted_agent_lessons.py --apply

    # 指定其它数据库文件 / agent_key
    uv run python scripts/cleanup_polluted_agent_lessons.py --db path/to/db.sqlite
    uv run python scripts/cleanup_polluted_agent_lessons.py --agent-key global

安全性
------
- 建议在 ``api/server.py`` / CLI 进程停止时运行：删除后若有进程防抖 buffer 中
  仍残留旧的错误 TOOL_RESULT，下一次 flush 可能重新写回个别 lesson
  （代码修复后新写入的是蒸馏文本，不再是巨型原文，影响可控）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.store.base import SearchItem
from langgraph.store.sqlite.aio import AsyncSqliteStore

# 允许以 `uv run python scripts/cleanup_polluted_agent_lessons.py` 直接运行：
# 此时 sys.path[0] 是脚本所在目录，需手动补上项目根目录才能 import memory 包。
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memory.models import ThreadFactItem

# 污染标记：lesson 内容含任一标记即判定为逐字入库的错误原文（历史 bug 产物）
_POLLUTION_MARKERS: tuple[str, ...] = (
    "[工具执行失败]",
    "请反思失败原因",
    "GraphInterrupt",
    "dangerous_command",
)

# 默认数据库路径（相对项目根目录）与 agent 级 namespace 标识
_DEFAULT_DB_RELATIVE: str = "data/checkpoints_async.sqlite"
_DEFAULT_AGENT_KEY: str = "global"

# asearch 默认 limit=10，必须显式分页才能读到全量数据
_READ_BATCH_SIZE: int = 1000

# 退出码
_EXIT_OK: int = 0
_EXIT_FAILED: int = 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "删除 agent 级长期记忆中被污染的 lesson（逐字入库的工具错误原文）。"
            "默认 dry-run，不删除任何数据。"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正执行删除（默认仅 dry-run，不写入任何数据；删除不可逆）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅报告将要删除的条目，不删除（默认行为；与 --apply 互斥）",
    )
    parser.add_argument(
        "--db",
        default=_DEFAULT_DB_RELATIVE,
        help=f"SQLite 数据库路径，相对路径基于项目根目录（默认 {_DEFAULT_DB_RELATIVE}）",
    )
    parser.add_argument(
        "--agent-key",
        default=_DEFAULT_AGENT_KEY,
        help=f"agent 级 namespace 标识（默认 {_DEFAULT_AGENT_KEY}）",
    )
    args = parser.parse_args(argv)
    if args.apply and args.dry_run:
        parser.error("--apply 与 --dry-run 互斥")
    return args


def _resolve_db_path(raw: str) -> Path:
    """把 CLI 传入的数据库路径解析为绝对路径（相对路径基于项目根目录）。"""
    path = Path(raw)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path.resolve()


async def _open_store(
    db_path: Path,
) -> tuple[AsyncSqliteStore, aiosqlite.Connection]:
    """按 ``memory/agent_memory.py`` 的方式构造 ``AsyncSqliteStore``。

    Args:
        db_path: SQLite 数据库文件的绝对路径。

    Returns:
        ``(store, conn)``：调用方负责在 ``finally`` 中关闭 ``conn``。
    """
    conn = await aiosqlite.connect(str(db_path))
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=10000")
    store = AsyncSqliteStore(conn)
    await store.setup()
    await conn.commit()  # 确保 setup 建表事务已提交
    return store, conn


async def _read_namespace(
    store: AsyncSqliteStore,
    namespace: tuple[str, ...],
) -> list[SearchItem]:
    """分页读取指定 namespace 下的全部条目（asearch 默认 limit=10）。"""
    items: list[SearchItem] = []
    offset = 0
    while True:
        batch = await store.asearch(namespace, limit=_READ_BATCH_SIZE, offset=offset)
        items.extend(batch)
        if len(batch) < _READ_BATCH_SIZE:
            break
        offset += len(batch)
    return items


def _is_polluted(fact: ThreadFactItem) -> bool:
    """判断一条 fact 是否为被污染的 lesson（逐字入库的工具错误原文）。"""
    if fact.category != "lesson":
        return False
    return any(marker in fact.content for marker in _POLLUTION_MARKERS)


def _preview(content: str, width: int = 72) -> str:
    """单行截断预览（去换行，超长加省略号）。"""
    flat = " ".join(content.split())
    if len(flat) > width:
        return flat[:width] + "…"
    return flat


async def _run(args: argparse.Namespace) -> int:
    """主流程：读取 agent 级 facts → 筛出污染 lesson → dry-run 报告 / 删除。"""
    db_path = _resolve_db_path(args.db)
    if not db_path.exists():
        print(f"[错误] 数据库文件不存在: {db_path}")
        return _EXIT_FAILED

    namespace: tuple[str, ...] = (args.agent_key, "global_facts")
    store, conn = await _open_store(db_path)
    try:
        items = await _read_namespace(store, namespace)
        facts = [ThreadFactItem.from_dict(item.value) for item in items]
        # Store key 与 fact_id 一致（写入侧约定），删除按 key 执行
        polluted = [
            (item, fact)
            for item, fact in zip(items, facts, strict=True)
            if _is_polluted(fact)
        ]

        print(f"namespace: {namespace}")
        print(f"agent 级 fact 总数: {len(facts)}")
        print(f"命中污染 lesson : {len(polluted)}")
        total_chars = sum(len(fact.content) for _, fact in polluted)
        print(f"污染内容合计  : {total_chars} 字符\n")

        for idx, (_, fact) in enumerate(polluted, 1):
            print(
                f"[{idx}] {fact.fact_id[:8]}  {fact.create_time}  "
                f"{len(fact.content)}字符  {_preview(fact.content)}"
            )

        if not polluted:
            print("\n无需清理。")
            return _EXIT_OK

        if not args.apply:
            print(
                "\n[dry-run] 未删除任何数据。确认无误后执行: "
                "uv run python scripts/cleanup_polluted_agent_lessons.py --apply"
            )
            return _EXIT_OK

        deleted = 0
        for item, _ in polluted:
            await store.adelete(namespace, key=item.key)
            deleted += 1
        await conn.commit()

        remaining = await _read_namespace(store, namespace)
        leftover = sum(
            1
            for item in remaining
            if _is_polluted(ThreadFactItem.from_dict(item.value))
        )
        print(f"\n[apply] 已删除 {deleted} 条污染 lesson，剩余 agent 级 facts {len(remaining)} 条")
        result: dict[str, Any] = {"leftover": leftover}
        if leftover:
            print(f"[警告] 仍有 {result['leftover']} 条污染残留（可能有进程在并发写回）")
            return _EXIT_FAILED
        return _EXIT_OK
    finally:
        await conn.close()


def main(argv: list[str] | None = None) -> int:
    """入口：解析参数并执行清理。"""
    args = _parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
