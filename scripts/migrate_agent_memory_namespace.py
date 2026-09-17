"""一次性数据迁移脚本：agent 级长期记忆 namespace 迁移。

背景
----
项目把长期记忆的 ``agent_key`` 与 ``process_type`` 解耦后，agent 级长期记忆
（``user_fact`` / ``lesson``）的共享 namespace 由旧的 ``("server", "global_facts")``
改为新的 ``("global", "global_facts")``。本脚本把旧 namespace 下的全部 facts
复制到新 namespace，保证历史记忆不丢失。

实现要点
--------
- 复用项目自己的 LangGraph ``AsyncSqliteStore``（构造方式与 ``memory/agent_memory.py``
  完全一致：独立 ``aiosqlite`` 连接 + WAL + ``busy_timeout=10000``），不直接对
  ``store`` 表执行 raw SQL，因此遵循 WAL 与 Store 的序列化约定。
- 复用 ``memory.models.ThreadFactItem`` 的 ``from_dict`` / ``to_dict`` 做序列化，
  不手工拼 JSON。
- 目标条目沿用源的 ``fact_id`` 作为 Store key（重复运行按 key 幂等），并把条目内
  的 ``thread_id`` 改写为 ``"global"``，与新 ``agent_key`` 保持溯源一致；其余字段
  （content / category / confidence / scope / create_time / last_used_at）原样保留。
- 默认 dry-run（只读），只有显式 ``--apply`` 才会写库；``--delete-source`` 仅在
  ``--apply`` 下生效，且必须在复制校验通过后才会删除源数据。

用法
----
    # 默认 dry-run：只报告将要执行的操作，不写入任何数据
    uv run python scripts/migrate_agent_memory_namespace.py --dry-run

    # 真正执行复制（需用户确认后手动执行）
    uv run python scripts/migrate_agent_memory_namespace.py --apply

    # 复制并校验通过后删除旧 namespace 数据
    uv run python scripts/migrate_agent_memory_namespace.py --apply --delete-source

    # 指定其它数据库文件
    uv run python scripts/migrate_agent_memory_namespace.py --db path/to/db.sqlite

安全性
------
- 可在 ``api/server.py`` 进程存活时运行：写入是幂等的、竞争窗口很短，
  SQLite 的 WAL 模式 + ``busy_timeout=10000`` 足以处理并发写。
- 迁移完成后，运行中的 ``api/server.py`` 进程会在下一次写入时开始写共享的
  ``global`` namespace（代码改动由另一个 agent 单独应用）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.store.base import SearchItem
from langgraph.store.sqlite.aio import AsyncSqliteStore

# 允许以 `uv run python scripts/migrate_agent_memory_namespace.py` 直接运行：
# 此时 sys.path[0] 是脚本所在目录，需手动补上项目根目录才能 import memory 包。
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from memory.models import ThreadFactItem

# 源 / 目标 namespace（LangGraph Store 会把 tuple 以点号拼接为 store 表的 prefix 列）
_SOURCE_NAMESPACE: tuple[str, ...] = ("server", "global_facts")
_DEST_NAMESPACE: tuple[str, ...] = ("global", "global_facts")

# 迁移后 agent 级记忆统一使用的 agent_key
_NEW_AGENT_KEY: str = "global"

# 默认数据库路径（相对项目根目录）
_DEFAULT_DB_RELATIVE: str = "data/checkpoints_async.sqlite"

# asearch 默认 limit=10，必须显式分页才能读到全量数据
_READ_BATCH_SIZE: int = 1000

# 退出码
_EXIT_OK: int = 0
_EXIT_FAILED: int = 1
_EXIT_USAGE: int = 2


@dataclass
class _MigrationPlan:
    """迁移计划：对比源 / 目标 namespace 后得出的待执行操作。

    Attributes:
        source_count: 源 namespace 条目总数
        dest_count_before: 迁移前目标 namespace 条目总数
        creates: 待新增条目 ``(key, value)``
        updates: 待覆盖更新条目 ``(key, value)``（目标已有同 key 但内容不同）
        unchanged: 目标已存在且内容完全一致的条目数（幂等跳过）
        source_keys: 源 namespace 的全部 key 集合
    """

    source_count: int
    dest_count_before: int
    creates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    updates: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    unchanged: int = 0
    source_keys: set[str] = field(default_factory=set)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "把 agent 级长期记忆从旧 namespace (server, global_facts) "
            "迁移到共享 namespace (global, global_facts)。默认 dry-run，不写入任何数据。"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正执行复制（默认仅 dry-run，不写入任何数据）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅报告将要执行的操作，不写入（默认行为；与 --apply 互斥）",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="复制并校验通过后删除旧 namespace 数据（仅在 --apply 下生效）",
    )
    parser.add_argument(
        "--db",
        default=_DEFAULT_DB_RELATIVE,
        help=f"SQLite 数据库路径，相对路径基于项目根目录（默认 {_DEFAULT_DB_RELATIVE}）",
    )
    return parser.parse_args(argv)


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

    与生产链路一致：独立 ``aiosqlite`` 连接 + WAL + ``busy_timeout=10000``，
    复用同一个 SQLite 文件（checkpoint 与长期记忆共库）。

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
    """分页读取指定 namespace 下的全部条目。

    ``asearch`` 默认 ``limit=10``，必须显式分页才能读到全量数据。
    """
    items: list[SearchItem] = []
    offset = 0
    while True:
        batch = await store.asearch(namespace, limit=_READ_BATCH_SIZE, offset=offset)
        items.extend(batch)
        if len(batch) < _READ_BATCH_SIZE:
            break
        offset += len(batch)
    return items


def _normalize_fact(item: SearchItem) -> ThreadFactItem:
    """把源条目转换为目标条目。

    - 复用源 Store key 作为 ``fact_id``（保证幂等与可追溯）
    - 把 ``thread_id`` 改写为新的 ``agent_key``（``"global"``）
    - 其余字段原样保留
    """
    fact = ThreadFactItem.from_dict(item.value)
    fact.fact_id = item.key
    fact.thread_id = _NEW_AGENT_KEY
    return fact


def _build_plan(
    source_items: list[SearchItem],
    dest_items: list[SearchItem],
) -> _MigrationPlan:
    """对比源 / 目标 namespace，生成迁移计划（只读，不写库）。"""
    dest_by_key: dict[str, dict[str, Any]] = {item.key: item.value for item in dest_items}
    plan = _MigrationPlan(
        source_count=len(source_items),
        dest_count_before=len(dest_items),
    )
    for item in source_items:
        plan.source_keys.add(item.key)
        target_value = _normalize_fact(item).to_dict()
        existing_value = dest_by_key.get(item.key)
        if existing_value is None:
            plan.creates.append((item.key, target_value))
        elif existing_value == target_value:
            plan.unchanged += 1
        else:
            plan.updates.append((item.key, target_value))
    return plan


async def _apply_plan(store: AsyncSqliteStore, plan: _MigrationPlan) -> None:
    """把迁移计划写入目标 namespace（按 key 幂等）。"""
    for key, value in plan.creates:
        await store.aput(_DEST_NAMESPACE, key=key, value=value)
    for key, value in plan.updates:
        await store.aput(_DEST_NAMESPACE, key=key, value=value)


async def _verify_destination(
    store: AsyncSqliteStore,
    expected_keys: set[str],
) -> tuple[bool, int, set[str]]:
    """重新读取目标 namespace，校验其 key 集合是源 key 集合的超集。

    Returns:
        ``(是否通过, 目标现有条数, 缺失的 key 集合)``
    """
    dest_items = await _read_namespace(store, _DEST_NAMESPACE)
    dest_keys = {item.key for item in dest_items}
    missing = expected_keys - dest_keys
    return not missing, len(dest_items), missing


async def _run_migration(args: argparse.Namespace) -> int:
    """执行迁移主流程，返回进程退出码。"""
    db_path = _resolve_db_path(args.db)
    do_delete = args.apply and args.delete_source

    print("=" * 60)
    print("agent 级长期记忆 namespace 迁移")
    print("=" * 60)
    print(f"数据库:         {db_path}")
    print(f"源 namespace:   {'.'.join(_SOURCE_NAMESPACE)}")
    print(f"目标 namespace: {'.'.join(_DEST_NAMESPACE)}")
    print(f"运行模式:       {'APPLY（真实写入）' if args.apply else 'DRY-RUN（只读，不写入）'}")
    print(f"删除源数据:     {'是' if do_delete else '否'}")
    print("-" * 60)

    if not db_path.exists():
        print(f"[错误] 数据库文件不存在: {db_path}")
        return _EXIT_USAGE

    if args.delete_source and not args.apply:
        print("[警告] --delete-source 仅在 --apply 下生效，本次不会删除任何源数据。")

    conn: aiosqlite.Connection | None = None
    try:
        store, conn = await _open_store(db_path)
        source_items = await _read_namespace(store, _SOURCE_NAMESPACE)
        dest_items = await _read_namespace(store, _DEST_NAMESPACE)
        plan = _build_plan(source_items, dest_items)

        print(f"源 namespace 条数:             {plan.source_count}")
        print(f"目标 namespace 条数（迁移前）: {plan.dest_count_before}")
        print(f"待新增:                        {len(plan.creates)} 条")
        print(f"待覆盖更新:                    {len(plan.updates)} 条")
        print(f"已存在且一致（跳过）:          {plan.unchanged} 条")

        if not args.apply:
            print("-" * 60)
            print("[DRY-RUN] 未写入任何数据。")
            if args.delete_source:
                print(
                    "[DRY-RUN] 若追加 --apply，将在校验通过后删除源 namespace 的 "
                    f"{plan.source_count} 条数据。"
                )
            print("[DRY-RUN] 实际执行请追加 --apply。")
            print("迁移结果: 成功（dry-run，无写入）")
            return _EXIT_OK

        print("-" * 60)
        await _apply_plan(store, plan)
        print(f"已新增: {len(plan.creates)} 条")
        print(f"已更新: {len(plan.updates)} 条")
        print(f"已跳过: {plan.unchanged} 条")

        passed, dest_count, missing = await _verify_destination(store, plan.source_keys)
        if not passed:
            print("-" * 60)
            print(
                f"[失败] 校验未通过：目标 namespace 现有 {dest_count} 条，"
                f"仍缺少 {len(missing)} 个源 fact_id。"
            )
            print("       目标 namespace 可能处于部分复制状态；本脚本幂等，可安全重跑修复。")
            print("迁移结果: 失败")
            return _EXIT_FAILED

        print("-" * 60)
        print(
            f"校验通过：目标 namespace 现有 {dest_count} 条，"
            f"已包含源全部 {len(plan.source_keys)} 个 fact_id。"
        )

        if do_delete:
            for key in sorted(plan.source_keys):
                await store.adelete(_SOURCE_NAMESPACE, key=key)
            print(f"源 namespace 删除: 已删除 {len(plan.source_keys)} 条")
        else:
            print("源 namespace 删除: 未启用（未指定 --delete-source）")

        print("迁移结果: 成功")
        return _EXIT_OK
    except Exception as error:
        # 顶层 CLI 边界：捕获一切异常转为中文报告 + 非零退出码
        print("-" * 60)
        print(f"[错误] 迁移失败: {type(error).__name__}: {error}")
        if args.apply:
            print("       若写入已开始，目标 namespace 可能处于部分复制状态；")
            print("       本脚本幂等（按 fact_id 覆盖 / 跳过），可安全重跑修复。")
        print("迁移结果: 失败")
        return _EXIT_FAILED
    finally:
        if conn is not None:
            await conn.close()


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数并在独立事件循环中执行迁移。"""
    args = _parse_args(argv)
    if args.apply and args.dry_run:
        print("[错误] 不能同时指定 --apply 和 --dry-run。")
        return _EXIT_USAGE
    return asyncio.run(_run_migration(args))


if __name__ == "__main__":
    raise SystemExit(main())
