"""迁移脚本：将旧 memory.json 长期记忆导入 LangGraph Store。

旧格式（memory.json）：
    [{"role": "user", "content": "...", "timestamp": "...", "metadata": {"important": true}}]

新格式（LangGraph Store）：
    namespace: (thread_id, "thread_facts")
    key: fact_id (uuid)
    value: ThreadFactItem.to_dict()

用法：
    # 预览模式（不写入，仅打印将迁移的内容）
    python scripts/migrate_memory_to_store.py --dry-run \\
        --memory-json /path/to/memory.json \\
        --checkpoint-file /path/to/checkpoint.db

    # 执行迁移
    python scripts/migrate_memory_to_store.py \\
        --memory-json /path/to/memory.json \\
        --checkpoint-file /path/to/checkpoint.db \\
        --thread-id migrated-from-json

    # 指定多个 thread_id（将记忆分配到不同会话）
    python scripts/migrate_memory_to_store.py \\
        --memory-json /path/to/memory.json \\
        --checkpoint-file /path/to/checkpoint.db \\
        --thread-id session-A --thread-id session-B

注意：
    - 迁移后旧 memory.json 不会被删除，需手动确认后删除
    - 默认 thread_id 为 "migrated-from-json"，所有旧记忆归入同一会话
    - 如需按内容自动分配 thread_id，请扩展 _classify_thread_id 函数
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any

# 将项目根目录加入 sys.path（脚本可独立运行）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import aiosqlite
from langgraph.store.sqlite.aio import AsyncSqliteStore

from agent.memory_models import MemoryCategory, ThreadFactItem

logger = logging.getLogger("migrate_memory")


def _naive_now() -> datetime:
    return datetime.now()  # noqa: DTZ005


def _classify_category(item: dict[str, Any]) -> str:
    """根据旧 memory.json 条目的 metadata 推断 MemoryCategory。

    旧格式只有 important=True/False 二值，需要启发式映射：
    - metadata.type == "summary" → IMPORTANT_CONVERSATION（压缩摘要）
    - important == True → IMPORTANT_CONVERSATION（用户显式标记）
    - 其余 → BUSINESS_ENTITY（保守保留）
    """
    metadata = item.get("metadata") or {}
    if metadata.get("type") == "summary":
        return MemoryCategory.IMPORTANT_CONVERSATION.value
    if metadata.get("important"):
        return MemoryCategory.IMPORTANT_CONVERSATION.value
    return MemoryCategory.BUSINESS_ENTITY.value


def _classify_thread_id(item: dict[str, Any], default_thread_id: str) -> str:
    """根据条目内容推断目标 thread_id。

    旧 memory.json 没有线程归属信息，默认全部归入同一会话。
    如需按内容分配，可在此函数中实现自定义逻辑（如按关键词匹配）。

    TODO: 可扩展为按 content 关键词自动分配到不同会话
    """
    return default_thread_id


def convert_item(
    item: dict[str, Any], thread_id: str
) -> ThreadFactItem:
    """将旧 memory.json 条目转换为 ThreadFactItem。

    Args:
        item: 旧格式条目 ``{"role": ..., "content": ..., "timestamp": ..., "metadata": ...}``
        thread_id: 目标会话线程 ID

    Returns:
        ThreadFactItem 实例
    """
    content = item.get("content", "")
    role = item.get("role", "unknown")
    timestamp = item.get("timestamp", _naive_now().isoformat())
    metadata = item.get("metadata") or {}

    # 如果是摘要条目，保留原内容
    if metadata.get("type") == "summary":
        text = content
    else:
        text = f"[{role}] {content}"

    return ThreadFactItem(
        fact_id=uuid.uuid4().hex,
        thread_id=thread_id,
        content=text,
        category=_classify_category(item),
        confidence=0.8,
        create_time=timestamp,
        last_used_at=timestamp,
    )


async def migrate(
    memory_json_path: str,
    checkpoint_file: str,
    thread_ids: list[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """执行迁移。

    Args:
        memory_json_path: 旧 memory.json 文件路径
        checkpoint_file: SQLite 文件路径（Store 数据将写入此文件）
        thread_ids: 目标 thread_id 列表（记忆将轮询分配到这些会话）
        dry_run: 为 True 时仅预览不写入

    Returns:
        迁移统计 ``{"total": int, "migrated": int, "skipped": int, "per_thread": dict}``
    """
    # 1. 读取旧 memory.json
    if not os.path.exists(memory_json_path):
        logger.error("memory.json 不存在: %s", memory_json_path)
        return {"total": 0, "migrated": 0, "skipped": 0, "per_thread": {}, "error": "file not found"}

    with open(memory_json_path, "r", encoding="utf-8") as f:
        old_items: list[dict[str, Any]] = json.load(f)

    logger.info("读取旧记忆 %d 条 from %s", len(old_items), memory_json_path)

    # 2. 转换为 ThreadFactItem
    default_tid = thread_ids[0] if thread_ids else "migrated-from-json"
    facts_by_thread: dict[str, list[ThreadFactItem]] = {}

    for idx, item in enumerate(old_items):
        tid = _classify_thread_id(item, default_tid)
        # 如果有多个 thread_id，轮询分配
        if len(thread_ids) > 1:
            tid = thread_ids[idx % len(thread_ids)]

        fact = convert_item(item, tid)
        facts_by_thread.setdefault(tid, []).append(fact)

    # 3. 预览模式
    if dry_run:
        print("\n=== 预览模式（不写入）===")
        for tid, facts in facts_by_thread.items():
            print(f"\n[thread={tid}] {len(facts)} 条:")
            for fact in facts:
                print(f"  [{fact.category}] {fact.content[:80]}...")
        return {
            "total": len(old_items),
            "migrated": 0,
            "skipped": 0,
            "per_thread": {tid: len(facts) for tid, facts in facts_by_thread.items()},
            "dry_run": True,
        }

    # 4. 初始化 AsyncSqliteStore
    parent = os.path.dirname(os.path.abspath(checkpoint_file))
    if parent:
        os.makedirs(parent, exist_ok=True)

    conn = await aiosqlite.connect(checkpoint_file)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=10000")
    store = AsyncSqliteStore(conn)
    await store.setup()
    await conn.commit()  # 确保 setup 建表事务已提交

    # 5. 写入 Store
    ns_key = "thread_facts"
    migrated = 0
    skipped = 0

    try:
        for tid, facts in facts_by_thread.items():
            ns = (tid, ns_key)
            for fact in facts:
                if not fact.content.strip():
                    skipped += 1
                    continue
                await store.aput(ns, key=fact.fact_id, value=fact.to_dict())
                migrated += 1
            logger.info("thread=%s: 写入 %d 条", tid, len(facts))
    finally:
        await conn.close()

    logger.info("迁移完成: %d 条写入, %d 条跳过", migrated, skipped)
    return {
        "total": len(old_items),
        "migrated": migrated,
        "skipped": skipped,
        "per_thread": {tid: len(facts) for tid, facts in facts_by_thread.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将旧 memory.json 长期记忆迁移到 LangGraph Store"
    )
    parser.add_argument(
        "--memory-json",
        required=True,
        help="旧 memory.json 文件路径",
    )
    parser.add_argument(
        "--checkpoint-file",
        required=True,
        help="SQLite 文件路径（Store 数据将写入此文件）",
    )
    parser.add_argument(
        "--thread-id",
        action="append",
        default=None,
        help="目标 thread_id（可多次指定，记忆将轮询分配）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式：仅打印将迁移的内容，不写入",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    thread_ids = args.thread_id or ["migrated-from-json"]

    result = asyncio.run(
        migrate(
            memory_json_path=args.memory_json,
            checkpoint_file=args.checkpoint_file,
            thread_ids=thread_ids,
            dry_run=args.dry_run,
        )
    )

    print("\n=== 迁移结果 ===")
    for key, value in result.items():
        print(f"  {key}: {value}")

    if result.get("dry_run"):
        print("\n（预览模式，未实际写入。去掉 --dry-run 执行迁移。）")
    elif result.get("migrated", 0) > 0:
        print(f"\n迁移成功！{result['migrated']} 条记忆已写入 Store。")
        print("旧 memory.json 文件保留，确认无误后可手动删除。")


if __name__ == "__main__":
    main()
