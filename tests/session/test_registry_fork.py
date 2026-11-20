"""会话分叉（afork_session）与展示名（title）测试。

覆盖：
- title 读写与清除语义（aset_title / aget_title）
- afork_session 完整复制：checkpoint 历史、session 配置、workspace、title、执行历史
- 分叉不影响源会话
- 工作流会话分叉后新 ID 保留 workflow 前缀
- 源会话不存在时返回 None
- 内存模式（MemorySaver）兜底路径（仅复制最新快照）

运行：
  pytest tests/session/test_registry_fork.py -v
"""
import asyncio
import os
import tempfile
import uuid
from datetime import UTC, datetime

import aiosqlite
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from session import SessionConfig, SessionRegistry, SessionStore

# --------------------------------------------------------------------------- #
# 辅助
# --------------------------------------------------------------------------- #


def _checkpoint(messages, checkpoint_id, parent_id=None):
    """构造最小可用 checkpoint dict（与 api 测试的写入结构一致）。"""
    ckpt = {
        "id": checkpoint_id,
        "ts": datetime.now(UTC).isoformat(),
        "channel_values": {"messages": list(messages)},
        "channel_versions": {"messages": "1"},
    }
    if parent_id is not None:
        ckpt["parent_checkpoint_id"] = parent_id
    return ckpt


async def _make_sqlite_registry():
    """建 SQLite 模式的 SessionRegistry（真实 SQL 复制路径）。"""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = await aiosqlite.connect(path)
    await conn.execute("PRAGMA journal_mode=WAL")
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    store = SessionStore()
    registry = SessionRegistry(
        saver, store, process_type="server", async_conn=conn
    )
    return registry, conn, path


# --------------------------------------------------------------------------- #
# title 读写
# --------------------------------------------------------------------------- #


def test_title_set_get_and_clear():
    store = SessionStore()
    registry = SessionRegistry(MemorySaver(), store, process_type="server")

    async def run():
        sid = registry.generate_session_id()
        assert await registry.aget_title(sid) is None
        # 设置（自动 strip）
        effective = await registry.aset_title(sid, "  我的会话  ")
        assert effective == "我的会话"
        assert await registry.aget_title(sid) == "我的会话"
        # 空白清除 → 回落 None
        effective = await registry.aset_title(sid, "   ")
        assert effective is None
        assert await registry.aget_title(sid) is None

    asyncio.run(run())


def test_title_batch_read_and_cleanup_on_delete():
    store = SessionStore()
    registry = SessionRegistry(MemorySaver(), store, process_type="server")

    async def run():
        s1, s2 = registry.generate_session_id(), registry.generate_session_id()
        await registry.aset_title(s1, "会话一")
        titles = await registry.aget_titles([s1, s2])
        assert titles == {s1: "会话一"}
        # 会话删除时 title 一并清理
        await store.adelete_session(s1)
        assert await registry.aget_title(s1) is None

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# afork_session：SQLite 全量复制
# --------------------------------------------------------------------------- #


def test_fork_copies_checkpoint_history_and_messages():
    """分叉复制完整 checkpoint 历史（含祖先链）与消息内容。"""

    async def run():
        registry, conn, path = await _make_sqlite_registry()
        try:
            sid = "server-thread-src00001"
            msgs1 = [HumanMessage(content="第一轮问题")]
            msgs2 = [HumanMessage(content="第一轮问题"), HumanMessage(content="第二轮问题")]
            await registry.checkpointer.aput(
                {"configurable": {"thread_id": sid, "checkpoint_ns": ""}},
                _checkpoint(msgs1, "ckpt-1"),
                {},
                {},
            )
            await registry.checkpointer.aput(
                {"configurable": {"thread_id": sid, "checkpoint_ns": ""}},
                _checkpoint(msgs2, "ckpt-2", parent_id="ckpt-1"),
                {},
                {},
            )

            new_sid = await registry.afork_session(sid)
            assert new_sid is not None
            assert new_sid != sid
            assert new_sid.startswith("server-thread-")

            # 新会话消息与源一致（读最新 checkpoint）
            fork_msgs = await registry.aget_messages(new_sid)
            src_msgs = await registry.aget_messages(sid)
            assert [m.content for m in fork_msgs] == [m.content for m in src_msgs]
            assert fork_msgs[-1].content == "第二轮问题"

            # 完整祖先链复制：checkpoints 表中新会话有 2 行
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?", (new_sid,)
            )
            count = (await cursor.fetchone())[0]
            await cursor.close()
            assert count == 2

            # 源会话不受影响
            assert len(await registry.aget_messages(sid)) == 2
        finally:
            await conn.close()
            try:
                os.unlink(path)
            except OSError:
                pass

    asyncio.run(run())


def test_fork_copies_config_workspace_title_and_history():
    """分叉复制 session 配置、workspace 绑定、展示名与执行历史。"""

    async def run():
        registry, conn, path = await _make_sqlite_registry()
        try:
            sid = "server-thread-src00002"
            workspace = os.path.dirname(os.path.abspath(path))
            await registry.checkpointer.aput(
                {"configurable": {"thread_id": sid, "checkpoint_ns": ""}},
                _checkpoint([HumanMessage(content="你好")], "ckpt-a"),
                {},
                {},
            )
            await registry.aset_session_config(
                sid, SessionConfig(provider="zhipu", model="glm-4-flash")
            )
            await registry.aset_workspace(workspace, session_id=sid)
            await registry.aset_title(sid, "重要会话")
            await registry._store.aappend_history(sid, {"step": 1, "tool": "read"})
            await registry._store.aadd_recorded_call_ids(sid, {"call-x"})

            new_sid = await registry.afork_session(sid)
            assert new_sid is not None

            # session 配置复制
            cfg = await registry.aget_session_config(new_sid)
            assert cfg is not None
            assert cfg.provider == "zhipu"
            assert cfg.model == "glm-4-flash"

            # workspace 绑定复制
            assert await registry.aget_workspace(new_sid) == workspace

            # 展示名复制
            assert await registry.aget_title(new_sid) == "重要会话"

            # 执行历史与去重集合复制
            history = await registry._store.aget_history(new_sid)
            assert history == [{"step": 1, "tool": "read"}]
            assert await registry._store.aget_recorded_call_ids(new_sid) == {"call-x"}
        finally:
            await conn.close()
            try:
                os.unlink(path)
            except OSError:
                pass

    asyncio.run(run())


def test_fork_missing_session_returns_none():
    store = SessionStore()

    async def run():
        registry, conn, path = await _make_sqlite_registry()
        try:
            assert await registry.afork_session("server-thread-notexist") is None
        finally:
            await conn.close()
            try:
                os.unlink(path)
            except OSError:
                pass
        # 内存 registry 同样返回 None
        mem = SessionRegistry(MemorySaver(), store)
        assert await mem.afork_session("thread-notexist") is None

    asyncio.run(run())


def test_fork_workflow_session_keeps_workflow_prefix():
    """工作流会话分叉后新 ID 仍带 workflow-{name} 前缀。"""

    async def run():
        registry, conn, path = await _make_sqlite_registry()
        try:
            sid = "server-workflow-simple-thread-src0003"
            await registry.checkpointer.aput(
                {"configurable": {"thread_id": sid, "checkpoint_ns": ""}},
                _checkpoint([HumanMessage(content="跑工作流")], "ckpt-w"),
                {},
                {},
            )
            new_sid = await registry.afork_session(sid)
            assert new_sid is not None
            assert new_sid.startswith("server-workflow-simple-thread-")
            assert registry.is_workflow_session(new_sid)
            assert registry.workflow_name_of(new_sid) == "simple"
        finally:
            await conn.close()
            try:
                os.unlink(path)
            except OSError:
                pass

    asyncio.run(run())


# --------------------------------------------------------------------------- #
# afork_session：内存模式兜底（MemorySaver）
# --------------------------------------------------------------------------- #


def test_fork_memory_saver_fallback():
    """内存模式（async_conn=None）通过 checkpointer API 复制最新快照。"""
    store = SessionStore()
    saver = MemorySaver()
    registry = SessionRegistry(saver, store, process_type="server")

    async def run():
        sid = "server-thread-mem00001"
        # new_versions 需列出写入的 channel，否则 MemorySaver 不落盘消息内容
        await saver.aput(
            {"configurable": {"thread_id": sid, "checkpoint_ns": ""}},
            _checkpoint([HumanMessage(content="内存会话")], f"ckpt-{uuid.uuid4()}"),
            {},
            {"messages": "1"},
        )
        new_sid = await registry.afork_session(sid)
        assert new_sid is not None
        # 直接经 checkpointer 验证快照复制（MemorySaver 的 get_tuple 要求显式
        # checkpoint_ns，registry.aget_messages 的 config 不带该键，不适用此处）
        tup = await saver.aget_tuple(
            {"configurable": {"thread_id": new_sid, "checkpoint_ns": ""}}
        )
        assert tup is not None
        msgs = tup.checkpoint["channel_values"]["messages"]
        assert len(msgs) == 1
        assert msgs[0].content == "内存会话"

    asyncio.run(run())
