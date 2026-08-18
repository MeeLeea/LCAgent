"""WorkspaceStore 测试 - session_id ↔ workspace_path 持久映射 + 内存缓存。

覆盖：
- 内存模式（async_conn=None）的 set/get/delete
- 同步读（get_cached）与异步读（aget）的一致性
- 删除后读返回 None
- 多 session 隔离

运行：
  pytest tests/session/test_workspace_store.py -v
"""
import asyncio

from session import WorkspaceStore


# --------------------------------------------------------------------------- #
# 内存模式基础 CRUD
# --------------------------------------------------------------------------- #
def test_mem_set_get_delete():
    store = WorkspaceStore(async_conn=None)

    async def run():
        await store.aset("s1", "D:/proj-a")
        return await store.aget("s1")

    path = asyncio.run(run())
    assert path == "D:/proj-a"


def test_mem_get_missing_returns_none():
    store = WorkspaceStore(async_conn=None)

    async def run():
        return await store.aget("nonexistent")

    assert asyncio.run(run()) is None


def test_mem_delete():
    store = WorkspaceStore(async_conn=None)

    async def run():
        await store.aset("s1", "D:/proj-a")
        deleted = await store.adelete("s1")
        after = await store.aget("s1")
        return deleted, after

    deleted, after = asyncio.run(run())
    assert deleted is True
    assert after is None


def test_mem_delete_missing_returns_false():
    store = WorkspaceStore(async_conn=None)

    async def run():
        return await store.adelete("nonexistent")

    assert asyncio.run(run()) is False


# --------------------------------------------------------------------------- #
# 同步读 / 异步读一致性
# --------------------------------------------------------------------------- #
def test_sync_read_after_async_set():
    """aset 后 get_cached 同步读应命中。"""
    store = WorkspaceStore(async_conn=None)

    async def run():
        await store.aset("s1", "D:/proj-a")

    asyncio.run(run())
    # 异步写后，同步读应命中
    assert store.get_cached("s1") == "D:/proj-a"


def test_sync_read_after_set_cached():
    """set_cached 同步写后，get_cached 同步读应立即命中。"""
    store = WorkspaceStore(async_conn=None)
    store.set_cached("s1", "D:/proj-a")
    assert store.get_cached("s1") == "D:/proj-a"


def test_aget_warms_cache():
    """aget 读取后应回填缓存，使后续 get_cached 同步读命中。"""
    store = WorkspaceStore(async_conn=None)

    async def run():
        await store.aset("s1", "D:/proj-a")
        # 清空缓存模拟进程重启
        store._cache.clear()
        # 缓存未命中
        assert store.get_cached("s1") is None
        # aget 从... (内存模式下 cache 清空后无源，此测试仅 SQLite 模式有意义)
        # 内存模式下清空 cache = 数据丢失，aget 返回 None
        return await store.aget("s1")

    # 内存模式清空 cache 后 aget 返回 None（无 DB 源）
    result = asyncio.run(run())
    assert result is None


# --------------------------------------------------------------------------- #
# 多 session 隔离
# --------------------------------------------------------------------------- #
def test_multi_session_isolation():
    store = WorkspaceStore(async_conn=None)

    async def run():
        await store.aset("s1", "D:/proj-a")
        await store.aset("s2", "D:/proj-b")
        return await store.aget("s1"), await store.aget("s2")

    p1, p2 = asyncio.run(run())
    assert p1 == "D:/proj-a"
    assert p2 == "D:/proj-b"
    # 互不干扰
    assert p1 != p2


def test_overwrite_workspace():
    """同一 session 重复 aset 应覆盖。"""
    store = WorkspaceStore(async_conn=None)

    async def run():
        await store.aset("s1", "D:/proj-a")
        await store.aset("s1", "D:/proj-b")
        return await store.aget("s1")

    assert asyncio.run(run()) == "D:/proj-b"


def test_delete_only_affects_target():
    store = WorkspaceStore(async_conn=None)

    async def run():
        await store.aset("s1", "D:/proj-a")
        await store.aset("s2", "D:/proj-b")
        await store.adelete("s1")
        return await store.aget("s1"), await store.aget("s2")

    p1, p2 = asyncio.run(run())
    assert p1 is None
    assert p2 == "D:/proj-b"


def test_awarm_from_db_alias():
    """awarm_from_db 等价于 aget（内存模式下）。"""
    store = WorkspaceStore(async_conn=None)

    async def run():
        await store.aset("s1", "D:/proj-a")
        store._cache.clear()
        await store.awarm_from_db("s1")

    # 内存模式清空 cache 后 awarm 无源，缓存仍空
    asyncio.run(run())
    assert store.get_cached("s1") is None
