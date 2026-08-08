"""AgentMemory 测试

覆盖保留的方法：
- __init__ / acreate：同步/异步构造
- _generate_thread_id：thread_id 生成与 process_type 前缀
- get_checkpointer / get_long_term_store：基础设施暴露
- aclose：异步关闭与幂等性
"""
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from memory.agent_memory import AgentMemory


def _run(coro):
    """在独立事件循环中执行协程(测试内不含运行中的 loop)。"""
    return asyncio.run(coro)


# ============ 同步构造 ============


def test_sync_init_memory_mode():
    """同步构造（内存模式）：thread_id 自动生成，checkpointer/store 可用。"""
    mem = AgentMemory(checkpoint_file=None, use_sqlite=False)
    assert mem.thread_id.startswith("thread-")
    assert mem._async_mode is False
    assert mem.get_checkpointer() is not None
    assert mem.get_long_term_store() is not None


def test_sync_init_with_process_type():
    """process_type 前缀体现在 thread_id 上。"""
    mem = AgentMemory(checkpoint_file=None, use_sqlite=False, process_type="server")
    assert mem.thread_id.startswith("server-thread-")


def test_sync_init_with_explicit_thread_id():
    """显式传入 thread_id 时直接使用。"""
    mem = AgentMemory(checkpoint_file=None, use_sqlite=False, thread_id="my-thread")
    assert mem.thread_id == "my-thread"


def test_sync_init_sqlite_checkpointer(tmp_path):
    """SQLite 模式下 checkpointer 和 store 可用。"""
    db_path = str(tmp_path / "sync.sqlite")
    mem = AgentMemory(checkpoint_file=db_path, use_sqlite=True)
    assert mem.get_checkpointer() is not None
    assert mem.get_long_term_store() is not None


# ============ _generate_thread_id ============


def test_generate_thread_id_uniqueness():
    """_generate_thread_id 每次返回不同值。"""
    mem = AgentMemory(checkpoint_file=None, use_sqlite=False)
    ids = {mem._generate_thread_id() for _ in range(20)}
    assert len(ids) == 20


def test_generate_thread_id_with_process_type():
    """process_type 存在时 thread_id 带前缀。"""
    mem = AgentMemory(checkpoint_file=None, use_sqlite=False, process_type="feishu")
    tid = mem._generate_thread_id()
    assert tid.startswith("feishu-thread-")


def test_generate_thread_id_without_process_type():
    """无 process_type 时 thread_id 以 thread- 开头。"""
    mem = AgentMemory(checkpoint_file=None, use_sqlite=False)
    tid = mem._generate_thread_id()
    assert tid.startswith("thread-")
    assert not tid.startswith("None")


# ============ 异步模式（acreate） ============


def test_acreate_memory_mode():
    """acreate 内存模式：checkpointer + store 可用。"""

    async def _scenario():
        mem = await AgentMemory.acreate(checkpoint_file=None, use_sqlite=False)
        try:
            assert mem._async_mode is True
            assert mem.thread_id.startswith("thread-")
            assert mem.get_checkpointer() is not None
            assert mem.get_long_term_store() is not None
            # 内存模式下无异步连接
            assert mem._async_conn is None
        finally:
            await mem.aclose()
        return True

    assert _run(_scenario())


def test_acreate_sqlite_mode(tmp_path):
    """acreate SQLite 模式：checkpointer + store 可用，连接可关闭。"""
    db_path = str(tmp_path / "async.sqlite")

    async def _scenario():
        mem = await AgentMemory.acreate(checkpoint_file=db_path, use_sqlite=True)
        try:
            assert mem._async_mode is True
            assert mem._async_conn is not None
            assert mem.get_checkpointer() is not None
            assert mem.get_long_term_store() is not None
        finally:
            await mem.aclose()
        # 关闭后连接应被置 None
        assert mem._async_conn is None
        return True

    assert _run(_scenario())


def test_aclose_idempotent(tmp_path):
    """aclose 可安全重复调用。"""
    db_path = str(tmp_path / "async_close.sqlite")

    async def _scenario():
        mem = await AgentMemory.acreate(checkpoint_file=db_path, use_sqlite=True)
        await mem.aclose()
        await mem.aclose()
        return True

    assert _run(_scenario())


def test_aclose_memory_mode_no_error():
    """内存模式下 aclose 不报错（无连接需关闭）。"""

    async def _scenario():
        mem = await AgentMemory.acreate(checkpoint_file=None, use_sqlite=False)
        await mem.aclose()
        return True

    assert _run(_scenario())
