import asyncio
import json
import os
import sys
import tempfile
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.memory import AgentMemory


def test_memory_basic():
    """测试长期记忆基本操作 + checkpointer 初始化 + close 文件可删除。"""
    fd, tmp = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        mem = AgentMemory(
            checkpoint_file=tmp,
            long_term_file=None,
            use_sqlite=True,
        )
        assert mem.thread_id.startswith("thread-")

        # 长期记忆(important 才写入)
        mem.add("user", "hi", {"important": True})
        assert len(mem.long_term_memory) == 1
        mem.clear_long_term()
        assert len(mem.long_term_memory) == 0

        # checkpointer 可用
        assert mem.get_checkpointer() is not None
        assert mem.get_config()["configurable"]["thread_id"] == mem.thread_id

        # 关闭连接
        mem.close()
    finally:
        # 连接已关闭，文件应可删除(Windows 上是泄漏的标志性测试)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError as error:
            # 删除失败说明连接仍被持有，这是泄漏，必须失败
            raise AssertionError(f"文件删除失败，连接可能未关闭: {error}") from error


def test_save_long_term_memory_is_atomic(tmp_path):
    # Given: 一个长期记忆文件路径。
    ltm_file = str(tmp_path / "ltm.json")
    mem = AgentMemory(checkpoint_file=None, long_term_file=ltm_file, use_sqlite=False)
    
    # When: 保存一条记忆。
    mem.add("user", "important thing", {"important": True})
    
    # Then: 文件存在且内容正确，临时文件已被清理。
    assert os.path.exists(ltm_file)
    with open(ltm_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["content"] == "important thing"
    
    # 临时文件应已被原子替换，不留残留
    tmp_files = [f for f in os.listdir(tmp_path) if ".memory_" in f or ".tmp" in f]
    assert len(tmp_files) == 0


# ============ 异步模式（acreate） ============


def _run(coro):
    """在独立事件循环中执行协程(测试内不含运行中的 loop)。"""
    return asyncio.run(coro)


def test_async_guard_raises_on_sync_methods(tmp_path):
    """异步实例(acreate)调用同步方法必须直接抛 RuntimeError，杜绝混跑。"""
    db_path = str(tmp_path / "async_guard.sqlite")

    async def _scenario():
        mem = await AgentMemory.acreate(checkpoint_file=db_path, use_sqlite=True)
        try:
            assert mem._async_mode is True
            for method, args in [
                ("add", ("user", "hi", {"important": True})),
                ("clear_long_term", ()),
                ("compress_memory", (lambda text, prompt: "",)),
            ]:
                try:
                    getattr(mem, method)(*args)
                except RuntimeError as error:
                    assert method in str(error)
                else:
                    raise AssertionError(f"异步实例调用 {method}() 未抛错")
            # close/__enter__ 同样受守卫
            try:
                mem.close()
            except RuntimeError:
                pass
            else:
                raise AssertionError("异步实例调用 close() 未抛错")
            try:
                with mem:
                    pass
            except RuntimeError:
                pass
            else:
                raise AssertionError("异步实例 __enter__ 未抛错")
        finally:
            await mem.aclose()
        return True

    assert _run(_scenario())


def test_async_aadd_and_aclear_long_term(tmp_path):
    """aadd 写入 memory.json，aclear_long_term 清空并删除文件。"""
    ltm_file = str(tmp_path / "ltm_async.json")

    def _read_json(path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def _scenario():
        mem = await AgentMemory.acreate(checkpoint_file=None, long_term_file=ltm_file, use_sqlite=False)
        try:
            await mem.aadd("user", "important thing", {"important": True})
            await mem.aadd("user", "ordinary thing")
            assert len(mem.long_term_memory) == 1
            assert os.path.exists(ltm_file)
            data = await asyncio.to_thread(_read_json, ltm_file)
            assert len(data) == 1
            assert data[0]["content"] == "important thing"

            await mem.aclear_long_term()
            assert mem.long_term_memory == []
            assert not os.path.exists(ltm_file)
        finally:
            await mem.aclose()
        return True

    assert _run(_scenario())


def test_async_acompress_memory(tmp_path):
    """acompress_memory 用异步回调生成摘要并替换原记忆。"""
    ltm_file = str(tmp_path / "ltm_compress_async.json")

    def _read_json(path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def _scenario():
        mem = await AgentMemory.acreate(checkpoint_file=None, long_term_file=ltm_file, use_sqlite=False)
        try:
            await mem.aadd("user", "第一条关键信息", {"important": True})
            await mem.aadd("user", "第二条关键信息", {"important": True})

            async def _summarize(text: str, prompt: str) -> str:
                return "压缩后的摘要"

            result = await mem.acompress_memory(_summarize)
            assert result["success"] is True
            assert result["original_count"] == 2
            assert result["summary"] == "压缩后的摘要"
            assert len(mem.long_term_memory) == 1
            assert mem.long_term_memory[0]["role"] == "system"
            # 摘要结果已持久化
            data = await asyncio.to_thread(_read_json, ltm_file)
            assert len(data) == 1
            assert "压缩后的摘要" in data[0]["content"]
        finally:
            await mem.aclose()
        return True

    assert _run(_scenario())


def test_async_acompress_memory_empty():
    """无长期记忆时 acompress_memory 返回失败结果而非抛错。"""

    async def _scenario():
        mem = await AgentMemory.acreate(checkpoint_file=None, use_sqlite=False)
        try:
            async def _summarize(text: str, prompt: str) -> str:
                return "不应被调用"

            result = await mem.acompress_memory(_summarize)
            assert result["success"] is False
            assert "error" in result
        finally:
            await mem.aclose()
        return True

    assert _run(_scenario())


def test_async_context_manager_closes_connection(tmp_path):
    """async with 离开上下文后应关闭 aiosqlite 连接。"""
    db_path = str(tmp_path / "async_ctx.sqlite")

    async def _scenario():
        async with await AgentMemory.acreate(checkpoint_file=db_path, use_sqlite=True) as mem:
            conn = mem._async_conn
            assert conn is not None
            # 连接在上下文中可用(能执行查询)
            cursor = await conn.execute("SELECT 1")
            await cursor.close()
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


def test_async_guard_does_not_affect_sync_instance():
    """同步实例(直接 __init__)不受守卫影响。"""
    mem = AgentMemory(checkpoint_file=None, use_sqlite=False)
    try:
        assert mem._async_mode is False
        mem.add("user", "hi", {"important": True})
        assert len(mem.long_term_memory) == 1
    finally:
        mem.close()
