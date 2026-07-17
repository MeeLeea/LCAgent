import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.memory import AgentMemory


def _create_checkpoint_rows(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT NOT NULL)")
        conn.execute("CREATE TABLE writes (thread_id TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO checkpoints (thread_id) VALUES (?)",
            [("target",), ("other",)],
        )
        conn.executemany(
            "INSERT INTO writes (thread_id) VALUES (?)",
            [("target",), ("target",), ("other",)],
        )


def _count_rows(db_path: str, table: str, thread_id: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE thread_id = ?",
            (thread_id,),
        )
        return int(cursor.fetchone()[0])


def test_memory_basic():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False).name
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

        # 会话列表至少包含当前 thread
        assert mem.thread_id in mem.list_threads()

        # 新会话
        old = mem.thread_id
        new = mem.new_thread()
        assert new != old
        assert new in mem.list_threads()

        # 对话导出(空会话也应有表头)
        exported = mem.export_thread()
        assert "对话导出" in exported
        assert mem.thread_id in exported

        # 导出指定会话
        exported2 = mem.export_thread(thread_id=old)
        assert "对话导出" in exported2
    finally:
        # SqliteSaver 仍持有连接,删除可能报 PermissionError(Windows),忽略即可
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def test_delete_thread_removes_checkpoints_and_writes_when_thread_exists(tmp_path):
    # Given: a real SQLite-backed AgentMemory with checkpoint rows and pending writes.
    db_path = str(tmp_path / "checkpoints.sqlite")
    _create_checkpoint_rows(db_path)
    mem = AgentMemory(checkpoint_file=db_path, thread_id="target", use_sqlite=True)

    # When: deleting the current target thread and then deleting a missing thread.
    deleted = mem.delete_thread("target")
    missing_deleted = mem.delete_thread("missing")

    # Then: target checkpoints and writes are gone, other rows remain, and current thread moves.
    assert deleted is True
    assert missing_deleted is False
    assert _count_rows(db_path, "checkpoints", "target") == 0
    assert _count_rows(db_path, "writes", "target") == 0
    assert _count_rows(db_path, "checkpoints", "other") == 1
    assert _count_rows(db_path, "writes", "other") == 1
    assert mem.thread_id == "other"
