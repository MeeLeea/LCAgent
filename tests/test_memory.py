import os
import json
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
    
    mem.close()


def test_workflow_thread_management_server_mode(tmp_path):
    """测试 server 模式下专属工作流会话的创建/识别/名称反解"""
    db_path = str(tmp_path / "wf.sqlite")
    mem = AgentMemory(checkpoint_file=db_path, use_sqlite=True, process_type="server")

    tid = mem.new_workflow_thread("simple")
    assert tid.startswith("server-workflow-simple-")
    assert mem.is_workflow_thread(tid)
    assert mem.workflow_name_of(tid) == "simple"
    # 新工作流会话也会出现在会话列表中
    assert tid in mem.list_threads()

    # 普通会话不应被识别为工作流会话
    chat_tid = mem.new_thread()
    assert not mem.is_workflow_thread(chat_tid)
    assert mem.workflow_name_of(chat_tid) is None

    # 名称中的 - / 空格被清洗，反解结果一致
    tid2 = mem.new_workflow_thread("my-workflow")
    assert mem.workflow_name_of(tid2) == "my_workflow"

    mem.close()


def test_workflow_thread_cli_mode():
    """测试无 process_type（CLI 模式）下工作流会话 id 与反解"""
    mem = AgentMemory(use_sqlite=False)
    try:
        tid = mem.new_workflow_thread("pipline")
        assert tid.startswith("workflow-")
        assert mem.is_workflow_thread(tid)
        assert mem.workflow_name_of(tid) == "pipline"

        # 普通 CLI 会话 thread-xxxx 不是工作流会话
        chat_tid = mem.new_thread()
        assert not mem.is_workflow_thread(chat_tid)
        assert mem.workflow_name_of(chat_tid) is None
    finally:
        mem.close()


def test_switch_thread_returns_true_only_when_checkpoint_exists(tmp_path):
    # Given: 数据库中有一个 thread 有 checkpoint，另一个没有。
    db_path = str(tmp_path / "switch.sqlite")
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT NOT NULL)")
        conn.execute("INSERT INTO checkpoints (thread_id) VALUES ('existing')")
    
    mem = AgentMemory(checkpoint_file=db_path, thread_id="current", use_sqlite=True)
    
    # When: 切换到有历史和无历史的 thread。
    has_history = mem.switch_thread("existing")
    no_history = mem.switch_thread("brand_new")
    
    # Then: 只有已存在的返回 True。
    assert has_history is True
    assert no_history is False
    assert mem.thread_id == "brand_new"
    
    mem.close()


def test_get_messages_with_thread_id_does_not_mutate_current_thread(monkeypatch):
    # Given: 当前会话为 t1，monkeypatch 让 get_tuple 对 t2 返回假消息。
    from langchain_core.messages import HumanMessage
    
    mem = AgentMemory(checkpoint_file=None, use_sqlite=False)
    mem.thread_id = "t1"
    original = mem.thread_id
    
    # monkeypatch get_tuple: t2 有消息，t1 没有
    def fake_get_tuple(config):
        tid = config.get("configurable", {}).get("thread_id")
        if tid == "t2":
            from types import SimpleNamespace
            return SimpleNamespace(
                checkpoint={"channel_values": {"messages": [HumanMessage(content="hello")]}}
            )
        return None
    
    monkeypatch.setattr(mem._checkpointer, "get_tuple", fake_get_tuple)
    
    # When: 读取另一个 thread 的消息。
    msgs = mem.get_messages(thread_id="t2")
    
    # Then: thread_id 不变，但成功读到了 t2 的消息。
    assert mem.thread_id == original
    assert mem.thread_id == "t1"
    assert len(msgs) == 1
    assert msgs[0].content == "hello"


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


def test_get_short_term_maps_tool_message_to_assistant_with_prefix(monkeypatch):
    # Given: checkpoint 里有 user, assistant, tool 三种消息。
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    mem = AgentMemory(checkpoint_file=None, use_sqlite=False)
    
    messages = [
        HumanMessage(content="请搜索"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": "c1"}]),
        ToolMessage(content="找到了答案", tool_call_id="c1"),
        AIMessage(content="好的"),
    ]
    
    def fake_get_tuple(config):
        from types import SimpleNamespace
        return SimpleNamespace(checkpoint={"channel_values": {"messages": messages}})
    
    monkeypatch.setattr(mem._checkpointer, "get_tuple", fake_get_tuple)
    
    # When: 调用 get_short_term。
    history = mem.get_short_term()
    
    # Then: ToolMessage 被映射为 assistant 且带前缀。
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "请搜索"}
    assert history[1]["role"] == "assistant"
    assert history[2]["role"] == "assistant"
    assert history[2]["content"].startswith("[工具结果] ")
    assert "找到了答案" in history[2]["content"]
    assert history[3] == {"role": "assistant", "content": "好的"}


def test_export_thread_does_not_add_tool_result_prefix(monkeypatch):
    # Given: 同样的 ToolMessage，export_thread 应该不加前缀(导出给人看，不混淆)。
    from langchain_core.messages import HumanMessage, ToolMessage
    
    mem = AgentMemory(checkpoint_file=None, use_sqlite=False)
    
    messages = [
        HumanMessage(content="查询"),
        ToolMessage(content="结果是42", tool_call_id="c1", name="calc"),
    ]
    
    def fake_get_tuple(config):
        from types import SimpleNamespace
        return SimpleNamespace(checkpoint={"channel_values": {"messages": messages}})
    
    monkeypatch.setattr(mem._checkpointer, "get_tuple", fake_get_tuple)
    
    # When: 导出。
    exported = mem.export_thread()
    
    # Then: 工具消息内容直接呈现，不带 [工具结果] 前缀。
    assert "结果是42" in exported
    assert "[工具结果]" not in exported
    assert "工具" in exported  # 角色标签应是"工具"
