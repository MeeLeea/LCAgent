"""Session 模块测试 - 验证 per-session 隔离、读写、裁剪与上下文构建。

运行：
  pytest tests/session/test_session_store.py -v
"""
import asyncio

from langgraph.checkpoint.memory import MemorySaver

from session import (
    SessionContext,
    SessionRegistry,
    SessionStore,
)


# --------------------------------------------------------------------------- #
# SessionStore: namespace 隔离
# --------------------------------------------------------------------------- #
def test_history_isolation_between_sessions():
    store = SessionStore(max_history=10)

    async def run():
        await store.aappend_history("s1", {"step": 1, "tool": "a"})
        await store.aappend_history("s1", {"step": 2, "tool": "b"})
        await store.aappend_history("s2", {"step": 1, "tool": "c"})
        h1 = await store.aget_history("s1")
        h2 = await store.aget_history("s2")
        return h1, h2

    h1, h2 = asyncio.run(run())
    assert len(h1) == 2
    assert len(h2) == 1
    assert h1[0]["tool"] == "a"
    assert h2[0]["tool"] == "c"
    # 互不干扰
    assert all(e["tool"] != "c" for e in h1)
    assert all(e["tool"] != "a" for e in h2)


def test_history_truncation_to_max():
    store = SessionStore(max_history=3)

    async def run():
        for i in range(5):
            await store.aappend_history("s", {"step": i})
        return await store.aget_history("s")

    history = asyncio.run(run())
    assert len(history) == 3
    # 保留最近 3 条（step 2,3,4）
    assert [e["step"] for e in history] == [2, 3, 4]


def test_recorded_call_ids_dedup():
    store = SessionStore()

    async def run():
        await store.aadd_recorded_call_ids("s1", {"call-a"})
        await store.aadd_recorded_call_ids("s1", {"call-a"})  # 重复
        await store.aadd_recorded_call_ids("s1", {"call-b"})
        await store.aadd_recorded_call_ids("s2", {"call-c"})
        ids1 = await store.aget_recorded_call_ids("s1")
        ids2 = await store.aget_recorded_call_ids("s2")
        return ids1, ids2

    ids1, ids2 = asyncio.run(run())
    assert ids1 == {"call-a", "call-b"}
    assert ids2 == {"call-c"}
    assert "call-c" not in ids1


def test_interrupt_mode_isolation():
    store = SessionStore()

    async def run():
        await store.aset_interrupt_mode("s1", "run")
        await store.aset_interrupt_mode("s2", "chat")
        m1 = await store.aget_interrupt_mode("s1")
        m2 = await store.aget_interrupt_mode("s2")
        # 清除 s1 不影响 s2
        await store.aclear_interrupt("s1")
        m1_after = await store.aget_interrupt_mode("s1")
        m2_after = await store.aget_interrupt_mode("s2")
        return m1, m2, m1_after, m2_after

    m1, m2, m1_after, m2_after = asyncio.run(run())
    assert m1 == "run"
    assert m2 == "chat"
    assert m1_after is None
    assert m2_after == "chat"


def test_delete_session_clears_all_state():
    store = SessionStore()

    async def run():
        await store.aappend_history("s1", {"step": 1})
        await store.aadd_recorded_call_ids("s1", {"call-x"})
        await store.aset_interrupt_mode("s1", "run")
        await store.adelete_session("s1")
        h = await store.aget_history("s1")
        ids = await store.aget_recorded_call_ids("s1")
        m = await store.aget_interrupt_mode("s1")
        return h, ids, m

    h, ids, m = asyncio.run(run())
    assert h == []
    assert ids == set()
    assert m is None


def test_empty_session_returns_defaults():
    store = SessionStore()

    async def run():
        h = await store.aget_history("never")
        ids = await store.aget_recorded_call_ids("never")
        m = await store.aget_interrupt_mode("never")
        return h, ids, m

    h, ids, m = asyncio.run(run())
    assert h == []
    assert ids == set()
    assert m is None


# --------------------------------------------------------------------------- #
# SessionContext
# --------------------------------------------------------------------------- #
def test_session_context_config_structure():
    cp = MemorySaver()
    ctx = SessionContext.create(
        session_id="thread-abc",
        checkpointer=cp,
        recursion_limit=30,
    )
    assert ctx.session_id == "thread-abc"
    assert ctx.checkpointer is cp
    assert ctx.config["configurable"]["thread_id"] == "thread-abc"
    assert ctx.config["recursion_limit"] == 30


# --------------------------------------------------------------------------- #
# SessionRegistry: session id 生成与工作流
# --------------------------------------------------------------------------- #
def test_registry_generate_session_id_with_process_type():
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type="server", recursion_limit=25)
    sid = reg.generate_session_id()
    assert sid.startswith("server-thread-")
    # 工作流会话
    wfid = reg.generate_session_id("data pipeline")
    assert "server-workflow-data_pipeline-" in wfid


def test_registry_workflow_helpers():
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type="feishu")
    wfid = reg.new_workflow_session("report-gen")
    assert reg.is_workflow_session(wfid)
    assert reg.workflow_name_of(wfid) == "report_gen"
    # 普通会话
    sid = reg.generate_session_id()
    assert not reg.is_workflow_session(sid)
    assert reg.workflow_name_of(sid) is None


def test_registry_get_context_uses_current_when_none():
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, recursion_limit=15)
    ctx = reg.get_context()
    assert ctx.session_id == reg.current_session_id
    assert ctx.config["recursion_limit"] == 15

    ctx2 = reg.get_context("explicit-id")
    assert ctx2.session_id == "explicit-id"


def test_registry_process_type_filter():
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type="scheduler")
    assert reg._matches_process_type("scheduler-thread-1")
    assert not reg._matches_process_type("server-thread-1")

    reg2 = SessionRegistry(cp, store, process_type=None)
    assert reg2._matches_process_type("anything")


def test_registry_alist_sessions_includes_current_for_memory():
    """内存 checkpointer 无 SQL，alist_sessions 应至少包含当前会话。"""
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type=None)

    sessions = asyncio.run(reg.alist_sessions())
    assert reg.current_session_id in sessions


# --------------------------------------------------------------------------- #
# SessionRegistry: 并发会话隔离（无状态化核心验证）
# --------------------------------------------------------------------------- #
import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage


def _put_checkpoint(cp, thread_id, messages):
    """向 checkpointer 写入一个含 messages 的 checkpoint（模拟 LangGraph 执行后落盘）。

    MemorySaver.put 会 pop channel_values 并按 new_versions 存储 blob，
    get_tuple 再按 channel_versions 重建 channel_values，因此两者版本必须一致。
    """
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    versions = {"messages": "1"}
    checkpoint = {
        "v": 1,
        "id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat(),
        "channel_values": {"messages": messages},
        "channel_versions": versions,
        "versions_seen": {},
    }
    cp.put(config, checkpoint, {"source": "input", "step": 0, "writes": {}}, versions)


def test_concurrent_message_isolation():
    """不同 session_id 的消息互不泄漏：aget_messages 只返回对应会话的消息。"""
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type=None)

    _put_checkpoint(cp, "s-alpha", [HumanMessage(content="hello A"), AIMessage(content="hi A")])
    _put_checkpoint(cp, "s-beta", [HumanMessage(content="hello B"), AIMessage(content="hi B")])

    async def run():
        msgs_a = await reg.aget_messages("s-alpha")
        msgs_b = await reg.aget_messages("s-beta")
        return msgs_a, msgs_b

    msgs_a, msgs_b = asyncio.run(run())
    assert len(msgs_a) == 2
    assert len(msgs_b) == 2
    assert msgs_a[0].content == "hello A"
    assert msgs_b[0].content == "hello B"
    assert all("B" not in m.content for m in msgs_a)
    assert all("A" not in m.content for m in msgs_b)


def test_concurrent_session_switching():
    """aswitch_session 切换后 current_session_id 正确更新，aget_messages 读对应会话。

    注意：MemorySaver 无 SQL 连接，astored_session_ids 返回空，
    因此 aswitch_session 的返回值恒为 False（无法查询存量）。
    但切换本身有效：current_session_id 更新 + aget_messages 读到正确会话。
    """
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type=None)

    _put_checkpoint(cp, "switch-a", [HumanMessage(content="msg A")])
    _put_checkpoint(cp, "switch-b", [HumanMessage(content="msg B")])

    async def run():
        await reg.aswitch_session("switch-a")
        msgs_a = await reg.aget_messages()
        await reg.aswitch_session("switch-b")
        msgs_b = await reg.aget_messages()
        return msgs_a, msgs_b

    msgs_a, msgs_b = asyncio.run(run())
    assert reg.current_session_id == "switch-b"
    assert msgs_a[0].content == "msg A"
    assert msgs_b[0].content == "msg B"


def test_concurrent_new_session_unique_ids():
    """连续 new_session 生成不同的 session_id。"""
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type=None)

    ids = [reg.new_session() for _ in range(5)]
    assert len(set(ids)) == 5
    assert reg.current_session_id == ids[-1]


def test_deletion_isolation():
    """删除一个会话不影响其他会话的消息。"""
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type=None)

    _put_checkpoint(cp, "del-a", [HumanMessage(content="keep A")])
    _put_checkpoint(cp, "del-b", [HumanMessage(content="keep B")])

    async def run():
        await reg.adelete_session("del-a")
        msgs_b = await reg.aget_messages("del-b")
        return msgs_b

    msgs_b = asyncio.run(run())
    assert len(msgs_b) == 1
    assert msgs_b[0].content == "keep B"


def test_export_isolation():
    """aexport_session 只导出指定会话的消息。"""
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type=None)

    _put_checkpoint(cp, "exp-a", [HumanMessage(content="export A content")])
    _put_checkpoint(cp, "exp-b", [HumanMessage(content="export B content")])

    async def run():
        text_a = await reg.aexport_session("exp-a")
        text_b = await reg.aexport_session("exp-b")
        return text_a, text_b

    text_a, text_b = asyncio.run(run())
    assert "export A content" in text_a
    assert "export B content" not in text_a
    assert "export B content" in text_b
    assert "export A content" not in text_b


def test_workflow_session_isolation_from_regular():
    """工作流会话与普通会话互不干扰。"""
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type="server")

    wf_id = reg.new_workflow_session("data-pipeline")
    regular_id = reg.new_session()

    assert reg.is_workflow_session(wf_id)
    assert not reg.is_workflow_session(regular_id)
    assert reg.workflow_name_of(wf_id) == "data_pipeline"
    assert reg.workflow_name_of(regular_id) is None


def test_asummarize_per_session():
    """asummarize 返回指定会话的消息数，不同会话互不干扰。"""
    cp = MemorySaver()
    store = SessionStore()
    reg = SessionRegistry(cp, store, process_type=None)

    _put_checkpoint(
        cp,
        "sum-a",
        [HumanMessage(content="m1"), AIMessage(content="m2"), HumanMessage(content="m3")],
    )
    _put_checkpoint(cp, "sum-b", [HumanMessage(content="only one")])

    async def run():
        sa = await reg.asummarize("sum-a")
        sb = await reg.asummarize("sum-b")
        return sa, sb

    sa, sb = asyncio.run(run())
    assert sa["checkpoint_messages"] == 3
    assert sb["checkpoint_messages"] == 1
    assert sa["session_id"] == "sum-a"
    assert sb["session_id"] == "sum-b"
