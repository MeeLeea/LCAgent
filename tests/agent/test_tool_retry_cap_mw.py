"""ToolRetryCapMW 相同参数工具失败熔断中间件测试。

验证通用 ReAct 死循环熔断逻辑：
- 无历史 / 失败次数 < MAX_IDENTICAL_FAILURES 时放行（主模型自行反思重试）
- 相同工具 + 相同参数失败达上限时拦截，返回失败 ToolMessage(status="error")
- 相同工具但参数不同 / 不同工具相同参数 / 成功结果 均不触发熔断
- ask_human 永不拦截
- state 为 None / 格式异常 / 无法关联 tool_calls 时 fail-open 放行，不崩溃
"""
import asyncio
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain.messages import AIMessage, ToolMessage

from agent.tool_retry_cap_mw import (
    MAX_IDENTICAL_FAILURES,
    ToolRetryCapMW,
    _count_identical_failures,
    _fingerprint,
    is_failed_tool_message,
)


def _make_request(
    tool_name: str,
    args: dict,
    state: object,
    call_id: str = "current_call",
) -> SimpleNamespace:
    """构造最小化 request（只含 tool_call 和 state，够中间件使用）。"""
    return SimpleNamespace(
        tool_call={"name": tool_name, "args": args, "id": call_id},
        state=state,
    )


def _make_failure_history(
    tool_name: str,
    args: dict,
    call_ids: list[str],
) -> dict:
    """构造「AIMessage(tool_calls) + 失败 ToolMessage」配对历史。"""
    messages: list[object] = []
    for call_id in call_ids:
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": args, "id": call_id}],
            )
        )
        messages.append(
            ToolMessage(
                content="[工具执行失败] 路径不存在",
                tool_call_id=call_id,
                name=tool_name,
                status="error",
            )
        )
    return {"messages": messages}


def _allow_handler(result_content: str = "ok"):
    """返回记录调用次数的放行 handler。"""
    called: list[object] = []

    def handler(req):
        called.append(req)
        return ToolMessage(
            content=result_content,
            tool_call_id="current_call",
            name=req.tool_call["name"],
        )

    return handler, called


# ============ 指纹与失败识别 ============

def test_fingerprint_is_deterministic_and_order_independent():
    """指纹与字典键顺序无关，参数相同即相同。"""
    a = _fingerprint("read_file", {"path": "a.txt", "head": 1})
    b = _fingerprint("read_file", {"head": 1, "path": "a.txt"})
    assert a == b
    assert _fingerprint("read_file", None) == _fingerprint("read_file", {})
    assert _fingerprint("read_file", {}) != _fingerprint("write_file", {})


def test_is_failed_tool_message_detects_status_and_markers():
    """status=error 与中文失败标记均识别；非 ToolMessage 返回 False。"""
    assert is_failed_tool_message(
        ToolMessage(content="x", tool_call_id="c1", status="error")
    )
    assert is_failed_tool_message(
        ToolMessage(content="[参数冲突] head 与 tail 互斥", tool_call_id="c1")
    )
    assert not is_failed_tool_message(
        ToolMessage(content="ok", tool_call_id="c1")
    )
    assert not is_failed_tool_message("not a message")
    assert not is_failed_tool_message(None)


# ============ 放行场景 ============

def test_allows_when_no_history():
    """无历史失败，放行。"""
    mw = ToolRetryCapMW()
    request = _make_request("read_file", {"path": "a.txt"}, {"messages": []})
    handler, called = _allow_handler()

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "ok"


def test_allows_below_limit():
    """历史相同失败 1 次（< MAX_IDENTICAL_FAILURES），放行。"""
    assert MAX_IDENTICAL_FAILURES == 2
    mw = ToolRetryCapMW()
    args = {"path": "missing.txt"}
    state = _make_failure_history("read_file", args, ["c1"])
    request = _make_request("read_file", args, state)
    handler, called = _allow_handler()

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "ok"


def test_allows_same_tool_different_args():
    """相同工具但参数不同，不计入同一指纹，放行。"""
    mw = ToolRetryCapMW()
    state = _make_failure_history("read_file", {"path": "a.txt"}, ["c1", "c2"])
    request = _make_request("read_file", {"path": "b.txt"}, state)
    handler, called = _allow_handler()

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "ok"


def test_allows_different_tool_identical_args():
    """不同工具即使参数相同，也不计入，放行。"""
    mw = ToolRetryCapMW()
    args = {"path": "a.txt"}
    state = _make_failure_history("read_file", args, ["c1", "c2"])
    request = _make_request("write_file", args, state)
    handler, called = _allow_handler()

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "ok"


def test_success_results_do_not_count():
    """非 error 的成功结果不计入失败次数，放行。"""
    mw = ToolRetryCapMW()
    args = {"path": "a.txt"}
    messages: list[object] = []
    for call_id in ["c1", "c2"]:
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": args, "id": call_id}],
            )
        )
        messages.append(
            ToolMessage(content="文件内容", tool_call_id=call_id, name="read_file")
        )
    request = _make_request("read_file", args, {"messages": messages})
    handler, called = _allow_handler()

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "ok"


def test_ask_human_is_never_blocked():
    """ask_human 即使历史相同失败达上限也放行。"""
    mw = ToolRetryCapMW()
    args = {"prompt": "请选择", "choices": []}
    state = _make_failure_history(
        "ask_human", args, [f"c{i}" for i in range(MAX_IDENTICAL_FAILURES + 3)]
    )
    request = _make_request("ask_human", args, state)
    handler, called = _allow_handler("human answered")

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "human answered"


# ============ 熔断场景 ============

def test_blocks_at_limit():
    """相同参数失败达 MAX_IDENTICAL_FAILURES 次，熔断拦截。"""
    mw = ToolRetryCapMW()
    args = {"path": "missing.txt"}
    state = _make_failure_history(
        "read_file", args, [f"c{i}" for i in range(MAX_IDENTICAL_FAILURES)]
    )
    request = _make_request("read_file", args, state, call_id="current_call")
    handler, called = _allow_handler("should not reach")

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 0
    assert "重复调用熔断" in result.content
    assert result.status == "error"
    assert result.name == "read_file"
    assert result.tool_call_id == "current_call"


def test_blocks_async_at_limit():
    """异步版同样在达上限时拦截。"""
    mw = ToolRetryCapMW()
    args = {"path": "missing.txt"}
    state = _make_failure_history(
        "read_file", args, [f"c{i}" for i in range(MAX_IDENTICAL_FAILURES)]
    )
    request = _make_request("read_file", args, state)
    called: list[object] = []

    async def handler(req):
        called.append(req)
        return ToolMessage(content="should not reach", tool_call_id="x", name="read_file")

    result = asyncio.run(mw.awrap_tool_call(request, handler))
    assert len(called) == 0
    assert "重复调用熔断" in result.content
    assert result.status == "error"


# ============ 边界：state 容错与 fail-open ============

def test_handles_none_and_malformed_state():
    """state 为 None / {"messages": None} / 无 messages 属性 均放行不崩溃。"""
    mw = ToolRetryCapMW()
    args = {"path": "a.txt"}
    for state in (None, {"messages": None}, SimpleNamespace(other=1)):
        request = _make_request("read_file", args, state)
        handler, called = _allow_handler()

        result = mw.wrap_tool_call(request, handler)
        assert len(called) == 1
        assert result.content == "ok"


def test_workspace_rejection_counts_as_failure():
    """WorkspaceSecurityMW 的「操作被拒绝」结果（无 status 字段）也计入失败并触发熔断。

    回归：路径逃逸/越界是最高频的失败场景，其 ToolMessage 既无 status="error"
    也不含方括号标记，若漏登记标记则熔断器对该场景完全失效。
    """
    mw = ToolRetryCapMW()
    args = {"path": "../outside.txt"}
    messages: list[object] = []
    for call_id in [f"c{i}" for i in range(MAX_IDENTICAL_FAILURES)]:
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": args, "id": call_id}],
            )
        )
        messages.append(
            ToolMessage(
                content="操作被拒绝：路径逃逸，目标不在工作空间内",
                tool_call_id=call_id,
                name="read_file",
            )
        )
    assert _count_identical_failures({"messages": messages}, "read_file", args) == (
        MAX_IDENTICAL_FAILURES
    )
    request = _make_request("read_file", args, {"messages": messages})
    handler, called = _allow_handler("should not reach")

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 0
    assert "重复调用熔断" in result.content
    assert result.status == "error"


def test_fail_open_when_no_correlatable_tool_calls():
    """存在失败 ToolMessage 但无 AIMessage.tool_calls 可关联时，放行（fail-open）。"""
    mw = ToolRetryCapMW()
    args = {"path": "missing.txt"}
    state = {
        "messages": [
            ToolMessage(
                content="[工具执行失败] 路径不存在",
                tool_call_id=f"c{i}",
                name="read_file",
                status="error",
            )
            for i in range(MAX_IDENTICAL_FAILURES + 2)
        ]
    }
    assert _count_identical_failures(state, "read_file", args) == 0
    request = _make_request("read_file", args, state)
    handler, called = _allow_handler()

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 1
    assert result.content == "ok"


# ============ 结构化 dict 失败识别（success: false） ============

def test_detects_structured_success_false():
    """dict 结果序列化为 {"success": false, ...} 时也识别为失败（无 status 参数）。"""
    assert is_failed_tool_message(
        ToolMessage(
            content='{"success": false, "returncode": 1, "error": "脚本执行失败 (exit 1): boom"}',
            tool_call_id="c1",
        )
    )


def test_structured_failure_blocks_third_identical_call():
    """终端工具以 {"success": false} dict 形式失败也计入，第 3 次相同调用被熔断。"""
    mw = ToolRetryCapMW()
    args = {"file_path": "bad.py"}
    messages: list[object] = []
    for call_id in [f"c{i}" for i in range(MAX_IDENTICAL_FAILURES)]:
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "run_python", "args": args, "id": call_id}],
            )
        )
        messages.append(
            ToolMessage(
                content='{"success": false, "returncode": 1, "error": "脚本执行失败 (exit 1): boom"}',
                tool_call_id=call_id,
                name="run_python",
            )
        )
    state = {"messages": messages}
    request = _make_request("run_python", args, state)
    handler, called = _allow_handler("should not reach")

    result = mw.wrap_tool_call(request, handler)
    assert len(called) == 0
    assert "重复调用熔断" in result.content
    assert result.status == "error"


def test_success_true_is_not_a_failure():
    """{"success": true} 结果不视为失败。"""
    assert not is_failed_tool_message(
        ToolMessage(
            content='{"success": true, "returncode": 0, "stdout": "ok"}',
            tool_call_id="c1",
        )
    )


def test_non_json_and_list_content_do_not_crash():
    """纯文本与 content 为内容块 list 时均不崩溃，返回 bool。"""
    assert not is_failed_tool_message(
        ToolMessage(content="plain text", tool_call_id="c1")
    )
    result = is_failed_tool_message(
        ToolMessage(
            content=[{"type": "text", "text": "hello"}],
            tool_call_id="c2",
        )
    )
    assert isinstance(result, bool)
