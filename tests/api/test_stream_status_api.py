"""GET /api/threads/{id}/stream-status 断点/挂起语义 API 测试。

覆盖：
- 响应字段齐全：thread_id / streaming / failure / pending_interrupt / resumable / resume_kind
- resume_kind 优先级：有挂起 → interrupt；有 failure → retry；仅可续跑 → resume；其余 → None
- streaming 为真时仍返回真实的 pending_interrupt / resumable（不短路）
- 非 workflow 会话不调用 workflow_sm，返回默认空值
- workflow_sm 缺失（未初始化）时优雅返回默认空值

不进入 TestClient 上下文（跳过 lifespan 的 build_agent），仅 patch 全局 agent / workflow_sm。

运行：
  pytest tests/api/test_stream_status_api.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from session.workflow_adapter import ResumeState

_TID = "server-workflow-simple-thread-abc"


@pytest.fixture
def env():
    """构造最小运行环境：agent（workflow 会话）+ workflow_sm + 空活跃流表。"""
    agent = MagicMock()
    agent.session.is_workflow_session.return_value = True
    workflow_sm = MagicMock()
    workflow_sm.aget_failure = AsyncMock(return_value=None)
    workflow_sm.aget_resume_state = AsyncMock(return_value=ResumeState())
    with (
        patch("api.server.agent", agent),
        patch("api.server.workflow_sm", workflow_sm),
        patch("api.server._active_streams", {}),
    ):
        from api.server import app

        yield TestClient(app), agent, workflow_sm


def _get_status(client: TestClient, tid: str = _TID) -> dict:
    resp = client.get(f"/api/threads/{tid}/stream-status")
    assert resp.status_code == 200
    return resp.json()


def test_status_fields_present_and_defaults(env):
    """无挂起、无 failure、不可续跑 → 全字段齐全且 resume_kind=None。"""
    client, _agent, _sm = env

    data = _get_status(client)

    assert data == {
        "thread_id": _TID,
        "streaming": False,
        "failure": None,
        "pending_interrupt": None,
        "resumable": False,
        "resume_kind": None,
    }


def test_resume_kind_interrupt_wins(env):
    """有挂起中断 → resume_kind=interrupt（优先于 failure / resumable）。"""
    client, _agent, sm = env
    sm.aget_failure.return_value = {"node": "n", "error": "e", "ts": 1.0}
    sm.aget_resume_state.return_value = ResumeState(
        pending_interrupt={"kind": "human_choice", "payload": {"prompt": "选?"}},
        resumable=True,
    )

    data = _get_status(client)

    assert data["pending_interrupt"] == {
        "kind": "human_choice",
        "payload": {"prompt": "选?"},
    }
    assert data["resumable"] is True
    assert data["resume_kind"] == "interrupt"


def test_resume_kind_retry_when_failure_resumable(env):
    """无挂起但有 failure 且可续跑 → resume_kind=retry。"""
    client, _agent, sm = env
    sm.aget_failure.return_value = {"node": "worker_exec", "error": "boom", "ts": 2.0}
    sm.aget_resume_state.return_value = ResumeState(resumable=True)

    data = _get_status(client)

    assert data["failure"] == {"node": "worker_exec", "error": "boom", "ts": 2.0}
    assert data["pending_interrupt"] is None
    assert data["resumable"] is True
    assert data["resume_kind"] == "retry"


def test_resume_kind_retry_when_failure_not_resumable(env):
    """无挂起、有 failure 但不可续跑 → 仍 resume_kind=retry（保持旧行为）。"""
    client, _agent, sm = env
    sm.aget_failure.return_value = {"node": "worker_exec", "error": "boom", "ts": 2.0}
    sm.aget_resume_state.return_value = ResumeState(resumable=False)

    data = _get_status(client)

    assert data["resumable"] is False
    assert data["resume_kind"] == "retry"


def test_resume_kind_resume_when_only_resumable(env):
    """无挂起、无 failure、可续跑 → resume_kind=resume（新入口）。"""
    client, _agent, sm = env
    sm.aget_resume_state.return_value = ResumeState(resumable=True)

    data = _get_status(client)

    assert data["failure"] is None
    assert data["resumable"] is True
    assert data["resume_kind"] == "resume"


def test_streaming_true_still_returns_real_values(env):
    """streaming 为真时仍返回真实 pending_interrupt / resumable（不短路）。"""
    client, _agent, sm = env
    sm.aget_resume_state.return_value = ResumeState(
        pending_interrupt={"kind": "user_confirmation", "payload": {"x": 1}},
        resumable=True,
    )
    with patch("api.server._active_streams", {_TID: object()}):
        data = _get_status(client)

    assert data["streaming"] is True
    assert data["pending_interrupt"] == {
        "kind": "user_confirmation",
        "payload": {"x": 1},
    }
    assert data["resumable"] is True
    assert data["resume_kind"] == "interrupt"


def test_non_workflow_session_skips_workflow_sm(env):
    """非 workflow 会话：不调用 workflow_sm，返回默认空值。"""
    client, agent, sm = env
    agent.session.is_workflow_session.return_value = False

    data = _get_status(client, tid="thread-plain")

    assert data["thread_id"] == "thread-plain"
    assert data["failure"] is None
    assert data["pending_interrupt"] is None
    assert data["resumable"] is False
    assert data["resume_kind"] is None
    sm.aget_failure.assert_not_awaited()
    sm.aget_resume_state.assert_not_awaited()


def test_workflow_sm_missing_returns_defaults(env):
    """workflow_sm 为 None（未初始化）时优雅返回默认空值。"""
    client, _agent, _sm = env
    with patch("api.server.workflow_sm", None):
        data = _get_status(client)

    assert data["failure"] is None
    assert data["pending_interrupt"] is None
    assert data["resumable"] is False
    assert data["resume_kind"] is None
