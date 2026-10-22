"""删除会话内某一轮（DELETE /api/threads/{id}/turns/{turn_index}）API 测试。

覆盖：
- 成功删除：透传 thread_id / turn_index，返回删除条数
- 轮序号越界（manager 抛 ValueError）→ 404
- turn_index 非整数 → 422（FastAPI 路径参数校验）
- 会话正在执行（per-thread 锁被占用）→ 409

不进入 TestClient 上下文（跳过 lifespan 的 build_agent），仅 patch 全局 agent。

运行：
  pytest tests/api/test_delete_turn_api.py -v
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_agent():
    """最小 agent mock：仅暴露 session_manager.adelete_turn。"""
    agent = MagicMock()
    agent.session_manager.adelete_turn = AsyncMock(return_value=2)
    return agent


@pytest.fixture
def client(mock_agent):
    with patch("api.server.agent", mock_agent):
        # 不使用 with 语句进入 lifespan（避免触发 build_agent 真实构建）
        yield _bare_client()


def _bare_client():
    from api.server import app

    return TestClient(app)


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #


def test_delete_turn_ok(client, mock_agent):
    """成功删除：透传参数并返回删除条数。"""
    resp = client.delete("/api/threads/thread-1/turns/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True
    assert data["thread_id"] == "thread-1"
    assert data["turn_index"] == 0
    assert data["removed_messages"] == 2
    mock_agent.session_manager.adelete_turn.assert_awaited_once_with("thread-1", 0)


def test_delete_turn_not_found(client, mock_agent):
    """轮序号越界（manager 抛 ValueError）→ 404。"""
    mock_agent.session_manager.adelete_turn = AsyncMock(
        side_effect=ValueError("轮序号越界: 5（该会话共 2 轮）")
    )
    resp = client.delete("/api/threads/thread-1/turns/5")
    assert resp.status_code == 404
    assert "轮序号越界" in resp.json()["detail"]


def test_delete_turn_invalid_index(client):
    """turn_index 非整数 → 422（路径参数类型校验）。"""
    resp = client.delete("/api/threads/thread-1/turns/abc")
    assert resp.status_code == 422


def test_delete_turn_conflict(client, mock_agent):
    """会话正在执行（per-thread 锁被占用）→ 409，不调用删除。"""
    locked = SimpleNamespace(locked=lambda: True)
    with patch("api.server._thread_lock", return_value=locked):
        resp = client.delete("/api/threads/thread-1/turns/0")
    assert resp.status_code == 409
    mock_agent.session_manager.adelete_turn.assert_not_awaited()
