"""
LangChainAgent API 服务端测试套件
====================================

覆盖特征点：
  - 健康检查与提供商/模型切换
  - 会话管理（CRUD）
  - 工具列表
  - SSE 流式聊天（普通对话 + 命令模式）
  - 工具调用事件
  - HITL 恢复
  - 命令执行

运行：
  pytest tests/test_api.py -v
"""
import asyncio
import json
import os
import sqlite3
import tempfile
from typing import Any, AsyncIterator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# 测试前需要 mock 全局状态，避免真实初始化
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def mock_agent():
    """Mock AgentCore 实例"""
    agent = MagicMock()
    agent.memory = MagicMock()
    agent.memory.thread_id = "test-thread-123"
    agent.memory.new_thread = MagicMock(return_value="new-thread-456")
    agent.memory.new_workflow_thread = MagicMock(return_value="server-workflow-simple-abc12345")
    agent.memory.list_threads = MagicMock(return_value=["thread-1", "thread-2"])
    agent.memory.delete_thread = MagicMock(return_value=True)
    agent.memory.is_workflow_thread = MagicMock(return_value=False)
    agent.memory.workflow_name_of = MagicMock(return_value=None)
    agent.get_available_tools = MagicMock(return_value=["calculator", "web_search", "ask_human"])
    agent.switch_llm = MagicMock()
    agent.aswitch_llm = AsyncMock(side_effect=agent.switch_llm)
    
    # Mock astream_chat：普通对话返回 token + done 事件
    async def mock_astream_chat(message: str) -> AsyncIterator[Dict[str, Any]]:
        if "工具调用" in message:
            # 模拟工具调用场景
            yield {"type": "token", "content": "正在"}
            yield {"type": "token", "content": "计算"}
            yield {
                "type": "tool_call",
                "tool_call_id": "call_123",
                "name": "calculator",
                "args": {"expression": "2+2"},
            }
            yield {
                "type": "tool_result",
                "tool_call_id": "call_123",
                "name": "calculator",
                "content": "4",
            }
            yield {"type": "token", "content": "结果是 4"}
            yield {"type": "done"}
        elif message.startswith("json:"):
            # 模拟执行型命令
            yield {"type": "token", "content": '{"result": "success"}'}
            yield {"type": "done"}
        else:
            # 普通对话
            yield {"type": "token", "content": "你好"}
            yield {"type": "token", "content": "，"}
            yield {"type": "token", "content": "有什么可以帮你"}
            yield {"type": "done"}
    
    agent.astream_chat = mock_astream_chat
    
    # Mock astream_resume：HITL 恢复
    async def mock_astream_resume(payload: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        user_response = payload.get("user_response", "")
        yield {"type": "token", "content": f"收到回复：{user_response}"}
        yield {"type": "token", "content": "，继续执行"}
        yield {"type": "done"}
    
    agent.astream_resume = mock_astream_resume

    # Mock metrics 收集器
    agent.metrics = MagicMock()
    agent.metrics.get_summary = MagicMock(return_value={
        "session": {"duration_seconds": 12.5, "turn_count": 3},
        "llm": {
            "total_calls": 5,
            "total_prompt_tokens": 1000,
            "total_completion_tokens": 500,
            "total_tokens": 1500,
            "total_duration_ms": 3000.0,
            "by_provider": {
                "zhipu": {
                    "count": 5,
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                    "avg_tokens": 300.0,
                    "total_ms": 3000.0,
                }
            },
        },
        "tools": {
            "total_calls": 2,
            "total_duration_ms": 500.0,
            "by_name": {
                "calculator": {
                    "count": 2,
                    "total_ms": 500.0,
                    "min_ms": 200.0,
                    "max_ms": 300.0,
                    "avg_ms": 250.0,
                    "failures": 0,
                    "timeouts": 0,
                }
            },
        },
        "compaction": {
            "total_count": 1,
            "total_messages_before": 60,
            "total_messages_after": 22,
            "total_duration_ms": 800.0,
            "messages_saved": 38,
        },
    })
    agent.metrics.reset = MagicMock()

    # Mock manually_compact（异步）
    agent.manually_compact = AsyncMock(return_value={
        "summary": "用户讨论了 API 设计和测试策略",
        "messages_before": 60,
        "messages_after": 22,
    })

    # Mock 记忆相关方法
    agent.get_memory_summary = MagicMock(return_value={
        "thread_id": "test-thread-123",
        "checkpoint_messages": 10,
        "checkpoint_backend": "sqlite",
        "checkpoint_file": "/tmp/test.sqlite",
        "long_term_count": 5,
        "total_threads": 3,
    })
    agent.compress_memory = MagicMock(return_value={
        "success": True,
        "original_count": 5,
        "original_chars": 2000,
        "compressed_chars": 500,
        "summary": "压缩后的摘要内容",
    })
    agent.memory.clear_long_term = MagicMock()
    agent.memory.clear_short_term = MagicMock()
    agent.memory.export_thread = MagicMock(return_value="用户: 测试消息\n助手: 回复")

    # Mock 技能列表
    agent.list_skills = MagicMock(return_value=[
        {"name": "pptx", "description": "PPT 生成技能"},
        {"name": "pdf", "description": "PDF 处理技能"},
    ])

    return agent


@pytest.fixture
def mock_llm():
    """Mock LLMClient 实例"""
    llm = MagicMock()
    llm.get_info = MagicMock(
        return_value={
            "provider": "zhipu",
            "provider_name": "智谱AI",
            "model": "glm-4-flash",
        }
    )
    llm.switch_model = MagicMock()
    return llm


@pytest.fixture
def temp_checkpoint_db():
    """临时 checkpoint 数据库"""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    
    # 初始化数据库结构
    conn = sqlite3.connect(path)
    saver = SqliteSaver(conn)
    
    # 写入测试数据
    test_messages = [
        HumanMessage(content="测试消息1"),
        AIMessage(content="回复1"),
        HumanMessage(content="测试消息2"),
        AIMessage(
            content="回复2",
            tool_calls=[
                {"id": "call_1", "name": "calculator", "args": {"expr": "1+1"}}
            ],
        ),
        ToolMessage(content="2", tool_call_id="call_1", name="calculator"),
    ]
    
    # SqliteSaver.put 需要完整的 checkpoint 结构，包含 id, ts 等字段
    from datetime import datetime, timezone
    import uuid
    
    checkpoint_id = str(uuid.uuid4())
    saver.put(
        {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}},
        {
            "id": checkpoint_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel_values": {"messages": test_messages},
        },
        {},
        {},  # new_versions 参数
    )
    
    conn.close()
    
    yield path
    
    # 清理
    try:
        os.unlink(path)
    except:
        pass


@pytest.fixture
def client(mock_agent, mock_llm, temp_checkpoint_db):
    """FastAPI 测试客户端"""
    # Mock 全局状态
    with patch("api.server.agent", mock_agent), \
         patch("api.server.llm", mock_llm), \
         patch("api.server.CHECKPOINT_FILE", temp_checkpoint_db), \
         patch("api.server.build_agent", return_value=(mock_agent, mock_llm)), \
         patch("api.server.LLMClient", return_value=mock_llm):
        
        from api.server import app
        
        with TestClient(app) as c:
            yield c


# --------------------------------------------------------------------------- #
# 健康检查与基础信息
# --------------------------------------------------------------------------- #
def test_health_check(client, mock_llm, mock_agent):
    """测试健康检查端点"""
    response = client.get("/api/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert data["provider"] == "zhipu"
    assert data["model"] == "glm-4-flash"
    assert data["thread_id"] == "test-thread-123"


def test_get_providers(client, mock_llm):
    """测试获取提供商列表"""
    with patch("api.server.load_providers") as mock_load:
        mock_load.return_value = {
            "zhipu": {
                "name": "智谱AI",
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "models": ["glm-4-flash", "glm-4-plus"],
                "api_key": "test-key",
            },
            "deepseek": {
                "name": "DeepSeek",
                "base_url": "https://api.deepseek.com",
                "models": ["deepseek-chat"],
                "env_key": "DEEPSEEK_API_KEY",
            },
        }
        
        response = client.get("/api/providers")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["providers"]) == 2
        assert data["current_provider"] == "zhipu"
        assert data["current_model"] == "glm-4-flash"
        
        # 检查脱敏：不应包含 api_key
        for provider in data["providers"]:
            assert "api_key" not in provider
            assert "has_key" in provider


def test_switch_provider(client, mock_agent, mock_llm):
    """测试切换提供商"""
    with patch("api.server.LLMClient") as mock_llm_class:
        new_llm = MagicMock()
        new_llm.get_info = MagicMock(
            return_value={
                "provider": "deepseek",
                "provider_name": "DeepSeek",
                "model": "deepseek-chat",
            }
        )
        mock_llm_class.return_value = new_llm
        
        response = client.post(
            "/api/providers/switch",
            json={"provider": "deepseek"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "deepseek"
        assert data["model"] == "deepseek-chat"
        
        # 验证调用
        mock_agent.switch_llm.assert_called_once_with(new_llm)


def test_switch_provider_invalid(client):
    """测试切换到无效提供商"""
    with patch("api.server.LLMClient") as mock_llm_class:
        mock_llm_class.side_effect = ValueError("未知提供商")
        
        response = client.post(
            "/api/providers/switch",
            json={"provider": "invalid"}
        )
        
        assert response.status_code == 400


def test_switch_model(client, mock_agent, mock_llm):
    """测试切换模型"""
    response = client.post(
        "/api/models/switch",
        json={"model": "glm-4-plus"}
    )
    
    assert response.status_code == 200
    
    # 验证调用
    mock_llm.switch_model.assert_called_once_with("glm-4-plus")
    mock_agent.switch_llm.assert_called_once_with(mock_llm)


# --------------------------------------------------------------------------- #
# 工具列表
# --------------------------------------------------------------------------- #
def test_get_tools(client, mock_agent):
    """测试获取工具列表"""
    response = client.get("/api/tools")
    assert response.status_code == 200
    
    data = response.json()
    assert "tools" in data
    assert "calculator" in data["tools"]
    assert "web_search" in data["tools"]
    assert "ask_human" in data["tools"]


# --------------------------------------------------------------------------- #
# 团队角色
# --------------------------------------------------------------------------- #
def test_get_roles(client, mock_agent):
    """测试列出可用团队角色与当前角色名"""
    mock_agent.name = "manager"
    with patch(
        "agent.role_sw.get_available_team_roles",
        return_value=["manager", "terminator", "worker"],
    ):
        response = client.get("/api/roles")

    assert response.status_code == 200
    data = response.json()
    assert data["roles"] == ["manager", "terminator", "worker"]
    assert data["current"] == "manager"


def test_switch_role(client, mock_agent):
    """测试切换团队角色：调用 arebuild_from_team_dir 并返回新角色"""
    mock_agent.name = "worker"
    mock_agent.arebuild_from_team_dir = AsyncMock()

    response = client.post("/api/roles/switch", json={"role": "worker"})

    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "worker"
    assert data["current"] == "worker"
    mock_agent.arebuild_from_team_dir.assert_awaited_once_with("worker", task="")


def test_switch_role_with_task(client, mock_agent):
    """测试切换角色时携带 task，task 透传给 arebuild_from_team_dir"""
    mock_agent.name = "manager"
    mock_agent.arebuild_from_team_dir = AsyncMock()

    response = client.post(
        "/api/roles/switch",
        json={"role": "manager", "task": "分析项目结构"},
    )

    assert response.status_code == 200
    mock_agent.arebuild_from_team_dir.assert_awaited_once_with(
        "manager", task="分析项目结构"
    )


def test_switch_role_unknown(client, mock_agent):
    """测试切换到未知角色返回 404"""
    mock_agent.arebuild_from_team_dir = AsyncMock(
        side_effect=KeyError("未找到 team 角色: ghost。可用角色: manager, worker")
    )
    with patch(
        "agent.role_sw.get_available_team_roles",
        return_value=["manager", "worker"],
    ):
        response = client.post("/api/roles/switch", json={"role": "ghost"})

    assert response.status_code == 404
    assert "ghost" in response.json()["detail"]


def test_switch_role_empty_prompt(client, mock_agent):
    """测试角色提示词文件为空返回 400"""
    mock_agent.arebuild_from_team_dir = AsyncMock(
        side_effect=FileNotFoundError("角色提示词文件为空或无法读取: team/broken/AGENT.md")
    )
    response = client.post("/api/roles/switch", json={"role": "broken"})

    assert response.status_code == 400
    assert "提示词" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# 工作流
# --------------------------------------------------------------------------- #
class FakeGraphNode:
    """模拟 graph.get_graph() 返回的节点"""

    def __init__(self, node_id: str):
        self.id = node_id


class FakeGraphEdge:
    """模拟 graph.get_graph() 返回的边"""

    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target


class FakeGraphObj:
    """模拟 graph.get_graph() 返回的图结构对象"""

    def __init__(self):
        self.nodes = {
            "__start__": FakeGraphNode("__start__"),
            "summarize": FakeGraphNode("summarize"),
            "manager_plan": FakeGraphNode("manager_plan"),
            "worker_exec": FakeGraphNode("worker_exec"),
            "terminator_final": FakeGraphNode("terminator_final"),
            "__end__": FakeGraphNode("__end__"),
        }
        self.edges = [
            FakeGraphEdge("__start__", "summarize"),
            FakeGraphEdge("summarize", "manager_plan"),
            FakeGraphEdge("manager_plan", "worker_exec"),
            FakeGraphEdge("worker_exec", "terminator_final"),
            FakeGraphEdge("terminator_final", "__end__"),
        ]


class FakeCompiledGraph:
    """模拟编译后的 LangGraph 图"""

    def get_graph(self):
        return FakeGraphObj()


@pytest.fixture(autouse=True)
def clear_workflow_cache():
    """每个工作流测试前清空 lru_cache,避免用例间缓存干扰"""
    from api.server import _workflow_snapshot

    _workflow_snapshot.cache_clear()
    yield
    _workflow_snapshot.cache_clear()


def test_get_workflow(client, mock_agent):
    """测试获取工作流结构与节点状态"""
    with patch("graph.registry.build_workflow", return_value=(FakeCompiledGraph(), {})):
        response = client.get("/api/workflow")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "simple"
    assert data["workflow_status"] == "idle"

    # 节点: 过滤掉 __start__/__end__ 哨兵
    node_ids = [n["id"] for n in data["nodes"]]
    assert node_ids == ["summarize", "manager_plan", "worker_exec", "terminator_final"]
    for node in data["nodes"]:
        assert node["status"] == "pending"
        assert node["label"] == node["id"]

    # 边: 哨兵节点映射为 START/END 标签,不再出现 __ 前缀
    for edge in data["edges"]:
        assert not edge["source"].startswith("__")
        assert not edge["target"].startswith("__")
    assert {"source": "START", "target": "summarize"} in data["edges"]
    assert {"source": "terminator_final", "target": "END"} in data["edges"]
    assert {"source": "summarize", "target": "manager_plan"} in data["edges"]


def test_get_workflow_cached(client, mock_agent):
    """测试工作流结构被缓存: 连续请求只构建一次"""
    calls = []

    def fake_build(name: str):
        calls.append(name)
        return FakeCompiledGraph(), {}

    with patch("graph.registry.build_workflow", side_effect=fake_build):
        r1 = client.get("/api/workflow")
        r2 = client.get("/api/workflow")

    assert r1.status_code == 200
    assert r2.status_code == 200
    # 缓存命中, build_workflow 只被调用一次
    assert len(calls) == 1
    assert calls[0] == "simple"


def test_get_workflow_unknown_name(client, mock_agent):
    """测试未知工作流名返回 404"""
    def fake_build(name: str):
        raise KeyError(f"未知工作流: {name}")

    with patch("graph.registry.build_workflow", side_effect=fake_build):
        response = client.get("/api/workflow?name=unknown_workflow")

    assert response.status_code == 404
    assert "未知工作流: unknown_workflow" in response.json()["detail"]


def test_get_workflow_build_error(client, mock_agent):
    """测试工作流构建异常返回 500"""
    def fake_build(name: str):
        raise RuntimeError("构建失败")

    with patch("graph.registry.build_workflow", side_effect=fake_build):
        response = client.get("/api/workflow")

    assert response.status_code == 500
    assert "工作流构建失败" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# 会话管理
# --------------------------------------------------------------------------- #
def test_list_threads(client, mock_agent, temp_checkpoint_db):
    """测试列出会话"""
    response = client.get("/api/threads")
    assert response.status_code == 200
    
    data = response.json()
    assert "threads" in data
    assert len(data["threads"]) == 2
    assert data["current"] == "test-thread-123"
    
    # 检查摘要信息
    for thread in data["threads"]:
        assert "thread_id" in thread
        assert "message_count" in thread
        assert "preview" in thread
        assert "type" in thread


def test_create_thread(client, mock_agent):
    """测试创建会话"""
    response = client.post("/api/threads")
    assert response.status_code == 200
    
    data = response.json()
    assert data["thread_id"] == "new-thread-456"
    
    mock_agent.memory.new_thread.assert_called_once()


def test_create_workflow_thread(client, mock_agent):
    """测试创建专属工作流会话"""
    response = client.post(
        "/api/threads",
        json={"type": "workflow", "workflow_name": "simple"},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["thread_id"] == "server-workflow-simple-abc12345"
    mock_agent.memory.new_workflow_thread.assert_called_once_with("simple")


def test_create_workflow_thread_default_name(client, mock_agent):
    """测试未指定工作流名时回退到 simple"""
    response = client.post(
        "/api/threads",
        json={"type": "workflow"},
    )
    assert response.status_code == 200
    mock_agent.memory.new_workflow_thread.assert_called_once_with("simple")


def test_list_workflows(client):
    """测试列出可用工作流名称"""
    with patch("graph.registry.WORKFLOWS", {"simple": None, "pipline": None}):
        response = client.get("/api/workflows")

    assert response.status_code == 200
    data = response.json()
    assert data["workflows"] == ["simple", "pipline"]


def test_delete_thread(client, mock_agent):
    """测试删除会话"""
    response = client.delete("/api/threads/thread-1")
    assert response.status_code == 200
    
    data = response.json()
    assert data["deleted"] is True
    assert data["thread_id"] == "thread-1"
    
    mock_agent.memory.delete_thread.assert_called_once_with("thread-1")


def test_delete_thread_not_found(client, mock_agent):
    """测试删除不存在的会话"""
    mock_agent.memory.delete_thread.return_value = False
    
    response = client.delete("/api/threads/nonexistent")
    assert response.status_code == 404


def test_get_thread_messages(client, mock_agent, temp_checkpoint_db):
    """测试读取会话消息"""
    from agent.memory import AgentMemory

    memory = AgentMemory(checkpoint_file=temp_checkpoint_db)
    mock_agent.memory.get_messages.side_effect = memory.get_messages
    try:
        response = client.get("/api/threads/thread-1/messages")
        assert response.status_code == 200

        data = response.json()
        assert data["thread_id"] == "thread-1"
        assert "messages" in data

        messages = data["messages"]
        assert len(messages) == 5

        # 验证消息结构
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "测试消息1"

        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "回复1"

        # 验证工具调用消息
        assert messages[3]["role"] == "assistant"
        assert "tool_calls" in messages[3]
        assert len(messages[3]["tool_calls"]) == 1
        assert messages[3]["tool_calls"][0]["name"] == "calculator"

        assert messages[4]["role"] == "tool"
        assert messages[4]["name"] == "calculator"
        assert messages[4]["content"] == "2"
    finally:
        memory.close()


# --------------------------------------------------------------------------- #
# SSE 流式聊天
# --------------------------------------------------------------------------- #
def test_chat_normal_conversation(client, mock_agent):
    """测试普通对话（SSE 流）"""
    response = client.post(
        "/api/chat",
        json={"message": "你好", "thread_id": "test-thread-123"}
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # 解析 SSE 事件
    events = []
    for line in response.iter_lines():
        # iter_lines() 返回字符串，不是字节
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    
    # 验证事件序列
    assert len(events) >= 2
    assert events[-1]["type"] == "done"
    
    # 验证 token 事件
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) > 0
    assert all("content" in e for e in token_events)


def test_chat_with_tool_calls(client, mock_agent):
    """测试包含工具调用的对话"""
    response = client.post(
        "/api/chat",
        json={"message": "帮我工具调用计算2+2", "thread_id": "test-thread-123"}
    )
    
    assert response.status_code == 200
    
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    
    # 验证工具调用事件
    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0]["name"] == "calculator"
    assert tool_call_events[0]["args"]["expression"] == "2+2"
    
    # 验证工具结果事件
    tool_result_events = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_result_events) == 1
    assert tool_result_events[0]["content"] == "4"
    
    # 验证最终完成
    assert events[-1]["type"] == "done"


def test_chat_create_new_thread(client, mock_agent):
    """测试未指定 thread_id 时自动创建"""
    response = client.post(
        "/api/chat",
        json={"message": "你好"}
    )
    
    assert response.status_code == 200
    
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    
    # 验证第一个事件是 thread_created
    assert events[0]["type"] == "thread_created"
    assert events[0]["thread_id"] == "new-thread-456"


def test_chat_execution_command(client, mock_agent):
    """测试执行型命令（json:）"""
    response = client.post(
        "/api/chat",
        json={"message": "/json: 生成配置", "thread_id": "test-thread-123"}
    )
    
    assert response.status_code == 200
    
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    
    # 验证返回 JSON 内容
    token_content = "".join(e["content"] for e in events if e["type"] == "token")
    assert "result" in token_content
    assert events[-1]["type"] == "done"


def test_chat_management_command(client, mock_agent):
    """测试管理型命令（help）"""
    with patch("api.server.dispatch_command") as mock_dispatch:
        mock_dispatch.return_value = "success"
        
        # Mock 输出捕获
        def side_effect(context, command):
            context.print_fn("可用命令列表")
            context.print_fn("- 输入 'help' 查看帮助")
            context.print_fn("- 输入 'info' 查看信息")
            return "success"
        
        mock_dispatch.side_effect = side_effect
        
        response = client.post(
            "/api/chat",
            json={"message": "/help", "thread_id": "test-thread-123"}
        )
        
        assert response.status_code == 200
        
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
        
        # 验证输出被包装成 token 事件
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) > 0
        
        # 验证表格格式转换
        content = "".join(e["content"] for e in token_events)
        assert "命令" in content or "help" in content


def test_chat_management_command_realtime_stream(client, mock_agent):
    """测试管理型命令输出实时推送:多段 print 输出是独立 token 事件而非合并一次"""
    with patch("api.server.dispatch_command") as mock_dispatch:
        def side_effect(context, command):
            context.print_fn("第一段输出")
            context.print_fn("第二段输出")
            return "success"

        mock_dispatch.side_effect = side_effect

        response = client.post(
            "/api/chat",
            json={"message": "/info", "thread_id": "test-thread-123"}
        )

        assert response.status_code == 200

        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        # 两段输出是独立 token 事件(实时推送),且按顺序到达,最后以 done 收尾
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["content"] == "第一段输出"
        assert token_events[1]["content"] == "第二段输出"
        assert events[-1]["type"] == "done"


def test_chat_workflow_events_forwarded(client, mock_agent):
    """测试工作流运行事件经 SSE 实时转发(workflow_node / workflow_status)"""
    with patch("api.server.dispatch_command") as mock_dispatch:
        def side_effect(context, command):
            context.print_fn("构建工作流: simple")
            # 模拟 run_workflow 内部经 workflow_event_cb 转发的事件
            if context.workflow_event_cb:
                context.workflow_event_cb({"type": "workflow_status", "status": "running"})
                context.workflow_event_cb({"type": "workflow_node", "node": "manager_plan", "status": "running"})
                context.workflow_event_cb({"type": "workflow_node", "node": "manager_plan", "status": "done"})
                context.workflow_event_cb({"type": "workflow_status", "status": "done"})
            context.print_fn("工作流执行完成")
            return "success"

        mock_dispatch.side_effect = side_effect

        response = client.post(
            "/api/chat",
            json={"message": "/workflow:simple 测试任务", "thread_id": "test-thread-123"}
        )

        assert response.status_code == 200

        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        types = [e["type"] for e in events]
        # 顺序:token → workflow_status → workflow_node → workflow_node → workflow_status → token → done
        assert types[0] == "token"
        assert types[1] == "workflow_status"
        assert events[1]["status"] == "running"
        assert types[2] == "workflow_node"
        assert events[2]["node"] == "manager_plan"
        assert events[2]["status"] == "running"
        assert types[3] == "workflow_node"
        assert events[3]["status"] == "done"
        assert types[4] == "workflow_status"
        assert events[4]["status"] == "done"
        assert types[-2] == "token"
        assert types[-1] == "done"


def test_chat_workflow_thread_auto_command(client, mock_agent):
    """测试专属工作流会话自动包装为 /workflow:<name> 命令"""
    mock_agent.memory.is_workflow_thread.return_value = True
    mock_agent.memory.workflow_name_of.return_value = "simple"

    with patch("api.server.dispatch_command") as mock_dispatch:
        def side_effect(context, command):
            context.print_fn(f"执行工作流命令: {command}")
            return "success"

        mock_dispatch.side_effect = side_effect

        response = client.post(
            "/api/chat",
            json={"message": "帮我分析项目", "thread_id": "server-workflow-simple-abc12345"},
        )

        assert response.status_code == 200

        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        token_content = "".join(e["content"] for e in events if e["type"] == "token")
        assert "workflow:simple 帮我分析项目" in token_content
        assert events[-1]["type"] == "done"


def test_chat_workflow_thread_explicit_command_not_wrapped(client, mock_agent):
    """测试工作流会话中显式以 / 开头的命令不被二次包装"""
    mock_agent.memory.is_workflow_thread.return_value = True
    mock_agent.memory.workflow_name_of.return_value = "simple"

    with patch("api.server.dispatch_command") as mock_dispatch:
        def side_effect(context, command):
            context.print_fn(f"显式命令: {command}")
            return "success"

        mock_dispatch.side_effect = side_effect

        response = client.post(
            "/api/chat",
            json={"message": "/help", "thread_id": "server-workflow-simple-abc12345"},
        )

        assert response.status_code == 200

        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        token_content = "".join(e["content"] for e in events if e["type"] == "token")
        assert "显式命令: help" in token_content
        assert "workflow:simple" not in token_content


# --------------------------------------------------------------------------- #
# HITL 恢复
# --------------------------------------------------------------------------- #
def test_chat_resume(client, mock_agent):
    """测试 HITL 恢复"""
    response = client.post(
        "/api/chat/resume",
        json={
            "payload": {"user_response": "确认执行"},
            "thread_id": "test-thread-123"
        }
    )
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    
    # 验证恢复流程
    assert len(events) >= 2
    assert events[-1]["type"] == "done"
    
    # 验证接收到用户回复
    token_content = "".join(e["content"] for e in events if e["type"] == "token")
    assert "确认执行" in token_content


def test_chat_resume_without_thread(client, mock_agent):
    """测试未指定 thread_id 的恢复"""
    response = client.post(
        "/api/chat/resume",
        json={"payload": {"user_response": "继续"}}
    )
    
    assert response.status_code == 200
    
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    
    assert events[-1]["type"] == "done"


# --------------------------------------------------------------------------- #
# 命令执行
# --------------------------------------------------------------------------- #
def test_execute_command_help(client, mock_agent):
    """测试执行 help 命令"""
    with patch("api.server.dispatch_command") as mock_dispatch:
        def side_effect(context, command):
            context.print_fn("可用命令：")
            context.print_fn("- 输入 'help' 查看帮助")
            context.print_fn("- 输入 'info' 查看配置信息")
            return "success"
        
        mock_dispatch.side_effect = side_effect
        
        response = client.post(
            "/api/command",
            json={"command": "/help", "thread_id": "test-thread-123"}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["outcome"] == "success"
        assert "命令" in data["output"] or "help" in data["output"]
        assert data["thread_id"] == "test-thread-123"


def test_execute_command_info(client, mock_agent):
    """测试执行 info 命令"""
    with patch("api.server.dispatch_command") as mock_dispatch:
        def side_effect(context, command):
            context.print_fn("当前配置：")
            context.print_fn("提供商：zhipu")
            context.print_fn("模型：glm-4-flash")
            return "success"
        
        mock_dispatch.side_effect = side_effect
        
        response = client.post(
            "/api/command",
            json={"command": "info"}
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "zhipu" in data["output"]
        assert "glm-4-flash" in data["output"]


def test_execute_command_threads_uses_current_thread(client, mock_agent):
    """测试 threads 命令不会因为 current 参数导致菜单模拟器崩溃"""
    with patch("api.server.dispatch_command") as mock_dispatch:
        def side_effect(context, command):
            selected = context.select_menu(
                "选择会话",
                [("thread-1 (当前)", "thread-1"), ("thread-2", "thread-2")],
                current="thread-1",
                action_keys={b"\x04": "delete"},
                hint="提示",
            )
            context.print_fn(f"selected={selected}")
            return "success"

        mock_dispatch.side_effect = side_effect

        response = client.post(
            "/api/command",
            json={"command": "threads", "thread_id": "thread-1"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "selected=thread-1" in data["output"]


def test_execute_command_with_exception(client, mock_agent):
    """测试命令执行异常"""
    with patch("api.server.dispatch_command") as mock_dispatch:
        mock_dispatch.side_effect = RuntimeError("命令执行失败")
        
        response = client.post(
            "/api/command",
            json={"command": "invalid"}
        )
        
        assert response.status_code == 500
        assert "命令执行失败" in response.text


# --------------------------------------------------------------------------- #
# 运行时指标
# --------------------------------------------------------------------------- #
def test_get_metrics(client, mock_agent):
    """测试获取运行时指标"""
    response = client.get("/api/metrics")
    assert response.status_code == 200

    data = response.json()
    assert "session" in data
    assert data["session"]["turn_count"] == 3
    assert "llm" in data
    assert data["llm"]["total_calls"] == 5
    assert data["llm"]["total_tokens"] == 1500
    assert "zhipu" in data["llm"]["by_provider"]
    assert "tools" in data
    assert data["tools"]["total_calls"] == 2
    assert "calculator" in data["tools"]["by_name"]
    assert "compaction" in data
    assert data["compaction"]["total_count"] == 1
    assert data["compaction"]["messages_saved"] == 38


def test_reset_metrics(client, mock_agent):
    """测试重置运行时指标"""
    response = client.post("/api/metrics/reset")
    assert response.status_code == 200

    data = response.json()
    assert data["reset"] is True
    mock_agent.metrics.reset.assert_called_once()


def test_get_metrics_no_agent(mock_llm):
    """测试 Agent 未初始化时返回 503"""
    with patch("api.server.agent", None), \
         patch("api.server.llm", mock_llm):
        from api.server import app
        with TestClient(app) as c:
            response = c.get("/api/metrics")
            assert response.status_code == 503


# --------------------------------------------------------------------------- #
# 上下文压缩
# --------------------------------------------------------------------------- #
def test_compact_success(client, mock_agent):
    """测试手动触发上下文压缩"""
    response = client.post("/api/compact", json={"thread_id": "test-thread-123"})
    assert response.status_code == 200

    data = response.json()
    assert data["compacted"] is True
    assert data["thread_id"] == "test-thread-123"
    assert data["messages_before"] == 60
    assert data["messages_after"] == 22
    assert "summary" in data


def test_compact_no_need(client, mock_agent):
    """测试消息数未超阈值时返回 compacted=False"""
    mock_agent.manually_compact = AsyncMock(return_value=None)
    response = client.post("/api/compact")
    assert response.status_code == 200

    data = response.json()
    assert data["compacted"] is False
    assert "message" in data


def test_compact_error(client, mock_agent):
    """测试压缩异常返回 500"""
    mock_agent.manually_compact = AsyncMock(side_effect=RuntimeError("压缩失败"))
    response = client.post("/api/compact")
    assert response.status_code == 500
    assert "压缩失败" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# 记忆管理
# --------------------------------------------------------------------------- #
def test_get_memory_summary(client, mock_agent):
    """测试获取记忆摘要"""
    response = client.get("/api/memory")
    assert response.status_code == 200

    data = response.json()
    assert data["thread_id"] == "test-thread-123"
    assert data["checkpoint_messages"] == 10
    assert data["long_term_count"] == 5
    assert data["total_threads"] == 3


def test_compress_memory(client, mock_agent):
    """测试压缩长期记忆"""
    response = client.post("/api/compress")
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["original_count"] == 5
    assert data["compressed_chars"] == 500
    mock_agent.compress_memory.assert_called_once()


def test_clear_memory_long(client, mock_agent):
    """测试清空长期记忆"""
    response = client.delete("/api/memory?scope=long")
    assert response.status_code == 200

    data = response.json()
    assert data["cleared"] is True
    assert data["scope"] == "long"
    mock_agent.memory.clear_long_term.assert_called_once()
    mock_agent.memory.clear_short_term.assert_not_called()


def test_clear_memory_short(client, mock_agent):
    """测试清空短期记忆"""
    response = client.delete("/api/memory?scope=short")
    assert response.status_code == 200

    data = response.json()
    assert data["scope"] == "short"
    mock_agent.memory.clear_short_term.assert_called_once()
    mock_agent.memory.clear_long_term.assert_not_called()


def test_clear_memory_all(client, mock_agent):
    """测试清空全部记忆"""
    response = client.delete("/api/memory?scope=all")
    assert response.status_code == 200

    data = response.json()
    assert data["scope"] == "all"
    mock_agent.memory.clear_long_term.assert_called_once()
    mock_agent.memory.clear_short_term.assert_called_once()


def test_clear_memory_invalid_scope(client, mock_agent):
    """测试无效 scope 返回 400"""
    response = client.delete("/api/memory?scope=invalid")
    assert response.status_code == 400


def test_clear_memory_default_scope(client, mock_agent):
    """测试默认 scope 为 long"""
    response = client.delete("/api/memory")
    assert response.status_code == 200
    assert response.json()["scope"] == "long"


# --------------------------------------------------------------------------- #
# 安全策略
# --------------------------------------------------------------------------- #
def test_get_safety(client):
    """测试获取安全策略配置"""
    with patch("api.server.safety_module") as mock_safety:
        mock_safety.load_config.return_value = {
            "mode": "blacklist",
            "confirm_dangerous": True,
            "blacklist": ["rm", "del"],
        }
        response = client.get("/api/safety")
        assert response.status_code == 200

        data = response.json()
        assert data["mode"] == "blacklist"
        assert data["confirm_dangerous"] is True


def test_update_safety_mode(client):
    """测试更新安全模式"""
    with patch("api.server.safety_module") as mock_safety:
        mock_safety.load_config.return_value = {
            "mode": "blacklist",
            "confirm_dangerous": True,
        }
        mock_safety.save_config.return_value = True
        response = client.put("/api/safety", json={"mode": "whitelist"})
        assert response.status_code == 200

        data = response.json()
        assert data["mode"] == "whitelist"
        mock_safety.save_config.assert_called_once()


def test_update_safety_confirm(client):
    """测试更新危险确认开关"""
    with patch("api.server.safety_module") as mock_safety:
        mock_safety.load_config.return_value = {
            "mode": "blacklist",
            "confirm_dangerous": True,
        }
        mock_safety.save_config.return_value = True
        response = client.put("/api/safety", json={"confirm_dangerous": False})
        assert response.status_code == 200

        data = response.json()
        assert data["confirm_dangerous"] is False


def test_update_safety_invalid_mode(client):
    """测试无效模式返回 400"""
    with patch("api.server.safety_module"):
        response = client.put("/api/safety", json={"mode": "invalid"})
        assert response.status_code == 400


def test_update_safety_save_failed(client):
    """测试保存失败返回 500"""
    with patch("api.server.safety_module") as mock_safety:
        mock_safety.load_config.return_value = {"mode": "blacklist", "confirm_dangerous": True}
        mock_safety.save_config.return_value = False
        response = client.put("/api/safety", json={"mode": "whitelist"})
        assert response.status_code == 500


# --------------------------------------------------------------------------- #
# 技能列表
# --------------------------------------------------------------------------- #
def test_get_skills(client, mock_agent):
    """测试获取技能列表"""
    response = client.get("/api/skills")
    assert response.status_code == 200

    data = response.json()
    assert len(data["skills"]) == 2
    assert data["skills"][0]["name"] == "pptx"
    assert data["skills"][1]["name"] == "pdf"


# --------------------------------------------------------------------------- #
# 会话导出
# --------------------------------------------------------------------------- #
def test_export_thread(client, mock_agent):
    """测试导出会话为文本"""
    response = client.get("/api/threads/thread-1/export")
    assert response.status_code == 200

    data = response.json()
    assert data["thread_id"] == "thread-1"
    assert data["format"] == "text"
    assert "测试消息" in data["content"]
    mock_agent.memory.export_thread.assert_called_once_with(thread_id="thread-1", fmt="text")


def test_export_thread_markdown(client, mock_agent):
    """测试导出会话为 Markdown"""
    response = client.get("/api/threads/thread-1/export?fmt=markdown")
    assert response.status_code == 200

    data = response.json()
    assert data["format"] == "markdown"
    mock_agent.memory.export_thread.assert_called_once_with(thread_id="thread-1", fmt="markdown")


def test_export_thread_invalid_format(client, mock_agent):
    """测试无效格式返回 400"""
    response = client.get("/api/threads/thread-1/export?fmt=pdf")
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# 辅助函数测试
# --------------------------------------------------------------------------- #
def test_format_help_as_table():
    """测试帮助文本转表格"""
    from api.server import _format_help_as_table
    
    help_text = """
    可用命令：
    - 输入 'help' 查看帮助
    - 输入 'info' 查看配置信息
    - 输入 'threads' 列出会话
    """
    
    result = _format_help_as_table(help_text)
    
    assert "##" in result
    assert "命令" in result
    assert "|" in result
    assert "`help`" in result
    assert "`info`" in result
    assert "`threads`" in result


def test_serialize_messages():
    """测试消息序列化"""
    from api.server import serialize_messages
    
    messages = [
        HumanMessage(content="用户消息"),
        AIMessage(content="助手回复"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "calc", "args": {"x": 1}}
            ]
        ),
        ToolMessage(content="结果", tool_call_id="c1", name="calc"),
    ]
    
    result = serialize_messages(messages)
    
    assert len(result) == 4
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "assistant"
    assert "tool_calls" in result[2]
    assert result[3]["role"] == "tool"


def test_thread_summary(temp_checkpoint_db):
    """测试会话摘要生成"""
    from types import SimpleNamespace

    from agent.memory import AgentMemory
    from api.server import thread_summary

    memory = AgentMemory(checkpoint_file=temp_checkpoint_db)
    try:
        with patch("api.server.agent", SimpleNamespace(memory=memory)):
            summary = thread_summary("thread-1")

            assert summary["thread_id"] == "thread-1"
            assert summary["message_count"] == 5
            assert len(summary["preview"]) > 0
    finally:
        memory.close()


def test_thread_summary_workflow_type(temp_checkpoint_db):
    """测试工作流会话摘要带类型与工作流名，普通会话不带"""
    from types import SimpleNamespace

    from agent.memory import AgentMemory
    from api.server import thread_summary

    memory = AgentMemory(checkpoint_file=temp_checkpoint_db, process_type="server")
    wf_tid = memory.new_workflow_thread("simple")
    try:
        with patch("api.server.agent", SimpleNamespace(memory=memory)):
            summary = thread_summary(wf_tid)
            assert summary["type"] == "workflow"
            assert summary["workflow_name"] == "simple"

            chat_summary = thread_summary("thread-1")
            assert chat_summary["type"] == "chat"
            assert "workflow_name" not in chat_summary
    finally:
        memory.close()


# --------------------------------------------------------------------------- #
# 并发与锁测试
# --------------------------------------------------------------------------- #
def test_chat_lock_serialization(mock_agent, mock_llm, temp_checkpoint_db):
    """测试聊天锁确保串行执行（简化版：验证锁存在即可）"""
    from api.server import chat_lock
    
    # 验证锁对象存在
    assert chat_lock is not None
    assert hasattr(chat_lock, 'locked')
    
    # 注：完整的并发测试需要 pytest-asyncio，此处仅验证锁机制存在


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
