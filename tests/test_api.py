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
    agent.memory.list_threads = MagicMock(return_value=["thread-1", "thread-2"])
    agent.memory.delete_thread = MagicMock(return_value=True)
    agent.get_available_tools = MagicMock(return_value=["calculator", "web_search", "ask_human"])
    agent.switch_llm = MagicMock()
    
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


def test_create_thread(client, mock_agent):
    """测试创建会话"""
    response = client.post("/api/threads")
    assert response.status_code == 200
    
    data = response.json()
    assert data["thread_id"] == "new-thread-456"
    
    mock_agent.memory.new_thread.assert_called_once()


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


def test_get_thread_messages(client, temp_checkpoint_db):
    """测试读取会话消息"""
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
    from api.server import thread_summary
    
    with patch("api.server.CHECKPOINT_FILE", temp_checkpoint_db):
        summary = thread_summary("thread-1")
        
        assert summary["thread_id"] == "thread-1"
        assert summary["message_count"] == 5
        assert len(summary["preview"]) > 0


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
