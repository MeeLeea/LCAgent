"""AgentEvent 事件模型测试

验证:
1. NODE_START / NODE_END / NODE_ERROR 工厂方法正确设置 event_type 与 node
2. node 字段默认值为空串，不影响既有事件类型
3. to_sse_dict 输出 workflow_node 格式（兼容前端协议）
4. NODE_* 事件不进入 is_terminal / is_memory_worthy（白名单排除）
5. 未知事件类型兜底带诊断字段
"""
from __future__ import annotations

from utils.events import AgentEvent, EventType


class TestNodeEvents:
    """NODE_* 事件工厂方法测试"""

    def test_node_start_factory(self):
        event = AgentEvent.node_start(node="manager_plan", thread_id="t1")
        assert event.event_type == EventType.NODE_START
        assert event.node == "manager_plan"
        assert event.thread_id == "t1"
        assert event.content == ""

    def test_node_end_factory(self):
        event = AgentEvent.node_end(node="worker_exec")
        assert event.event_type == EventType.NODE_END
        assert event.node == "worker_exec"
        assert event.content == ""

    def test_node_end_factory_with_content(self):
        """node_end 携带节点产出 content（供前端渲染节点结果块）"""
        event = AgentEvent.node_end(node="worker_exec", content="执行结果文本")
        assert event.event_type == EventType.NODE_END
        assert event.node == "worker_exec"
        assert event.content == "执行结果文本"

    def test_node_error_factory(self):
        event = AgentEvent.node_error(node="terminator_final")
        assert event.event_type == EventType.NODE_ERROR
        assert event.node == "terminator_final"

    def test_node_default_empty(self):
        """既有事件类型（如 TOKEN）不携带 node，node 默认空串"""
        event = AgentEvent.token("hello")
        assert event.node == ""


class TestNodeEventSseDict:
    """NODE_* 事件 to_sse_dict 输出测试（兼容前端 workflow_node 协议）"""

    def test_node_start_sse(self):
        event = AgentEvent.node_start(node="manager_plan")
        assert event.to_sse_dict() == {
            "type": "workflow_node",
            "node": "manager_plan",
            "status": "running",
        }

    def test_node_end_sse(self):
        event = AgentEvent.node_end(node="manager_plan")
        assert event.to_sse_dict() == {
            "type": "workflow_node",
            "node": "manager_plan",
            "status": "done",
        }

    def test_node_end_sse_with_content(self):
        """node_end 携带产出时，to_sse_dict 附带 content 键（向后兼容扩展）"""
        event = AgentEvent.node_end(node="manager_plan", content="计划内容")
        assert event.to_sse_dict() == {
            "type": "workflow_node",
            "node": "manager_plan",
            "status": "done",
            "content": "计划内容",
        }

    def test_node_error_sse(self):
        event = AgentEvent.node_error(node="manager_plan")
        assert event.to_sse_dict() == {
            "type": "workflow_node",
            "node": "manager_plan",
            "status": "error",
        }

    def test_all_event_types_have_sse_mapping(self):
        """所有 EventType 枚举值都有明确 to_sse_dict 映射，不得输出 unknown。

        锁死不变量：新增枚举必须同步添加 to_sse_dict 分支，否则测试失败。
        """
        for event_type in EventType:
            event = AgentEvent(event_type=event_type)
            sse = event.to_sse_dict()
            assert sse["type"] != "unknown", (
                f"EventType.{event_type.name} 缺少 to_sse_dict 分支"
            )


class TestNodeEventExclusion:
    """NODE_* 事件不进入 is_terminal / is_memory_worthy"""

    def test_node_events_not_terminal(self):
        """节点结束不等于整个工作流结束，NODE_* 不应标记为流终止"""
        assert not AgentEvent.node_start(node="n").is_terminal
        assert not AgentEvent.node_end(node="n").is_terminal
        assert not AgentEvent.node_error(node="n").is_terminal

    def test_node_events_not_memory_worthy(self):
        """节点进度不是记忆素材，NODE_* 不应提交给 MemoryManager"""
        assert not AgentEvent.node_start(node="n").is_memory_worthy
        assert not AgentEvent.node_end(node="n").is_memory_worthy
        assert not AgentEvent.node_error(node="n").is_memory_worthy