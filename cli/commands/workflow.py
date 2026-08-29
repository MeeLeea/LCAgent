"""
Workflow 命令处理 - 多 Agent 工作流执行

通过统一 SessionManager 门面（workflow_sm，绑定 WorkflowAdapter）执行：
锁 / 记忆提交与沉淀 / checkpoint 跨轮次上下文 / 节点事件流全部由门面承载，
命令层只负责输出渲染与事件转发。
"""
from __future__ import annotations

import logging
from typing import Any

from .types import HANDLED, CommandContext, CommandOutcome

logger = logging.getLogger(__name__)


def _render_interrupt_choices(ev: dict[str, Any], context: CommandContext) -> None:
    """渲染 interrupt 事件的 prompt + choices（参考 human_input.render_human_interrupt）。"""
    prompt = str(ev.get("prompt") or "需要人工输入")
    context.print()
    for line in prompt.split("\n"):
        context.print(line.strip())
    choices = ev.get("choices") or []
    for choice in choices:
        if isinstance(choice, dict):
            cid = str(choice.get("id", ""))
            label = str(choice.get("label", cid))
            context.print(f"  [{cid}] {label}")


def _collect_interrupt_choice(
    ev: dict[str, Any],
    context: CommandContext,
) -> dict[str, Any]:
    """收集用户对 interrupt 的选择，返回 resume payload。

    约定（对照 tools/safety.py:606-609 与 cli/human_input.py:82-98）：
      {"choice_id": "<id>"}  -> 选择某个 choice
      {"cancelled": True}    -> 用户取消（Esc/空输入）
    """
    _render_interrupt_choices(ev, context)
    choices = ev.get("choices") or []
    if not choices:
        # 非结构化中断回退到自由文本
        text = context.input("请输入: ").strip()
        return {"text": text} if text else {"cancelled": True}

    options = []
    for choice in choices:
        if isinstance(choice, dict):
            label = str(choice.get("label") or choice.get("id", ""))
            options.append((label, str(choice.get("id", ""))))
        else:
            options.append((str(choice), str(choice)))

    selected = context.select_menu(
        str(ev.get("prompt") or "请选择"), options
    )
    if selected is None:
        return {"cancelled": True}
    return {"choice_id": selected}


async def run_workflow(context: CommandContext, name: str, task: str) -> dict:
    """
    运行指定工作流 - 异步版本

    经 ``context.workflow_sm.arun_stream`` 执行：workflow 专属会话在共享
    SessionRegistry 下持久化（checkpoint），记忆提交/沉淀由门面自动完成。

    Args:
        context: 命令上下文（需注入 workflow_sm）
        name: 工作流名称
        task: 用户任务

    Returns:
        工作流执行结果字典(包含 final_answer)
    """
    sm = context.workflow_sm
    if sm is None:
        raise RuntimeError("CommandContext 未注入 workflow_sm，无法执行工作流")

    context.print(f"\n构建工作流: {name}")

    # 确定工作流 thread_id：当前会话已是 workflow 专属会话（API 层通过
    # /api/session?type=workflow 创建）则复用其 thread_id 实现持久化绑定；
    # 否则生成新的工作流 thread_id（CLI 场景）。
    current_sid = context.agent.session.current_session_id
    if context.agent.session.is_workflow_session(current_sid):
        workflow_thread_id = current_sid
    else:
        workflow_thread_id = context.agent.session.generate_session_id(name)

    context.print(f"\n执行任务: {task}")
    context.print("-" * 50)

    # 节点/整体状态事件转发(供 SSE 等实时通道更新前端节点高亮)
    def _emit(event: dict[str, str]) -> None:
        if context.workflow_event_cb:
            context.workflow_event_cb(event)

    _emit({"type": "workflow_status", "status": "running"})
    final_answer = ""
    try:
        # 多轮 HITL：首次用 arun_stream，遇到 interrupt 收集用户答案后
        # 切到 aresume_stream 恢复；resume 可能再次中断，循环直至 done/error。
        resume_payload: dict[str, Any] | None = None
        while True:
            if resume_payload is None:
                stream = sm.arun_stream(task, thread_id=workflow_thread_id)
            else:
                stream = sm.aresume_stream(
                    resume_payload, thread_id=workflow_thread_id
                )
                resume_payload = None

            got_done = False
            async for ev_dict in stream:
                ev_type = ev_dict.get("type")
                if ev_type == "workflow_node":
                    node = ev_dict.get("node", "")
                    status = ev_dict.get("status", "")
                    if status == "running":
                        context.print(f"▸ 节点开始: {node}")
                    elif status == "done":
                        context.print(f"✓ 节点完成: {node}")
                    elif status == "error":
                        context.print(f"✗ 节点失败: {node}")
                    _emit(ev_dict)
                elif ev_type == "done":
                    final_answer = ev_dict.get("content", "")
                    got_done = True
                elif ev_type == "error":
                    context.print(
                        f"\n工作流执行错误: {ev_dict.get('content', '')}"
                    )
                    _emit(ev_dict)
                    got_done = True
                elif ev_type == "interrupt":
                    # 展示 prompt + choices 并收集用户答案（参考
                    # cli/human_input.py 的 render_human_interrupt +
                    # read_human_resume 模式；resume 负载约定
                    # {"choice_id": "<id>"} / {"cancelled": True}，
                    # 对照 tools/safety.py:606-609）
                    resume_payload = _collect_interrupt_choice(ev_dict, context)
                    _emit(ev_dict)
                    # 跳出内层 for，由外层 while 调 aresume_stream 续跑
                    break

            if resume_payload is None or got_done:
                break
    finally:
        # 无论成功失败都复位整体状态,避免前端 UI 停留在"运行中"
        _emit({"type": "workflow_status", "status": "done"})
    return {"final_answer": final_answer}


async def workflow_command(context: CommandContext, user_input: str) -> CommandOutcome:
    """
    处理 workflow 相关命令 - 异步版本
    
    支持命令:
        workflow              - 列出可用工作流
        workflow:<name> <task> - 运行指定工作流
    """
    # 导入放在函数内避免循环依赖
    from graph.registry import list_workflows
    
    if user_input.strip() == "workflow":
        # 列出可用工作流(含描述)
        context.print("\n可用工作流:")
        for wf_name, desc in list_workflows():
            if desc:
                context.print(f"  - {wf_name}: {desc}")
            else:
                context.print(f"  - {wf_name}")
        context.print("\n用法: workflow:<name> <task>")
        context.print("示例: workflow:simple 帮我分析一下项目结构")
        return HANDLED
    
    if user_input.startswith("workflow:"):
        # 解析命令: workflow:<name> <task>
        rest = user_input[9:].strip()  # 去掉 "workflow:"
        
        if not rest:
            context.print("\n用法: workflow:<name> <task>")
            context.print("示例: workflow:simple 帮我分析一下项目结构")
            return HANDLED
        
        # 分割工作流名称和任务
        parts = rest.split(None, 1)
        if len(parts) < 2:
            context.print("\n错误: 缺少任务描述")
            context.print("用法: workflow:<name> <task>")
            return HANDLED
        
        workflow_name = parts[0]
        task = parts[1]
        
        # 检查工作流是否存在
        from graph.registry import WORKFLOWS
        if workflow_name not in WORKFLOWS:
            available = ", ".join(name for name, _ in list_workflows())
            context.print(f"\n错误: 未知工作流 '{workflow_name}'")
            context.print(f"可用工作流: {available}")
            return HANDLED
        
        # 执行工作流
        try:
            result = await run_workflow(context, workflow_name, task)
            
            # 输出最终答案
            context.print("\n" + "=" * 50)
            context.print("工作流执行完成")
            context.print("=" * 50)
            context.print(f"\n{result.get('final_answer', '无结果')}")
            
        except Exception as e:
            context.print(f"\n工作流执行失败: {e}")
        
        return HANDLED
    
    # 不应该到这里,但保险起见返回未处理
    return CommandOutcome(handled=False, should_break=False)
