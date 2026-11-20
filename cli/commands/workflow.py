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


def _choice_options(choices: Any) -> list[tuple[str, str]]:
    """把 interrupt 的 choices 列表归一化为 select_menu 的 (label, id) 选项"""
    options: list[tuple[str, str]] = []
    for choice in choices or []:
        if isinstance(choice, dict):
            label = str(choice.get("label") or choice.get("id", ""))
            options.append((label, str(choice.get("id", ""))))
        else:
            options.append((str(choice), str(choice)))
    return options


def _render_interrupt_choices(ev: dict[str, Any], context: CommandContext) -> None:
    """渲染 interrupt 事件的 prompt + 选项（参考 human_input.render_human_interrupt）

    两种形态：
    - ``user_confirmation``（分组）：``items`` 内每个问题有独立选项，逐问题列出；
    - ``human_choice`` / ``dangerous_command``（单选）：顶层 ``choices`` 平铺。
    """
    prompt = str(ev.get("prompt") or "需要人工输入")
    context.print()
    for line in prompt.split("\n"):
        context.print(line.strip())
    items = ev.get("items") or []
    if items:
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", ""))
            question = str(item.get("question") or item_id)
            context.print(f"  [{item_id}] {question}")
            for cid, label in _choice_options(item.get("choices")):
                context.print(f"    - [{cid}] {label}")
        return
    for cid, label in _choice_options(ev.get("choices")):
        context.print(f"  [{cid}] {label}")


def _collect_interrupt_choice(
    ev: dict[str, Any],
    context: CommandContext,
) -> dict[str, Any]:
    """收集用户对 interrupt 的选择，返回 resume payload。

    约定：
    - ``user_confirmation``（``items`` 分组）：
      ``{"answers": {item_id: {"choice_id": selected_id}}}``
      （对照 tools/human_confirmation.py 的 resume 契约）
    - ``human_choice`` / ``dangerous_command``（顶层 choices 单选）：
      ``{"choice_id": "<id>"}``（对照 tools/safety.py 与 cli/human_input.py）
    - ``{"cancelled": True}`` -> 用户取消（Esc/空输入）
    """
    _render_interrupt_choices(ev, context)
    items = ev.get("items") or []
    if items:
        # 分组模式：每个问题独立选择，全部回答后一次性提交
        answers: dict[str, dict[str, str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", ""))
            options = _choice_options(item.get("choices"))
            if not options:
                continue
            selected = context.select_menu(
                str(item.get("question") or item_id), options
            )
            if selected is None:
                return {"cancelled": True}
            answers[item_id] = {"choice_id": selected}
        return {"answers": answers} if answers else {"cancelled": True}

    choices = ev.get("choices") or []
    if not choices:
        # 非结构化中断回退到自由文本
        text = context.input("请输入: ").strip()
        return {"text": text} if text else {"cancelled": True}

    selected = context.select_menu(str(ev.get("prompt") or "请选择"), _choice_options(choices))
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
        # workspace 按 session_id 存储在 session_workspaces 表，新生成的工作流会话
        # 默认没有任何绑定；若不继承，WorkspaceSecurityMW 会读到
        # configurable.workspace_path=None 而直接放行（相对路径不会被解析进 workspace）。
        # 因此把当前会话已绑定的 workspace 复制到新的工作流会话。
        inherited_workspace = await context.agent.session.aget_workspace(current_sid)
        if inherited_workspace:
            try:
                await context.agent.session.aset_workspace(
                    inherited_workspace, workflow_thread_id
                )
            except ValueError as exc:
                # 原绑定路径可能已被删除或不再是目录；继承失败不中断工作流，
                # 降级为「无 workspace」继续运行。
                logger.warning(
                    "工作流会话 %s 继承 workspace 失败（降级为无 workspace 运行）: %s",
                    workflow_thread_id,
                    exc,
                )
                # 仅记日志时用户无从感知，明确提示本次运行不受 workspace 限制
                context.print(
                    f"⚠ 工作流会话继承工作空间失败，本次以无工作空间运行: {exc}"
                )

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
            paused = False
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
                elif ev_type == "heartbeat":
                    # 静默期心跳：节点内 LLM 长流式 / 工具长执行期间事件泵补发，
                    # 必须转发给 SSE 客户端（前端仅重置看门狗、不渲染 UI），
                    # 否则工作流长节点会被前端 90s 看门狗误判"响应超时"拉黑线程。
                    _emit(ev_dict)
                elif ev_type == "interrupt":
                    # 先转发事件：SSE 需要立即把中断卡片推给前端
                    _emit(ev_dict)
                    if not context.interactive:
                        # 非交互入口（HTTP API）：不在后端替用户作答。直接退出，
                        # 把 pending interrupt 留在外层图里，等前端调用
                        # /api/chat/resume 携带答案恢复（否则会用 choices[0]
                        # 秒答，用户根本没机会选择）。
                        logger.info(
                            "[workflow] 非交互入口遇到中断，保留 pending interrupt 等待 resume"
                        )
                        paused = True
                        break
                    # 交互入口（CLI）：渲染选项并收集答案（resume 负载约定见
                    # _collect_interrupt_choice），由外层 while 续跑
                    resume_payload = _collect_interrupt_choice(ev_dict, context)
                    # 跳出内层 for，由外层 while 调 aresume_stream 续跑
                    break

            if paused or resume_payload is None or got_done:
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
    from graph.common import list_workflows
    
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
        from graph.common import WORKFLOWS
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
