"""
LangChainAgent FastAPI 服务端
=============================

把原本纯 CLI 的 AgentCore 包装成 HTTP API，供 TypeScript 前端调用。

核心能力：
  - SSE 流式聊天（token 级增量 + 工具调用事件）
  - 会话(Thread)管理：列表 / 新建 / 删除 / 历史消息
  - 提供商与模型切换、工具列表

启动：
  python -m api.server                      # 默认提供商 zhipu
  python -m api.server --provider deepseek  # 指定提供商
  python -m api.server --port 8000 --host 0.0.0.0

前端开发时 Vite 代理到本服务（默认 http://127.0.0.1:8000）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

logger = logging.getLogger("api.server")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 确保项目根目录在 sys.path 中（支持 python -m api.server 与 python api/server.py）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import AgentCore
from agent.config import load_agent_config, resolve_path
from agent.llm_client import LLMClient, load_providers
from agent.message_utils import stringify_content  # 消息内容序列化
from cli.commands import CommandContext, dispatch_command
from cli.commands.provider import create_llm
from memory import MemoryContext
from tools import safety as safety_module

# --------------------------------------------------------------------------- #
# 路径常量（与 main.py 保持一致）
# --------------------------------------------------------------------------- #
LLM_FILE = os.path.join(BASE_DIR, "config", "llm_config.json")
MCP_CONFIG_FILE = os.path.join(BASE_DIR, "config", "mcp_servers.json")
AGENT_CONFIG_FILE = os.path.join(BASE_DIR, "agent", "agent_config.json")
SERVER_CONFIG_FILE = os.path.join(BASE_DIR, "config", "server_config.json")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "data", "checkpoints_async.sqlite")
WEB_DIST = os.path.join(BASE_DIR, "web", "dist")


def load_server_config() -> dict[str, Any]:
    """加载服务端口配置（config/server_config.json）。

    文件不存在或解析失败时回退到默认值（127.0.0.1:8000），不抛异常。
    """
    defaults: dict[str, Any] = {"host": "127.0.0.1", "port": 8000}
    if not os.path.exists(SERVER_CONFIG_FILE):
        return defaults
    try:
        with open(SERVER_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if isinstance(data.get("host"), str):
                defaults["host"] = data["host"]
            if isinstance(data.get("port"), int):
                defaults["port"] = data["port"]
    except (OSError, json.JSONDecodeError):
        pass
    return defaults

# --------------------------------------------------------------------------- #
# 全局状态
# --------------------------------------------------------------------------- #
agent: AgentCore | None = None
llm: LLMClient | None = None
_startup_provider: str | None = None
# 串行化对话轮次：AgentCore 是有状态单例，同一时刻只能跑一轮。
# 全局管理锁：管理型命令/管理接口串行（dispatch_command 会操作共享管理状态）
chat_lock = asyncio.Lock()

# per-thread 锁：普通对话/恢复/压缩按会话并发，同会话内串行
# 不同会话各自持有独立锁，互不阻塞；同一会话的请求仍严格排队
_thread_locks: dict[str, asyncio.Lock] = {}
_THREAD_LOCKS_MAX = 200


def _thread_lock(thread_id: str) -> asyncio.Lock:
    """获取指定会话的专用锁（不存在则创建）。

    单线程事件循环内 get-or-create 是原子的，无需额外加锁。
    超过容量上限时淘汰未持锁的旧锁，防止长期运行后内存无限增长。
    """
    lock = _thread_locks.get(thread_id)
    if lock is None:
        lock = asyncio.Lock()
        _thread_locks[thread_id] = lock
        if len(_thread_locks) > _THREAD_LOCKS_MAX:
            _prune_thread_locks()
    return lock


def _prune_thread_locks() -> None:
    """清理未持有且过量的 per-thread 锁。"""
    if len(_thread_locks) <= _THREAD_LOCKS_MAX:
        return
    idle = [tid for tid, lock in _thread_locks.items() if not lock.locked()]
    excess = len(_thread_locks) - _THREAD_LOCKS_MAX
    for tid in idle[:excess]:
        del _thread_locks[tid]


async def build_agent(provider: str) -> tuple[AgentCore, LLMClient]:
    """根据提供商初始化 LLM 与 Agent（逻辑与 main.py 一致，去掉 CLI 打印）。"""
    new_llm = LLMClient(provider=provider, config_file=LLM_FILE)
    cfg = load_agent_config(AGENT_CONFIG_FILE)
    agent_prompt_file = cfg.get("agent_prompt_file")
    skills_dir = resolve_path(cfg["skills_dir"], BASE_DIR)
    mcp_config_file = resolve_path(cfg["mcp_config_file"], BASE_DIR)
    # 三层架构：先创建 MemoryContext（记忆基础设施），再创建 AgentCore（纯执行内核）
    memory_ctx = await MemoryContext.acreate(
        checkpoint_file=CHECKPOINT_FILE,
        short_term_size=cfg["memory_size"],
        use_sqlite=True,
        process_type="server",
        llm_getter=lambda: new_llm,
        buffer_delay_seconds=cfg.get("memory_buffer_delay_seconds", 20),
        max_buffer_messages=cfg.get("memory_max_buffer_messages", 30),
        max_facts_per_thread=cfg.get("memory_max_facts_per_thread", 50),
        recall_limit=cfg.get("memory_recall_limit", 10),
    )
    new_agent = await AgentCore.acreate(
        llm_client=new_llm,
        name=cfg["name"],
        max_iterations=cfg["max_iterations"],
        verbose=cfg["verbose"],
        mcp_config_file=mcp_config_file,
        enable_mcp=cfg["enable_mcp"],
        skills_dir=skills_dir,
        auto_match_skills=cfg["auto_match_skills"],
        max_context_messages=cfg["max_context_messages"],
        context_trim_keep=cfg["context_trim_keep"],
        process_type="server",
        agent_prompt_file=agent_prompt_file,
        max_execution_history=cfg.get("max_execution_history", 100),
        tool_timeout=cfg.get("tool_timeout", 120),
        checkpointer=memory_ctx.checkpointer,
        store=memory_ctx.store,
        extra_middleware=[memory_ctx.read_middleware],
        initial_thread_id=memory_ctx.thread_id,
        async_conn=memory_ctx.async_conn,
    )
    # 注入 MemoryManager → SessionManager 懒初始化时会自动接收
    new_agent.set_memory_manager(memory_ctx.memory_manager)
    new_agent._memory_context = memory_ctx  # 供 aclose 时关闭 SQLite 连接
    # 动态绑定：记忆组件直接读取 agent 当前 LLM，切换 provider 后自动同步
    # （修复 /api/providers/switch 后记忆抽取仍用启动时旧 LLMClient 的问题）
    memory_ctx.bind_llm(lambda: new_agent.llm)
    return new_agent, new_llm


@asynccontextmanager
async def lifespan(_: FastAPI):
    """在 FastAPI 生命周期内创建和释放 Agent 资源。"""
    global agent, llm
    provider = _startup_provider or pick_default_provider()
    logger.info("初始化提供商: %s", provider)
    agent, llm = await build_agent(provider)
    safety_module.set_confirm_backend(safety_module.interrupt_confirm)
    info = llm.get_info()
    logger.info("模型: %s / %s", info["provider_name"], info["model"])
    logger.info("工具: %s", ", ".join(agent.get_available_tools()))
    try:
        yield
    finally:
        if agent is not None:
            await agent.session_manager.aclose()
            # 关闭 MemoryContext（释放 SQLite 连接等底层资源）
            mem_ctx = getattr(agent, "_memory_context", None)
            if mem_ctx is not None:
                await mem_ctx.aclose()
        agent = None
        llm = None


def pick_default_provider() -> str:
    """选择默认提供商：优先有 api_key 的，否则回退 zhipu。"""
    providers = load_providers(LLM_FILE)
    for key, conf in providers.items():
        if conf.get("api_key") or os.environ.get(conf.get("env_key", "")):
            return key
    return "zhipu" if "zhipu" in providers else (next(iter(providers), "zhipu"))


def serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """把 LangGraph 消息对象序列化为前端可消费的 JSON。"""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": stringify_content(m.content)})
        elif isinstance(m, AIMessage):
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": stringify_content(m.content),
            }
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")}
                    for tc in tool_calls
                ]
            out.append(entry)
        elif isinstance(m, ToolMessage):
            out.append({
                "role": "tool",
                "id": getattr(m, "tool_call_id", ""),
                "name": getattr(m, "name", "") or "",
                "content": stringify_content(m.content),
            })
    return out


async def thread_summary(thread_id: str) -> dict[str, Any]:
    """单个会话的摘要信息（消息数 + 预览 + 会话类型）。"""
    msgs = await agent.session.aget_messages(session_id=thread_id) if agent else []
    preview = ""
    for m in msgs:
        if isinstance(m, HumanMessage):
            preview = stringify_content(m.content).strip().replace("\n", " ")[:50]
            break
    if not preview and msgs:
        preview = stringify_content(msgs[-1].content).strip().replace("\n", " ")[:50]
    summary: dict[str, Any] = {
        "thread_id": thread_id,
        "message_count": len(msgs),
        "preview": preview,
        "type": "chat",
    }
    # 专属工作流会话：标注类型并带上绑定的工作流名，前端据此区分展示
    if agent and agent.session.is_workflow_session(thread_id):
        summary["type"] = "workflow"
        summary["workflow_name"] = agent.session.workflow_name_of(thread_id)
    return summary


# --------------------------------------------------------------------------- #
# Pydantic 模型
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class CreateThreadRequest(BaseModel):
    type: str | None = "chat"
    workflow_name: str | None = None
    workspace_path: str | None = None


class ResumeRequest(BaseModel):
    payload: dict[str, Any]
    thread_id: str | None = None


class SwitchProviderRequest(BaseModel):
    provider: str


class SwitchModelRequest(BaseModel):
    model: str


class CommandRequest(BaseModel):
    command: str
    thread_id: str | None = None


class SwitchRoleRequest(BaseModel):
    role: str
    task: str | None = None


class SafetyUpdateRequest(BaseModel):
    mode: str | None = None
    confirm_dangerous: bool | None = None


class SetWorkspaceRequest(BaseModel):
    path: str


# --------------------------------------------------------------------------- #
# FastAPI 应用
# --------------------------------------------------------------------------- #
app = FastAPI(title="LangChainAgent API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    info = llm.get_info() if llm else {}
    return {
        "status": "ok",
        "provider": info.get("provider"),
        "model": info.get("model"),
        "thread_id": agent.session.current_session_id if agent else None,
    }


@app.get("/api/providers")
async def get_providers():
    """列出全部提供商与模型（脱敏，不含 api_key）。"""
    providers = load_providers(LLM_FILE)
    current = llm.get_info() if llm else {}
    items = []
    for key, conf in providers.items():
        items.append({
            "key": key,
            "name": conf.get("name", key),
            "base_url": conf.get("base_url", ""),
            "models": conf.get("models", []),
            "has_key": bool(conf.get("api_key") or os.environ.get(conf.get("env_key", ""))),
        })
    return {
        "providers": items,
        "current_provider": current.get("provider"),
        "current_provider_name": current.get("provider_name"),
        "current_model": current.get("model"),
    }


@app.post("/api/providers/switch")
async def switch_provider(req: SwitchProviderRequest):
    logger.info("切换提供商: %s", req.provider)
    async with chat_lock:
        global llm
        try:
            new_llm = LLMClient(provider=req.provider, config_file=LLM_FILE)
        except Exception as e:
            logger.error("切换失败: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
        await agent.aswitch_llm(new_llm)
        llm = new_llm
    info = llm.get_info()
    logger.info("已切换 → %s / %s", info["provider_name"], info["model"])
    return info


@app.post("/api/models/switch")
async def switch_model(req: SwitchModelRequest):
    logger.info("切换模型: %s", req.model)
    async with chat_lock:
        try:
            llm.switch_model(req.model)
            await agent.aswitch_llm(llm)
        except Exception as e:
            logger.error("切换失败: %s", e)
            raise HTTPException(status_code=400, detail=str(e))
    info = llm.get_info()
    logger.info("已切换 → %s / %s", info["provider_name"], info["model"])
    return info


@app.get("/api/tools")
async def get_tools():
    return {"tools": agent.get_available_tools() if agent else []}


@app.get("/api/roles")
async def get_roles():
    """列出 team/ 下的可用团队角色与当前角色名（对应 CLI 的 role 命令）。"""
    from agent.role_sw import get_available_team_roles

    return {
        "roles": get_available_team_roles(),
        "current": agent.name if agent else None,
    }


@app.post("/api/roles/switch")
async def switch_role(req: SwitchRoleRequest):
    """切换主对话 Agent 的团队角色。

    就地把 AgentCore 重建为 team/<role>/ 定义的角色（提示词/LLM）。
    可选 task：切换后由角色自动匹配注入相应技能。

    错误映射：未知角色 → 404；角色提示词文件为空 → 400；其他异常 → 500。
    """
    logger.info("切换团队角色: %s", req.role)
    async with chat_lock:
        try:
            await agent.arebuild_from_team_dir(req.role, task=req.task or "")
        except KeyError as e:
            from agent.role_sw import get_available_team_roles

            available = ", ".join(get_available_team_roles()) or "(无)"
            logger.warning("角色不存在 [%s]，可用: %s", req.role, available)
            raise HTTPException(status_code=404, detail=f"{e}")
        except FileNotFoundError as e:
            logger.error("角色提示词读取失败 [%s]: %s", req.role, e)
            raise HTTPException(status_code=400, detail=f"{e}")
        except (RuntimeError, ValueError) as e:
            logger.error("切换角色失败 [%s]: %s", req.role, e)
            raise HTTPException(status_code=500, detail=f"{e}")
    logger.info("已切换到团队角色: %s", req.role)
    return {"role": req.role, "current": agent.name if agent else None}


# LangGraph 内部哨兵节点与前端友好标签的映射
_SENTINEL_LABELS = {"__start__": "START", "__end__": "END"}


@lru_cache(maxsize=8)
def _workflow_snapshot(name: str) -> dict:
    """
    构建工作流并提取结构快照（带进程级缓存）。

    缓存的是最终 JSON 快照而非 graph 实例，避免 LLM Client 等资源长期占用。
    哨兵节点(__start__/__end__)从 nodes 中过滤,并在 edges 中映射为 START/END 标签。
    节点状态为初始 pending；运行时的实时进度由 /api/chat 的 SSE 事件
    (workflow_node / workflow_status)推送，前端据此更新本快照的节点状态。

    从全局 agent.session 获取 checkpointer 注入 build_workflow，使图编译带持久化，
    与 CLI 执行路径保持一致。缓存仅存储结构 JSON（nodes/edges），
    checkpointer 不影响图结构，因此 lru_cache 按 name 缓存安全。

    Args:
        name: 工作流名称

    Returns:
        包含 name/nodes/edges 的结构字典

    Raises:
        KeyError: 工作流名称不存在
    """
    from graph.registry import build_workflow

    # 从全局 agent 获取 checkpointer（API 层持久化注入）
    checkpointer = None
    if agent is not None:
        checkpointer = getattr(agent.session, "checkpointer", None)

    graph, _ = build_workflow(name, checkpointer=checkpointer)

    graph_obj = graph.get_graph()
    nodes = [
        {"id": n.id, "label": n.id, "status": "pending"}
        for n in graph_obj.nodes.values()
        # 过滤掉内部哨兵节点(__start__/__end__),仅展示真实业务节点
        if not n.id.startswith("__")
    ]
    edges = [
        {
            "source": _SENTINEL_LABELS.get(e.source, e.source),
            "target": _SENTINEL_LABELS.get(e.target, e.target),
        }
        for e in graph_obj.edges
    ]
    return {"name": name, "nodes": nodes, "edges": edges}


@app.get("/api/workflow")
async def get_workflow(name: str = "simple"):
    """获取工作流结构图与节点状态（节点初始为 pending；运行进度通过 /api/chat 的 SSE 事件实时推送）。"""
    try:
        snapshot = _workflow_snapshot(name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("构建工作流失败 [%s]: %s", name, e)
        raise HTTPException(status_code=500, detail=f"工作流构建失败: {e}")

    snapshot["workflow_status"] = "idle"
    return snapshot


@app.get("/api/workflows")
async def list_workflows():
    """列出全部可用工作流名称（供前端切换选择）。"""
    from graph.registry import WORKFLOWS

    return {"workflows": list(WORKFLOWS.keys())}


@app.get("/api/threads")
async def list_threads():
    """列出所有会话（按消息数倒序，便于最近活跃的靠前）。"""
    ids = await agent.session.alist_sessions() if agent else []
    summaries = [await thread_summary(tid) for tid in ids]
    summaries.sort(key=lambda x: x["message_count"], reverse=True)
    return {"threads": summaries, "current": agent.session.current_session_id if agent else None}


@app.post("/api/threads")
async def create_thread(req: CreateThreadRequest | None = None):
    """新建会话，返回 thread_id。

    type=workflow 时创建专属工作流会话。
    workspace_path 指定会话绑定的外部工作空间路径（可选，用于多会话工作目录隔离）。
    """
    async with chat_lock:
        if req and req.type == "workflow":
            workflow_name = req.workflow_name or "simple"
            tid = agent.session.new_workflow_session(
                workflow_name,
                workspace_path=req.workspace_path if req else None,
            )
            agent.set_current_session(tid)
        else:
            tid = agent.session.new_session(
                workspace_path=req.workspace_path if req else None,
            )
            agent.set_current_session(tid)
    logger.info("创建会话: %s (workspace=%s)", tid, req.workspace_path if req else None)
    return {"thread_id": tid}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    logger.info("删除会话: %s", thread_id)
    ok = await agent.session.adelete_session(thread_id)
    if not ok:
        logger.warning("删除失败: %s", thread_id)
        raise HTTPException(status_code=404, detail="会话不存在或删除失败")
    return {"deleted": True, "thread_id": thread_id}


@app.get("/api/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    msgs = await agent.session.aget_messages(session_id=thread_id) if agent else []
    return {"thread_id": thread_id, "messages": serialize_messages(msgs)}


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _get_total_tokens() -> int:
    """获取当前 LLM 累计 total_tokens（供前端输入栏实时展示）"""
    metrics = getattr(agent, "metrics", None) if agent else None
    if metrics is None:
        return 0
    try:
        return metrics.get_summary()["llm"]["total_tokens"]
    except (KeyError, TypeError):
        return 0


def _enrich_done(ev: dict[str, Any]) -> dict[str, Any]:
    """为 done 事件附加 total_tokens，前端据此更新输入栏 token 计数"""
    if ev.get("type") == "done":
        ev["total_tokens"] = _get_total_tokens()
    return ev


def _format_help_as_table(help_text: str) -> str:
    """将帮助文本转换成 Markdown 表格格式。"""
    lines = help_text.strip().split('\n')

    # 提取命令行（以 "- 输入" 开头的行）
    commands = []
    for line in lines:
        line = line.strip()
        if line.startswith("- 输入"):
            # 移除 "- 输入 " 前缀
            cmd_desc = line[5:].strip()
            # 分割命令和描述：命令通常用单引号包裹
            if cmd_desc.startswith("'"):
                end_quote = cmd_desc.find("'", 1)
                if end_quote != -1:
                    cmd = cmd_desc[1:end_quote]
                    desc = cmd_desc[end_quote + 1:].strip()
                else:
                    cmd, desc = cmd_desc, ""
            else:
                parts = cmd_desc.split(" ", 1)
                cmd = parts[0]
                desc = parts[1] if len(parts) > 1 else ""
            commands.append((cmd, desc))

    if not commands:
        return help_text

    # 构建 Markdown 表格
    table_lines = [
        "## 可用命令\n",
        "| 命令 | 说明 |",
        "|------|------|",
    ]
    for cmd, desc in commands:
        cmd = cmd.replace('|', '\\|')
        desc = desc.replace('|', '\\|')
        table_lines.append(f"| `{cmd}` | {desc} |")

    return "\n".join(table_lines)


def _is_execution_command(command: str) -> bool:
    """判断命令是否为"执行型"（需要走流式 Runner，而非管理型同步命令）。

    执行型命令会把任务交给 Agent 流式执行：
    - ``json:`` / ``react:`` / ``cot:`` 直接进入任务执行
    - ``skill:<name> <task>`` 带任务文本时执行；``skill:clear`` / 仅列出技能时是管理型

    Args:
        command: 去掉了前导 ``/`` 的命令字符串

    Returns:
        是否为执行型命令。是则走 session_manager.achat_stream 流式通道，否则走 dispatch_command 管理通道。
    """
    low = command.lower()
    if low.startswith(("json:", "react:", "cot:")):
        return True
    if low.startswith("skill:"):
        # skill:<name> <task> 是执行型；skill:clear 或 skill:<name>（无任务）是管理型
        rest = command[6:].strip()
        parts = rest.split(None, 1)
        skill_name = parts[0].strip() if parts else ""
        task_text = parts[1].strip() if len(parts) > 1 else ""
        return bool(skill_name and task_text and skill_name.lower() not in ("clear", "清空", "reset"))
    return False


def _unsupported_runner(agent_obj: object, text: str) -> str:
    """Web 端 Runner 兜底：不支持交互式执行时返回明确提示，避免静默 `None`。

    当管理型命令内部嵌套执行型调用（如 ``role:<name> <task>``）时，
    返回标记文本代替 `None`，由命令层打印出来，用户能看到原因而非空结果。
    """
    return "[Web 通道暂不支持该嵌套执行，请改用 json:/react:/普通对话发送]"


# 双方共用：捕获执行型命令统一走流式，其余走管理型 dispatch_command
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式聊天。支持普通对话和命令执行（如 /skill:pptx <task>、/json:、/react: 等）。"""

    async def event_stream():
        # 确定会话 ID（new_thread 在事件循环内原子执行，无需加锁）
        is_new_thread = req.thread_id is None
        tid = req.thread_id if req.thread_id else agent.session.new_session()
        agent.set_current_session(tid)

        message = req.message.strip()

        # 专属工作流会话：未显式以 / 开头时，自动包装为 /workflow:<name> 命令执行
        if agent.session.is_workflow_session(tid):
            workflow_name = agent.session.workflow_name_of(tid)
            if workflow_name and not message.startswith('/'):
                message = f"/workflow:{workflow_name} {message}"
                logger.info("工作流会话 [%s] 自动包装命令: %s", tid, message)

        # 区分命令类型：
        # - 执行型命令（json:/react:/cot:/skill:<task>）与普通对话走流式通道，用 per-thread 锁（并发多会话）
        # - 管理型命令走 dispatch_command，用全局锁（其内部会切换/新建当前会话等共享管理状态）
        command_mode = message.startswith('/')
        command = message.lstrip('/') if command_mode else ""
        is_management = command_mode and not _is_execution_command(command)
        lock = chat_lock if is_management else _thread_lock(tid)

        preview = message.replace("\n", " ")[:50]
        logger.info("收到聊天 [%s]: %s", tid, preview)

        async with lock:
            if is_new_thread:
                yield _sse({"type": "thread_created", "thread_id": tid})

            if command_mode:
                # 命令模式：通过 dispatch_command 处理
                logger.info("检测到命令 [%s]: %s", tid, command)
                
                # 文本输出包装器：管理型命令的输出包成 token 事件
                output_buffer = []
                # 管理型命令实时输出队列：print 输出与工作流结构化事件入队，主循环边收边推，
                # 避免 dispatch_command 长耗时（如工作流）运行期间前端收不到任何进度
                output_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
                # help 命令需要全量输出做表格转换，禁用实时推送，最后统一推送表格
                is_help = command.lower() == "help"

                def capture_output(text: str):
                    output_buffer.append(text)
                    if not is_help:
                        output_queue.put_nowait(("print", text))

                def emit_workflow_event(event: dict[str, str]):
                    """工作流节点/整体状态事件：实时转发给前端 SSE"""
                    output_queue.put_nowait(("event", event))

                try:
                    context = CommandContext(
                        agent=agent,
                        print_fn=capture_output,
                        input_fn=lambda prompt="": "",  # 不支持交互输入
                        select_menu=lambda title, choices: choices[0] if choices else "",
                        create_llm=lambda provider: create_llm(provider, LLM_FILE),
                        list_providers=lambda: load_providers(LLM_FILE),
                        run_structured_until_completion=_unsupported_runner,  # 嵌套执行返回明确提示,不静默 None
                        chat_until_completion=_unsupported_runner,  # 同上
                        safety_backend=safety_module,
                        base_dir=BASE_DIR,
                        config_file=AGENT_CONFIG_FILE,
                        mcp_config_file=MCP_CONFIG_FILE,
                        workflow_event_cb=emit_workflow_event,
                    )
                    
                    # 执行型命令（json:/react:/cot:/skill:<task>）走流式通道
                    if _is_execution_command(command):
                        # 执行型命令：走流式 runner，显式传 thread_id 保证多会话隔离
                        logger.info("执行型命令 [%s]，走流式通道", tid)
                        # 把命令原文传给 achat_stream（它会自动匹配技能）
                        async for ev in agent.session_manager.achat_stream(command, thread_id=tid):
                            yield _sse(_enrich_done(ev))
                        logger.info("完成 [%s]", tid)
                        return
                    else:
                        # 管理型命令：dispatch_command 在当前事件循环执行，print 输出/工作流事件
                        # 经队列实时推送，保证长耗时命令（如 workflow）运行期间前端持续收到进度
                        logger.info("管理型命令 [%s]", tid)

                        async def _run_dispatch() -> None:
                            """执行 dispatch_command，结果/异常经队列送回主循环"""
                            try:
                                outcome = await dispatch_command(context, command)
                                output_queue.put_nowait(("outcome", outcome))
                            except Exception as e:
                                logger.error("命令执行异常 [%s]: %s", tid, e)
                                output_queue.put_nowait(("error", e))

                        dispatch_task = asyncio.create_task(_run_dispatch())

                        # 实时消费队列：print → token 事件；workflow 事件 → 结构化 SSE 事件
                        outcome = None
                        while True:
                            kind, payload = await output_queue.get()
                            if kind == "print":
                                yield _sse({"type": "token", "content": payload})
                            elif kind == "event":
                                yield _sse(payload)
                            elif kind == "outcome":
                                outcome = payload
                                break
                            else:  # error
                                yield _sse({"type": "error", "content": f"命令执行失败: {payload}"})
                                return

                        await dispatch_task  # 确保命令任务完全退出

                        # help 命令：全量输出统一转表格后推送（实时通道已跳过）
                        if is_help:
                            output = _format_help_as_table("\n".join(output_buffer))
                            if output:
                                yield _sse({"type": "token", "content": output})

                        yield _sse({"type": "done", "total_tokens": _get_total_tokens()})
                        logger.info("命令完成 [%s]: %s", tid, outcome)
                        return
                        
                except Exception as e:
                    logger.error("命令执行异常 [%s]: %s", tid, e)
                    yield _sse({"type": "error", "content": f"命令执行失败: {e}"})
                    return
            
            # 普通对话模式（显式传 thread_id 实现多会话隔离）
            try:
                async for event in agent.session_manager.achat_stream(message, thread_id=tid):
                    yield _sse(_enrich_done(event))
                logger.info("完成 [%s]", tid)
            except Exception as e:
                logger.error("异常 [%s]: %s", tid, e)
                yield _sse({"type": "error", "content": f"内部错误: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/chat/resume")
async def chat_resume(req: ResumeRequest):
    """SSE 流式恢复被 ask_human 中断的会话。"""

    async def event_stream():
        # 恢复必须带 thread_id（前端在会话被中断时已知线程）；缺失时回退当前会话
        tid = req.thread_id or agent.session.current_session_id
        logger.info("恢复会话 [%s]", tid)
        async with _thread_lock(tid):
            try:
                async for event in agent.session_manager.aresume_stream(req.payload, thread_id=tid):
                    yield _sse(event)
                logger.info("恢复完成 [%s]", tid)
            except Exception as e:
                logger.error("恢复异常 [%s]: %s", tid, e)
                yield _sse({"type": "error", "content": f"内部错误: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/command")
async def execute_command(req: CommandRequest):
    """执行 CLI 命令（如 /help, /info, /threads 等）。"""
    async with chat_lock:
        if req.thread_id:
            agent.set_current_session(req.thread_id)
        tid = req.thread_id or agent.session.current_session_id
        
        # 去掉前导 / 符号（CLI 命令不需要 /）
        command = req.command.lstrip('/')
        logger.info("执行命令 [%s]: %s", tid, command)
        
        # 构造命令上下文（模拟 CLI 环境）
        output_lines = []
        def capture_output(text: str):
            output_lines.append(text)
        
        # 对于需要交互的命令，提供一个假的 input 函数（返回空字符串或默认值）
        def fake_input(prompt: str = "") -> str:
            output_lines.append(f"[交互提示] {prompt}")
            return ""
        
        # 对于菜单选择，优先返回当前项，否则返回第一项
        def fake_select_menu(
            title: str,
            choices: list[str],
            current: str | None = None,
            action_keys: dict | None = None,
            hint: str | None = None,
        ) -> str | tuple[str, str] | None:
            if action_keys:
                # 命令模式下不模拟二次动作，直接忽略快捷键分支
                action_keys = None
            if current is not None:
                for choice in choices:
                    if isinstance(choice, tuple) and len(choice) == 2:
                        label, value = choice
                        if value == current or str(label) == current:
                            return value
                    elif choice == current:
                        return choice
            if choices:
                first = choices[0]
                if isinstance(first, tuple) and len(first) == 2:
                    return first[1]
                return first
            return None
        
        # 不支持的命令会 fallback 到聊天模式，提示用户用正常对话
        async def unsupported_runner(agent_obj, text: str) -> str:
            capture_output("未知命令。请直接发送消息进行对话，或输入 /help 查看可用命令。")
            return ""
        
        try:
            context = CommandContext(
                agent=agent,
                print_fn=capture_output,
                input_fn=fake_input,
                select_menu=fake_select_menu,
                create_llm=lambda provider: create_llm(provider, LLM_FILE),
                list_providers=lambda: load_providers(LLM_FILE),
                run_structured_until_completion=unsupported_runner,
                chat_until_completion=unsupported_runner,
                safety_backend=safety_module,
                base_dir=BASE_DIR,
                config_file=AGENT_CONFIG_FILE,
                mcp_config_file=MCP_CONFIG_FILE,
            )
            
            outcome = await dispatch_command(context, command)
            output = "\n".join(output_lines)
            
            # 如果是 help 命令，将输出转换成表格格式
            if command.lower() == "help":
                output = _format_help_as_table(output)
            
            logger.info("命令完成 [%s]: %s", tid, outcome)
            return {
                "success": outcome != "quit",
                "outcome": outcome,
                "output": output,
                "thread_id": tid,
            }
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error("命令执行异常 [%s]: %s\n%s", tid, e, error_trace)
            raise HTTPException(status_code=500, detail=f"{e}\n\n{error_trace}")


# --------------------------------------------------------------------------- #
# 运行时指标
# --------------------------------------------------------------------------- #
@app.get("/api/metrics")
async def get_metrics():
    """获取运行时指标（LLM 调用、工具执行、压缩统计的结构化 JSON）"""
    metrics = getattr(agent, "metrics", None) if agent else None
    if metrics is None:
        raise HTTPException(status_code=503, detail="指标收集不可用")
    return metrics.get_summary()


@app.post("/api/metrics/reset")
async def reset_metrics():
    """重置所有运行时指标"""
    metrics = getattr(agent, "metrics", None) if agent else None
    if metrics is None:
        raise HTTPException(status_code=503, detail="指标收集不可用")
    metrics.reset()
    logger.info("指标已重置")
    return {"reset": True}


# --------------------------------------------------------------------------- #
# 上下文压缩
# --------------------------------------------------------------------------- #
@app.post("/api/compact")
async def compact_context(thread_id: str | None = None):
    """手动触发指定会话的上下文压缩（增量摘要 + 工具输出 Prune）

    与 before_model 中间件使用相同的压缩逻辑，适用于对话过长时主动释放 token。
    """
    tid = thread_id or agent.session.current_session_id
    logger.info("手动压缩上下文 [%s]", tid)
    async with _thread_lock(tid):
        try:
            result = await agent.session_manager.manually_compact(thread_id=tid)
        except Exception as e:
            logger.error("压缩失败 [%s]: %s", tid, e)
            raise HTTPException(status_code=500, detail=f"压缩失败: {e}")
        if result is None:
            return {"compacted": False, "message": "消息数未超阈值或无消息可压缩", "thread_id": tid}
        logger.info("压缩完成 [%s]: %d → %d 条消息", tid, result["messages_before"], result["messages_after"])
        return {"compacted": True, "thread_id": tid, **result}


# --------------------------------------------------------------------------- #
# 记忆管理
# --------------------------------------------------------------------------- #
@app.get("/api/memory")
async def get_memory_summary():
    """获取记忆摘要统计（当前会话消息数、长期记忆条数、checkpoint 信息）"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    return await agent.session_manager.aget_memory_summary()


@app.post("/api/compress")
async def compress_long_term_memory():
    """压缩长期记忆（用 LLM 生成摘要并替换原始记忆条目）

    acompress_memory 内部将同步 LLM 调用放入线程池，避免阻塞事件循环。
    """
    async with chat_lock:
        logger.info("压缩长期记忆")
        try:
            result = await agent.session_manager.acompress_memory()
        except Exception as e:
            logger.error("长期记忆压缩失败: %s", e)
            raise HTTPException(status_code=500, detail=f"压缩失败: {e}")
        if result.get("success"):
            logger.info("长期记忆压缩完成: %d 条 → %d 字符", result.get("original_count", 0), result.get("compressed_chars", 0))
        return result


@app.delete("/api/memory")
async def clear_memory(scope: str = "long"):
    """清空记忆

    Args:
        scope: long=仅长期记忆, short=仅短期记忆(当前会话), all=全部
    """
    async with chat_lock:
        if scope in ("long", "长期"):
            cleared = await agent.session_manager.aclear_long_term_memory()
            logger.info("已清空长期记忆 (%d 条 facts)", cleared)
        elif scope in ("short", "短期"):
            # 短期记忆 = 当前会话 checkpoint；开启新会话替代删除
            tid = agent.session.new_session()
            agent.set_current_session(tid)
            logger.info("已清空短期记忆（新会话: %s）", tid)
        elif scope in ("all", "全部"):
            cleared = await agent.session_manager.aclear_long_term_memory()
            tid = agent.session.new_session()
            agent.set_current_session(tid)
            logger.info("已清空全部记忆 (长期 %d 条 facts + 短期，新会话: %s)", cleared, tid)
        else:
            raise HTTPException(status_code=400, detail="scope 必须为 long|short|all")
        return {"cleared": True, "scope": scope}


# --------------------------------------------------------------------------- #
# 安全策略
# --------------------------------------------------------------------------- #
@app.get("/api/safety")
async def get_safety():
    """获取安全策略配置"""
    return safety_module.load_config()


@app.put("/api/safety")
async def update_safety(req: SafetyUpdateRequest):
    """更新安全策略配置（mode 和/或 confirm_dangerous）"""
    config = safety_module.load_config()
    if req.mode is not None:
        if req.mode not in ("blacklist", "whitelist"):
            raise HTTPException(status_code=400, detail="mode 必须为 blacklist|whitelist")
        config["mode"] = req.mode
    if req.confirm_dangerous is not None:
        config["confirm_dangerous"] = req.confirm_dangerous
    if safety_module.save_config(config):
        logger.info("安全策略已更新: mode=%s, confirm=%s", config.get("mode"), config.get("confirm_dangerous"))
        return config
    raise HTTPException(status_code=500, detail="保存失败")


# --------------------------------------------------------------------------- #
# 技能列表
# --------------------------------------------------------------------------- #
@app.get("/api/skills")
async def get_skills():
    """列出所有本地可用技能"""
    return {"skills": agent.list_skills() if agent else []}


# --------------------------------------------------------------------------- #
# 会话导出
# --------------------------------------------------------------------------- #
@app.get("/api/threads/{thread_id}/export")
async def export_thread(thread_id: str, fmt: str = "text"):
    """导出指定会话的对话为可读文本

    Args:
        thread_id: 会话 ID
        fmt: 导出格式，支持 text（默认）与 markdown
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    if fmt not in ("text", "markdown"):
        raise HTTPException(status_code=400, detail="fmt 必须为 text|markdown")
    msgs = await agent.session.aget_messages(session_id=thread_id)
    blocks = []
    for m in msgs:
        if isinstance(m, HumanMessage):
            role = "用户"
        elif isinstance(m, AIMessage):
            role = "助手"
        elif isinstance(m, SystemMessage):
            role = "系统"
        else:
            role = "工具"
        text = stringify_content(m.content).strip()
        if not text:
            continue
        if fmt == "markdown":
            blocks.append(f"**{role}**:\n\n{text}")
        else:
            blocks.append(f"【{role}】\n{text}")
    sep = "\n\n---\n\n" if fmt == "markdown" else "\n\n"
    header = f"# 对话导出 - {thread_id}\n\n" if fmt == "markdown" else f"对话导出 - {thread_id}\n{'=' * 40}\n"
    content = header + sep.join(blocks)
    return {"thread_id": thread_id, "format": fmt, "content": content}


# --------------------------------------------------------------------------- #
# 工作空间绑定（Workspace）
# --------------------------------------------------------------------------- #
@app.get("/api/threads/{thread_id}/workspace")
async def get_workspace(thread_id: str):
    """查看指定会话绑定的工作空间路径（对应 CLI ``workspace`` 命令）。

    Returns:
        thread_id 与 workspace（未绑定时为 None）
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    ws = await agent.session.aget_workspace(session_id=thread_id)
    logger.info("查询工作空间 [%s]: %s", thread_id, ws or "(未绑定)")
    return {"thread_id": thread_id, "workspace": ws}


@app.post("/api/threads/{thread_id}/workspace")
async def set_workspace(thread_id: str, req: SetWorkspaceRequest):
    """设置/修改指定会话的工作空间绑定（对应 CLI ``workspace <path>`` 命令）。

    路径校验由 ``SessionRegistry.aset_workspace`` 完成：必须是已存在的目录、
    非系统关键目录。校验失败（ValueError）返回 400。

    Returns:
        thread_id 与规范化后的绝对路径
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    logger.info("绑定工作空间 [%s]: %s", thread_id, req.path)
    try:
        real = await agent.session.aset_workspace(req.path, session_id=thread_id)
    except ValueError as e:
        logger.warning("绑定失败 [%s]: %s", thread_id, e)
        raise HTTPException(status_code=400, detail=str(e))
    return {"thread_id": thread_id, "workspace": real}


@app.delete("/api/threads/{thread_id}/workspace")
async def clear_workspace(thread_id: str):
    """清除指定会话的工作空间绑定（对应 CLI ``workspace:clear`` 命令）。

    Returns:
        thread_id 与 cleared 状态（True=原绑定存在已清除，False=原本无绑定）
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    existed = await agent.session.aclear_workspace(session_id=thread_id)
    logger.info("清除工作空间 [%s]: existed=%s", thread_id, existed)
    return {"thread_id": thread_id, "cleared": existed}


@app.get("/api/workspace/browse")
async def browse_workspace(path: str | None = None):
    """浏览文件系统目录结构（仅返回子目录），供前端文件检索器选择工作空间。

    与 ``/api/threads/{thread_id}/workspace`` 配合：本端点负责"检索"目录树，
    用户选定目录后由前端调用 POST ``workspace`` 完成绑定并记录。

    安全限制：
      - 路径必须是已存在的目录，否则返回 400
      - 仅返回子目录（不返回文件），避免泄露文件级信息
      - 跳过隐藏目录（``.`` 开头）与无权限目录
      - 无 ``path`` 时：Windows 返回可用盘符列表，其他平台从根目录 ``/`` 开始

    Args:
        path: 要浏览的目录绝对路径，为空时返回浏览起点

    Returns:
        path: 当前规范化路径（起点时为空串）
        entries: 子目录列表 ``[{name, path, has_children}]``
        is_root: 是否为浏览起点（盘符列表/根目录）
    """
    import string
    import sys

    def _has_subdir(full: str) -> bool:
        """快速判断目录是否含子目录（命中即返回，最多扫 256 个条目）。"""
        try:
            for i, sub in enumerate(os.listdir(full)):
                if i >= 256:
                    break
                if os.path.isdir(os.path.join(full, sub)):
                    return True
        except (PermissionError, OSError):
            pass
        return False

    # 无 path：返回浏览起点
    if not path:
        if sys.platform == "win32":
            drives = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({"name": f"{letter}:", "path": drive, "has_children": True})
            return {"path": "", "entries": drives, "is_root": True}
        path = "/"

    real_path = os.path.realpath(os.path.abspath(path))
    if not os.path.isdir(real_path):
        raise HTTPException(status_code=400, detail=f"路径不存在或不是目录: {real_path}")

    entries = []
    try:
        for name in sorted(os.listdir(real_path)):
            full = os.path.join(real_path, name)
            if not os.path.isdir(full):
                continue
            # 跳过隐藏目录
            if name.startswith("."):
                continue
            entries.append({"name": name, "path": full, "has_children": _has_subdir(full)})
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"无权限访问: {real_path}")

    return {"path": real_path, "entries": entries, "is_root": False}


# 生产环境：如果前端已构建（web/dist），则由本服务直接托管静态文件
if os.path.isdir(WEB_DIST):
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


def main():
    # 配置日志 - 必须在 uvicorn.run() 之前
    # 1. 生成日志文件名：按日期分目录，文件名为时间 + 哈希
    import hashlib
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).astimezone()  # noqa: UP017 - 本环境 Python 构建缺少 datetime.UTC 属性
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    time_hash = hashlib.md5(str(now.timestamp()).encode()).hexdigest()[:8]
    
    log_date_dir = os.path.join(BASE_DIR, "api", "log", date_str)
    os.makedirs(log_date_dir, exist_ok=True)
    
    log_file = os.path.join(log_date_dir, f"{time_str}_{time_hash}.log")
    
    # 2. 配置日志：同时输出到终端（stderr）和文件
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stderr),  # 终端输出
            logging.FileHandler(log_file, encoding="utf-8"),  # 文件输出
        ],
        force=True,  # 强制重新配置
    )
    
    logger.info("日志文件: %s", log_file)
    
    server_cfg = load_server_config()
    parser = argparse.ArgumentParser(description="LangChainAgent API Server")
    parser.add_argument("--provider", default=None, help="初始 LLM 提供商")
    parser.add_argument("--host", default=server_cfg["host"], help=f"绑定地址（默认读自 {SERVER_CONFIG_FILE}）")
    parser.add_argument("--port", type=int, default=server_cfg["port"], help=f"绑定端口（默认读自 {SERVER_CONFIG_FILE}）")
    args = parser.parse_args()

    global _startup_provider
    _startup_provider = args.provider or pick_default_provider()
    logger.info("初始化提供商: %s", _startup_provider)
    logger.info("服务启动: http://%s:%s", args.host, args.port)

    import uvicorn
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port, 
        log_level="info",
        access_log=False  # 关闭访问日志（GET/POST 请求）
    )


if __name__ == "__main__":
    main()
