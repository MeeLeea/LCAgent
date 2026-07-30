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
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 确保项目根目录在 sys.path 中（支持 python -m api.server 与 python api/server.py）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agent import AgentCore
from agent.message_utils import stringify_content  # 消息内容序列化
from agent.config import load_agent_config, resolve_path
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from llm_client import LLMClient, load_providers
from tools import safety as safety_module

# --------------------------------------------------------------------------- #
# 路径常量（与 main.py 保持一致）
# --------------------------------------------------------------------------- #
LLM_FILE = os.path.join(BASE_DIR, "config", "llm_config.json")
MCP_CONFIG_FILE = os.path.join(BASE_DIR, "config", "mcp_servers.json")
AGENT_CONFIG_FILE = os.path.join(BASE_DIR, "config", "agent_config.json")
SERVER_CONFIG_FILE = os.path.join(BASE_DIR, "config", "server_config.json")
MEMORY_FILE = os.path.join(BASE_DIR, "memory", "memory.json")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "memory", "checkpoints.sqlite")
WEB_DIST = os.path.join(BASE_DIR, "web", "dist")


def load_server_config() -> Dict[str, Any]:
    """加载服务端口配置（config/server_config.json）。

    文件不存在或解析失败时回退到默认值（127.0.0.1:8000），不抛异常。
    """
    defaults: Dict[str, Any] = {"host": "127.0.0.1", "port": 8000}
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
    except (json.JSONDecodeError, IOError):
        pass
    return defaults

# --------------------------------------------------------------------------- #
# 全局状态
# --------------------------------------------------------------------------- #
agent: Optional[AgentCore] = None
llm: Optional[LLMClient] = None
# 串行化对话轮次：AgentCore 是有状态单例，同一时刻只能跑一轮。
chat_lock = asyncio.Lock()


def build_agent(provider: str) -> tuple[AgentCore, LLMClient]:
    """根据提供商初始化 LLM 与 Agent（逻辑与 main.py 一致，去掉 CLI 打印）。"""
    new_llm = LLMClient(provider=provider, config_file=LLM_FILE)
    cfg = load_agent_config(AGENT_CONFIG_FILE)
    skills_dir = resolve_path(cfg["skills_dir"], BASE_DIR)
    mcp_config_file = resolve_path(cfg["mcp_config_file"], BASE_DIR)
    new_agent = AgentCore(
        llm_client=new_llm,
        memory_size=cfg["memory_size"],
        long_term_memory_file=MEMORY_FILE,
        checkpoint_file=CHECKPOINT_FILE,
        max_iterations=cfg["max_iterations"],
        verbose=cfg["verbose"],
        mcp_config_file=mcp_config_file,
        enable_mcp=cfg["enable_mcp"],
        skills_dir=skills_dir,
        auto_match_skills=cfg["auto_match_skills"],
        max_context_messages=cfg["max_context_messages"],
        context_trim_keep=cfg["context_trim_keep"],
    )
    return new_agent, new_llm


def pick_default_provider() -> str:
    """选择默认提供商：优先有 api_key 的，否则回退 zhipu。"""
    providers = load_providers(LLM_FILE)
    for key, conf in providers.items():
        if conf.get("api_key") or os.environ.get(conf.get("env_key", "")):
            return key
    return "zhipu" if "zhipu" in providers else (next(iter(providers), "zhipu"))


def serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """把 LangGraph 消息对象序列化为前端可消费的 JSON。"""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": stringify_content(m.content)})
        elif isinstance(m, AIMessage):
            entry: Dict[str, Any] = {
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


def thread_summary(thread_id: str) -> Dict[str, Any]:
    """单个会话的摘要信息（消息数 + 预览）。"""
    msgs = agent.memory.get_messages(thread_id=thread_id) if agent else []
    preview = ""
    for m in msgs:
        if isinstance(m, HumanMessage):
            preview = stringify_content(m.content).strip().replace("\n", " ")[:50]
            break
    if not preview and msgs:
        preview = stringify_content(msgs[-1].content).strip().replace("\n", " ")[:50]
    return {"thread_id": thread_id, "message_count": len(msgs), "preview": preview}


# --------------------------------------------------------------------------- #
# Pydantic 模型
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ResumeRequest(BaseModel):
    payload: Dict[str, Any]
    thread_id: Optional[str] = None


class SwitchProviderRequest(BaseModel):
    provider: str


class SwitchModelRequest(BaseModel):
    model: str


# --------------------------------------------------------------------------- #
# FastAPI 应用
# --------------------------------------------------------------------------- #
app = FastAPI(title="LangChainAgent API", version="1.0.0")
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
        "thread_id": agent.memory.thread_id if agent else None,
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
    print(f"\n[切换] 收到切换提供商请求: {req.provider}", flush=True)
    async with chat_lock:
        global agent, llm
        try:
            new_llm = LLMClient(provider=req.provider, config_file=LLM_FILE)
        except Exception as e:
            print(f"[切换] 提供商切换失败: {e}", flush=True)
            raise HTTPException(status_code=400, detail=str(e))
        agent.switch_llm(new_llm)
        llm = new_llm
    info = llm.get_info()
    print(f"[切换] 提供商已切换 → {info['provider_name']} / {info['model']}", flush=True)
    return info


@app.post("/api/models/switch")
async def switch_model(req: SwitchModelRequest):
    print(f"\n[切换] 收到切换模型请求: {req.model}", flush=True)
    async with chat_lock:
        try:
            llm.switch_model(req.model)
            agent.switch_llm(llm)
        except Exception as e:
            print(f"[切换] 模型切换失败: {e}", flush=True)
            raise HTTPException(status_code=400, detail=str(e))
    info = llm.get_info()
    print(f"[切换] 模型已切换 → {info['provider_name']} / {info['model']}", flush=True)
    return info


@app.get("/api/tools")
async def get_tools():
    return {"tools": agent.get_available_tools() if agent else []}


@app.get("/api/threads")
async def list_threads():
    """列出所有会话（按消息数倒序，便于最近活跃的靠前）。"""
    ids = agent.memory.list_threads() if agent else []
    summaries = [thread_summary(tid) for tid in ids]
    summaries.sort(key=lambda x: x["message_count"], reverse=True)
    return {"threads": summaries, "current": agent.memory.thread_id if agent else None}


@app.post("/api/threads")
async def create_thread():
    """新建会话，返回 thread_id。"""
    async with chat_lock:
        tid = agent.memory.new_thread()
    return {"thread_id": tid}


@app.delete("/api/threads/{thread_id}")
async def delete_thread(thread_id: str):
    ok = agent.memory.delete_thread(thread_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在或删除失败")
    return {"deleted": True, "thread_id": thread_id}


@app.get("/api/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    msgs = agent.memory.get_messages(thread_id=thread_id) if agent else []
    return {"thread_id": thread_id, "messages": serialize_messages(msgs)}


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式聊天。"""

    async def event_stream():
        async with chat_lock:
            tid = req.thread_id
            if tid:
                agent.memory.thread_id = tid
            else:
                tid = agent.memory.new_thread()
                yield _sse({"type": "thread_created", "thread_id": tid})
            try:
                async for event in agent.astream_chat(req.message):
                    yield _sse(event)
            except Exception as e:
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
        async with chat_lock:
            if req.thread_id:
                agent.memory.thread_id = req.thread_id
            try:
                async for event in agent.astream_resume(req.payload):
                    yield _sse(event)
            except Exception as e:
                yield _sse({"type": "error", "content": f"内部错误: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# 生产环境：如果前端已构建（web/dist），则由本服务直接托管静态文件
if os.path.isdir(WEB_DIST):
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")


def main():
    server_cfg = load_server_config()
    parser = argparse.ArgumentParser(description="LangChainAgent API Server")
    parser.add_argument("--provider", default=None, help="初始 LLM 提供商")
    parser.add_argument("--host", default=server_cfg["host"], help=f"绑定地址（默认读自 {SERVER_CONFIG_FILE}）")
    parser.add_argument("--port", type=int, default=server_cfg["port"], help=f"绑定端口（默认读自 {SERVER_CONFIG_FILE}）")
    args = parser.parse_args()

    global agent, llm
    provider = args.provider or pick_default_provider()
    print(f"[API] 初始化提供商: {provider}")
    agent, llm = build_agent(provider)
    info = llm.get_info()
    print(f"[API] 模型: {info['provider_name']} / {info['model']}")
    print(f"[API] 工具: {', '.join(agent.get_available_tools())}")
    print(f"[API] 服务启动: http://{args.host}:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
