"""
LangChainAgent 飞书远程控制机器人 (lark-oapi v2)

通过飞书 WebSocket 长连接远程控制 LangChainAgent。
直接复用 main.py 的 build_agent()，所有 tools/skills/MCP 与 CLI 一致。

用法: cd LangChainAgent && python tools/remote_agent_control.py
"""

from __future__ import annotations

import asyncio, json, os, re, subprocess, sys, threading, traceback
from typing import Any, Dict, Optional

import pyautogui
import lark_oapi as lark
from lark_oapi.api.im.v1.model.create_file_request import CreateFileRequest
from lark_oapi.api.im.v1.model.create_file_request_body import CreateFileRequestBody
from lark_oapi.api.im.v1.model.create_image_request import CreateImageRequest
from lark_oapi.api.im.v1.model.create_image_request_body import CreateImageRequestBody
from lark_oapi.api.im.v1.model.create_message_request import CreateMessageRequest
from lark_oapi.api.im.v1.model.create_message_request_body import CreateMessageRequestBody
from lark_oapi.api.im.v1.processor import P2ImMessageReceiveV1Processor
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
from lark_oapi.ws import Client as LarkWSClient

# ── 项目路径 ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from main import BASE_DIR as _, LLM_FILE, AGENT_CONFIG_FILE, MEMORY_FILE, CHECKPOINT_FILE, build_agent  # noqa: E402

REMOTE_CONFIG_FILE = os.path.join(BASE_DIR, "config", "remote_control.json")
REMOTE_THREAD_FILE = os.path.join(BASE_DIR, "memory", "remote_thread_id.txt")

# ── 飞书远程配置 ──
def _load_remote_config():
    if not os.path.exists(REMOTE_CONFIG_FILE):
        raise FileNotFoundError(f"缺少: {REMOTE_CONFIG_FILE}\n  请复制 .example 并填入真实值")
    with open(REMOTE_CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feishu, agent_cfg = cfg.get("feishu", {}), cfg.get("agent", {})
    return feishu.get("app_id", ""), feishu.get("app_secret", ""), feishu.get("allow_open_id", []), agent_cfg.get("provider", "")

APP_ID, APP_SECRET, ALLOW_OPEN_ID, AGENT_PROVIDER = _load_remote_config()

# ── SDK 客户端 ──
_http = lark.Client.builder().app_id(APP_ID).app_secret(APP_SECRET).log_level(lark.LogLevel.ERROR).build()

# ── 全局状态 ──
_agent: Optional[Any] = None
_agent_lock = threading.Lock()
_agent_info: Dict[str, Any] = {}
_processing: set[str] = set()
_processing_lock = threading.Lock()
_interrupt_cache: dict[str, dict] = {}
_model_menu_cache: dict[str, list[tuple[str, str]]] = {}
_seen_msg: set[str] = set()

# ===================== 消息发送 =====================

def _send_lark_msg(chat_id: str, msg_type: str, content: dict) -> None:
    """通用飞书消息发送。"""
    try:
        _http.im.v1.message.create(
            CreateMessageRequest.builder()
            .receive_id_type("chat_id")
            .request_body(CreateMessageRequestBody.builder()
                          .content(json.dumps(content)).msg_type(msg_type).receive_id(chat_id).build())
            .build()
        )
    except Exception as e:
        print(f"[飞书] 发送失败: {e}")

_send_text = lambda c, t: _send_lark_msg(c, "text", {"text": t})

def _send_screenshot(chat_id: str) -> None:
    try:
        img = pyautogui.screenshot()
        path = os.path.join(BASE_DIR, "snap_temp.png")
        img.save(path)
        with open(path, "rb") as f:
            resp = _http.im.v1.image.create(
                CreateImageRequest.builder()
                .request_body(CreateImageRequestBody.builder().image_type("message").image(f).build())
                .build()
            )
        os.remove(path)
        if resp.success():
            _send_lark_msg(chat_id, "image", {"image_key": resp.data.image_key})
        else:
            _send_text(chat_id, f"截图失败: {resp.msg}")
    except Exception as e:
        _send_text(chat_id, f"截图失败: {e}")

def _send_image_file(chat_id: str, path: str) -> bool:
    try:
        with open(path, "rb") as f:
            resp = _http.im.v1.image.create(
                CreateImageRequest.builder()
                .request_body(CreateImageRequestBody.builder().image_type("message").image(f).build())
                .build()
            )
        if resp.success():
            _send_lark_msg(chat_id, "image", {"image_key": resp.data.image_key})
            return True
        return False
    except Exception as e:
        print(f"[图片] {path}: {e}")
        return False

def _send_file(chat_id: str, path: str) -> bool:
    try:
        if not os.path.isfile(path):
            return _send_text(chat_id, f"❌ 文件不存在: {path}")
        if os.path.getsize(path) > 100 * 1024 * 1024:
            return _send_text(chat_id, f"❌ 文件过大，上限 100MB")
        with open(path, "rb") as f:
            resp = _http.im.v1.file.create(
                CreateFileRequest.builder()
                .request_body(CreateFileRequestBody.builder()
                              .file_name(os.path.basename(path)).file_type("stream").file(f).build())
                .build()
            )
        if resp.success():
            _send_lark_msg(chat_id, "file", {"file_key": resp.data.file_key})
            return True
        _send_text(chat_id, f"❌ 上传失败: {resp.msg}")
        return False
    except Exception as e:
        print(f"[文件] {path}: {e}")
        return _send_text(chat_id, f"❌ 发送失败: {e}")

# ===================== Agent 生命周期 =====================

def _auto_detect_provider() -> str:
    from llm_client import load_providers
    providers = load_providers(LLM_FILE)
    if not providers:
        return print("[Agent] 无 provider，回退 zhipu") or "zhipu"
    if AGENT_PROVIDER and AGENT_PROVIDER in providers:
        return print(f"[Agent] 显式: {providers[AGENT_PROVIDER]['name']}") or AGENT_PROVIDER
    for key in providers:
        if os.environ.get(str(providers[key].get("env_key", ""))) or _has_api_key(key):
            return print(f"[Agent] 自动: {providers[key]['name']} ({key})") or key
    key = next(iter(providers))
    return print(f"[Agent] 回退: {providers[key]['name']} ({key})") or key

def _has_api_key(key: str) -> bool:
    if not os.path.exists(LLM_FILE): return False
    try:
        with open(LLM_FILE) as f:
            return bool(json.load(f).get("providers", {}).get(key, {}).get("api_key"))
    except Exception:
        return False

def _patch_safety_config() -> None:
    import tools.safety as m
    if not os.path.exists(REMOTE_CONFIG_FILE):
        return
    with open(REMOTE_CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    safety_cfg = cfg.get("safety")
    if safety_cfg:
        m._config_cache = dict(m.DEFAULT_CONFIG)
        m._config_cache.update(safety_cfg)
        print(f"[Agent] 安全: {REMOTE_CONFIG_FILE} -> safety")

def start_agent() -> str:
    """初始化或重启 Agent。在独立线程中运行以避免 asyncio 事件循环冲突。"""
    global _agent, _agent_info
    _patch_safety_config()
    try:
        # 在独立线程中构建 Agent，解决 MCP 加载的 asyncio.run() 与 WS 事件循环冲突
        result_container: list[tuple] = []
        error_container: list[Exception] = []

        def _build():
            try:
                # 为 MCP 异步加载创建独立事件循环
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                tid = _load_remote_thread_id()
                agent, llm = build_agent(_auto_detect_provider())
                agent.verbose = False
                if tid:
                    agent.memory.switch_thread(tid)
                else:
                    _save_remote_thread_id(agent.memory.thread_id)
                # 主动修复可能残留的孤儿 tool_calls（上次中断可能遗留）
                try:
                    agent._repair_rejected_tool_calls(agent._invoke_config())
                    agent._clear_pending_interrupt()
                except Exception:
                    pass
                result_container.append((agent, llm, agent.memory.thread_id))
            except Exception as e:
                error_container.append(e)
            finally:
                loop.close()

        t = threading.Thread(target=_build, daemon=True)
        t.start()
        t.join(timeout=30)

        if error_container:
            raise error_container[0]
        if not result_container:
            raise RuntimeError("Agent 初始化超时（30s）")

        agent, llm, tid = result_container[0]
        with _agent_lock:
            _agent = agent
            _agent_info = {"provider": llm.provider, "model": llm.model, "status": "running"}
        return f"✅ Agent 已启动\n  模型: {llm.model}\n  会话: {tid}"
    except Exception as e:
        with _agent_lock:
            _agent, _agent_info = None, {"status": "error", "error": str(e)}
        return f"❌ 启动失败: {e}"

restart_agent = lambda: (_agent_info.update({"status": "restarting"}) or start_agent())
get_agent = lambda: _agent  # protected by _agent_lock externally

def _load_remote_thread_id() -> str:
    if os.path.exists(REMOTE_THREAD_FILE):
        with open(REMOTE_THREAD_FILE) as f: return f.read().strip()
    return ""

def _save_remote_thread_id(tid: str) -> None:
    os.makedirs(os.path.dirname(REMOTE_THREAD_FILE), exist_ok=True)
    with open(REMOTE_THREAD_FILE, "w") as f: f.write(tid)

# ===================== 文件自动发送 =====================

_FIND_PATTERNS = [
    (re.compile(r"""(?:['"\s]|^)([\w\-\\\/:.]+\.(?:png|jpg|jpeg|gif|bmp))(?:['"\s]|$)""", re.I), _send_image_file),
    (re.compile(r"""(?:['"\s]|^)([\w\-\\\/:.]+\.(?:pdf|docx?|xlsx?|pptx?|txt|csv|log|json|
        yaml|yml|py|js|ts|html|css|zip|rar|7z|tar|gz|exe|msi))(?:['"\s]|$)""", re.I | re.X), _send_file),
]

def _auto_send_files(chat_id: str, text: str) -> int:
    sent = 0
    for pat, sender in _FIND_PATTERNS:
        seen = set()
        for m in pat.finditer(text):
            p = os.path.abspath(m.group(1)) if not os.path.isabs(m.group(1)) else m.group(1)
            if p in seen or not os.path.isfile(p): continue
            seen.add(p)
            if sender(chat_id, p): sent += 1
            if len(seen) >= 5: break
    return sent

# ===================== 中断处理 =====================

def _clear_pending_interrupt(agent: Any) -> None:
    """清理未完成的中断，确保 checkpoint 干净。"""
    try:
        agent._repair_rejected_tool_calls(agent._invoke_config())
        agent._clear_pending_interrupt()
    except Exception as e:
        print(f"[中断] 清理失败: {e}")
    _interrupt_cache.clear()

def _handle_turn_result(chat_id: str, agent: Any, turn: Any, mode: str) -> None:
    if turn.is_completed:
        out = (turn.output or "无输出")[:4000]
        _send_text(chat_id, out if mode == "chat" else f"✅ 完成\n\n{out}")
        _auto_send_files(chat_id, turn.output or "")
    elif turn.is_interrupted:
        interrupts = getattr(turn, "interrupts", []) or []
        if not interrupts: return _send_text(chat_id, "⏸️ 中断，无选项")
        v = getattr(interrupts[0], "value", {})
        if not isinstance(v, dict) or v.get("kind") != "human_choice":
            _interrupt_cache[chat_id] = {"agent": agent, "is_text": True}
            return _send_text(chat_id, "⏸️ 需要人工输入，发送消息回复")
        choices = v.get("choices", [])
        if not choices:
            _interrupt_cache[chat_id] = {"agent": agent, "is_text": True}
            return _send_text(chat_id, f"⏸️ {v.get('prompt', '请选择')}")
        lines = [f"⏸️ {v.get('prompt', '请选择')}", ""]
        for i, c in enumerate(choices, 1):
            lines.append(f"  {i}. {c.get('label', c.get('id', i))}")
        lines += ["", f"回复编号 (1-{len(choices)}) 或「取消」"]
        _send_text(chat_id, "\n".join(lines))
        _interrupt_cache[chat_id] = {"agent": agent, "choices": choices}
    elif turn.status == "cancelled":
        _send_text(chat_id, f"🚫 已取消: {turn.output}")

def _resume_interrupt(chat_id: str, content: str) -> None:
    info = _interrupt_cache.pop(chat_id, None)
    if not info: return
    agent, choices = info["agent"], info.get("choices")
    if content.strip().lower() in ("取消", "cancel", "c"):
        _clear_pending_interrupt(agent)
        return _send_text(chat_id, "✅ 已取消")
    if choices:
        if content.strip().isdigit():
            idx = int(content.strip()) - 1
            if 0 <= idx < len(choices):
                payload = {"choice_id": choices[idx]["id"]}
            else:
                _interrupt_cache[chat_id] = info
                return _send_text(chat_id, f"⚠️ 编号超出 1-{len(choices)}")
        else:
            _clear_pending_interrupt(agent)
            return _send_text(chat_id, f"⚠️ 已取消。回复编号 (1-{len(choices)}) 重新选择")
    else:
        payload = {"text": content}
    try:
        _handle_turn_result(chat_id, agent, agent.resume_structured(payload),
                            getattr(agent, "_pending_interrupt_mode", "chat"))
    except Exception as e:
        _send_text(chat_id, f"❌ 恢复失败: {e}")
        _clear_pending_interrupt(agent)

# ===================== 指令处理器 =====================

HELP_TEXT = (
    "📋 Agent控制 | 启动agent 重启agent 停止 状态 模型\n"
    "  会话      | 会话 会话列表 会话切换 <id>\n"
    "  任务执行  | run/运行/执行 <任务>  chat <消息>\n"
    "  系统控制  | 截图  文件 <路径>  cmd <…>  ps <…>  帮助"
)

# ── 会话（Thread）管理 ──

def _handle_threads(chat_id: str, arg: str = "") -> None:
    agent = get_agent()
    if not agent: return _send_text(chat_id, "⚠️ 请先启动agent")
    mem = agent.memory

    if not arg or arg == "列表":
        threads = mem.list_threads()
        cur = mem.thread_id
        if not threads:
            return _send_text(chat_id, "没有已保存的会话")
        lines = [f"📋 {len(threads)} 个会话：", ""]
        for t in threads:
            msgs = len(mem.get_messages()) if t == cur else 0  # 只统计当前线程
            lines.append(f"  • {t}{' ← 当前' if t == cur else ''}")
        _send_text(chat_id, "\n".join(lines))
    elif arg.startswith("切换 "):
        tid = arg.split(maxsplit=1)[1].strip()
        if mem.switch_thread(tid):
            _save_remote_thread_id(tid)
            _send_text(chat_id, f"✅ 已切换到 {tid}")
        else:
            _send_text(chat_id, f"⚠️ 会话不存在: {tid}，发送「会话列表」查看")
    else:
        lines = [f"📋 当前: {mem.thread_id}", f"  总会话数: {len(mem.list_threads())}"]
        info = mem.summarize()
        lines.append(f"  当前消息数: {info.get('checkpoint_messages', 0)}")
        _send_text(chat_id, "\n".join(lines))

def _handle_status(chat_id: str) -> None:
    with _agent_lock: info = dict(_agent_info)
    if not info: return _send_text(chat_id, "Agent 未启动")
    if info["status"] == "error": return _send_text(chat_id, f"❌ {info.get('error')}")
    agent = get_agent()
    _send_text(chat_id, (
        f"📊 {info['status']} | {info.get('provider')} / {info.get('model')}"
        + (f" | 工具:{len(agent.tools)} 步骤:{len(agent.execution_history)}" if agent else "")
    ))

def _handle_stop(chat_id: str) -> None:
    if chat_id in _interrupt_cache:
        agent = get_agent()
        if agent: _clear_pending_interrupt(agent)
        _interrupt_cache.pop(chat_id, None)
        return _send_text(chat_id, "✅ 已取消")
    with _processing_lock: running = chat_id in _processing
    _send_text(chat_id, "⏸️ 发送「重启agent」强制终止" if running else "没有运行中的任务")

def _handle_model_menu(chat_id: str, inp: str = "") -> None:
    agent = get_agent()
    if not agent: return _send_text(chat_id, "⚠️ 请先启动agent")
    if not inp:  # 列出菜单
        from llm_client import load_providers
        try: providers = load_providers(LLM_FILE)
        except Exception as e: return _send_text(chat_id, f"❌ {e}")
        flat = [(k, m) for k, v in providers.items() for m in v.get("models", [])]
        if not flat: return _send_text(chat_id, "无可用模型")
        _model_menu_cache[chat_id] = flat
        cur = (agent.llm.provider, agent.llm.model)
        lines = ["📋 回复编号切换：", ""]
        for i, (p, m) in enumerate(flat, 1):
            lines.append(f"  {i:>2}. {p}: {m}{' ← 当前' if (p, m) == cur else ''}")
        return _send_text(chat_id, "\n".join(lines))
    # 接收选择
    flat = _model_menu_cache.get(chat_id)
    if not flat: return _send_text(chat_id, "请先发送「模型」")
    if inp.isdigit() and 0 <= (idx := int(inp) - 1) < len(flat):
        return _do_switch(chat_id, agent, *flat[idx])
    matches = [(p, m) for p, m in flat if inp.lower() in m.lower()]
    if len(matches) == 1: return _do_switch(chat_id, agent, *matches[0])
    if len(matches) > 1:
        _model_menu_cache[chat_id] = matches
        return _send_text(chat_id, "\n".join(["🔍 多个匹配："] + [f"  {i:>2}. {p}: {m}" for i, (p, m) in enumerate(matches, 1)]))
    _send_text(chat_id, f"❌ 未匹配「{inp}」")

def _do_switch(chat_id: str, agent, pk: str, mn: str) -> None:
    old = f"{agent.llm.provider}: {agent.llm.model}"
    try:
        if agent.llm.provider != pk:
            from utils.commands.provider import create_llm
            agent.llm = create_llm(pk, LLM_FILE)
            if agent.llm.model != mn: agent.llm.switch_model(mn)
        else:
            agent.llm.switch_model(mn)
        agent.agent_executor = agent._create_agent_executor()
        with _agent_lock: _agent_info.update(provider=agent.llm.provider, model=agent.llm.model)
        _send_text(chat_id, f"✅ {old} → {pk}: {mn}")
    except Exception as e:
        _send_text(chat_id, f"❌ 切换失败: {e}")

def _exec_cmd_or_ps(chat_id: str, cmd: str, shell: bool) -> None:
    try:
        out = subprocess.check_output(cmd if shell else ["powershell", "-Command", cmd],
                                      shell=shell, text=True, timeout=30)
        _send_text(chat_id, f"✅ {(out[:1800])}\n...(截断)" if len(out) > 1800 else f"✅ {out}")
    except subprocess.TimeoutExpired:
        _send_text(chat_id, "⏰ 超时 (30s)")
    except Exception as e:
        _send_text(chat_id, f"❌ {e}")

def _agent_task(chat_id: str, method: str, payload: str) -> None:
    def _worker():
        try:
            agent = get_agent()
            if not agent: return _send_text(chat_id, "⚠️ 请先启动agent")
            _clear_pending_interrupt(agent)
            turn = agent.run_structured(payload) if method == "run" else agent.chat_structured(payload)
            _handle_turn_result(chat_id, agent, turn, method)
        except Exception as e:
            _send_text(chat_id, f"❌ {e}")
        finally:
            with _processing_lock: _processing.discard(chat_id)
    threading.Thread(target=_worker, daemon=True).start()

# ===================== 飞书回调 & 路由 =====================

def _on_message(data) -> None:
    try:
        msg = data.event.message
        if msg.message_id in _seen_msg: return
        _seen_msg.add(msg.message_id)
        if len(_seen_msg) > 500: _seen_msg.clear()
        raw = msg.content.strip()
        content = json.loads(raw).get("text", raw) if raw.startswith("{") else raw
        print(f"[飞书] {data.event.sender.sender_id.open_id} | {content}")
        if data.event.sender.sender_id.open_id not in ALLOW_OPEN_ID:
            return _send_text(msg.chat_id, "⚠️ 权限不足")
        _dispatch(msg.chat_id, content)
    except Exception:
        traceback.print_exc()

# 命令检测（用于区分中断回复和全新指令）
_CMD_PREFIXES = ("run ", "运行 ", "执行 ", "chat ", "启动", "重启", "状态", "模型", "会话", "截图",
                 "文件 ", "file ", "cmd ", "ps ", "帮助", "help")
_CMD_EXACT = ("状态", "模型", "截图", "帮助", "启动agent", "重启agent", "取消", "停止", "stop", "cancel")

def _dispatch(chat_id: str, content: str) -> None:
    low = content.lower()
    is_cmd = any(low.startswith(p) for p in _CMD_PREFIXES) or low in _CMD_EXACT

    # 中断缓存 → 当恢复回复
    if chat_id in _interrupt_cache:
        if is_cmd:  # 发新命令 → 先取消中断
            agent = get_agent()
            if agent: _clear_pending_interrupt(agent)
            _interrupt_cache.pop(chat_id, None)
        else:  # 非命令 → 尝试恢复
            with _processing_lock:
                if chat_id in _processing: return _send_text(chat_id, "⏳ 请稍候...")
                _processing.add(chat_id)
            def _worker():
                try: _resume_interrupt(chat_id, content)
                finally:
                    with _processing_lock: _processing.discard(chat_id)
            threading.Thread(target=_worker, daemon=True).start()
            return

    # ── 同步指令路由 ──
    if low in ("启动agent", "启动", "start"):     return _send_text(chat_id, start_agent())
    if low in ("重启agent", "重启", "restart"):   return _send_text(chat_id, "🔄\n" + restart_agent())
    if low in ("状态", "status"):                  return _handle_status(chat_id)
    if low in ("模型", "model"):                   return _handle_model_menu(chat_id)
    if low.startswith("会话"):                     return _handle_threads(chat_id, content[2:].strip())
    if content.strip().isdigit():                   return _handle_model_menu(chat_id, content.strip())
    if low in ("停止", "取消", "stop", "cancel"):  return _handle_stop(chat_id)
    if low in ("截图", "screenshot"):              return _send_screenshot(chat_id) or _send_text(chat_id, "📸")
    if low.startswith(("文件 ", "file ")):
        p = content.split(maxsplit=1)[1] if " " in content else ""
        return _send_file(chat_id, p) if p else _send_text(chat_id, "用法: 文件 <路径>")
    if low.startswith("cmd "):                     return _exec_cmd_or_ps(chat_id, content[4:].strip(), True)
    if low.startswith("ps ") or low.startswith("ps\n"):
        return _exec_cmd_or_ps(chat_id, content[3:].strip(), False)
    if low in ("帮助", "help", "?", "？"):         return _send_text(chat_id, HELP_TEXT)

    # ── 后台 Agent 任务 ──
    with _processing_lock:
        if chat_id in _processing: return _send_text(chat_id, "⏳ 请稍候...")
        _processing.add(chat_id)

    # run / chat 别名匹配
    for prefix, method in (("run ", "run"), ("运行 ", "run"), ("执行 ", "run"),
                           ("chat ", "chat")):
        if low.startswith(prefix):
            _send_text(chat_id, "🤖 思考中...")
            return _agent_task(chat_id, method, content[len(prefix):].strip())
    for prefix, method in (("run\n", "run"), ("运行\n", "run"), ("执行\n", "run"),
                           ("chat\n", "chat")):
        if low.startswith(prefix):
            _send_text(chat_id, "🤖 思考中...")
            return _agent_task(chat_id, method, content.split("\n", 1)[1].strip())

    # 默认 = chat
    _send_text(chat_id, "🤖 思考中...")
    _agent_task(chat_id, "chat", content)

# ===================== 入口 =====================

def run_remote_bot() -> None:
    print("=" * 45, "\n  LangChainAgent 飞书远程控制", "\n" + "=" * 45)
    print(f"  项目: {BASE_DIR}\n  配置: {REMOTE_CONFIG_FILE}\n")
    print("[启动] " + start_agent() + "\n")
    handler = EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(_on_message).build()
    print("  WebSocket 连接中... 请在飞书发消息获取 open_id\n")
    LarkWSClient(app_id=APP_ID, app_secret=APP_SECRET, event_handler=handler, log_level=lark.LogLevel.ERROR).start()

if __name__ == "__main__":
    run_remote_bot()
