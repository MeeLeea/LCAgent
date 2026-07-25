"""
LangChainAgent 飞书远程控制机器人 (lark-oapi v2)

通过飞书 WebSocket 长连接远程控制 LangChainAgent。

架构：
  ┌──────────────┐    WebSocket     ┌───────────────┐
  │  飞书客户端   │ ◄─────────────► │  本机常驻进程   │
  └──────────────┘                  │  AgentCore     │
                                    │   .run()       │
                                    │   .chat()      │
                                    └───────────────┘

用法：
    cd LangChainAgent
    python tools/remote_agent_control.py

前置条件：
    1. 飞书开放平台创建「企业自建应用」→ 开通「im:message」权限
    2. 复制 config/remote_control.json.example → config/remote_control.json，填入 app_id / app_secret
    3. 启动后在飞书发消息，控制台打印 open_id → 加入 allow_open_id 白名单
    4. 配置 config/llm_config.json（至少一个 provider 有 api_key）

参考：tools/xiaolan_chatbot.py（旧 SDK）→ 已迁移到 lark-oapi v2
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import traceback
from typing import Any, Dict, Optional

import pyautogui

import lark_oapi as lark
from lark_oapi.api.im.v1.model.create_image_request import CreateImageRequest
from lark_oapi.api.im.v1.model.create_image_request_body import CreateImageRequestBody
from lark_oapi.api.im.v1.model.create_message_request import CreateMessageRequest
from lark_oapi.api.im.v1.model.create_message_request_body import CreateMessageRequestBody
from lark_oapi.api.im.v1.processor import P2ImMessageReceiveV1Processor
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
from lark_oapi.ws import Client as LarkWSClient

# ---------- 项目路径 & 复用 main.py 的 Agent 构建逻辑 ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from main import (              # noqa: E402  ─ 复用 main.py 以保证行为一致
    BASE_DIR as _BASE_DIR,
    LLM_FILE,
    MCP_CONFIG_FILE,
    AGENT_CONFIG_FILE,
    MEMORY_FILE,
    CHECKPOINT_FILE,
    build_agent,
)

# 远程控制专属配置
REMOTE_CONFIG_FILE = os.path.join(BASE_DIR, "config", "remote_control.json")
# 持久化 thread_id，保证重启 bot 后能继续同一会话
REMOTE_THREAD_FILE = os.path.join(BASE_DIR, "memory", "remote_thread_id.txt")


# ===================== 配置加载 =====================


def _load_remote_config() -> Dict[str, Any]:
    """加载飞书远程控制配置（config/remote_control.json）。"""
    if not os.path.exists(REMOTE_CONFIG_FILE):
        raise FileNotFoundError(
            f"缺少配置文件: {REMOTE_CONFIG_FILE}\n"
            "  请复制 config/remote_control.json.example 并填入真实值"
        )
    with open(REMOTE_CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    feishu = cfg.get("feishu", {})
    agent_cfg = cfg.get("agent", {})
    return {
        "app_id": feishu.get("app_id", ""),
        "app_secret": feishu.get("app_secret", ""),
        "allow_open_id": feishu.get("allow_open_id", []),
        "agent_provider": agent_cfg.get("provider", ""),
    }


_rc = _load_remote_config()
APP_ID: str = str(_rc["app_id"])
APP_SECRET: str = str(_rc["app_secret"])
ALLOW_OPEN_ID: list[str] = _rc["allow_open_id"]
AGENT_PROVIDER: str = _rc["agent_provider"]

# ---------- SDK 客户端 ----------
# HTTP 客户端（发消息、传文件）
_http_client = lark.Client.builder() \
    .app_id(APP_ID) \
    .app_secret(APP_SECRET) \
    .log_level(lark.LogLevel.ERROR) \
    .build()

# ---------- 全局 Agent 实例 ----------
_agent: Optional[Any] = None
_agent_lock = threading.Lock()
_agent_info: Dict[str, Any] = {}

# 防止同一会话并发处理（如用户快速连续发多条消息）
_processing: set[str] = set()
_processing_lock = threading.Lock()


# ===================== 飞书消息发送 =====================

def _send_text(chat_id: str, text: str) -> None:
    """发送文本消息到飞书。"""
    try:
        req = CreateMessageRequest.builder() \
            .receive_id_type("chat_id") \
            .request_body(CreateMessageRequestBody.builder()
                          .content(json.dumps({"text": text}))
                          .msg_type("text")
                          .receive_id(chat_id)
                          .build()) \
            .build()
        _http_client.im.v1.message.create(req)
    except Exception as e:
        print(f"[飞书] 发送消息失败: {e}")


def _send_screenshot(chat_id: str) -> None:
    """截取当前桌面并发送到飞书。"""
    try:
        img = pyautogui.screenshot()
        img_path = os.path.join(BASE_DIR, "snap_temp.png")
        img.save(img_path)

        with open(img_path, "rb") as f:
            req = CreateImageRequest.builder() \
                .request_body(CreateImageRequestBody.builder()
                              .image_type("message")
                              .image(f)
                              .build()) \
                .build()
            resp = _http_client.im.v1.image.create(req)

        os.remove(img_path)

        if resp.success():
            image_key = resp.data.image_key
            _send_image_msg(chat_id, image_key)
        else:
            _send_text(chat_id, f"截图上传失败: {resp.msg}")
    except Exception as e:
        _send_text(chat_id, f"截图失败: {str(e)}")


def _send_image_msg(chat_id: str, image_key: str) -> None:
    """发送已上传的图片。"""
    req = CreateMessageRequest.builder() \
        .receive_id_type("chat_id") \
        .request_body(CreateMessageRequestBody.builder()
                      .content(json.dumps({"image_key": image_key}))
                      .msg_type("image")
                      .receive_id(chat_id)
                      .build()) \
        .build()
    _http_client.im.v1.message.create(req)


# ===================== Agent 生命周期 =====================


def _auto_detect_provider() -> str:
    """自动选择第一个已配置 api_key 的 provider，打印选择结果。"""
    from llm_client import load_providers

    providers = load_providers(LLM_FILE)
    if not providers:
        print("[Agent] 未发现任何 provider 配置，回退到 zhipu")
        return "zhipu"

    # 优先使用显式配置
    if AGENT_PROVIDER:
        if AGENT_PROVIDER in providers:
            info = providers[AGENT_PROVIDER]
            print(f"[Agent] 使用显式配置: {info['name']} ({AGENT_PROVIDER})")
            return AGENT_PROVIDER
        else:
            print(f"[Agent] ⚠️ AGENT_PROVIDER={AGENT_PROVIDER} 不在可用列表中，改为自动检测")

    # 遍历配置文件，找第一个有 api_key 的
    if os.path.exists(LLM_FILE):
        try:
            with open(LLM_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            provider_cfg = raw.get("providers", {})
            for key in providers:
                env_key = str(providers[key].get("env_key", ""))
                has_key = bool(os.environ.get(env_key)) or bool(
                    provider_cfg.get(key, {}).get("api_key")
                )
                if has_key:
                    info = providers[key]
                    print(f"[Agent] 自动选择: {info['name']} ({key})  可用模型: {', '.join(info.get('models', []))}")
                    return key
        except (json.JSONDecodeError, IOError):
            pass

    # 回退：取第一个
    key = next(iter(providers))
    info = providers[key]
    print(f"[Agent] 未检测到 api_key，回退到: {info['name']} ({key})")
    return key


def _load_remote_thread_id() -> str:
    """读取持久化的 thread_id，不存在返回空字符串。"""
    if os.path.exists(REMOTE_THREAD_FILE):
        with open(REMOTE_THREAD_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def _save_remote_thread_id(thread_id: str) -> None:
    """持久化 thread_id。"""
    parent = os.path.dirname(REMOTE_THREAD_FILE)
    os.makedirs(parent, exist_ok=True)
    with open(REMOTE_THREAD_FILE, "w", encoding="utf-8") as f:
        f.write(thread_id)


def _patch_safety_config() -> None:
    """将安全模块指向远程 bot 专用配置（关闭交互确认，避免控制台阻塞）。"""
    import tools.safety as safety_module

    remote_safety = os.path.join(BASE_DIR, "config", "safety_remote.json")
    if os.path.exists(remote_safety):
        safety_module.CONFIG_PATH = remote_safety
        safety_module.reload_config()
        print("[Agent] 安全配置: config/safety_remote.json（已关闭交互确认）")


def start_agent() -> str:
    """初始化或重启 Agent（复用 main.build_agent，确保行为一致）。"""
    global _agent, _agent_info
    with _agent_lock:
        try:
            # 远程 bot 使用独立的安全配置（关闭交互确认，避免控制台阻塞）
            _patch_safety_config()
            persisted_thread_id = _load_remote_thread_id()
            provider = _auto_detect_provider()
            agent, llm = build_agent(provider)
            # 远程控制关闭 verbose，避免刷屏
            agent.verbose = False
            # 恢复持久化的 thread_id，保证跨重启会话不丢失
            if persisted_thread_id:
                agent.memory.switch_thread(persisted_thread_id)
            else:
                _save_remote_thread_id(agent.memory.thread_id)
            _agent = agent
            _agent_info = {
                "provider": llm.provider,
                "model": llm.model,
                "status": "running",
            }
            thread_id = agent.memory.thread_id
            return f"✅ Agent 已启动\n  提供商: {llm.provider}\n  模型: {llm.model}\n  会话: {thread_id}"
        except Exception as e:
            _agent = None
            _agent_info = {"status": "error", "error": str(e)}
            return f"❌ Agent 启动失败: {str(e)}"


def restart_agent() -> str:
    """重启 Agent。"""
    global _agent_info
    _agent_info["status"] = "restarting"
    return start_agent()


def get_agent() -> Optional[Any]:
    """获取当前 Agent 实例（线程安全）。"""
    with _agent_lock:
        return _agent


# ===================== 指令处理器 =====================


def _find_image_paths(text: str) -> list[str]:
    """从文本中提取图片路径（匹配 .png/.jpg/.jpeg/.gif/.bmp），返回存在的绝对路径。"""
    import re

    IMG_PATTERN = re.compile(
        r"""(?:['"\s]|^)([\w\-\\\/:.]+\.(?:png|jpg|jpeg|gif|bmp))(?:['"\s]|$)""",
        re.IGNORECASE,
    )
    seen = set()
    found: list[str] = []
    for m in IMG_PATTERN.finditer(text):
        raw = m.group(1)
        # 解析为绝对路径
        path = os.path.abspath(raw) if not os.path.isabs(raw) else raw
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            found.append(path)
            if len(found) >= 5:  # 最多发 5 张，防止刷屏
                break
    return found


def _send_image_file(chat_id: str, path: str) -> bool:
    """上传本地图片文件并通过飞书发送。成功返回 True。"""
    try:
        with open(path, "rb") as f:
            req = CreateImageRequest.builder() \
                .request_body(CreateImageRequestBody.builder()
                              .image_type("message")
                              .image(f)
                              .build()) \
                .build()
            resp = _http_client.im.v1.image.create(req)
        if resp.success():
            _send_image_msg(chat_id, resp.data.image_key)
            return True
        else:
            print(f"[图片] 上传失败: {path} → {resp.msg}")
            return False
    except Exception as e:
        print(f"[图片] 上传异常: {path} → {e}")
        return False


def _send_images_from_output(chat_id: str, text: str) -> int:
    """扫描输出文本中的图片路径并发送。返回成功发送的图片数量。"""
    paths = _find_image_paths(text)
    sent = 0
    for p in paths:
        if _send_image_file(chat_id, p):
            sent += 1
    if sent > 0:
        print(f"[图片] 已发送 {sent} 张图片")
    return sent


def _handle_run(chat_id: str, task: str) -> None:
    """后台线程：执行 Agent 任务并推送结果。"""
    agent = get_agent()
    if agent is None:
        _send_text(chat_id, "⚠️ Agent 未启动，请先发送「启动agent」")
        return
    try:
        turn = agent.run_structured(task)
    except Exception as e:
        _send_text(chat_id, f"❌ 任务执行异常: {str(e)}")
        return
    if turn.is_completed:
        output = turn.output or "(无输出)"
        if len(output) > 4000:
            output = output[:4000] + "\n\n...(输出过长已截断)"
        _send_text(chat_id, f"✅ 任务完成\n\n{output}")
        _send_images_from_output(chat_id, turn.output or "")
    elif turn.is_interrupted:
        _send_text(chat_id, "⏸️ 任务被中断，需人工输入，请在本机 CLI 处理。")
    elif turn.status == "cancelled":
        _send_text(chat_id, f"🚫 任务已取消: {turn.output}")


def _handle_chat(chat_id: str, message: str) -> None:
    """后台线程：Agent 对话并推送结果。"""
    agent = get_agent()
    if agent is None:
        _send_text(chat_id, "⚠️ Agent 未启动，请先发送「启动agent」")
        return
    try:
        turn = agent.chat_structured(message)
    except Exception as e:
        _send_text(chat_id, f"❌ 对话异常: {str(e)}")
        return
    if turn.is_completed:
        output = turn.output or "(无回复)"
        if len(output) > 4000:
            output = output[:4000] + "\n\n...(回复过长已截断)"
        _send_text(chat_id, output)
        _send_images_from_output(chat_id, turn.output or "")
    elif turn.is_interrupted:
        _send_text(chat_id, "⏸️ Agent 需人工输入，请在本机 CLI 继续。")
    elif turn.status == "cancelled":
        _send_text(chat_id, f"🚫 已取消: {turn.output}")


def _run_in_background(chat_id: str, method: str, payload: str) -> None:
    """将耗时操作放到后台线程，不阻塞 WebSocket 回调。"""
    def _worker() -> None:
        try:
            if method == "run":
                _handle_run(chat_id, payload)
            elif method == "chat":
                _handle_chat(chat_id, payload)
        finally:
            with _processing_lock:
                _processing.discard(chat_id)
    threading.Thread(target=_worker, daemon=True).start()


def _handle_status(chat_id: str) -> None:
    with _agent_lock:
        info = dict(_agent_info)
    if not info:
        _send_text(chat_id, "Agent 尚未初始化。发送「启动agent」开始。")
        return
    if info.get("status") == "error":
        _send_text(chat_id, f"❌ Agent 状态异常: {info.get('error', '未知错误')}")
        return
    lines = [
        f"📊 Agent 状态",
        f"  状态: {info.get('status', 'unknown')}",
        f"  提供商: {info.get('provider', 'N/A')}",
        f"  模型: {info.get('model', 'N/A')}",
    ]
    agent = get_agent()
    if agent is not None:
        lines.append(f"  工具数: {len(agent.tools)}")
        lines.append(f"  历史步骤: {len(agent.execution_history)}")
    _send_text(chat_id, "\n".join(lines))


def _handle_help(chat_id: str) -> None:
    _send_text(chat_id, (
        "📋 可用指令：\n\n"
        "【Agent 控制】\n"
        "  启动agent  - 初始化/启动 LangChainAgent\n"
        "  重启agent  - 重启 Agent\n"
        "  状态       - 查看 Agent 运行状态\n"
        "  模型       - 查看所有可用模型（编号回复切换）\n\n"
        "【任务执行】\n"
        "  run <任务> - 让 Agent 执行任务\n"
        "  chat <消息> - 与 Agent 对话\n\n"
        "【系统控制】\n"
        "  截图       - 获取本机桌面截图\n"
        "  cmd <命令> - 执行 CMD 命令\n"
        "  ps <命令>  - 执行 PowerShell 命令\n"
        "  帮助       - 显示此帮助"
    )) 







# 模型菜单缓存：chat_id → [(provider_key, model_name), ...]
_model_menu_cache: dict[str, list[tuple[str, str]]] = {}


def _handle_model_menu(chat_id: str, user_input: str = "") -> None:
    """统一的模型选择：发送列表或接收选择。

    - 无参数或"模型": 列出所有 供应商:模型 编号菜单
    - 发送编号: 切换对应模型
    - 发送模型名: 模糊匹配切换
    """
    agent = get_agent()
    if agent is None:
        _send_text(chat_id, "⚠️ Agent 未启动，请先发送「启动agent」")
        return

    # ---------- 列出菜单 ----------
    if not user_input:
        from llm_client import load_providers

        try:
            providers = load_providers(LLM_FILE)
        except Exception as e:
            _send_text(chat_id, f"❌ 读取配置失败: {str(e)}")
            return

        if not providers:
            _send_text(chat_id, "未发现任何供应商配置")
            return

        # 构建扁平列表: [(provider_key, model_name), ...]
        flat: list[tuple[str, str]] = []
        for key, info in providers.items():
            for m in info.get("models", []):
                flat.append((key, m))

        if not flat:
            _send_text(chat_id, "没有任何可用模型")
            return

        _model_menu_cache[chat_id] = flat

        current_pair = (agent.llm.provider, agent.llm.model)
        lines = ["📋 可用模型（回复编号切换）：", ""]
        for i, (p, m) in enumerate(flat, 1):
            mark = "  ← 当前" if (p, m) == current_pair else ""
            lines.append(f"  {i:>2}. {p}: {m}{mark}")
        _send_text(chat_id, "\n".join(lines))
        return

    # ---------- 接收选择 ----------
    flat = _model_menu_cache.get(chat_id)
    if flat is None:
        _send_text(chat_id, "请先发送「模型」查看可用列表")
        return

    # 1) 编号选择
    inp = user_input.strip()
    if inp.isdigit():
        idx = int(inp) - 1
        if 0 <= idx < len(flat):
            provider_key, model_name = flat[idx]
            _do_switch(chat_id, agent, provider_key, model_name)
            return
        _send_text(chat_id, f"⚠️ 编号超出范围 (1-{len(flat)})，请重新发送「模型」")
        return

    # 2) 模型名模糊匹配
    matches = [(p, m) for p, m in flat if inp.lower() in m.lower()]
    if len(matches) == 1:
        provider_key, model_name = matches[0]
        _do_switch(chat_id, agent, provider_key, model_name)
        return
    if len(matches) > 1:
        lines = ["🔍 多个匹配，请回复编号：", ""]
        for i, (p, m) in enumerate(matches, 1):
            lines.append(f"  {i:>2}. {p}: {m}")
        _model_menu_cache[chat_id] = matches
        _send_text(chat_id, "\n".join(lines))
        return

    _send_text(chat_id, f"❌ 未匹配到模型「{inp}」，请重新发送「模型」查看列表")


def _do_switch(chat_id: str, agent, provider_key: str, model_name: str) -> None:
    """执行实际的供应商/模型切换。"""
    old = f"{agent.llm.provider}: {agent.llm.model}"
    new = f"{provider_key}: {model_name}"

    try:
        if agent.llm.provider != provider_key:
            # 跨供应商：需重建 LLM 客户端
            from utils.commands.provider import create_llm

            new_llm = create_llm(provider_key, LLM_FILE)
            agent.llm = new_llm
            # 如果新供应商默认模型不是目标模型，再切换一次
            if agent.llm.model != model_name:
                agent.llm.switch_model(model_name)
        else:
            # 同供应商：只切换模型
            agent.llm.switch_model(model_name)

        agent.agent_executor = agent._create_agent_executor()
        with _agent_lock:
            _agent_info["provider"] = agent.llm.provider
            _agent_info["model"] = agent.llm.model

        _send_text(chat_id, f"✅ 已切换\n  {old}\n  → {new}")
    except Exception as e:
        _send_text(chat_id, f"❌ 切换失败: {str(e)}")


# ===================== 飞书事件处理 =====================

# 去重：WS 可能重复推送同一条消息，用 message_id 去重
_seen_message_ids: set[str] = set()
_MAX_SEEN_IDS = 500


def _on_message(data: P2ImMessageReceiveV1Processor.type()) -> None:
    """处理收到的飞书消息（在 ws 回调线程中调用，务必快速返回）。"""
    try:
        event = data.event
        msg_id = event.message.message_id

        # 去重：同一条消息不处理两次
        if msg_id in _seen_message_ids:
            return
        _seen_message_ids.add(msg_id)
        if len(_seen_message_ids) > _MAX_SEEN_IDS:
            _seen_message_ids.clear()  # 简单轮替，避免无限增长

        sender_open_id = event.sender.sender_id.open_id
        chat_id = event.message.chat_id

        # 解析消息内容
        raw = event.message.content.strip()
        if raw.startswith("{"):
            try:
                content = json.loads(raw).get("text", "")
            except json.JSONDecodeError:
                content = raw
        else:
            content = raw

        print(f"[飞书] 收到消息 | user:{sender_open_id} | content:{content}")

        # 权限校验
        if sender_open_id not in ALLOW_OPEN_ID:
            _send_text(chat_id, "⚠️ 权限不足，禁止操作本机！")
            print(f"[飞书] 拒绝未授权用户: {sender_open_id}")
            return

        # 指令路由
        _dispatch(chat_id, content)

    except Exception as e:
        traceback.print_exc()


def _dispatch(chat_id: str, content: str) -> None:
    """根据消息内容分发到对应处理器。耗时操作走后台线程。"""
    low = content.lower()

    # Agent 控制（瞬间完成，同步处理）
    if low in ("启动agent", "启动", "start"):
        _send_text(chat_id, start_agent())
        return
    if low in ("重启agent", "重启", "restart"):
        _send_text(chat_id, "🔄 正在重启 Agent...")
        _send_text(chat_id, restart_agent())
        return
    if low in ("状态", "status"):
        _handle_status(chat_id)
        return
    if low == "模型" or low == "model":
        _handle_model_menu(chat_id)
        return
    # 纯数字 → 当作模型编号选择
    if content.strip().isdigit():
        _handle_model_menu(chat_id, content.strip())
        return

    # 系统控制（瞬间完成，同步处理）
    if low in ("截图", "screenshot", "snap"):
        _send_text(chat_id, "📸 正在截图...")
        _send_screenshot(chat_id)
        return
    if low.startswith("cmd "):
        cmd = content[4:].strip()
        try:
            output = subprocess.check_output(cmd, shell=True, text=True, timeout=30)
            text = f"✅ 执行结果:\n{output[:1800]}"
            if len(output) > 1800:
                text += "\n...(输出过长已截断)"
            _send_text(chat_id, text)
        except subprocess.TimeoutExpired:
            _send_text(chat_id, "⏰ 命令执行超时（30 秒）")
        except Exception as e:
            _send_text(chat_id, f"❌ 命令执行失败: {str(e)}")
        return
    if low.startswith("ps ") or low.startswith("ps\n"):
        ps_cmd = content[3:].strip()
        try:
            output = subprocess.check_output(
                ["powershell", "-Command", ps_cmd],
                text=True, timeout=30
            )
            text = f"✅ PowerShell:\n{output[:1800]}"
            if len(output) > 1800:
                text += "\n...(输出过长已截断)"
            _send_text(chat_id, text)
        except subprocess.TimeoutExpired:
            _send_text(chat_id, "⏰ 命令执行超时（30 秒）")
        except Exception as e:
            _send_text(chat_id, f"❌ PowerShell 执行失败: {str(e)}")
        return
    if low in ("帮助", "help", "?", "？"):
        _handle_help(chat_id)
        return

    # --- 以下为耗时 Agent 操作，放入后台线程 ---

    # 防止同一会话并发
    with _processing_lock:
        if chat_id in _processing:
            _send_text(chat_id, "⏳ 上一个任务还在处理中，请稍候...")
            return
        _processing.add(chat_id)

    if low.startswith("run "):
        task = content[4:].strip()
        if task:
            _send_text(chat_id, f"🤖 正在执行任务...\n\n任务: {task[:200]}")
            _run_in_background(chat_id, "run", task)
        else:
            with _processing_lock:
                _processing.discard(chat_id)
            _send_text(chat_id, "用法: run <任务描述>")
    elif low.startswith("run\n"):
        task = content.split("\n", 1)[1].strip()
        if task:
            _send_text(chat_id, f"🤖 正在执行任务...\n\n任务: {task[:200]}")
            _run_in_background(chat_id, "run", task)
        else:
            with _processing_lock:
                _processing.discard(chat_id)
            _send_text(chat_id, "用法: run (换行) <任务描述>")
    elif low.startswith("chat "):
        msg = content[5:].strip()
        if msg:
            _send_text(chat_id, "🤖 思考中...")
            _run_in_background(chat_id, "chat", msg)
        else:
            with _processing_lock:
                _processing.discard(chat_id)
            _send_text(chat_id, "用法: chat <消息>")
    elif low.startswith("chat\n"):
        msg = content.split("\n", 1)[1].strip()
        if msg:
            _send_text(chat_id, "🤖 思考中...")
            _run_in_background(chat_id, "chat", msg)
        else:
            with _processing_lock:
                _processing.discard(chat_id)
            _send_text(chat_id, "用法: chat (换行) <消息>")
    else:
        # 默认当作 chat 处理
        _send_text(chat_id, "🤖 思考中...")
        _run_in_background(chat_id, "chat", content)


# ===================== 入口 =====================

def run_remote_bot() -> None:
    """启动 LangChainAgent 飞书远程控制机器人。供 main.py --remote 调用。"""
    print("=" * 55)
    print("  LangChainAgent 飞书远程控制机器人 (v2)")
    print("=" * 55)
    print(f"  项目路径: {BASE_DIR}")
    print(f"  配置文件: {REMOTE_CONFIG_FILE}")
    print()

    # 启动时自动初始化 Agent
    print("[启动] 正在初始化 Agent...")
    result = start_agent()
    print(f"  {result}")
    print()

    # 构建事件处理器（WS 模式不需要 encrypt_key/verification_token）
    handler = EventDispatcherHandler.builder("", "") \
        .register_p2_im_message_receive_v1(_on_message) \
        .build()

    print("  WebSocket 正在连接飞书云端...")
    print("  请在飞书中发送消息以获取 open_id，然后加入 config/remote_control.json 的 allow_open_id 白名单")
    print()

    ws_client = LarkWSClient(
        app_id=APP_ID,
        app_secret=APP_SECRET,
        event_handler=handler,
        log_level=lark.LogLevel.ERROR,
    )
    ws_client.start()


if __name__ == "__main__":
    run_remote_bot()
