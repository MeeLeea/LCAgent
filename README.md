# LangChainAgent

基于 **LangChain 1.x + LangGraph** 框架的智能 Agent 项目，支持：
- 多 LLM 提供商（智谱/千问/DeepSeek/Kimi）
- 本地工具调用（搜索、文件读写、计算、终端命令）
- **MCP Server 工具动态加载**（文件夹管理、可扩展任意 MCP 服务）
- **LangGraph Checkpoint 持久化**（SQLite 自动保存，程序重启可恢复对话）
- 长期记忆管理（compress 压缩摘要）

## 项目结构

```
LangChainAgent/
├── main.py                  # 入口文件，交互式命令行
├── llm_client.py            # 统一大模型封装（多提供商）
├── llm_config.json          # API密钥配置文件
├── mcp_servers.json         # MCP Server 配置文件
├── requirements.txt         # 依赖列表
├── data/
│   ├── checkpoints.sqlite   # Checkpoint 持久化数据库（运行时自动生成）
│   └── memory.json          # 长期记忆文件（用于 compress 摘要）
├── agent/
│   ├── __init__.py
│   ├── memory.py            # 记忆模块（Checkpoint + 长期记忆）
│   └── agent_core.py        # Agent核心调度
└── tools/
    ├── __init__.py          # 本地工具注册
    ├── search.py            # 联网搜索工具
    ├── file_tool.py         # 文件读写工具
    ├── calculator.py        # 数学计算工具
    ├── terminal_tools.py    # 终端命令工具（shell/python/bat/ps1）
    ├── get_local_time.py    # 获取本地时间工具
    ├── mcp_loader.py        # MCP 工具加载器
    └── workspace_tool.py    # 工作目录管理 MCP Server
```

## 模块说明

### main.py — 入口

交互式命令行，负责：
- 启动时选择 LLM 提供商
- 解析用户命令（`react:`/`cot:`/`switch:`/`info`/`tools`/`clear`/`mcp` 等）
- 调用 Agent 执行任务

### llm_client.py — 大模型封装

基于 `langchain_openai.ChatOpenAI` 的统一封装：

| 类/函数 | 说明 |
|---------|------|
| `LLMClient` | 核心客户端类，支持4个提供商 |
| `LLMClient.chat()` | 发送对话请求 |
| `LLMClient.chat_with_history()` | 带历史记录的对话 |
| `LLMClient.get_chat_model()` | 获取 LangChain `ChatOpenAI` 实例 |
| `LLMClient.switch_provider()` | 运行时切换提供商 |
| `create_client()` | 创建客户端的便捷函数 |
| `list_providers()` | 列出所有支持的提供商 |

**支持的提供商：**

| 提供商 | 名称 | API地址 | 环境变量 | 默认模型 |
|--------|------|---------|----------|----------|
| `zhipu` | 智谱AI | open.bigmodel.cn | `ZHIPU_API_KEY` | glm-4 |
| `qwen` | 通义千问 | dashscope.aliyuncs.com | `DASHSCOPE_API_KEY` | qwen-plus |
| `deepseek` | DeepSeek | api.deepseek.com | `DEEPSEEK_API_KEY` | deepseek-chat |
| `kimi` | Kimi | api.moonshot.cn | `MOONSHOT_API_KEY` | moonshot-v1-8k |

---

## 记忆系统（Memory）

### 设计架构

Agent 采用 **LangGraph Checkpoint + 长期记忆** 双层设计，由 [agent/memory.py](agent/memory.py) 的 `AgentMemory` 类实现：

```
┌──────────────────────────────────────────────────────────────┐
│                       AgentMemory                              │
│                                                                │
│  ┌───────────────────────────┐  ┌─────────────────────────┐  │
│  │   Checkpoint (自动)        │  │   长期记忆 (手动)         │  │
│  │   SQLite 持久化            │  │   memory.json            │  │
│  │                           │  │                           │  │
│  │   • Agent 每步执行后自动写 │  │   • 仅 important=True 写入│  │
│  │   • 完整状态(消息+工具调用)│  │   • 仅 react/cot 触发     │  │
│  │   • 按 thread_id 隔离      │  │   • 用于 compress 摘要    │  │
│  │   • 程序重启可恢复         │  │   • 跨会话保留关键信息    │  │
│  └───────────────────────────┘  └─────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 两层存储对比

| 类型 | 存储方式 | 触发时机 | 保存内容 | 持久化 | 用途 |
|------|---------|---------|---------|--------|------|
| **Checkpoint** | `data/checkpoints.sqlite` (SQLite) | Agent 每步执行后自动 | 完整状态(消息+工具调用链+中间变量) | ✅ 永久 | 程序重启恢复对话、多会话隔离 |
| **长期记忆** | `data/memory.json` (JSON) | 手动标记 `important=True` | 仅 react/cot 的最终结果 | ✅ 永久 | 跨会话保留关键决策、用于 compress |

### 三种模式的记忆行为

| 模式 | 调用方法 | 写入 Checkpoint | 写入 memory.json | 说明 |
|------|---------|----------------|------------------|------|
| `react:任务` | `run()` | ✅ 自动 | ✅ 手动触发 | ReAct 结果重要，永久保存 |
| `cot:任务` | `cot()` | ❌ 不写 | ✅ 手动触发 | CoT 不走 Agent，仅存结论 |
| 普通输入 | `chat()` | ✅ 自动 | ❌ 不写 | 闲聊仅存 checkpoint，用完即丢 |

> **关键区别**：Checkpoint 由 LangGraph 自动管理，无需手动干预；长期记忆需显式传 `metadata={"important": True}` 才会写入 `memory.json`。

### Checkpoint 持久化原理

Agent 创建时传入 `checkpointer`，LangGraph 在每次 `invoke()` 后自动保存状态到 SQLite：

```python
# agent/agent_core.py
agent = create_react_agent(
    model=chat_model,
    tools=self.tools,
    prompt=self._get_system_prompt(),
    checkpointer=self.memory.get_checkpointer(),  # ← 自动持久化
)

# 调用时传 thread_id,LangGraph 自动恢复该会话历史
result = self.agent_executor.invoke(
    {"messages": [HumanMessage(content=task)]},
    config=self.memory.get_config()  # {"configurable": {"thread_id": "..."}}
)
```

**核心能力**：

| 能力 | 说明 |
|------|------|
| 自动持久化 | Agent 每步执行后自动写入 SQLite，无需手动 |
| 程序重启恢复 | 重启后用相同 `thread_id` 即可恢复完整对话历史 |
| 多会话隔离 | 不同 `thread_id` 完全独立，可管理多个对话 |
| 完整状态保存 | 保存消息、工具调用链、中间变量(不止文本) |
| 中断恢复 | 程序崩溃可从最近 checkpoint 续跑 |

### 长期记忆(memory.json)的触发原理

Checkpoint 是自动全量保存，长期记忆是**手动精选**——只有 `important=True` 的内容才写入：

```python
# agent/memory.py
def add(self, role, content, metadata=None):
    # 只在 important=True 时才写入 memory.json
    if metadata and metadata.get("important", False):
        self.long_term_memory.append(item)
        self._save_long_term_memory()
    # 普通(非important)由 checkpoint 自动处理,这里不再存
```

在 [agent/agent_core.py](agent/agent_core.py) 中：

```python
# run() 和 cot() 模式 - 存长期
self.memory.add("user", task)
self.memory.add("assistant", output, {"important": True})  # ← 触发持久化

# chat() 模式 - 不存长期(由 checkpoint 自动保存)
self.memory.add("user", message)
self.memory.add("assistant", output)  # ← 无 important,不写 memory.json
```

### 为什么 cot 不写入 checkpoint

`cot()` 方法直接调用 `self.llm.chat_with_history()`，**绕过了 `agent_executor.invoke()`**，因此 checkpoint 不会记录。这是设计上的选择：

- **cot 的语义**：纯推理，不调用工具
- **避免被 Agent 拦截**：走 Agent 通道 LLM 可能自己决定调用工具，违背 cot 初衷
- **checkpoint 是为 Agent 设计的**：cot 没有工具调用链，用 memory.json 存结论即可

**副作用**：cot 之间无法续接(第二次 cot 看不到第一次 cot 的对话)，只能通过 memory.json 的长期记忆间接看到摘要。

### 文件位置

由 [main.py](main.py) 配置：

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(BASE_DIR, "data", "checkpoints.sqlite")  # Checkpoint 数据库
MEMORY_FILE = os.path.join(BASE_DIR, "data", "memory.json")            # 长期记忆
```

| 文件 | 格式 | 内容 |
|------|------|------|
| `data/checkpoints.sqlite` | SQLite | Agent 执行状态(按 thread_id 隔离) |
| `data/memory.json` | JSON 数组 | react/cot 的最终结果(用于 compress) |

`memory.json` 文件格式：

```json
[
  {
    "role": "user",
    "content": "react:帮我搜索Python教程",
    "timestamp": "2026-07-06T10:30:00.123456",
    "metadata": {"important": true}
  },
  {
    "role": "assistant",
    "content": "已为你找到以下Python教程...",
    "timestamp": "2026-07-06T10:30:02.654321",
    "metadata": {"important": true}
  }
]
```

> 两个文件都会**自动创建父目录**，无需手动建 `data/` 文件夹。

### 会话(Thread)管理

基于 checkpoint 的 `thread_id` 机制，支持多会话管理：

| 命令 | 作用 | 示例 |
|------|------|------|
| `thread` | 列出所有会话(标记当前 *) | `thread` |
| `thread:new` | 开启新会话(原会话保留) | `thread:new` |
| `thread:switch <id>` | 切换到指定会话(恢复历史) | `thread:switch thread-abc123` |

**示例**：

```
你: thread
所有会话 (共 2 个):
------------------------------------------------------------
  * thread-a4d099d2
    thread-b1dc3b2a
------------------------------------------------------------
* = 当前会话

你: thread:new
已开启新会话: thread-c7e8f1a3
原会话 thread-a4d099d2 已保留,可用 'thread:switch thread-a4d099d2' 切回

你: thread:switch thread-a4d099d2
已切换到会话: thread-a4d099d2 (恢复 6 条历史消息)
```

### 记忆管理命令

| 命令 | 作用 |
|------|------|
| `clear` 或 `clear long` | 清空长期记忆 + 删除 `memory.json` |
| `clear short` | 清空当前会话(开启新 thread 替代删除) |
| `clear all` | 全部清空 |
| `compress` 或 `压缩` | 压缩长期记忆(LLM 摘要后替换原内容) |
| `thread` | 列出所有会话 |
| `thread:new` | 开启新会话 |
| `thread:switch <id>` | 切换会话(恢复历史) |

用 `info` 命令可查看记忆状态：

```
你: info
当前提供商: DeepSeek
当前模型:   deepseek-chat
API地址:    https://api.deepseek.com

--- 记忆状态 ---
当前会话:   thread-a4d099d2
Checkpoint: sqlite → D:\work\LangChainAgent\data\checkpoints.sqlite
已存消息:   8 条
长期记忆:   5 条
总会话数:   2
```

### 压缩长期记忆（compress）

随着 `react:` / `cot:` 不断积累，`memory.json` 会越来越大。`compress` 命令通过 LLM 把所有长期记忆压缩成一份摘要，再写回 `memory.json`。

> **注意**：compress 只压缩 `memory.json`，不影响 `checkpoints.sqlite`。

#### 工作流程

```
memory.json (N条原始记忆)
       ↓
拼接为文本 + 压缩提示词
       ↓
发送给 LLM
       ↓
LLM 生成摘要
       ↓
用摘要替换原 N 条记忆（变成 1 条）
       ↓
保存回 memory.json
```

#### 压缩提示词

系统提示词要求 LLM：

1. 保留所有关键信息、用户意图、重要决策和事实
2. 去除重复和冗余内容
3. 按主题分条目组织，使用 `- ` 开头
4. 保持事实准确，不添加推测内容
5. 用中文输出

#### 使用示例

```
你: compress

开始压缩长期记忆 (共 8 条)...
压缩完成！
  原记忆条数:   8 条
  原字符数:     4523 字符
  压缩后字符数: 687 字符
  压缩率:       84.8%

--- 摘要内容 ---
- 用户要求创建 LangChainAgent 项目
- 已配置 4 个 LLM 提供商（智谱/千问/DeepSeek/Kimi）
- 添加了 MCP workspace 工具（6 个文件夹管理工具）
- 修复了 StructuredTool 同步调用问题
- ...
--- 已保存到 memory.json ---
```

#### 压缩后的 memory.json 格式

```json
[
  {
    "role": "system",
    "content": "[历史记忆摘要 2026-07-06T12:00:00]\n- 用户要求创建...\n- 已配置...",
    "timestamp": "2026-07-06T12:00:00",
    "metadata": {
      "important": true,
      "type": "summary",
      "original_count": 8,
      "original_chars": 4523,
      "compressed_chars": 687
    }
  }
]
```

`metadata` 中保留了压缩前的条数和字符数，便于追溯。

#### 特点与注意事项

| 特性 | 说明 |
|------|------|
| 保留关键信息 | system prompt 要求 LLM 保留用户意图、决策、事实 |
| 结构化输出 | 按主题分条，便于后续查阅 |
| 可追溯 | `metadata` 保存原始条数和字符数 |
| 可多次压缩 | 再次 `compress` 会对已有摘要再压缩 |
| LLM 失败不丢数据 | 调用失败则原记忆不变，不会写回 |
| **不可逆** | 压缩后原条目无法恢复，建议重要数据先备份 `memory.json` |
| 不影响 checkpoint | 只压缩 memory.json，checkpoints.sqlite 保留完整历史 |

> 💡 **建议**：在长期记忆较多时（如 50+ 条）使用，平时少量记忆无需压缩。

### 记忆相关 API

| 方法 | 说明 |
|------|------|
| `get_checkpointer()` | 获取 checkpointer 实例(传给 create_react_agent) |
| `get_config()` | 返回 `{"configurable": {"thread_id": ...}}`(传给 invoke) |
| `get_messages()` | 从 checkpoint 获取当前 thread 的所有消息 |
| `get_langchain_messages()` | 等同于 get_messages()（兼容旧 API） |
| `get_short_term(limit)` | 从 checkpoint 取消息转为 dict 格式 |
| `get_long_term(limit)` | 获取最近 N 条长期记忆 |
| `get_all_context(long_term_limit)` | 获取完整上下文(长期+短期) |
| `add(role, content, metadata)` | 添加记忆，`important=True` 触发写 memory.json |
| `new_thread()` | 开启新会话 |
| `switch_thread(thread_id)` | 切换到指定会话 |
| `list_threads()` | 列出所有会话 ID |
| `clear_short_term()` | 开启新会话(替代删除) |
| `clear_long_term()` | 清空长期记忆并删除文件 |
| `summarize()` | 返回记忆统计信息(含 thread_id、消息数、会话数) |
| `compress_memory()` | 压缩长期记忆（LLM 摘要后替换原内容） |

---

## 工具系统（Tools）

Agent 的工具分为两类：

```
┌────────────────────────────────────────────────┐
│                 Agent 工具池                    │
│                                                │
│   ┌──────────────────┐  ┌──────────────────┐  │
│   │   本地工具         │  │   MCP 工具         │  │
│   │  (Local Tools)    │  │  (MCP Tools)     │  │
│   │                   │  │                   │  │
│   │  @tool 装饰器定义  │  │  从 MCP Server    │  │
│   │  直接 import 调用  │  │  动态加载         │  │
│   └──────────────────┘  └──────────────────┘  │
└────────────────────────────────────────────────┘
```

### 1. 本地工具（Local Tools）

使用 LangChain `@tool` 装饰器定义，启动时直接加载：

| 工具 | 文件 | 功能 | 参数 |
|------|------|------|------|
| `search` | [tools/search.py](tools/search.py) | 联网搜索 | `query`, `num_results` |
| `read_file` | [tools/file_tool.py](tools/file_tool.py) | 读取文件 | `file_path` |
| `write_file` | [tools/file_tool.py](tools/file_tool.py) | 写入文件 | `file_path`, `content`, `mode` |
| `calculate` | [tools/calculator.py](tools/calculator.py) | 数学计算 | `expression` |

### 2. MCP 工具（MCP Server Tools）

通过 **MCP（Model Context Protocol）** 协议从独立进程加载，支持动态增删。

#### 已注册的 MCP Server

| Server 名称 | 工具数 | 传输方式 | 说明 |
|-------------|--------|---------|------|
| `workspace` | 6 | stdio | 文件夹创建/删除/移动/复制/列举（Python 实现） |
| `filesystem` | - | stdio | 官方文件系统服务（默认禁用，需 Node.js） |
| `fetch` | - | stdio | 官方网页抓取服务（默认禁用，需 Node.js） |

#### workspace MCP Server 提供的工具

由 [tools/workspace_tool.py](tools/workspace_tool.py) 实现，基于 `FastMCP`：

| 工具 | 功能 | 参数 |
|------|------|------|
| `create_workspace` | 创建文件夹 | `folder_name`, `parent_dir` |
| `get_current_workspace` | 获取当前工作目录 | 无 |
| `list_directory` | 列出目录内容 | `path` |
| `delete_workspace` | 删除文件夹 | `folder_path`, `recursive` |
| `move_workspace` | 移动/重命名文件夹 | `src_path`, `dest_path` |
| `copy_workspace` | 复制文件夹 | `src_path`, `dest_path` |

#### MCP 配置文件

[mcp_servers.json](mcp_servers.json) 定义所有 MCP Server：

```json
{
    "servers": {
        "workspace": {
            "transport": "stdio",
            "command": "D:\\work\\LangChainAgent\\.venv\\Scripts\\python.exe",
            "args": ["D:\\work\\LangChainAgent\\tools\\workspace_tool.py"],
            "enabled": true
        },
        "filesystem": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "D:\\work"],
            "enabled": false
        }
    }
}
```

| 字段 | 说明 |
|------|------|
| `transport` | 传输方式：`stdio` / `sse` / `streamable_http` |
| `command` | stdio 模式下的启动命令 |
| `args` | 命令参数列表 |
| `enabled` | 是否启用（`false` 则不加载） |
| `url` | sse/http 模式下的服务器地址 |

#### MCP 管理命令

在交互界面中动态管理 MCP Server：

| 命令 | 作用 | 示例 |
|------|------|------|
| `mcp` | 查看 MCP Server 状态 | `mcp` |
| `mcp:reload` | 重新加载所有 MCP 工具 | `mcp:reload` |
| `mcp:add` | 添加 stdio MCP Server | `mcp:add myserver npx -y @modelcontextprotocol/server-fetch` |
| `mcp:add` | 添加 sse MCP Server | `mcp:add myserver sse:http://localhost:8000/sse` |
| `mcp:add` | 添加 http MCP Server | `mcp:add myserver http:http://localhost:8000/mcp` |
| `mcp:remove` | 删除 MCP Server | `mcp:remove fetch` |
| `mcp:toggle` | 启用/禁用 | `mcp:toggle fetch on` / `mcp:toggle fetch off` |

### 工具调用机制

Agent 基于 **LangGraph `create_react_agent`** 实现，工具调用流程：

```
用户输入
   ↓
AgentCore.chat() / run()
   ↓
agent_executor.invoke({"messages": [...]})
   ↓
LLM 收到 system prompt + 工具列表 + 用户消息
   ↓
LLM 决定是否调用工具
   ├─ 是 → 返回 tool_call → Agent 执行工具 → 结果回传 LLM → 最终回复
   └─ 否 → 直接返回文本回复
```

### 三种模式与工具的关系

| 模式 | 工具可用 | 步骤打印 | 长期记忆 | 适用场景 |
|------|---------|---------|---------|---------|
| 普通输入（`chat()`） | ✅ 有 | ❌ 静默 | ❌ 仅短期 | 日常对话、轻量操作 |
| `react:任务`（`run()`） | ✅ 有 | ✅ 打印每步 | ✅ 存长期 | 复杂任务、需观察推理过程 |
| `cot:任务`（`cot()`） | ❌ 无 | ❌ 静默 | ✅ 存长期 | 纯推理、分析类问题 |

> **重要**：`chat()` 模式也走 Agent 执行，LLM 会自动判断是否调用工具。无需强制加 `react:` 前缀也能创建文件、目录等。

### System Prompt 强化

为确保 LLM 主动调用工具而非拒绝，system prompt 中明确要求：

```
1. 当用户要求创建文件、读写文件、创建目录等操作时，你【必须】调用相应工具
2. 绝对不要回复'我无法访问你的文件系统'、'请你自己保存'之类的话
3. 你确实拥有这些工具的能力，工具会在用户本地执行
4. 如果用户要保存内容到文件，直接调用 write_file 工具
5. 如果用户要创建目录，直接调用 create_workspace 工具
```

---

## 快速开始

### 1. 创建虚拟环境

```powershell
cd D:\work\LangChainAgent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> 如遇脚本执行限制：`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 配置 API 密钥

编辑 [llm_config.json](llm_config.json)，填入你的密钥：

```json
{
    "zhipu": {
        "api_key": "你的智谱密钥",
        "model": "glm-4"
    },
    "deepseek": {
        "api_key": "你的DeepSeek密钥",
        "model": "deepseek-chat"
    }
}
```

> 也可以通过环境变量配置，例如 `set DEEPSEEK_API_KEY=sk-xxx`

### 4. 运行

```powershell
python main.py
```

启动后选择提供商即可进入交互模式。

---

## 使用方式

### 交互命令一览

| 命令 | 说明 |
|------|------|
| `react:任务` | Agent 模式，自动调用工具，打印步骤，存长期记忆 |
| `cot:任务` | 链式思考模式，纯推理不调用工具 |
| `switch:提供商名` | 运行时切换 LLM（如 `switch:deepseek`） |
| `info` | 查看当前模型和记忆状态(含 thread_id、会话数) |
| `tools` | 查看可用工具列表（含 MCP 工具） |
| `clear [long\|short\|all]` | 清理记忆（默认 long） |
| `compress` 或 `压缩` | 压缩长期记忆（LLM 摘要后替换原内容） |
| `thread` | 列出所有会话 |
| `thread:new` | 开启新会话(原会话保留) |
| `thread:switch <id>` | 切换到指定会话(恢复历史) |
| `mcp` | 查看 MCP Server 状态 |
| `mcp:reload` | 重新加载 MCP 工具 |
| `mcp:add <name> ...` | 添加 MCP Server |
| `mcp:remove <name>` | 删除 MCP Server |
| `mcp:toggle <name> <on\|off>` | 启用/禁用 MCP Server |
| `quit` / `exit` | 退出 |
| 其他输入 | 普通对话模式（也支持工具调用，但不打印步骤） |

### 运行示例

```
==================================================
  LangChain Agent (基于LangChain框架)
==================================================

可用的大模型提供商:
--------------------------------------------------
  1. [ ] zhipu      (智谱AI)
  2. [ ] qwen       (通义千问)
  3. [✓] deepseek   (DeepSeek)
  4. [ ] kimi       (Kimi (Moonshot))
--------------------------------------------------

请选择提供商 (1-4) 或直接回车使用默认[智谱]: 3

[MCP] 已加载 6 个工具: create_workspace, get_current_workspace, list_directory, delete_workspace, move_workspace, copy_workspace

Agent 已就绪！
当前提供商: DeepSeek
当前模型:   deepseek-chat

本地工具: search, read_file, write_file, calculate, run_shell, run_python, run_cmd, get_local_time
MCP工具:  create_workspace, get_current_workspace, list_directory, delete_workspace, move_workspace, copy_workspace

你: 在 D:\work 下创建一个叫 my_project 的文件夹
助手: 已为你创建文件夹 my_project，路径：D:\work\my_project

你: react:计算 (123 + 456) * 2
--- 步骤 1 ---
工具: calculate
输入: {'expression': '(123 + 456) * 2'}
结果: 1158
最终答案: (123 + 456) * 2 = 1158

你: cot:分析Python和Java的区别
最终答案: Python和Java的区别在于...

你: info
当前提供商: DeepSeek
当前模型:   deepseek-chat
API地址:    https://api.deepseek.com

--- 记忆状态 ---
当前会话:   thread-a4d099d2
Checkpoint: sqlite → D:\work\LangChainAgent\data\checkpoints.sqlite
已存消息:   8 条
长期记忆:   2 条
总会话数:   1

你: thread:new
已开启新会话: thread-c7e8f1a3
原会话 thread-a4d099d2 已保留

你: thread
所有会话 (共 2 个):
------------------------------------------------------------
    thread-a4d099d2
  * thread-c7e8f1a3
------------------------------------------------------------

你: mcp
MCP Servers:
------------------------------------------------------------
  [✓启用] workspace (stdio)
           D:\work\LangChainAgent\.venv\Scripts\python.exe D:\work\LangChainAgent\tools\workspace_tool.py
  [✗禁用] filesystem (stdio)
           npx -y @modelcontextprotocol/server-filesystem D:\work
------------------------------------------------------------
已加载 MCP 工具数: 6

你: quit
再见!
```

> **重启程序后**：用 `thread` 查看所有历史会话，`thread:switch <id>` 可恢复之前的对话上下文。

---

## 代码使用示例

```python
from llm_client import create_client
from agent import AgentCore

# 创建客户端和Agent
llm = create_client(provider="deepseek", config_file="llm_config.json")
agent = AgentCore(
    llm_client=llm,
    memory_size=10,
    long_term_memory_file="data/memory.json",        # 长期记忆(用于 compress)
    checkpoint_file="data/checkpoints.sqlite",       # Checkpoint 持久化
    max_iterations=25,
    mcp_config_file="mcp_servers.json",
    enable_mcp=True
)

# 普通对话（自动判断是否调用工具，自动写 checkpoint，不存长期记忆）
response = agent.chat("帮我创建一个叫 test 的文件夹")

# Agent模式（自动调用工具，打印步骤，写 checkpoint + 存长期记忆）
result = agent.run("计算 123 * 456 并把结果写入 result.txt")

# CoT模式（纯推理，不调用工具，不写 checkpoint，存长期记忆）
result = agent.cot("分析机器学习的应用场景")

# 切换LLM
from llm_client import create_client
new_llm = create_client(provider="qwen", config_file="llm_config.json")
agent.switch_llm(new_llm)

# 会话管理
agent.memory.new_thread()                    # 开启新会话
agent.memory.switch_thread("thread-abc123")  # 切换到已有会话
print(agent.memory.list_threads())           # 列出所有会话

# 记忆管理
agent.memory.clear_long_term()   # 清空长期记忆
agent.memory.clear_short_term()  # 开启新会话(替代删除)
print(agent.memory.summarize())  # 查看记忆统计(含 thread_id、消息数)

# 压缩长期记忆
result = agent.compress_memory()
print(f"压缩率: {1 - result['compressed_chars']/result['original_chars']:.1%}")

# MCP 工具管理
agent.reload_mcp_tools()         # 重新加载 MCP 工具
print(agent.get_available_tools())  # 查看所有工具
```

---

## 扩展工具

### 方式 1：添加本地工具

在 `tools/` 目录新建文件，使用 `@tool` 装饰器定义工具：

```python
# tools/my_tool.py
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """工具描述（Agent会读取这个docstring来理解工具用途）"""
    return f"处理结果: {param}"
```

在 [tools/__init__.py](tools/__init__.py) 中导入并添加到 `all_tools`：

```python
from .my_tool import my_tool
all_tools = [search, read_file, write_file, calculate, my_tool]
```

重启后 Agent 即可自动调用新工具。

### 方式 2：添加 MCP Server

**A. 使用现有 MCP Server**（如官方提供的）：

在交互界面输入：

```
mcp:add fetch npx -y @modelcontextprotocol/server-fetch
```

或直接编辑 [mcp_servers.json](mcp_servers.json)：

```json
{
    "servers": {
        "fetch": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-fetch"],
            "enabled": true
        }
    }
}
```

然后输入 `mcp:reload` 热加载。

**B. 自定义 MCP Server**（Python + FastMCP）：

```python
# tools/my_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def my_tool(param: str) -> dict:
    """工具描述"""
    return {"result": param}

if __name__ == "__main__":
    mcp.run()
```

注册到 `mcp_servers.json`：

```json
{
    "servers": {
        "myserver": {
            "transport": "stdio",
            "command": "python",
            "args": ["tools/my_server.py"],
            "enabled": true
        }
    }
}
```

运行 `mcp:reload` 即可加载。

### 本地工具 vs MCP 工具对比

| 特性 | 本地工具 | MCP 工具 |
|------|---------|---------|
| 定义方式 | `@tool` 装饰器 | `@mcp.tool()` + FastMCP |
| 运行位置 | Agent 同进程 | 独立子进程 |
| 加载方式 | import 后直接使用 | 通过 MCP 协议动态加载 |
| 增删 | 需改代码 + 重启 | 改配置 + `mcp:reload` 热加载 |
| 跨语言支持 | 仅 Python | 任意语言（Node/Go 等） |
| 适用场景 | 简单、轻量工具 | 复杂服务、第三方集成 |

---

## 技术栈

- Python 3.10+
- LangChain 1.x（`langchain`, `langchain-core`, `langchain-openai`）
- LangGraph 1.x（`create_react_agent`）
- LangGraph Checkpoint（`langgraph-checkpoint-sqlite`，SQLite 持久化）
- langchain-mcp-adapters 0.3+（MCP 工具适配）
- mcp 1.9+（FastMCP Server）
- OpenAI SDK（兼容所有4个提供商的 API 格式）
