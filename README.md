# LangChainAgent

基于 **LangChain 1.x + LangGraph** 框架的智能 Agent 项目，支持：

- 配置驱动的多 LLM 提供商（见 `config/llm_config.json`），运行时可切换提供商/模型
- 本地工具调用（搜索、文件读写、计算、终端命令、文件打开、技能读取）
- **MCP Server 工具动态加载**（已预置 workspace 文件夹管理服务，可扩展任意 MCP 服务）
- **LangGraph Checkpoint 持久化**（SQLite 自动保存，程序重启可恢复对话）
- **LangGraph Human-in-the-loop**（`ask_human` 暂停图执行，CLI 结构化选择后 `Command(resume)` 继续）
- 长期记忆管理（compress 压缩摘要）与长上下文自动裁剪
- 多会话隔离（thread_id 机制，方向键菜单切换/删除/导出）
- 安全护栏（危险终端命令拦截/确认、路径保护）

---

## 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [核心概念：三种模式](#核心概念三种模式)
- [记忆系统](#记忆系统memory)
  - [设计架构](#设计架构)
  - [上下文注入机制](#上下文注入机制重要)
  - [两层存储对比](#两层存储对比)
  - [三种模式的记忆行为](#三种模式的记忆行为)
  - [Checkpoint 持久化原理](#checkpoint-持久化原理)
  - [长期记忆触发原理](#长期记忆触发原理)
  - [为什么 cot 不写入 checkpoint](#为什么-cot-不写入-checkpoint)
  - [文件位置](#文件位置)
  - [会话(Thread)管理](#会话thread管理)
  - [记忆管理命令](#记忆管理命令)
  - [压缩长期记忆(compress)](#压缩长期记忆compress)
  - [记忆相关 API](#记忆相关-api)
- [工具系统](#工具系统tools)
  - [本地工具](#1-本地工具local-tools)
  - [MCP 工具](#2-mcp-工具mcp-server-tools)
  - [技能阅读 Skills](#3-技能阅读skills)
  - [安全护栏 Safety](#4-安全护栏safety)
  - [工具调用机制](#工具调用机制)
  - [System Prompt 强化](#system-prompt-强化)
- [Human-in-the-loop（HITL）](#human-in-the-loophitl)
  - [执行流程](#执行流程)
  - [ask_human 工具](#ask_human-工具)
  - [结构化 API](#结构化-api)
  - [CLI 编排](#cli-编排)
  - [并行 interrupt 与多轮暂停](#并行-interrupt-与多轮暂停)
  - [线程隔离](#线程隔离)
  - [与普通多轮聊天、安全护栏的区别](#与普通多轮聊天安全护栏的区别)
  - [代码示例](#代码示例)
  - [限制与注意事项](#限制与注意事项)
- [交互命令参考](#交互命令参考)
- [运行示例](#运行示例)
- [代码使用示例](#代码使用示例)
- [扩展工具](#扩展工具)
- [运行时配置（agent_config.json）](#运行时配置agent_configjson)
- [测试](#测试)
- [技术栈](#技术栈)

---

## 快速开始

### 1. 创建虚拟环境

```bash
cd <项目路径>
python -m venv .venv
```

| 平台   | 激活命令                                         |
|--------|--------------------------------------------------|
| Windows | `.\.venv\Scripts\Activate.ps1`                   |
| Linux   | `source .venv/bin/activate`                      |

> Windows 如遇脚本执行限制：`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备配置文件

项目克隆后 `config/` 目录下只有 `.exp.json` 模板，需手动复制为运行时配置：

```bash
cp config/agent_config.exp.json   config/agent_config.json
cp config/llm_config.exp.json     config/llm_config.json
cp config/mcp_servers.exp.json    config/mcp_servers.json
cp config/safety.exp.json         config/safety.json
```

### 4. 配置 API 密钥

编辑 `config/llm_config.json`，填入你要使用的提供商密钥。推荐通过环境变量配置，也可直接写在文件中：

```json
{
    "providers": {
        "zhipu": {
            "name": "智谱AI",
            "base_url": "https://open.bigmodel.cn/api/paas/v4/",
            "env_key": "ZHIPU_API_KEY",
            "model": "glm-4-flash",
            "models": ["glm-4", "glm-4-flash", "glm-4-long"],
            "api_key": ""
        },
        "deepseek": {
            "name": "DeepSeek",
            "base_url": "https://api.deepseek.com",
            "env_key": "DEEPSEEK_API_KEY",
            "model": "deepseek-chat",
            "models": ["deepseek-chat", "deepseek-reasoner"],
            "api_key": ""
        }
    },
    "tavily": {
        "api_key": ""
    }
}
```

> **建议使用环境变量**：`set DEEPSEEK_API_KEY=sk-xxx`（Windows）或 `export DEEPSEEK_API_KEY=sk-xxx`（Linux）。`api_key` 留空即可，程序会自动读取对应环境变量。  
> 增删提供商只需编辑 `config/llm_config.json` 的 `providers` 字段。每次新增/移除提供商后重启程序生效。

### 5. 运行

```bash
python main.py
```

启动后用方向键选择提供商即可进入交互模式。

---

## 项目结构

```
LangChainAgent/
├── main.py                  # 入口文件，交互式命令行
├── llm_client.py            # 统一大模型封装（多提供商 + 多模型）
├── config/                  # 配置目录（需从 *.exp.json 复制；见快速开始）
│   ├── llm_config.json      # API密钥配置文件
│   ├── agent_config.json    # 运行时配置(迭代上限/技能目录/长上下文裁剪等)
│   ├── mcp_servers.json     # MCP Server 配置(已预置 workspace 本地服务器,enabled)
│   └── safety.json          # 安全护栏配置（运行时按需生成）
├── requirements.txt         # 运行依赖(含开发/测试依赖 pytest)
├── pytest.ini               # pytest 配置
├── .agents/skills/          # 本地技能目录（每个子目录一个 SKILL.md）
├── data/
│   ├── checkpoints.sqlite   # Checkpoint 持久化数据库（运行时自动生成）
│   └── memory.json          # 长期记忆文件（用于 compress 摘要）
├── agent/
│   ├── __init__.py
│   ├── memory.py            # 记忆模块（Checkpoint + 长期记忆）
│   └── agent_core.py        # Agent核心调度
├── tools/
│   ├── __init__.py          # 本地工具注册
│   ├── search.py            # 联网搜索工具(Tavily API)
│   ├── file_tool.py         # 文件读写工具
│   ├── calculator.py        # 数学计算工具
│   ├── terminal_tools.py    # 终端命令工具（shell/python/bat/ps1,含安全护栏）
│   ├── get_local_time.py    # 获取本地时间工具
│   ├── open_file.py         # 文件打开工具（系统默认程序/DB Browser）
│   ├── skills.py            # SkillManager（扫描/匹配/渲染技能）
│   ├── skill_tool.py        # read_skill 工具（LLM 自助读取技能指引）
│   ├── safety.py            # 安全护栏(黑名单/白名单/交互确认/路径保护)
│   ├── mcp_loader.py        # MCP 工具加载器
│   └── workspace_tool.py    # 工作目录管理 MCP Server
├── utils/
│   ├── __init__.py
│   ├── cli_menu.py          # 通用终端方向键选择菜单
│   ├── human_input.py       # ask_human 工具 + HITL 暂停/恢复编排
│   └── commands/            # CLI 命令分发与领域处理器
│       ├── types.py         # CommandContext / CommandOutcome
│       ├── dispatcher.py    # 按既有优先级路由命令
│       ├── core.py          # help / info / tools / 就绪信息
│       ├── threads.py       # 会话切换、创建、删除、导出
│       ├── memory.py        # clear / compress
│       ├── provider.py      # 提供商与模型切换
│       ├── mcp.py           # MCP 管理命令
│       ├── skills.py        # Skill 管理命令
│       ├── safety.py        # 安全策略命令
│       └── execution.py     # json / react / cot / 普通对话
├── tests/                   # 离线单元测试(pytest)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_safety.py
│   ├── test_skills.py
│   ├── test_search.py
│   ├── test_terminal.py
│   └── test_memory.py
├── config.py                # 运行时配置加载(config/agent_config.json)
└── exports/                 # 对话导出目录(运行 export 时生成)
```

### 模块职责

| 模块                                                        | 职责                                                                                                                        |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| [main.py](main.py)                                           | 交互式命令行入口：初始化运行时、读取输入并调用命令分发器                                                                    |
| [llm_client.py](llm_client.py)                               | 从`config/llm_config.json` 读取提供商配置，支持运行时切换提供商/模型                                                      |
| [config.py](config.py)                                       | 加载`config/agent_config.json`，统一运行时配置                                                                            |
| [agent/memory.py](agent/memory.py)                           | 双层记忆：Checkpoint（自动）+ memory.json（手动）+ 对话导出                                                                 |
| [agent/agent_core.py](agent/agent_core.py)                   | Agent 核心：`run()` / `chat()` / `cot()` 三种模式 + `AgentTurnResult` 结构化暂停/恢复 API + 技能注入 + 长上下文裁剪 |
| [tools/skills.py](tools/skills.py)                           | `SkillManager`：扫描/匹配/渲染本地技能                                                                                    |
| [tools/skill_tool.py](tools/skill_tool.py)                   | `read_skill` 工具：LLM 在任务中自助读取技能指引                                                                           |
| [tools/](tools/)                                             | 本地工具 + MCP 工具加载 + 技能管理                                                                                          |
| [utils/cli_menu.py](utils/cli_menu.py)                       | 通用终端方向键选择菜单                                                                                                      |
| [utils/human_input.py](utils/human_input.py)                 | `ask_human` 工具、interrupt 展示和循环恢复编排                                                                            |
| [utils/commands/dispatcher.py](utils/commands/dispatcher.py) | 按兼容顺序匹配命令并路由到领域处理器                                                                                        |
| [utils/commands/types.py](utils/commands/types.py)           | 命令依赖上下文、活动 LLM 状态和分发结果类型                                                                                 |
| [utils/commands/](utils/commands/)                           | 会话、记忆、模型、MCP、技能、安全及 Agent 执行命令                                                                          |

### llm_client.py 主要 API

| 类/函数                           | 说明                                     |
| --------------------------------- | ---------------------------------------- |
| `LLMClient`                     | 核心客户端类，支持配置文件中的所有提供商 |
| `LLMClient.chat()`              | 发送对话请求                             |
| `LLMClient.chat_with_history()` | 带历史记录的对话                         |
| `LLMClient.get_chat_model()`    | 获取 LangChain`ChatOpenAI` 实例        |
| `LLMClient.switch_provider()`   | 运行时切换提供商                         |
| `LLMClient.switch_model(model)` | 运行时切换当前提供商的模型               |
| `LLMClient.list_models()`       | 列出当前提供商的可用模型                 |
| `create_client()`               | 创建客户端的便捷函数                     |
| `list_providers()`              | 列出所有支持的提供商                     |

**支持的提供商：**

提供商不再硬编码在代码中，全部来自 [config/llm_config.json](config/llm_config.json) 的 `providers` 字段。当前配置包含：

| 提供商       | 名称            | 环境变量              | 默认模型                |
| ------------ | --------------- | --------------------- | ----------------------- |
| `zhipu`    | 智谱AI          | `ZHIPU_API_KEY`     | `glm-4.7-flash`       |
| `qwen`     | 通义千问        | `DASHSCOPE_API_KEY` | `Qwen2.5-7B-Instruct` |
| `deepseek` | DeepSeek        | `DEEPSEEK_API_KEY`  | `deepseek-chat`       |
| `kimi`     | Kimi (Moonshot) | `MOONSHOT_API_KEY`  | `moonshot-v1-8k`      |
| `yunwu`    | 云雾            | `YUNWU_API_KEY`     | `gpt-5.5`             |

---

## 核心概念：三种模式

Agent 有三种执行模式，对应三种不同的交互入口：

| 模式            | 命令           | 调用方法   | 工具可用 | 步骤打印    | 适用场景                 |
| --------------- | -------------- | ---------- | -------- | ----------- | ------------------------ |
| **ReAct** | `react:任务` | `run()`  | ✅ 有    | ✅ 打印每步 | 复杂任务、需观察推理过程 |
| **CoT**   | `cot:任务`   | `cot()`  | ❌ 无    | ❌ 静默     | 纯推理、分析类问题       |
| **Chat**  | 其他输入       | `chat()` | ✅ 有    | ❌ 静默     | 日常对话、轻量操作       |

> **重要**：`chat()` 模式也走 Agent 执行，LLM 会自动判断是否调用工具。无需强制加 `react:` 前缀也能创建文件、目录等。`react:` 与 `chat()` 的区别仅在于**是否打印步骤 + 是否存长期记忆**。

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

### 上下文注入机制（重要）

一个常见疑问：**Agent 在对话时，历史消息从哪里来？** 答案取决于模式。

#### `react:` 和 `chat()` → 只用 checkpoint

```python
# run() / chat()
result = self.agent_executor.invoke(
    {"messages": [HumanMessage(content=task)]},   # ← 只传当前这一条
    config=self.memory.get_config()               # ← 传 thread_id
)
```

- 调用时**只传当前这条新消息**
- 历史消息由 **LangGraph 从 checkpoint 自动恢复**（按 thread_id 取出该会话所有历史消息，拼到新消息前面）
- **memory.json 完全不参与** LLM 上下文

```
invoke(新消息, thread_id)
   ↓
LangGraph: 从 checkpoint 读该 thread 的全部历史
   ↓
[历史消息1, 历史消息2, ..., 新消息] → 传给 LLM
   ↓
LLM 回复 → 写回 checkpoint
```

#### `cot:` → checkpoint + memory.json

```python
# cot()
context = self.memory.get_short_term() + self.memory.get_long_term(3)
response = self.llm.chat_with_history(
    user_input=task,
    history=context,   # ← 短期(checkpoint) + 长期(memory.json)
    ...
)
```

- `get_short_term()` → 从 **checkpoint** 取当前 thread 的消息（转 dict）
- `get_long_term(3)` → 从 **memory.json** 取最近 3 条长期记忆
- 两者拼接后作为 history 传给 LLM

#### 技能指引与长上下文摘要的注入

`react:` 和 `chat()` 在每次执行前会根据任务重建 Agent：

```python
self.agent_executor = self._create_agent_executor(
    self._compute_skill_block(task)
)
```

因此除了 checkpoint 历史外，system prompt 还可能包含：

| 来源         | 触发方式                                  | 说明                                              |
| ------------ | ----------------------------------------- | ------------------------------------------------- |
| 手动技能     | `skill:<name>`                          | 后续对话都会注入该技能指引，直到`skill:clear`   |
| 自动匹配技能 | `auto_match_skills=true`                | 根据任务与技能描述的关键词重叠度自动注入相关技能  |
| 长上下文摘要 | `max_context_messages > 0` 且消息超阈值 | 旧消息被摘要后注入 system prompt，并开启新 thread |

#### 对比表

| 模式       | checkpoint  | memory.json    | 技能指引  | 长上下文摘要 | 说明                             |
| ---------- | ----------- | -------------- | --------- | ------------ | -------------------------------- |
| `react:` | ✅ 自动注入 | ❌ 不参与      | ✅ 可注入 | ✅ 可注入    | LangGraph 自动恢复该 thread 历史 |
| 普通对话   | ✅ 自动注入 | ❌ 不参与      | ✅ 可注入 | ✅ 可注入    | 同上，但不打印步骤、不存长期     |
| `cot:`   | ✅ 手动取   | ✅ 取最近 3 条 | ❌ 不注入 | ❌ 不注入    | 绕过 Agent，纯 LLM 推理          |

> ⚠️ **重要副作用**：由于 `react:` 和普通对话**不读 memory.json**，意味着你在会话 A 里用 `react:` 做的事（已存入 memory.json），切到会话 B 再用 `react:` 时，**LLM 看不到**会话 A 的关键信息。只有 `cot:` 模式或 `compress` 命令才会读 memory.json。也就是说 **memory.json 的跨会话"记忆"能力目前只对 cot 生效**。

### 两层存储对比

| 类型                 | 存储方式                             | 触发时机                   | 保存内容                           | 持久化  | 用途                              |
| -------------------- | ------------------------------------ | -------------------------- | ---------------------------------- | ------- | --------------------------------- |
| **Checkpoint** | `data/checkpoints.sqlite` (SQLite) | Agent 每步执行后自动       | 完整状态(消息+工具调用链+中间变量) | ✅ 永久 | 程序重启恢复对话、多会话隔离      |
| **长期记忆**   | `data/memory.json` (JSON)          | 手动标记`important=True` | 仅 react/cot 的最终结果            | ✅ 永久 | 跨会话保留关键决策、用于 compress |

### 三种模式的记忆行为

| 模式           | 调用方法   | 写入 Checkpoint | 写入 memory.json | 说明                          |
| -------------- | ---------- | --------------- | ---------------- | ----------------------------- |
| `react:任务` | `run()`  | ✅ 自动         | ✅ 手动触发      | ReAct 结果重要，永久保存      |
| `cot:任务`   | `cot()`  | ❌ 不写         | ✅ 手动触发      | CoT 不走 Agent，仅存结论      |
| 普通输入       | `chat()` | ✅ 自动         | ❌ 不写          | 闲聊仅存 checkpoint，用完即丢 |

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

| 能力         | 说明                                                                           |
| ------------ | ------------------------------------------------------------------------------ |
| 自动持久化   | Agent 每步执行后自动写入 SQLite，无需手动                                      |
| 程序重启恢复 | 重启后用相同`thread_id` 即可恢复完整对话历史                                 |
| 多会话隔离   | 不同`thread_id` 完全独立，可管理多个对话                                     |
| 完整状态保存 | 保存消息、工具调用链、中间变量(不止文本)                                       |
| 图暂停/恢复  | `ask_human` 触发 interrupt 后，用同一 `thread_id` 的 checkpoint 继续当前图 |

### 长期记忆触发原理

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

| 文件                        | 格式      | 内容                                |
| --------------------------- | --------- | ----------------------------------- |
| `data/checkpoints.sqlite` | SQLite    | Agent 执行状态(按 thread_id 隔离)   |
| `data/memory.json`        | JSON 数组 | react/cot 的最终结果(用于 compress) |

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
>
> **查看 checkpoints.sqlite**：可用 [DB Browser for SQLite](https://sqlitebrowser.org/dl/) 打开，或让 Agent 调用 `open_sqlite` 工具自动打开。

### 会话(Thread)管理

基于 checkpoint 的 `thread_id` 机制，支持多会话管理：

| 命令                   | 作用                               | 示例                            |
| ---------------------- | ---------------------------------- | ------------------------------- |
| `thread`             | 方向键选择切换会话(显示消息数预览) | `thread`                      |
| `thread:new`         | 开启新会话(原会话保留)             | `thread:new`                  |
| `thread:delete <id>` | 删除指定会话(二次确认,不可恢复)    | `thread:delete thread-abc123` |

> 切换会话统一走 `thread` 方向键菜单，无需手动输入 thread_id。
>
> 在 `thread` 菜单中，使用方向键移动高亮项，按 `Enter` 切换会话，按 `Esc` 取消；按 `Ctrl+D` 可删除当前高亮会话。删除前会要求输入 `y` 或 `yes` 确认。

**示例**：

```
你: thread
 选择会话 (共 2 个,↑↓ 选择,Enter 切换)
  (↑↓ 选择, Enter 切换, Ctrl+D 删除, Esc 取消)
    thread-a4d099d2  [6 条消息]
  ❯ thread-b1dc3b2a  [0 条消息] (当前)
----------------------------------------
(按 Enter 切换)

你: thread:new
已开启新会话: thread-c7e8f1a3
原会话 thread-a4d099d2 已保留,可用 'thread' 切回

你: thread:delete thread-a4d099d2
确认删除会话 'thread-a4d099d2'? 此操作不可恢复 [y/N]: y
已删除会话: thread-a4d099d2
```

在会话菜单中删除高亮会话的操作示例：

```
你: thread
  (↑↓ 选择, Enter 切换, Ctrl+D 删除, Esc 取消)
  ❯ thread-b1dc3b2a  [0 条消息] (当前)

确认删除会话 'thread-b1dc3b2a'? 此操作不可恢复 [y/N]: y
已删除会话: thread-b1dc3b2a
```

### 记忆管理命令

| 命令                        | 作用                                 |
| --------------------------- | ------------------------------------ |
| `clear` 或 `clear long` | 清空长期记忆 + 删除`memory.json`   |
| `clear short`             | 清空当前会话(开启新 thread 替代删除) |
| `clear all`               | 全部清空                             |
| `compress` 或 `压缩`    | 压缩长期记忆(LLM 摘要后替换原内容)   |
| `thread`                  | 方向键选择切换会话                   |
| `thread:new`              | 开启新会话                           |
| `thread:delete <id>`      | 删除指定会话(二次确认)               |

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
- 已通过 config/llm_config.json 配置多个 LLM 提供商
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

| 特性              | 说明                                                    |
| ----------------- | ------------------------------------------------------- |
| 保留关键信息      | system prompt 要求 LLM 保留用户意图、决策、事实         |
| 结构化输出        | 按主题分条，便于后续查阅                                |
| 可追溯            | `metadata` 保存原始条数和字符数                       |
| 可多次压缩        | 再次`compress` 会对已有摘要再压缩                     |
| LLM 失败不丢数据  | 调用失败则原记忆不变，不会写回                          |
| **不可逆**  | 压缩后原条目无法恢复，建议重要数据先备份`memory.json` |
| 不影响 checkpoint | 只压缩 memory.json，checkpoints.sqlite 保留完整历史     |

> 💡 **建议**：在长期记忆较多时（如 50+ 条）使用，平时少量记忆无需压缩。

### 记忆相关 API

| 方法                                 | 说明                                                      |
| ------------------------------------ | --------------------------------------------------------- |
| `get_checkpointer()`               | 获取 checkpointer 实例(传给 create_react_agent)           |
| `get_config()`                     | 返回`{"configurable": {"thread_id": ...}}`(传给 invoke) |
| `get_messages()`                   | 从 checkpoint 获取当前 thread 的所有消息                  |
| `get_langchain_messages()`         | 等同于 get_messages()（兼容旧 API）                       |
| `get_short_term(limit)`            | 从 checkpoint 取消息转为 dict 格式                        |
| `get_long_term(limit)`             | 获取最近 N 条长期记忆                                     |
| `get_all_context(long_term_limit)` | 获取完整上下文(长期+短期)                                 |
| `add(role, content, metadata)`     | 添加记忆，`important=True` 触发写 memory.json           |
| `new_thread()`                     | 开启新会话                                                |
| `switch_thread(thread_id)`         | 切换到指定会话(代码层 API,CLI 用`thread` 菜单)          |
| `delete_thread(thread_id)`         | 删除指定会话(删当前会话时自动切换到其他会话)              |
| `list_threads()`                   | 列出所有会话 ID                                           |
| `export_thread(thread_id, fmt)`    | 导出指定会话为可读文本/Markdown                           |
| `clear_short_term()`               | 开启新会话(替代删除)                                      |
| `clear_long_term()`                | 清空长期记忆并删除文件                                    |
| `summarize()`                      | 返回记忆统计信息(含 thread_id、消息数、会话数)            |
| `compress_memory()`                | 压缩长期记忆（LLM 摘要后替换原内容）                      |

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

| 工具               | 文件                                              | 功能                                      | 参数                                                 |
| ------------------ | ------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------- |
| `search`         | [tools/search.py](tools/search.py)                 | 联网搜索(Tavily API)                      | `query`, `num_results`, `search_depth`         |
| `read_file`      | [tools/file_tool.py](tools/file_tool.py)           | 读取文件                                  | `file_path`                                        |
| `write_file`     | [tools/file_tool.py](tools/file_tool.py)           | 写入文件                                  | `file_path`, `content`, `mode`                 |
| `calculate`      | [tools/calculator.py](tools/calculator.py)         | 数学计算                                  | `expression`                                       |
| `run_shell`      | [tools/terminal_tools.py](tools/terminal_tools.py) | 执行 shell 命令                           | `command`, `cwd`, `timeout`                    |
| `run_python`     | [tools/terminal_tools.py](tools/terminal_tools.py) | 执行 Python 脚本文件                      | `file_path`, `script_args`, `cwd`, `timeout` |
| `run_cmd`        | [tools/terminal_tools.py](tools/terminal_tools.py) | 执行 .bat/.cmd/.ps1 文件                  | `file_path`, `script_args`, `cwd`, `timeout` |
| `get_local_time` | [tools/get_local_time.py](tools/get_local_time.py) | 获取本地时间                              | 无                                                   |
| `open_file`      | [tools/open_file.py](tools/open_file.py)           | 用系统默认/指定程序打开文件或文件夹       | `file_path`, `app_path`                          |
| `open_sqlite`    | [tools/open_file.py](tools/open_file.py)           | 用 DB Browser for SQLite 打开 .sqlite/.db | `file_path`                                        |
| `read_skill`     | [tools/skill_tool.py](tools/skill_tool.py)         | 读取本地技能(SKILL.md)的指引正文          | `skill_name`(可空)                                 |
| `ask_human`      | [utils/human_input.py](utils/human_input.py)       | 暂停 LangGraph 图并请求人工结构化选择     | `prompt`, `choices`                              |

> `open_sqlite` 会自动查找 DB Browser for SQLite 路径（环境变量 `SQLITE_BROWSER_PATH` → 常见安装位置 → PATH），找不到则返回下载链接。

### 2. MCP 工具（MCP Server Tools）

通过 **MCP（Model Context Protocol）** 协议从独立进程加载，支持动态增删。

#### 已注册的 MCP Server

| Server 名称    | 工具数 | 传输方式 | 说明                                                |
| -------------- | ------ | -------- | --------------------------------------------------- |
| `workspace`  | 6      | stdio    | 文件夹创建/删除/移动/复制/列举（Python 实现）       |
| `filesystem` | -      | stdio    | 可选官方文件系统服务（需手动添加/启用，需 Node.js） |
| `fetch`      | -      | stdio    | 可选官方网页抓取服务（需手动添加/启用，需 Node.js） |

#### workspace MCP Server 提供的工具

由 [tools/workspace_tool.py](tools/workspace_tool.py) 实现，基于 `FastMCP`：

| 工具                      | 功能              | 参数                            |
| ------------------------- | ----------------- | ------------------------------- |
| `create_workspace`      | 创建文件夹        | `folder_name`, `parent_dir` |
| `get_current_workspace` | 获取当前工作目录  | 无                              |
| `list_directory`        | 列出目录内容      | `path`                        |
| `delete_workspace`      | 删除文件夹        | `folder_path`, `recursive`  |
| `move_workspace`        | 移动/重命名文件夹 | `src_path`, `dest_path`     |
| `copy_workspace`        | 复制文件夹        | `src_path`, `dest_path`     |

#### MCP 配置文件

[config/mcp_servers.json](config/mcp_servers.json) 定义所有 MCP Server：

```json
{
    "servers": {
        "workspace": {
            "transport": "stdio",
            "command": "python",
            "args": ["tools/workspace_tool.py"],
            "enabled": true
        }
    }
}
```

> 说明：配置文件中的 `command: "python"`（或 `"python3"`）会在加载时自动替换为**当前运行的解释器**（即激活的 venv），因此无需写死绝对路径。stdio 的 `args` 路径相对于项目根目录。

| 字段          | 说明                                                |
| ------------- | --------------------------------------------------- |
| `transport` | 传输方式：`stdio` / `sse` / `streamable_http` |
| `command`   | stdio 模式下的启动命令                              |
| `args`      | 命令参数列表                                        |
| `enabled`   | 是否启用（`false` 则不加载）                      |
| `url`       | sse/http 模式下的服务器地址                         |

#### MCP 管理命令

在交互界面中动态管理 MCP Server：

| 命令           | 作用                  | 示例                                                           |
| -------------- | --------------------- | -------------------------------------------------------------- |
| `mcp`        | 查看 MCP Server 状态  | `mcp`                                                        |
| `mcp:reload` | 重新加载所有 MCP 工具 | `mcp:reload`                                                 |
| `mcp:add`    | 添加 stdio MCP Server | `mcp:add myserver npx -y @modelcontextprotocol/server-fetch` |
| `mcp:add`    | 添加 sse MCP Server   | `mcp:add myserver sse:http://localhost:8000/sse`             |
| `mcp:add`    | 添加 http MCP Server  | `mcp:add myserver http:http://localhost:8000/mcp`            |
| `mcp:remove` | 删除 MCP Server       | `mcp:remove fetch`                                           |
| `mcp:toggle` | 启用/禁用             | `mcp:toggle fetch on` / `mcp:toggle fetch off`             |

### 3. 技能阅读（Skills）

项目内置 `.agents/skills/` 目录，存放标准格式的技能（`SKILL.md`，含 `name`/`description` frontmatter + 正文指引），例如 `find-skills`、`git-commit`、`pptx`。

Agent 可以在任务中**阅读并使用这些技能指引**，支持三种方式：

| 方式                     | 说明                                                                                                                     |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| **工具自动调用**   | `read_skill` 工具：LLM 在任务中自行决定何时读取某技能的完整指引（不传名称则列出全部可用技能）                          |
| **命令行手动加载** | `skill:<name>` 把某技能注入当前会话的 system prompt；`skill` 列出全部；`skill:clear` 清空                          |
| **任务自动匹配**   | 每次执行任务时，根据任务描述与技能`description` 做关键词匹配（确定性打分，不调 LLM），自动注入相关技能指引（默认开启） |

#### 已注册的技能（本地）

由 [tools/skills.py](tools/skills.py) 的 `SkillManager` 扫描 `.agents/skills/` 自动发现：

| 技能            | 说明                                   |
| --------------- | -------------------------------------- |
| `find-skills` | 发现并安装 agent skills                |
| `git-commit`  | 从 git diff 生成双语提交信息并执行提交 |
| `pptx`        | 创建/读取/编辑`.pptx` 演示文稿       |

> 中文任务通过内置中→英关键词扩展（如 提交→commit/git、演示→ppt/slides）实现与英文描述的匹配。

#### 交互命令

| 命令                    | 作用                                                     |
| ----------------------- | -------------------------------------------------------- |
| `skill` 或 `skills` | 列出所有本地可用技能（名称 + 描述）                      |
| `skill:<name>`        | 将指定技能加载进当前会话（注入后续对话的 system prompt） |
| `skill:clear`         | 清空手动加载的技能                                       |

#### `read_skill` 工具

LLM 在任务中可调用：

- `read_skill()`（不传名称）→ 返回所有可用技能列表，供其判断该用哪个
- `read_skill("git-commit")` → 返回该技能的完整 `SKILL.md` 指引正文

#### 自动匹配原理

```
用户输入任务
    ↓
SkillManager.match_skills(task)
  • 中文关键词扩展为英文(提交→commit/git ...)
  • 与每个技能的 name+description 做重叠度(Jaccard)打分
  • 取分数>0 的前 N 个技能
    ↓
将命中技能正文拼接为「技能指引」块
    ↓
注入本次任务的 system prompt(手动加载的技能也会合并进去)
```

> 关闭自动匹配：`agent.set_auto_match(False)`；手动加载的技能不受此开关影响。

#### 技能相关 API

| 方法                                 | 说明                                                      |
| ------------------------------------ | --------------------------------------------------------- |
| `agent.list_skills()`              | 列出所有本地技能（名称 + 描述 + 路径）                    |
| `agent.load_skill(name)`           | 手动加载技能到当前会话（注入 system prompt + 重建 Agent） |
| `agent.clear_skills()`             | 清空手动加载的技能                                        |
| `agent.set_auto_match(enabled)`    | 开关自动匹配（默认开）                                    |
| `SkillManager.match_skills(task)`  | 根据任务匹配相关技能（确定性打分）                        |
| `SkillManager.render_block(names)` | 把若干技能渲染为可注入的指引块                            |

### 4. 安全护栏（Safety）

Agent 可自动执行终端命令与文件操作，为防止破坏性操作，内置两级安全策略（由 [tools/safety.py](tools/safety.py) 实现）：

| 级别                            | 规则                                                                                                                                                     | 行为                                                 |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **BLOCKLIST（始终拒绝）** | `rm -rf /`、`format`、`mkfs`、`dd if=`、`shutdown`、`fork bomb`、`:(){`、``curl\|sh``/``wget\|sh``、`del /f /s`、`rd /s /q` 等灾难性命令 | 直接拦截，返回拒绝错误                               |
| **CONFIRM（需确认）**     | `sudo`、`rm`、`chmod`、`chown`、`mv`、`kill`、`taskkill`、`schtasks` 等危险但有时需要的命令                                              | 交互式确认（输入`y` 才执行，空/超时/EOF 默认拒绝） |

> 执行任意脚本文件（`.py` / `.ps1` / `.bat`）因无法静态分析内容，默认也需确认。

#### 配置（config/safety.json）

```json
{
    "mode": "blacklist",
    "confirm_dangerous": true,
    "blacklist": [],
    "whitelist": ["echo", "dir", "ls", "python", "pip", "git", "cat", "type"]
}
```

字段含义：

- `mode`: `blacklist`(默认) 或 `whitelist`
- `confirm_dangerous`: 是否对危险命令交互确认
- `blacklist`: 追加的拒绝正则(与内置合并)
- `whitelist`: 白名单模式下允许的首命令
- `blacklist` 模式：仅拦截 BLOCKLIST + 追加黑名单，其余放行（危险命令需确认）
- `whitelist` 模式：只有白名单内的首命令才允许，其余一律拒绝

#### 路径保护

`delete_workspace`（MCP 子进程）无法交互确认，改为**路径保护**：禁止删除系统目录（`C:\Windows` 等）、用户主目录、项目根目录及根目录本身。

#### 交互命令

| 命令                                  | 作用                                         |
| ------------------------------------- | -------------------------------------------- |
| `safety`                            | 查看当前安全策略（模式 / 确认开关 / 黑名单） |
| `safety:mode <blacklist\|whitelist>` | 切换模式                                     |
| `safety:confirm <on\|off>`           | 开关危险命令确认                             |

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

若调用的是 `ask_human`，工具不会立即返回普通观察结果，而是通过 LangGraph interrupt 暂停图执行。CLI 收集选择后调用 `resume_structured()`，用 `Command(resume=...)` 把选择送回同一图状态，再继续后续工具调用或生成最终回复。

### System Prompt 强化

为确保 LLM 主动调用工具而非拒绝，system prompt 中明确要求：

```
1. 当用户要求创建文件、读写文件、创建目录等操作时，你【必须】调用相应工具
2. 绝对不要回复'我无法访问你的文件系统'、'请你自己保存'之类的话
3. 你确实拥有这些工具的能力，工具会在用户本地执行
4. 创建文件、脚本、文件夹默认位置是 ./tests/
5. 如果用户要保存内容到文件，直接调用 write_file 工具
6. 如果用户要创建目录，直接调用 create_workspace 工具
7. 测试/运行脚本时直接调用终端工具
8. 危险命令会被安全策略拦截或要求确认
9. 专业任务应优先用 read_skill 读取相关技能指引
10. 需要人工确认、选择或补充信息时，应调用 ask_human 并提供结构化 choices
```

---

## Human-in-the-loop（HITL）

当 Agent 在执行过程中需要**人工确认、二选一、补充信息**才能继续时，会调用本地工具 `ask_human` 触发 LangGraph **interrupt**，让图在工具节点暂停。CLI 收集到结构化选择后，用 `Command(resume=...)` 把答案送回**同一个 checkpoint 线程**继续执行。

> 这不是普通多轮聊天。普通多轮聊天是「用户发一条新消息 → Agent 重新执行一轮」；HITL 是「**同一次 LangGraph 执行**在工具节点暂停 → CLI 收集选择 → 用 `Command(resume)` 恢复同一个图状态」。因此 HITL 不会丢失中间工具调用链，恢复后能直接接着执行后续步骤。

### 执行流程

```
用户输入任务（react: / 普通对话）
    ↓
AgentCore.run_structured() / chat_structured()
    ↓
LangGraph invoke() 开始执行
    ↓
LLM 决定调用 ask_human 工具
    ↓
ask_human 内部调用 langgraph.types.interrupt(payload) → 图暂停
    ↓
invoke() 返回 __interrupt__ → AgentTurnResult.status = "interrupted"
    ↓
CLI 渲染 interrupt（render_human_interrupt）
    ↓
select_menu 展示结构化选项（↑↓ 选择，Enter 确认，Esc 取消）
    ↓
收集 resume payload（{"choice_id": "..."} 或 {"cancelled": True}）
    ↓
AgentCore.resume_structured(payload) → Command(resume=payload)
    ↓
同一 thread_id 的 checkpoint 继续执行
    ↓
（若再次遇到 ask_human → 回到「CLI 渲染 interrupt」循环）
    ↓
AgentTurnResult.status = "completed" → 输出最终答案
```

### ask_human 工具

定义在 [utils/human_input.py](utils/human_input.py)，是一个 `StructuredTool`，LLM 把它当作普通工具调用：

| 参数       | 类型             | 说明                         |
| ---------- | ---------------- | ---------------------------- |
| `prompt` | `str`          | 展示给用户的提问文本         |
| `choices`| `list[Choice]` | 结构化选项列表（id + label） |

工具内部不返回普通观察值，而是调用 `langgraph.types.interrupt()` 暂停图执行。**interrupt 的 payload 固定结构**：

```python
{
    "kind": "human_choice",
    "prompt": "请选择要执行的操作",
    "choices": [
        {"id": "approve", "label": "确认执行"},
        {"id": "cancel", "label": "取消"}
    ]
}
```

`ask_human` 的返回值就是 resume 时传入的 payload（如 `{"choice_id": "approve"}`），LLM 在恢复后会读到这个返回值，从而知道用户的选择。

### 结构化 API

`AgentCore` 暴露结构化方法，让调用方能区分「完成」「暂停」「取消」三种状态：

| 名称                           | 作用                                                                                                                  |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `AgentTurnResult`            | 一轮图执行结果，`status` 为 `completed` / `interrupted` / `cancelled`；完成时读 `output`，暂停时读 `interrupts` |
| `run_structured(task)`       | ReAct/任务模式入口，返回 `AgentTurnResult`，完成后写 checkpoint + 长期记忆                                           |
| `chat_structured(message)`   | 普通对话入口，返回 `AgentTurnResult`，完成后写 checkpoint，不写长期记忆                                              |
| `resume_structured(payload)` | 对暂停中的图调用 `Command(resume=payload)`，继续同一个 checkpoint 线程                                               |

`AgentTurnResult` 是 frozen dataclass，三种状态的语义：

| 状态            | 触发条件                             | 后续动作                          |
| --------------- | ------------------------------------ | --------------------------------- |
| `completed`   | LLM 生成最终回复，无 interrupt       | 读 `output`，按模式写记忆         |
| `interrupted` | `ask_human` 触发 interrupt           | 读 `interrupts`，收集答案后 resume |
| `cancelled`   | 用户在安全护栏确认中拒绝了危险命令   | 读 `output`（取消提示），不 resume |

### CLI 编排

[utils/human_input.py](utils/human_input.py) 提供了一组辅助函数，把「invoke → 渲染 → 收集 → resume」封装成循环：

| 函数                                                  | 作用                                                  |
| ----------------------------------------------------- | ----------------------------------------------------- |
| `run_human_input_loop(agent, message, render, read)` | 启动对话并持续恢复 interrupt，直到本轮完成          |
| `run_structured_until_completion(agent, message)`    | ReAct 模式版本，同上                                 |
| `chat_until_completion(agent, message)`              | 普通对话模式的默认入口                               |
| `complete_human_input_turn(turn, render, read)`     | 循环处理 interrupt 直到 `status == "completed"`      |
| `render_human_interrupt(interrupt)`                  | 把 interrupt payload 渲染成终端提示                  |
| `read_human_resume(interrupt)`                       | 用 `select_menu` 收集结构化选择，返回 resume payload |

#### 结构化选择的展示与收集

`read_human_resume` 的逻辑：

1. 校验 interrupt 是否为 `human_choice`，否则回退到自由文本输入 `{"text": "..."}`
2. 从 `choices` 提取 `(label, id)` 对，调用 [utils/cli_menu.py](utils/cli_menu.py) 的 `select_menu(prompt, options)` 展示方向键菜单
3. 用户 `Enter` 确认 → 返回 `{"choice_id": "<id>"}`
4. 用户按 `Esc` → 返回 `{"cancelled": True}`（Agent 据此知道用户取消了）

菜单展示效果：

```
是否删除旧的临时文件?
  (↑↓ 选择, Enter 确认, Esc 取消)
  ❯ 删除
    跳过
----------------------------------------
是否删除旧的临时文件? ❯ 删除
```

### 并行 interrupt 与多轮暂停

LangGraph 支持在同一轮执行中产生**多个并行 interrupt**（例如 LLM 一次调用了两个 `ask_human`）。CLI 必须一次性收集所有答案，并按 `interrupt.id` 映射后用**一次** `Command(resume=...)` 提交：

```python
# 单 interrupt：直接传值
resume = {"choice_id": "approve"}

# 并行 interrupt：必须按 interrupt.id 映射
resume = {
    interrupt.id: {"choice_id": "approve"}
    for interrupt in turn.interrupts
}
```

`_resume_payload_for_interrupts` 会自动区分这两种情况：单 interrupt 直接传值，多 interrupt 按 `interrupt.id` 映射，避免答案错配。

**多轮暂停**也支持：恢复后若又遇到新的 `ask_human`，`complete_human_input_turn` 会继续循环处理，直到 `status == "completed"`。例如一个图节点里连续两次 `ask_human`，需要两次 resume 才能完成。

### 线程隔离

`resume_structured` 会校验当前 `thread_id` 与产生 interrupt 的线程一致，跨线程恢复会抛 `ValueError`：

```python
def resume_structured(self, payload):
    config = self._invoke_config()
    pending_thread_id = getattr(self, "_pending_interrupt_thread_id", None)
    current_thread_id = self._thread_id_from_config(config)
    if pending_thread_id is not None and current_thread_id != pending_thread_id:
        raise ValueError("Cannot resume interrupt on a different thread")
    ...
```

这是为了防止「在会话 A 触发 interrupt 后切到会话 B，再尝试 resume」导致的状态错乱。原会话的暂停状态仍保留在 checkpoint 中，切回原会话后可以继续 resume。

### 与普通多轮聊天、安全护栏的区别

| 机制              | 触发主体                  | 暂停层级              | 恢复方式                       | 中间状态保留           |
| ----------------- | ------------------------- | --------------------- | ------------------------------ | ---------------------- |
| **普通多轮聊天** | 用户主动发问              | 无暂停，每轮独立      | 用户发送下一条消息             | 仅靠 checkpoint 历史   |
| **HITL**        | LLM 调用 `ask_human`    | 图执行暂停（interrupt）| `Command(resume=payload)`     | 保留完整工具调用链     |
| **安全护栏确认** | 工具执行前拦截            | 工具内部抛异常        | 抛异常，整轮取消，不 resume    | 工具调用被标记为 error |

> 安全护栏的确认（如 `rm` 命令需输入 `y`）与 HITL 是**两套独立机制**：安全护栏在工具执行**前**拦截，拒绝则抛 `UserRejectedCommandError` 让整轮取消（`status="cancelled"`）；HITL 是 LLM **主动**调用 `ask_human` 暂停图，恢复后继续执行。两者不互相调用。

### 代码示例

#### 直接调用结构化 API

```python
# 启动一轮可能触发 interrupt 的对话
turn = agent.chat_structured("需要人工选择时请先问我")
while turn.is_interrupted:
    if len(turn.interrupts) == 1:
        # 单 interrupt：直接传 choice_id
        resume = {"choice_id": "approve"}
    else:
        # 并行 interrupt：按 interrupt.id 映射
        resume = {
            interrupt.id: {"choice_id": "approve"}
            for interrupt in turn.interrupts
        }
    turn = agent.resume_structured(resume)
print(turn.output)
```

#### 使用 CLI 辅助函数

```python
from utils.human_input import run_structured_until_completion, chat_until_completion

# ReAct 模式（自动循环处理 interrupt 直到完成）
output = run_structured_until_completion(agent, "react:删除旧文件前先让我确认")

# 普通对话模式
output = chat_until_completion(agent, "需要人工选择时请先问我")
```

### 限制与注意事项

| 限制                      | 说明                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| **不跨进程恢复**        | CLI 只负责正在运行进程内的结构化选择与恢复；重启后未完成的人工选择界面不会自动重建                  |
| **重启后可读历史**      | 重启后仍可通过 `thread` 菜单恢复历史对话上下文，但图执行不会自动续接                               |
| **同一 AgentCore 实例** | `resume_structured` 要求在同一个 `AgentCore` 实例中调用，跨实例恢复不支持                          |
| **跨线程拒绝**          | `resume_structured` 校验 `thread_id`，跨线程恢复会抛 `ValueError`                                  |
| **与 cot 无关**         | `cot()` 模式不走 LangGraph，不会触发 interrupt；HITL 仅对 `run_structured`/`chat_structured` 生效 |
| **测试覆盖**            | [tests/test_human_input.py](tests/test_human_input.py) 覆盖了 interrupt、resume、并行选择、线程隔离等场景 |

---

## 交互命令参考

| 命令                                        | 说明                                                                       |
| ------------------------------------------- | -------------------------------------------------------------------------- |
| `react:任务`                              | Agent 模式，自动调用工具，打印步骤，存长期记忆                             |
| `cot:任务`                                | 链式思考模式，纯推理不调用工具                                             |
| `switch:提供商名`                         | 运行时切换 LLM 提供商（如`switch:deepseek`）                             |
| `model`                                   | 方向键选择切换当前提供商的模型                                             |
| `model:<name>`                            | 直接切换模型（如`model:glm-4-flash`）                                    |
| `help`                                    | 查看完整命令说明                                                           |
| `info`                                    | 查看当前模型和记忆状态(含 thread_id、会话数)                               |
| `tools`                                   | 查看可用工具列表（含 MCP 工具）                                            |
| `clear [long\|short\|all]`                  | 清理记忆（默认 long）                                                      |
| `compress` 或 `压缩`                    | 压缩长期记忆（LLM 摘要后替换原内容）                                       |
| `thread`                                  | 方向键选择切换会话(显示消息数预览);`Enter` 切换, `Ctrl+D` 删除高亮会话 |
| `thread:new`                              | 开启新会话(原会话保留)                                                     |
| `thread:delete <id>`                      | 删除指定会话(二次确认,不可恢复)                                            |
| `mcp`                                     | 查看 MCP Server 状态                                                       |
| `mcp:reload`                              | 重新加载 MCP 工具                                                          |
| `mcp:add <name> ...`                      | 添加 MCP Server                                                            |
| `mcp:remove <name>`                       | 删除 MCP Server                                                            |
| `mcp:toggle <name> <on\|off>`              | 启用/禁用 MCP Server                                                       |
| `skill` 或 `skills`                     | 查看所有本地可用技能                                                       |
| `skill:<name>`                            | 将某技能加载进当前会话(注入 system prompt)                                 |
| `skill:<name> <任务>`                     | 加载技能并立即以 Agent 模式执行该任务(如`skill:git-commit 提交README`)   |
| `skill:clear`                             | 清空手动加载的技能                                                         |
| `safety`                                  | 查看当前安全策略                                                           |
| `safety:mode <blacklist\|whitelist>`       | 切换安全模式                                                               |
| `safety:confirm <on\|off>`                 | 开关危险命令确认                                                           |
| `export` 或 `export:<thread_id> [路径]` | 导出对话为 Markdown(默认存`exports/`)                                    |
| `json:<任务>`                             | 让 Agent 以 JSON 对象返回结果并解析展示                                    |
| `quit` / `exit`                         | 退出                                                                       |
| 其他输入                                    | 普通对话模式（也支持工具调用，但不打印步骤）                               |

---

## 运行示例

```
==================================================
  LangChain Agent (基于LangChain框架)
==================================================

选择 LLM 提供商
  (↑↓ 选择, Enter 确认, Esc 取消)
    [✓] zhipu      (智谱AI)
        qwen       (通义千问)
  ❯   [✓] deepseek   (DeepSeek)        ← 绿色高亮,方向键移动
        kimi       (Kimi (Moonshot))
----------------------------------------

[MCP] workspace: 加载了 6 个工具
[MCP] 已加载 6 个工具: create_workspace, get_current_workspace, list_directory, delete_workspace, move_workspace, copy_workspace

Agent 已就绪！
当前提供商: DeepSeek
当前模型:   deepseek-chat

本地工具: search, read_file, write_file, calculate, run_shell, run_python, run_cmd, get_local_time, open_file, open_sqlite, read_skill, ask_human
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

你: model
当前提供商 [DeepSeek] 可用模型:
  (↑↓ 选择, Enter 确认, Esc 取消)
    deepseek-chat
  ❯ deepseek-reasoner     ← 切换到推理模型
----------------------------------------
已切换模型: deepseek-reasoner (提供商: DeepSeek)

你: info
当前提供商: DeepSeek
当前模型:   deepseek-reasoner
API地址:    https://api.deepseek.com

--- 记忆状态 ---
当前会话:   thread-a4d099d2
Checkpoint: sqlite → D:\work\LangChainAgent\data\checkpoints.sqlite
已存消息:   8 条
长期记忆:   2 条
总会话数:   1

你: thread:new
已开启新会话: thread-c7e8f1a3
原会话 thread-a4d099d2 已保留,可用 'thread' 切回

你: thread
选择会话 (共 2 个,↑↓ 选择,Enter 切换)
  (↑↓ 选择, Enter 确认, Esc 取消)
    thread-a4d099d2  [8 条消息]
  ❯ thread-c7e8f1a3  [0 条消息] (当前)
----------------------------------------

你: react:帮我打开 checkpoints.sqlite 数据库
--- 步骤 1 ---
工具: open_sqlite
输入: {'file_path': 'data/checkpoints.sqlite'}
结果: {"success": true, "message": "已用 DB Browser for SQLite 打开数据库"}
最终答案: 已为你打开 checkpoints.sqlite 数据库

你: react:删除旧的临时文件前先让我确认
--- 步骤 1 ---
工具: ask_human
输入: {'prompt': '是否删除旧的临时文件?', 'choices': [{'id': 'delete', 'label': '删除'}, {'id': 'skip', 'label': '跳过'}]}

是否删除旧的临时文件?
  (↑↓ 选择, Enter 确认, Esc 取消)
  ❯ 删除
    跳过
----------------------------------------
是否删除旧的临时文件? ❯ 删除
最终答案: 已确认删除旧的临时文件。

你: mcp
MCP Servers:
------------------------------------------------------------
  [✓启用] workspace (stdio)
           python tools/workspace_tool.py
------------------------------------------------------------
已加载 MCP 工具数: 6

你: quit
再见!
```

> **重启程序后**：用 `thread` 方向键菜单选择历史会话，恢复之前的对话上下文。

---

## 代码使用示例

```python
from llm_client import create_client
from agent import AgentCore

# 创建客户端和Agent
llm = create_client(provider="deepseek", config_file="config/llm_config.json")
agent = AgentCore(
    llm_client=llm,
    memory_size=10,
    long_term_memory_file="data/memory.json",        # 长期记忆(用于 compress)
    checkpoint_file="data/checkpoints.sqlite",       # Checkpoint 持久化
    max_iterations=15,
    mcp_config_file="config/mcp_servers.json",
    enable_mcp=True,
    skills_dir=".agents/skills",
    auto_match_skills=True,
    max_context_messages=0,                           # 0=关闭长上下文裁剪
    context_trim_keep=12
)

# 普通对话（自动判断是否调用工具，自动写 checkpoint，不存长期记忆）
response = agent.chat("帮我创建一个叫 test 的文件夹")

# Agent模式（自动调用工具，打印步骤，写 checkpoint + 存长期记忆）
result = agent.run("计算 123 * 456 并把结果写入 result.txt")

# CoT模式（纯推理，不调用工具，不写 checkpoint，存长期记忆）
result = agent.cot("分析机器学习的应用场景")

# Human-in-the-loop 结构化入口：chat_structured/run_structured 返回 AgentTurnResult
turn = agent.chat_structured("需要人工选择时请先问我")
while turn.is_interrupted:
    # 单个 interrupt 可直接 resume；并行多个 interrupt 要按 interrupt.id 提交映射
    if len(turn.interrupts) == 1:
        resume = {"choice_id": "approve"}
    else:
        resume = {
            interrupt.id: {"choice_id": "approve"}
            for interrupt in turn.interrupts
        }
    # CLI 中由 select_menu 生成选择；Esc 会向 Agent 发送 {"cancelled": True}
    turn = agent.resume_structured(resume)
print(turn.output)

# 切换LLM提供商
from llm_client import create_client
new_llm = create_client(provider="qwen", config_file="config/llm_config.json")
agent.switch_llm(new_llm)

# 切换模型(同一提供商内)
llm.switch_model("glm-4-flash")    # 切到 glm-4-flash
print(llm.list_models())           # 查看当前提供商的可用模型
agent.switch_llm(llm)              # 重建 Agent 以使用新模型

# 会话管理
agent.memory.new_thread()                    # 开启新会话
agent.memory.switch_thread("thread-abc123")  # 切换到已有会话
agent.memory.delete_thread("thread-xxx")     # 删除指定会话
print(agent.memory.list_threads())           # 列出所有会话
print(agent.memory.export_thread(fmt="markdown"))  # 导出当前会话为 Markdown 文本

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

# 技能管理
print(agent.list_skills())       # 列出所有本地技能
agent.load_skill("git-commit")   # 手动加载技能到当前会话
agent.clear_skills()             # 清空手动加载的技能
agent.set_auto_match(False)      # 关闭任务自动匹配
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

或直接编辑 [config/mcp_servers.json](config/mcp_servers.json)：

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

注册到 `config/mcp_servers.json`：

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

| 特性       | 本地工具          | MCP 工具                      |
| ---------- | ----------------- | ----------------------------- |
| 定义方式   | `@tool` 装饰器  | `@mcp.tool()` + FastMCP     |
| 运行位置   | Agent 同进程      | 独立子进程                    |
| 加载方式   | import 后直接使用 | 通过 MCP 协议动态加载         |
| 增删       | 需改代码 + 重启   | 改配置 +`mcp:reload` 热加载 |
| 跨语言支持 | 仅 Python         | 任意语言（Node/Go 等）        |
| 适用场景   | 简单、轻量工具    | 复杂服务、第三方集成          |

---

## 运行时配置（agent_config.json）

原先硬编码在 `main.py` 的运行时参数已外置到 `config/agent_config.json`，由 [config.py](config.py) 的 `load_agent_config` 加载并与默认值合并（缺省键不报错）。

| 键                       | 类型 | 默认值                      | 说明                                                  |
| ------------------------ | ---- | --------------------------- | ----------------------------------------------------- |
| `max_iterations`       | int  | 15                          | 单次`invoke` 最大推理步数（即 `recursion_limit`） |
| `skills_dir`           | str  | `.agents/skills`          | 技能目录（相对项目根或绝对路径）                      |
| `auto_match_skills`    | bool | true                        | 任务自动匹配并注入相关技能                            |
| `enable_mcp`           | bool | true                        | 是否加载 MCP 工具                                     |
| `memory_size`          | int  | 10                          | 兼容旧 API 的记忆容量                                 |
| `verbose`              | bool | true                        | 是否打印详细过程                                      |
| `mcp_config_file`      | str  | `config/mcp_servers.json` | MCP 配置文件（相对项目根或绝对路径）                  |
| `max_context_messages` | int  | 0                           | 长上下文裁剪阈值（0 = 关闭）                          |
| `context_trim_keep`    | int  | 12                          | 裁剪时保留的最近消息条数                              |

修改后重启 `main.py` 即可生效。

### 长上下文裁剪（Long-Context Trimming）

当某个会话的消息数超过 `max_context_messages` 时，Agent 会自动：

1. 用 LLM 将较早的消息压缩成一份中文摘要；
2. 开启**新会话**，并把摘要注入后续 system prompt（保留上下文精华）；
3. 仅保留最近 `context_trim_keep` 条消息，从而避免撞上 LLM 上下文窗口。

> 触发时会在终端打印提示（含新旧 `thread_id`）。默认 `max_context_messages=0`（关闭），需要时在 `config/agent_config.json` 中设一个合理值（如 60）即可开启。

### 对话导出（Export）

```
export                 # 导出当前会话到 exports/<thread_id>.md
export:<thread_id>     # 导出指定会话
export:<thread_id> 路径/文件.md   # 导出到指定路径
```

由 [agent/memory.py](agent/memory.py) 的 `AgentMemory.export_thread()` 实现，将 checkpoint 中的对话渲染为可读文本/Markdown。

### JSON 模式（Structured Output）

```
json:<任务描述>
```

Agent 会按要求**只输出一个合法 JSON 对象**（不含 ``` 标记与解释文字），执行后自动用 `LLMClient.extract_json` 解析并美化打印；解析失败则回退显示原始输出。适用于需要结构化返回（如生成配置、报表、API 响应）的场景。

---

## 测试

项目提供一套**离线**单元测试（无需 API Key、不联网），覆盖核心逻辑：

| 测试文件                       | 覆盖内容                                                 |
| ------------------------------ | -------------------------------------------------------- |
| `tests/test_config.py`       | 运行时配置：默认值合并、路径解析                         |
| `tests/test_safety.py`       | 安全护栏：黑名单拒绝、白名单放行、危险命令确认、路径保护 |
| `tests/test_skills.py`       | `SkillManager`：列出/读取/匹配(中→英别名)/渲染技能    |
| `tests/test_search.py`       | `search` 工具：无 Key 降级、Tavily 返回结构(mock)      |
| `tests/test_cli_commands.py` | CLI 命令分发：路由优先级、状态变更和各领域处理器         |
| `tests/test_human_input.py`  | LangGraph HITL：interrupt、恢复、并行选择和线程隔离      |
| `tests/test_terminal.py`     | 终端工具：输出截断、护栏拒绝、安全执行(mock subprocess)  |
| `tests/test_memory.py`       | `AgentMemory`：长期记忆、会话管理(SQLite)              |

运行：

```powershell
.\.venv\Scripts\pip.exe install -r requirements.txt
.\.venv\Scripts\pytest.exe
```

测试配置见 `pytest.ini`（`pythonpath = .` 保证 `import tools`/`import agent` 可用），`conftest.py` 提供安全配置缓存隔离的 autouse fixture。

---

## 技术栈

- Python 3.10+
- LangChain 1.x（`langchain`, `langchain-core`, `langchain-openai`）
- LangGraph 1.x（`create_react_agent`）
- LangGraph Checkpoint（`langgraph-checkpoint-sqlite`，SQLite 持久化）
- langchain-mcp-adapters 0.3+（MCP 工具适配）
- mcp 1.9+ / fastmcp 2.x（FastMCP Server）
- OpenAI SDK（用于 OpenAI 兼容接口）
- Tavily Python SDK（联网搜索）
- pytest（离线单元测试）
