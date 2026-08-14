# LangChainAgent

基于 **LangChain 1.x + LangGraph** 框架的智能 Agent 项目，支持：

- 配置驱动的多 LLM 提供商（见 `config/llm_config.json`），运行时可切换提供商/模型
- 本地工具调用（搜索、文件读写、计算、终端命令、文件打开、技能读取）
- **MCP Server 工具动态加载**（可扩展任意 MCP 服务）
- **LangGraph Checkpoint 持久化**（服务器异步运行时使用 `data/checkpoints_async.sqlite`，程序重启可恢复对话）
- **LangGraph Human-in-the-loop**（`ask_human` 暂停图执行，CLI 结构化选择后 `Command(resume)` 继续）
- 长期记忆管理（compress 压缩摘要）与长上下文自动裁剪
- **长上下文压缩中间件**（增量摘要 + 工具输出 Prune，摘要随 checkpoint 持久化、per-thread 隔离；`before_model` 自动触发或 `compact` 命令手动触发）
- **MCP 连接池**（per-server 隔离、健康探测、单 server 自动重连，替代全量重载）
- 多会话隔离（thread_id 机制，方向键菜单切换/删除/导出）
- **并发多会话**（per-thread 锁 + 独立编译图 + 按线程中断状态，Web 多标签页/多客户端可同时对话互不阻塞，详见「异步 Public API → 并发多会话」）
- 命令模式下的会话切换会优先保持当前会话，不会因为菜单参数不兼容而失败
- 安全护栏（危险终端命令拦截/确认、路径保护）
- **全套异步 Public API**（`arun` / `achat` / `aresume` / `arun_structured` / `achat_structured` 等）
- **运行时指标收集**（LLM 调用 / 工具执行 / 压缩统计，`metrics` 命令查询）
- **结构化日志**（trace_id / thread_id 上下文注入，asyncio 安全）
- **工具超时保护**与**统一异常层次**（`LCAgentError` 及其子类）

---

## 目录

- [LangChainAgent](#langchainagent)
  - [目录](#目录)
  - [快速开始](#快速开始)
    - [1. 创建虚拟环境](#1-创建虚拟环境)
    - [2. 安装依赖](#2-安装依赖)
    - [3. 准备配置文件](#3-准备配置文件)
    - [4. 配置 API 密钥](#4-配置-api-密钥)
    - [5. 运行](#5-运行)
  - [项目结构](#项目结构)
    - [模块职责](#模块职责)
    - [agent/llm\_client.py 主要 API](#agentllm_clientpy-主要-api)
  - [核心概念：三种模式](#核心概念三种模式)
  - [记忆系统（Memory）](#记忆系统memory)
    - [设计架构](#设计架构)
    - [上下文注入机制（重要）](#上下文注入机制重要)
      - [`react:` 和 `chat()` → 事件驱动写入 + prompt 注入读取](#react-和-chat--事件驱动写入--prompt-注入读取)
      - [`cot:` → 仅短期记忆](#cot--仅短期记忆)
      - [技能指引与长上下文摘要的注入](#技能指引与长上下文摘要的注入)
      - [对比表](#对比表)
    - [两层存储对比](#两层存储对比)
    - [三种模式的记忆行为](#三种模式的记忆行为)
    - [Checkpoint 持久化原理](#checkpoint-持久化原理)
    - [长期记忆写入流水线](#长期记忆写入流水线)
    - [为什么 cot 不写入 checkpoint](#为什么-cot-不写入-checkpoint)
    - [文件位置](#文件位置)
    - [记忆管理命令](#记忆管理命令)
    - [压缩长期记忆（compress）](#压缩长期记忆compress)
      - [工作流程](#工作流程)
      - [压缩提示词](#压缩提示词)
      - [使用示例](#使用示例)
      - [压缩后的 Store 格式](#压缩后的-store-格式)
      - [特点与注意事项](#特点与注意事项)
    - [记忆相关 API](#记忆相关-api)
  - [会话管理（Session）](#会话管理session)
  - [工具系统（Tools）](#工具系统tools)
    - [1. 本地工具（Local Tools）](#1-本地工具local-tools)
    - [2. MCP 工具（MCP Server Tools）](#2-mcp-工具mcp-server-tools)
      - [已注册的 MCP Server](#已注册的-mcp-server)
      - [MCP 配置文件](#mcp-配置文件)
      - [MCP 管理命令](#mcp-管理命令)
    - [3. 技能阅读（Skills）](#3-技能阅读skills)
      - [已注册的技能（本地）](#已注册的技能本地)
      - [交互命令](#交互命令)
      - [`read_skill` 工具](#read_skill-工具)
      - [自动匹配原理](#自动匹配原理)
      - [技能相关 API](#技能相关-api)
    - [4. 安全护栏（Safety）](#4-安全护栏safety)
      - [配置（config/safety.json）](#配置configsafetyjson)
      - [路径保护](#路径保护)
      - [交互命令](#交互命令-1)
    - [工具调用机制](#工具调用机制)
    - [System Prompt 强化](#system-prompt-强化)
    - [5. 扩展工具](#5-扩展工具)
      - [方式 1：添加本地工具](#方式-1添加本地工具)
      - [方式 2：添加 MCP Server](#方式-2添加-mcp-server)
      - [本地工具 vs MCP 工具对比](#本地工具-vs-mcp-工具对比)
  - [可观测性与可靠性](#可观测性与可靠性)
    - [长上下文压缩中间件（Compaction）](#长上下文压缩中间件compaction)
    - [MCP 连接池（MCPPool）](#mcp-连接池mcppool)
    - [运行时指标（Metrics）](#运行时指标metrics)
    - [结构化日志（Logging）](#结构化日志logging)
    - [工具超时保护（Tool Timeout）](#工具超时保护tool-timeout)
    - [统一异常层次](#统一异常层次)
  - [异步 Public API](#异步-public-api)
  - [定时任务调度（Scheduler）](#定时任务调度scheduler)
    - [工具接口](#工具接口)
    - [快速开始](#快速开始-1)
    - [模块结构](#模块结构)
  - [Human-in-the-loop（HITL）](#human-in-the-loophitl)
    - [执行流程](#执行流程)
    - [ask\_human 工具](#ask_human-工具)
    - [结构化 API](#结构化-api)
    - [CLI 编排](#cli-编排)
      - [结构化选择的展示与收集](#结构化选择的展示与收集)
    - [并行 interrupt 与多轮暂停](#并行-interrupt-与多轮暂停)
    - [线程隔离](#线程隔离)
    - [与普通多轮聊天、安全护栏的区别](#与普通多轮聊天安全护栏的区别)
    - [代码示例](#代码示例)
      - [直接调用结构化 API](#直接调用结构化-api)
      - [使用 CLI 辅助函数](#使用-cli-辅助函数)
    - [限制与注意事项](#限制与注意事项)
  - [交互命令参考](#交互命令参考)
  - [运行示例](#运行示例)
    - [1. 启动与基础对话](#1-启动与基础对话)
    - [2. 三种执行模式](#2-三种执行模式)
    - [3. 模型与状态管理](#3-模型与状态管理)
    - [4. 会话管理](#4-会话管理)
    - [5. Human-in-the-loop（HITL）](#5-human-in-the-loophitl)
    - [6. 对话导出](#6-对话导出)
    - [7. JSON 模式](#7-json-模式)
    - [8. MCP 管理](#8-mcp-管理)
    - [9. 退出](#9-退出)
  - [代码使用示例](#代码使用示例)
  - [运行时配置](#运行时配置)
    - [1. `agent_config.json` — Agent 运行时参数](#1-agent_configjson--agent-运行时参数)
      - [长上下文裁剪（Long-Context Trimming）](#长上下文裁剪long-context-trimming)
    - [2. `llm_config.json` — LLM 服务商配置](#2-llm_configjson--llm-服务商配置)
    - [3. `mcp_servers.json` — MCP 服务器配置](#3-mcp_serversjson--mcp-服务器配置)
    - [4. `safety.json` — 安全护栏](#4-safetyjson--安全护栏)
    - [5. `remote_control.json` — 远程控制（飞书）](#5-remote_controljson--远程控制飞书)
    - [6. `scheduler_config.json` — 定时任务调度](#6-scheduler_configjson--定时任务调度)
  - [测试](#测试)
    - [运行测试](#运行测试)
  - [技术栈](#技术栈)

---

## 快速开始

### 1. 创建虚拟环境

```bash
cd <项目路径>
python -m venv .venv
```

| 平台    | 激活命令                         |
| ------- | -------------------------------- |
| Windows | `.\.venv\Scripts\Activate.ps1` |
| Linux   | `source .venv/bin/activate`    |

> Windows 如遇脚本执行限制：`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备配置文件

项目克隆后 `config/` 目录下只有 `.example` 模板（不含真实密钥），需手动复制为运行时配置：

```bash
cp config/llm_config.json.example     config/llm_config.json
cp config/mcp_servers.json.example    config/mcp_servers.json
cp config/safety.json.example         config/safety.json
cp config/scheduler_config.json.example config/scheduler_config.json
```

> 注：`agent/agent_config.json` 已包含默认配置，无需从模板复制。

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
├── main.py                  # 入口文件，交互式命令行 + 工作流构建
├── config/                  # 配置目录（需从 .example 复制；见快速开始）
│   ├── llm_config.json      # API 密钥配置文件
│   ├── mcp_servers.json     # MCP Server 配置
│   ├── safety.json          # 安全护栏策略
│   ├── remote_control.json  # 远程控制（飞书）配置
│   └── scheduler_config.json# 定时任务调度配置
├── scheduler/               # 定时任务调度模块
│   ├── store.py             # SQLite CRUD + 原子抢占
│   ├── executor.py          # AgentCore 异步执行桥接(acreate/arun/aclose 单 loop)
│   ├── engine.py            # APScheduler 引擎
│   └── run.py               # 独立进程入口
├── data/                  # 运行时数据库目录（自动生成）
│   ├── checkpoints_async.sqlite # Checkpoint 持久化数据库 + 长期记忆 Store（同一文件）
│   └── scheduled_tasks.sqlite# 定时任务数据库
├── memory/                 # 记忆模块（三层架构 Memory 层，与 agent 平级）
│   ├── __init__.py         # 包导出
│   ├── agent_memory.py     # AgentMemory：checkpointer + Store 基础设施
│   ├── config.py           # Memory 配置默认值
│   ├── context.py          # MemoryContext：统一工厂（入口程序调用）
│   ├── lock_pool.py        # ThreadMemoryLockPool：per-thread 并发锁池
│   ├── manager.py          # MemoryManager：统一门面（召回/消费/压缩/清理）
│   ├── middleware.py       # 读写中间件（防抖 buffer + Fact 抽取 + prompt 注入）
│   ├── models.py           # 数据模型与事件分类判定（MemoryCategory / ThreadFactItem）
│   └── store.py            # ThreadMemoryStore：Store 业务封装（per-thread 隔离）
├── agent/
│   ├── __init__.py
│   ├── agent_config.json    # Agent 运行时参数
│   ├── AGENT.md             # Agent 核心系统提示词（行为规则）
│   ├── llm_client.py        # 统一大模型封装（多提供商 + 多模型）
│   ├── config.py            # 运行时配置加载(agent/agent_config.json)
│   ├── compaction.py        # 长上下文压缩中间件（增量摘要 + 工具输出 Prune）
│   ├── metrics.py           # 运行时指标收集（LLM/工具/压缩统计，线程安全）
│   ├── logging_config.py    # 结构化日志（trace_id/thread_id 上下文注入）
│   ├── exceptions.py        # 统一异常层次（LCAgentError 及其子类）
│   ├── message_utils.py     # LLM 异常信息提取（中文化错误提示）
│   └── agent_core.py        # Agent 核心调度：run/chat/cot 三种模式 + HITL + 异步 API
├── session/                 # 会话管理模块（三层架构 Session 层）
│   ├── context.py           # SessionContext：单会话运行时上下文（session_id + config + checkpointer）
│   ├── store.py             # SessionStore：基于 LangGraph Store 的 per-session 瞬态状态
│   ├── registry.py          # SessionRegistry：会话生命周期管理（生成/查询/删除/消息读取）
│   ├── workspace_store.py   # WorkspaceStore：session_id ↔ workspace_path 映射
│   └── manager.py           # SessionManager：对外门面 & 会话调度（封装 Agent + Memory）
├── team/                    # 多 Agent 团队协作模块
│   ├── __init__.py          # 导出 ManagerAgent/WorkerAgent/TerminatorAgent
│   ├── factory.py           # 团队 Agent 工厂函数
│   ├── manager/             # Manager Agent（任务拆解）
│   │   ├── manager.py
│   │   ├── agent_config.json
│   │   └── AGENT.md
│   ├── worker/              # Worker Agent（任务执行）
│   │   ├── manager.py
│   │   ├── agent_config.json
│   │   └── AGENT.md
│   └── terminator/          # Terminator Agent（结果汇总）
│       ├── terminator.py
│       ├── agent_config.json
│       └── AGENT.md
├── graph/                   # LangGraph 工作流编排
│   ├── common.py            # 工作流通用能力：异步执行辅助 + 技能注入 + 跨轮次记忆压缩
│   ├── simple.py            # 监督者模式工作流（Manager→Worker→Terminator，异步节点）
│   ├── pipline.py           # 流水线模式工作流（异步节点，与 simple 同构）
│   └── registry.py          # 工作流/Agent 注册表与构建入口
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
│   ├── create_tools.py      # 动态生成工具代码，保存为 .py 并自动注册到 __init__.py
│   ├── safety.py            # 安全护栏(黑名单/白名单/交互确认/路径保护)
│   ├── mcp_loader.py        # MCP 配置管理与工具加载器
│   ├── mcp_pool.py          # MCP 连接池（per-server 隔离 + 健康探测 + 自动重连）
│   ├── tool_wrapper.py      # 工具超时包装器（统一超时保护，超时返回 JSON 错误）
│   └── scheduler_tool.py    # 定时任务工具（schedule_task/list/cancel/delete/cleanup）
├── cli/
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
│       ├── workflow.py      # 工作流命令（workflow / workflow:<name> <task>）
│       └── execution.py     # json / react / cot / 普通对话
├── tests/                   # 单元测试(pytest；含 1 个在线连通性测试)
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_safety.py
│   ├── test_skills.py
│   ├── test_search.py
│   ├── test_terminal.py
│   ├── test_memory.py
│   ├── test_cli_commands.py
│   ├── test_human_input.py
│   ├── test_scheduler.py
│   ├── test_workflow.py
│   ├── test_agent_core_regressions.py
│   ├── test_config_templates.py
│   ├── test_calculator.py
│   ├── test_api.py
│   ├── test_llm_client.py
│   ├── test_message_utils.py
│   ├── test_compaction.py
│   ├── test_create_tool.py
│   ├── test_graph_rebuild.py
│   ├── test_mcp_pool.py
│   ├── test_threads_preview.py
│   ├── test_metrics.py
│   ├── test_tool_wrapper.py
│   ├── test_logging_config.py
│   ├── test_exceptions_and_close.py
│   ├── test_provider_models.py        # 在线连通性测试(需 API Key)
└── exports/                 # 对话导出目录(运行 export 命令时生成)
```

### 模块职责

| 模块                                                    | 职责                                                                                                                                                                                                                                       |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [main.py](main.py)                                       | 交互式命令行入口 + Agent 构建接口                                                                                                                                                                                                          |
| [agent/llm_client.py](agent/llm_client.py)               | 从`config/llm_config.json` 读取提供商配置，支持运行时切换提供商/模型                                                                                                                                                                     |
| [agent/config.py](agent/config.py)                       | 加载`agent/agent_config.json`，统一运行时配置                                                                                                                                                                                            |
| [memory/](memory/)                                       | 三层架构 Memory 层：`AgentMemory`（checkpointer + Store 基础设施）/ `MemoryContext`（统一工厂）/ `MemoryManager`（统一门面）/ `ThreadMemoryStore`（Store 业务封装）/ 读写中间件（防抖 + Fact 抽取 + prompt 注入）/ per-thread 锁池 |
| [agent/compaction.py](agent/compaction.py)               | 长上下文压缩中间件：增量摘要 + 工具输出 Prune + 保留近期消息，摘要随 checkpoint 持久化、per-thread 隔离                                                                                                                                    |
| [agent/metrics.py](agent/metrics.py)                     | `MetricsCollector`：线程安全的运行时指标收集（LLM 调用 / 工具执行 / 压缩统计）                                                                                                                                                           |
| [agent/logging_config.py](agent/logging_config.py)       | 结构化日志：`contextvars` 实现 trace_id / thread_id 异步安全注入                                                                                                                                                                         |
| [agent/exceptions.py](agent/exceptions.py)               | 统一异常层次：`LCAgentError` 基类及 MCP/超时/压缩/中断/状态等子类                                                                                                                                                                        |
| [agent/agent_core.py](agent/agent_core.py)               | Agent 核心：`run()` / `chat()` / `cot()` 三种模式 + 全套异步 API + `AgentTurnResult` 结构化暂停/恢复 + 技能注入 + 压缩/裁剪                                                                                                        |
| [session/](session/)                                     | 三层架构 Session 层：`SessionContext`（单会话运行时上下文）/ `SessionStore`（per-session 瞬态状态）/ `SessionRegistry`（生命周期管理）/ `WorkspaceStore`（工作空间映射）/ `SessionManager`（对外门面 & 会话调度）                |
| [team/](team/)                                           | 多 Agent 团队协作：ManagerAgent（拆解）/ WorkerAgent（执行）/ TerminatorAgent（汇总）+ 工厂函数                                                                                                                                            |
| [graph/common.py](graph/common.py)                       | 工作流通用能力：`ainvoke_team_agent` 异步执行辅助、`SkillInjector` 技能注入、`arun_compiled_workflow` 跨轮次记忆压缩                                                                                                                 |
| [graph/simple.py](graph/simple.py)                       | LangGraph 监督者模式工作流编排（Manager→Worker→Terminator，异步节点）                                                                                                                                                                    |
| [graph/pipline.py](graph/pipline.py)                     | LangGraph 流水线模式工作流编排（异步节点，与 simple 同构）                                                                                                                                                                                 |
| [graph/registry.py](graph/registry.py)                   | 工作流/Agent 注册表：`register_workflow` / `register_agent` / `build_workflow`                                                                                                                                                       |
| [tools/skills.py](tools/skills.py)                       | `SkillManager`：扫描/匹配/渲染本地技能                                                                                                                                                                                                   |
| [tools/skill_tool.py](tools/skill_tool.py)               | `read_skill` 工具：LLM 在任务中自助读取技能指引                                                                                                                                                                                          |
| [tools/mcp_pool.py](tools/mcp_pool.py)                   | `MCPPool`：per-server 连接管理 + 健康探测 + 自动重连，替代全量重载                                                                                                                                                                       |
| [tools/tool_wrapper.py](tools/tool_wrapper.py)           | 工具超时包装：统一超时保护，超时返回 JSON 错误而非抛异常                                                                                                                                                                                   |
| [tools/](tools/)                                         | 本地工具 + MCP 工具加载 + 技能管理                                                                                                                                                                                                         |
| [cli/cli_menu.py](cli/cli_menu.py)                       | 通用终端方向键选择菜单                                                                                                                                                                                                                     |
| [cli/human_input.py](cli/human_input.py)                 | `ask_human` 工具、interrupt 展示和循环恢复编排                                                                                                                                                                                           |
| [cli/commands/dispatcher.py](cli/commands/dispatcher.py) | 按兼容顺序匹配命令并路由到领域处理器                                                                                                                                                                                                       |
| [cli/commands/types.py](cli/commands/types.py)           | 命令依赖上下文、活动 LLM 状态和分发结果类型                                                                                                                                                                                                |
| [cli/commands/](cli/commands/)                           | 会话、记忆、模型、MCP、技能、安全及 Agent 执行命令                                                                                                                                                                                         |
| [scheduler/](scheduler/)                                 | 定时任务调度：TaskStore（SQLite CRUD）、SchedulerEngine（APScheduler 轮询）、独立进程入口                                                                                                                                                  |

### agent/llm_client.py 主要 API

| 类/函数                           | 说明                                     |
| --------------------------------- | ---------------------------------------- |
| `LLMClient`                     | 核心客户端类，支持配置文件中的所有提供商 |
| `LLMClient.chat()`              | 发送对话请求                             |
| `LLMClient.chat_with_history()` | 带历史记录的对话                         |
| `LLMClient.get_chat_model()`    | 获取 LangChain`ChatOpenAI` 实例        |
| `LLMClient.switch_provider()`   | 运行时切换提供商                         |
| `LLMClient.switch_model(model)` | 运行时切换当前提供商的模型               |
| `LLMClient.list_models()`       | 列出当前提供商的可用模型                 |
| `LLMClient(...)`                | 创建客户端的构造方法                     |
| `list_providers()`              | 列出所有支持的提供商                     |

**支持的提供商：**

提供商不再硬编码在代码中，全部来自 [config/llm_config.json](config/llm_config.json) 的 `providers` 字段。当前配置包含（实际内容以文件为准）：

| 提供商       | 名称            | 环境变量              | 默认模型                |
| ------------ | --------------- | --------------------- | ----------------------- |
| `zhipu`    | 智谱AI          | `ZHIPU_API_KEY`     | `glm-4.7-flash`       |
| `qwen`     | 通义千问        | `DASHSCOPE_API_KEY` | `Qwen2.5-7B-Instruct` |
| `deepseek` | DeepSeek        | `DEEPSEEK_API_KEY`  | `deepseek-chat`       |
| `kimi`     | Kimi (Moonshot) | `MOONSHOT_API_KEY`  | `kimi-k3`             |
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

记忆系统为独立的 **三层架构 Memory 层**（[memory/](memory/)，与 `agent/` `session/` 平级），由 [memory/manager.py](memory/manager.py) 的 `MemoryManager` 作为统一门面，底层封装 **LangGraph Checkpoint（短期）+ BaseStore 长期记忆（facts）** 双层存储：

```
┌──────────────────────────────────────────────────────────────────┐
│            memory/ 包（三层架构 Memory 层，与 agent 平级）         │
│                                                                  │
│  统一门面  MemoryManager  ── recall / consume_event / compress   │
│                                                                  │
│  ┌────────────────────────┐  ┌──────────────────────────────┐   │
│  │  Checkpoint (自动)      │  │  长期记忆 Store (事件驱动)    │   │
│  │  AsyncSqliteSaver      │  │  AsyncSqliteStore             │   │
│  │  checkpoints_async.sql │  │  namespace=(thread_id,        │   │
│  │  (MemoryContext 注入)  │  │   "thread_facts")             │   │
│  │                        │  │                              │   │
│  │  • Agent 每步执行后自动写│  │  • 写: ThreadMemoryWrite     │   │
│  │  • 完整状态(消息+工具调用)│  │    Middleware 防抖→LLM 抽取  │   │
│  │  • 按 thread_id 隔离    │  │  • 读: ThreadMemoryRead      │   │
│  │  • 程序重启可恢复       │  │    Middleware 注入 SystemMsg │   │
│  │                        │  │  • 去重 + LRU 淘汰(50条)     │   │
│  └────────────────────────┘  └──────────────────────────────┘   │
│                                                                  │
│  基础设施: AgentMemory(checkpointer+Store) / MemoryContext(工厂)  │
│           ThreadMemoryLockPool(per-thread 锁) / models.py(分类)  │
└──────────────────────────────────────────────────────────────────┘
```

**数据流**（读写分离）：

```
写: SessionManager 消费 AgentEvent 流（submit_user_message / consume_event）
     → 非阻塞投递到 WriteMiddleware 防抖 buffer（20s 窗口）
     → LLM Fact 抽取（格式化事实 + 分类 + 置信度）
     → 无效过滤 / 本地去重 → 批量写入 ThreadMemoryStore
     → LRU 淘汰（单 thread 超 50 条时）

读: ThreadMemoryReadMiddleware.awrap_model_call（每个 model 调用前）
     → query_facts(thread_id) 读取该 thread 全部 facts
     → 格式化为文本追加到 SystemMessage（【长期记忆】块）
     → 非阻塞更新 last_used_at（供 LRU 淘汰）
```

**组件职责**：

| 组件                            | 文件                                            | 职责                                                                                                                                 |
| ------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `MemoryManager`               | [memory/manager.py](memory/manager.py)           | 统一门面：`recall` / `recall_text` / `submit_user_message` / `consume_event` / `compress` / `clear` / `flush_all`      |
| `MemoryContext`               | [memory/context.py](memory/context.py)           | 统一工厂：入口程序（main/api/scheduler）调用`acreate()` 组装全部组件，暴露 checkpointer / store / read_middleware / memory_manager |
| `AgentMemory`                 | [memory/agent_memory.py](memory/agent_memory.py) | checkpointer（`AsyncSqliteSaver`）+ 长期记忆 Store（`AsyncSqliteStore`）基础设施                                                 |
| `ThreadMemoryStore`           | [memory/store.py](memory/store.py)               | Store 业务封装：facts 的增/查/批量写/LRU 淘汰/摘要替换/会话级清空                                                                    |
| `ThreadMemoryWriteMiddleware` | [memory/middleware.py](memory/middleware.py)     | 写服务：事件接收 + 防抖 buffer + Fact 抽取流水线（非 AgentMiddleware）                                                               |
| `ThreadMemoryReadMiddleware`  | [memory/middleware.py](memory/middleware.py)     | 读中间件（AgentMiddleware）：`awrap_model_call` 注入 facts 到 SystemMessage                                                        |
| `ThreadMemoryLockPool`        | [memory/lock_pool.py](memory/lock_pool.py)       | per-thread`asyncio.Lock` 池：串行化同一 thread 的写入，不同 thread 并行                                                            |
| `models.py`                   | [memory/models.py](memory/models.py)             | `MemoryCategory` / `ThreadFactItem` / `MemoryInputEvent` / `judge_long_term_memory` 分类判定                                 |
| `config.py`                   | [memory/config.py](memory/config.py)             | 运行时参数默认值（buffer 延迟 / 上限 / fact 上限 / 召回条数）                                                                        |

> 除此之外还有一层 **Compaction 压缩中间件**（[agent/compaction.py](agent/compaction.py)）负责控制**单会话内的上下文长度**：当 checkpoint 恢复的消息数超过阈值时，`before_model` 自动把旧消息增量摘要成 `state.summary`（随 checkpoint 持久化、per-thread 隔离），并 Prune 过长的历史工具输出，无需新开 thread。注意这与记忆系统的 `compress` 命令是两回事（前者压缩会话上下文，后者压缩长期记忆 facts）。详见[可观测性与可靠性 → 长上下文压缩中间件](#长上下文压缩中间件compaction)。

### 上下文注入机制（重要）

一个常见疑问：**Agent 在对话时，历史消息从哪里来？** 答案取决于模式。

#### `react:` 和 `chat()` → 事件驱动写入 + prompt 注入读取

Agent 模式下，LLM 上下文由 **checkpoint 恢复的历史消息 + ReadMiddleware 注入的长期记忆 facts** 共同组成：

```python
# arun_events()：graph 恢复 checkpoint 历史
config = self._invoke_config(thread_id)          # {"configurable": {"thread_id": ...}}
async for ev in self._arun_graph_events({"messages": [HumanMessage(content=message)]}, config, ...)
```

- **历史消息**：LangGraph 从 checkpoint 自动恢复（按 thread_id 取出该会话所有历史消息，拼到新消息前面）
- **长期记忆**：`ThreadMemoryReadMiddleware` 在每个 model 调用前（`awrap_model_call`）从 Store 读取该 thread 的 facts，格式化为文本追加到 SystemMessage，随请求一起发给 LLM
- **写入路径**：SessionManager 在事件流消费时调用 `submit_user_message()` / `consume_event()` 非阻塞投递，经防抖 + LLM 抽取后写入 Store

```
invoke(新消息, thread_id)
   ↓
[历史消息1, ..., 新消息] + SystemMessage(【长期记忆】块) → 传给 LLM
   ↓
LLM 回复 → 写回 checkpoint
   ↓
DONE / TOOL_RESULT 事件 → SessionManager.consume_event() → 防抖 → LLM 抽取 → 写 Store
```

#### `cot:` → 仅短期记忆

```python
# acot()（异步）
short_term = await self.session.aget_short_term()   # ← 仅 checkpointer 的短期历史
response = self.llm.chat_with_history(
    user_input=task,
    history=short_term,
    system_prompt=system_prompt,
    ...
)
```

- `session.aget_short_term()` → 从 **SessionRegistry**（封装 checkpointer）取当前会话的消息（转 dict）
- **不经过 Agent 执行**、不触发读写中间件，因此长期记忆 facts **不参与** cot 的上下文
- cot 的输入输出也不消费（提交）到记忆流水线

#### 技能指引与长上下文摘要的注入

`react:` 和 `chat()` 在每次执行前会根据任务重建 Agent：

```python
self.agent_executor = self._create_agent_executor(
    self._compute_skill_block(task)
)
```

因此除了 checkpoint 历史外，system prompt 还可能包含：

| 来源         | 触发方式                                              | 说明                                                                                                                |
| ------------ | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 长期记忆     | 事件驱动自动沉淀，model 调用前注入                    | ReadMiddleware 注入 Store facts（`【长期记忆】` 块）                                                              |
| 手动技能     | `skill:<name>`                                      | 后续对话都会注入该技能指引，直到`skill:clear`                                                                     |
| 自动匹配技能 | `auto_match_skills=true`                            | 根据任务与技能描述的关键词重叠度自动注入相关技能                                                                    |
| 长上下文压缩 | 消息数超`max_messages` 时 `before_model` 自动触发 | 由 Compaction 中间件增量摘要旧消息，写入`state.summary`（随 checkpoint 持久化、per-thread 隔离），无需新开 thread |

#### 对比表

| 模式       | checkpoint  | Store facts | 技能指引  | 长上下文压缩  | 说明                                    |
| ---------- | ----------- | ----------- | --------- | ------------- | --------------------------------------- |
| `react:` | ✅ 自动恢复 | ✅ 自动注入 | ✅ 可注入 | ✅ 中间件触发 | `is_run_mode=True`，DONE 事件标记重要 |
| 普通对话   | ✅ 自动恢复 | ✅ 自动注入 | ✅ 可注入 | ✅ 中间件触发 | 不标记重要，写入照常（事件驱动）        |
| `cot:`   | ✅ 手动取   | ❌ 不参与   | ❌ 不注入 | ❌ 不触发     | 绕过 Agent，纯 LLM 推理，不写长期       |

> ⚠️ **重要副作用**：`react:` / 普通对话的记忆只按 **thread_id** 隔离（namespace `(thread_id, "thread_facts")`）。切到新的 thread（会话 B）后，ReadMiddleware 只注入会话 B 自己的 facts，**看不到**会话 A 沉淀的长期记忆。跨会话共享记忆目前不支持——长期记忆与 checkpoint 一样按会话隔离。

### 两层存储对比

| 类型                 | 存储方式                                   | 触发时机                               | 保存内容                                 | 持久化  | 用途                                      |
| -------------------- | ------------------------------------------ | -------------------------------------- | ---------------------------------------- | ------- | ----------------------------------------- |
| **Checkpoint** | `data/checkpoints_async.sqlite` (SQLite) | Agent 每步执行后自动                   | 完整状态(消息+工具调用链+中间变量)       | ✅ 永久 | 程序重启恢复对话、多会话隔离              |
| **长期记忆**   | 同一 SQLite 文件（`AsyncSqliteStore`）   | 事件驱动 + LLM 抽取（防抖 20s 后批量） | facts（content + category + confidence） | ✅ 永久 | 跨会话保留关键信息、注入 prompt、compress |

### 三种模式的记忆行为

| 模式           | 调用方法                          | 写入 Checkpoint | 写入长期记忆 Store                    | 说明                                       |
| -------------- | --------------------------------- | --------------- | ------------------------------------- | ------------------------------------------ |
| `react:任务` | `arun_events(is_run_mode=True)` | ✅ 自动         | ✅ 事件驱动（用户消息标记 important） | ReAct 结果重要，DONE 事件标记 is_important |
| `cot:任务`   | `acot()`                        | ❌ 不写         | ❌ 不写                               | CoT 绕过 Agent，纯 LLM 推理，无事件流      |
| 普通输入       | `achat_stream()`                | ✅ 自动         | ✅ 事件驱动（不标记 important）       | 对话同样参与记忆抽取，只是不强调重要       |

> **关键区别**：Checkpoint 由 LangGraph 自动管理，无需手动干预；长期记忆由**事件驱动流水线**自动沉淀（分类判定 + LLM 抽取 + 去重），`important` 标记只是提高"值得评估"的优先级，不再决定是否写入。

### Checkpoint 持久化原理

checkpointer 由入口程序（[main.py](main.py) / [api/server.py](api/server.py) / [scheduler/run.py](scheduler/run.py)）通过 `MemoryContext.acreate()` 创建后注入 `AgentCore`，LangGraph 在每次执行后自动保存状态到 SQLite：

```python
# main.py：先创建 MemoryContext，再注入 AgentCore
memory_ctx = await MemoryContext.acreate(
    checkpoint_file=CHECKPOINT_FILE,
    llm_getter=lambda: llm,
    ...
)
agent = await AgentCore.acreate(
    llm_client=llm,
    checkpointer=memory_ctx.checkpointer,            # ← 自动持久化
    store=memory_ctx.store,                          # ← 长期记忆 Store
    extra_middleware=[memory_ctx.read_middleware],   # ← 长期记忆读取中间件
    initial_thread_id=memory_ctx.thread_id,
    async_conn=memory_ctx.async_conn,
)
agent.set_memory_manager(memory_ctx.memory_manager)  # ← 注入 MemoryManager
```

创建 `AgentCore` 后，还需把记忆组件的 LLM 来源动态绑定到当前 Agent（三个入口 `main.py` / `api/server.py` / `scheduler/run.py` 均已内置）：

```python
memory_ctx.bind_llm(lambda: agent.llm)  # 记忆组件直接读取 agent 当前 LLM
```

这样**运行时切换提供商/模型**（API `/api/providers/switch`、CLI `switch` 命令、team 角色切换）后，记忆链路（事实抽取 / 召回 / 压缩）会跟随 `agent.llm` 同步切换，避免记忆抽取仍使用启动时的旧 `LLMClient` 向旧提供商发请求。

调用时传 thread_id，LangGraph 自动恢复该会话历史（`_invoke_config` 构造 `{"configurable": {"thread_id": ...}}`）。

**核心能力**：

| 能力         | 说明                                                                           |
| ------------ | ------------------------------------------------------------------------------ |
| 自动持久化   | Agent 每步执行后自动写入 SQLite，无需手动                                      |
| 程序重启恢复 | 重启后用相同`thread_id` 即可恢复完整对话历史                                 |
| 多会话隔离   | 不同`thread_id` 完全独立，可管理多个对话                                     |
| 完整状态保存 | 保存消息、工具调用链、中间变量(不止文本)                                       |
| 图暂停/恢复  | `ask_human` 触发 interrupt 后，用同一 `thread_id` 的 checkpoint 继续当前图 |

### 长期记忆写入流水线

长期记忆不再是"手动标记 important"的简单追加，而是**事件驱动 + LLM 抽取**的完整流水线。写入在 [memory/middleware.py](memory/middleware.py) 的 `ThreadMemoryWriteMiddleware` 中实现：

```
submit_event(thread_id, role, content, important)   # SessionManager 非阻塞调用
   ↓ 投递到防抖 buffer（同 thread 新事件重置 20s 计时）
   ↓ 限流：单 thread buffer 上限 30 条
   ↓
_a_run_pipeline()  （buffer 超时后批量处理）
   ↓
① 分类判定  judge_long_term_memory(MemoryInputEvent)
   ↓ 非 SKIP 的事件进入下一步
② LLM Fact 抽取  _a_extract_facts()
   ↓ 从原始消息提取 {"content", "category", "confidence"} JSON 列表
③ 无效内容过滤（空 content 丢弃）
④ 本地去重（与 Store 已存在的 fact content 比对，批次内去重）
⑤ 批量写入  save_facts_batch() → ThreadMemoryStore
   ↓
⑥ LRU 淘汰  prune_facts()（单 thread 超 50 条时淘汰最久未使用）
```

**事件来源**：SessionManager 消费 AgentEvent 流时调用：

```python
# session/manager.py（achat_stream / arun_stream 中）
if self._memory is not None:
    await self._memory.submit_user_message(tid, message, important=is_run_mode)
    # 执行过程中：
    if event.is_memory_worthy:          # DONE / TOOL_RESULT 事件
        await self._memory.consume_event(event)
```

- `is_run_mode=True`（`react:` 模式）→ 用户消息标记 `important=True`
- `DONE` / `TOOL_RESULT` 事件（`AgentEvent.is_memory_worthy`）→ 提交给记忆流水线评估
- `TOKEN` / `TOOL_CALL` / `INTERRUPT` 事件不单独提交

**分类判定**（[memory/models.py](memory/models.py) 的 `judge_long_term_memory`）：

| 分类                       | 取值          | 判定条件                                                  |
| -------------------------- | ------------- | --------------------------------------------------------- |
| `USER_FACT`              | `user_fact` | 用户告知的个人信息、习惯、偏好                            |
| `LESSON_EXPERIENCE`      | `lesson`    | 工具踩坑、稳定推理结论、不可行方案（同类失败 ≥2 次）     |
| `BUSINESS_ENTITY`        | `business`  | 项目配置、关键路径、接口、长期目标                        |
| `IMPORTANT_CONVERSATION` | `conv`      | 用户显式说"记住"、重要技术决策                            |
| `SKIP`                   | `skip`      | 临时资源 / 未确认猜想 / 一次性子任务 / 单次失败 → 不写入 |

> **LLM 抽取是第二道闸门**：分类判定只决定"值得评估"，最终是否入库由 LLM 从原始对话中抽取结构化事实决定，并附带 `category` 与 `confidence`（置信度），无效分类回退为 `conv`。

### 为什么 cot 不写入 checkpoint

`acot()` 方法直接调用 `self.llm.chat_with_history()`，**绕过了 `agent_executor`（LangGraph 执行）**，因此 checkpoint 不会记录。这是设计上的选择：

- **cot 的语义**：纯推理，不调用工具
- **避免被 Agent 拦截**：走 Agent 通道 LLM 可能自己决定调用工具，违背 cot 初衷
- **checkpoint 是为 Agent 设计的**：cot 没有工具调用链，也没有事件流，无需沉淀

**副作用**：cot 之间无法续接（第二次 cot 看不到第一次 cot 的对话），只能用 `session.aget_short_term()` 手动取历史作为上下文。

### 文件位置

由 [main.py](main.py) 配置 checkpoint 文件路径；长期记忆与会话工作目录映射均**复用同一 SQLite 文件**，按表隔离：

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(BASE_DIR, "data", "checkpoints_async.sqlite")
```

所有持久化数据集中在**单一 SQLite 文件** `data/checkpoints_async.sqlite` 中：

| 表 / 数据                    | 管理者               | 内容                                                               |
| ---------------------------- | -------------------- | ------------------------------------------------------------------ |
| `checkpoints` / `writes` | `AsyncSqliteSaver` | Agent 执行状态（消息 + 工具调用链，按`thread_id` 隔离）          |
| LangGraph Store 表           | `AsyncSqliteStore` | 长期记忆 facts（per-thread namespace 隔离，替代旧`memory.json`） |
| `session_workspaces`       | `WorkspaceStore`   | `session_id ↔ workspace_path` 映射（多会话工作目录隔离）        |

> **长期记忆存储**：长期记忆由 LangGraph `BaseStore`（`AsyncSqliteStore`，见 [memory/store.py](memory/store.py) 的 `ThreadMemoryStore`）管理，与 checkpoint 复用同一 SQLite 文件，按 `(thread_id, "thread_facts")` namespace 天然实现 per-thread 隔离。`compress` 命令现改为对该 Store 中的 facts 做摘要替换（`replace_with_summary`）。

> checkpoint 与 Store 各持有**独立的 `aiosqlite` 连接**（均启用 WAL 模式 + `busy_timeout=10000`），支持多进程（CLI / API / 调度器 / 飞书）并发读写。

> `data/` 父目录会在首次连接时**自动创建**，无需手动建文件夹。
>
> **查看 SQLite 内容**：可用 [DB Browser for SQLite](https://sqlitebrowser.org/dl/) 打开，或让 Agent 调用 `open_sqlite` 工具自动打开。

### 记忆管理命令

| 命令                        | 作用                                                    |
| --------------------------- | ------------------------------------------------------- |
| `clear` 或 `clear long` | 清空当前线程的长期记忆 facts（`MemoryManager.clear`） |
| `clear short`             | 清空当前会话(开启新 thread 替代删除)                    |
| `clear all`               | 全部清空（长期 facts + 新会话）                         |
| `compress` 或 `压缩`    | 压缩长期记忆(LLM 摘要后替换单个摘要 fact)               |
| `thread`                  | 方向键选择切换会话                                      |
| `thread:new`              | 开启新会话                                              |
| `thread:delete <id>`      | 删除指定会话(二次确认)                                  |

用 `info` 命令可查看记忆状态：

```
你: info
当前提供商: DeepSeek
当前模型:   deepseek-chat
API地址:    https://api.deepseek.com

--- 记忆状态 ---
当前会话:   thread-a4d099d2
Checkpoint: sqlite → D:\work\LangChainAgent\data\checkpoints_async.sqlite
已存消息:   8 条
长期记忆:   5 条
总会话数:   2
```

### 压缩长期记忆（compress）

随着对话不断积累，Store 中的 facts 会越来越多（单 thread 上限 50 条，超出 LRU 淘汰）。`compress` 命令通过 LLM 把所有 facts 压缩成一份摘要，再替换为**单条摘要 fact**。

> **注意**：compress 只压缩**长期记忆 facts**，不影响 checkpoint（完整对话历史保留在 `checkpoints_async.sqlite`）。

#### 工作流程

```
Store facts (N条原始记忆)
       ↓
MemoryManager.compress(thread_id) 拼接为文本 + 压缩提示词
       ↓
LLM 生成摘要（阻塞调用放入 asyncio.to_thread，不阻塞事件循环）
       ↓
ThreadMemoryStore.replace_with_summary(thread_id, summary)
       ↓
删除全部旧 facts → 写入 1 条摘要 fact（category=conv, confidence=1.0）
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
- 修复了 StructuredTool 同步调用问题
- ...
--- 已保存到长期记忆 Store ---
```

#### 压缩后的 Store 格式

摘要作为单条 `ThreadFactItem` 写入 Store（namespace `(thread_id, "thread_facts")`）：

| 字段           | 值                            |
| -------------- | ----------------------------- |
| `content`    | `[历史记忆摘要]\n{summary}` |
| `category`   | `conv`（重要对话）          |
| `confidence` | `1.0`                       |
| `fact_id`    | 新生成的 uuid hex             |

`MemoryManager.compress()` 返回 `{"success", "original_count", "original_chars", "compressed_chars", "summary"}`，其中保留压缩前的条数与字符数，便于追溯。

#### 特点与注意事项

| 特性              | 说明                                                               |
| ----------------- | ------------------------------------------------------------------ |
| 保留关键信息      | system prompt 要求 LLM 保留用户意图、决策、事实                    |
| 结构化输出        | 按主题分条，便于后续查阅                                           |
| 可追溯            | 返回`original_count` / `original_chars` / `compressed_chars` |
| 可多次压缩        | 再次`compress` 会对已有摘要 fact 再压缩                          |
| LLM 失败不丢数据  | 调用失败则原 facts 不变，不会替换                                  |
| **不可逆**  | 压缩后原 facts 无法恢复，重要数据可先用`export` 备份             |
| 不影响 checkpoint | 只压缩 Store facts，checkpoints_async.sqlite 保留完整历史          |

> 💡 **建议**：在长期记忆较多时（如 50 条上限附近）使用，平时少量记忆无需压缩。

### 记忆相关 API

记忆系统的对外接口分三层：**入口程序 → `MemoryContext`（工厂）→ `MemoryManager`（门面）→ `ThreadMemoryStore`（存储）**。SessionManager 是上层唯一调用方，Agent 不直接触碰记忆组件。

#### MemoryContext（统一工厂，入口程序使用）

| 成员                                 | 说明                                                                             |
| ------------------------------------ | -------------------------------------------------------------------------------- |
| `await MemoryContext.acreate(...)` | 异步创建全部记忆组件（checkpointer + Store + 锁池 + 读写中间件 + MemoryManager） |
| `ctx.checkpointer`                 | LangGraph checkpointer（传给`AgentCore` / `SessionRegistry`）                |
| `ctx.store`                        | LangGraph`BaseStore`（传给 `create_agent(store=...)`）                       |
| `ctx.read_middleware`              | `ThreadMemoryReadMiddleware`（传给 `extra_middleware`）                      |
| `ctx.thread_id`                    | 初始会话线程 ID                                                                  |
| `ctx.async_conn`                   | 异步 SQLite 连接（供 SessionRegistry 共享）                                      |
| `ctx.memory_manager`               | `MemoryManager` 实例（注入 SessionManager）                                    |
| `await ctx.aclose()`               | 刷新记忆 buffer、关闭中间件、释放 SQLite 连接                                    |

#### MemoryManager（统一门面，SessionManager 使用）

| 方法                                                               | 说明                                                                                    |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `await recall(thread_id, limit=None)`                            | 召回该 thread 的长期记忆 facts（按创建时间升序，截取最近 limit 条，默认 10）            |
| `await recall_text(thread_id, limit=None)`                       | 召回并格式化为文本片段（`【长期记忆】` 块，供注入 prompt）                            |
| `await submit_user_message(thread_id, content, important=False)` | 提交用户消息到写中间件（非阻塞，防抖）                                                  |
| `await consume_event(event)`                                     | 消费 AgentEvent（`is_memory_worthy` 的 DONE / TOOL_RESULT），提交到写中间件（非阻塞） |
| `await compress(thread_id)`                                      | 压缩该 thread 长期记忆（LLM 摘要替换为单条 fact）                                       |
| `await clear(thread_id)`                                         | 清空该 thread 全部 facts，返回清除数量                                                  |
| `await count_facts(thread_id)`                                   | 统计该 thread 的 fact 条数                                                              |
| `await flush_all()`                                              | 立即处理所有 buffer 事件（Agent 关闭前调用）                                            |
| `await shutdown()`                                               | 关闭写中间件（取消定时器、清理 buffer）                                                 |

#### ThreadMemoryReadMiddleware（读，LangGraph 中间件）

| 方法                                   | 说明                                                                                                             |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `awrap_model_call(request, handler)` | 每个 model 调用前从 Store 读取该 thread facts，组装为`【长期记忆】` 文本块追加到 SystemMessage，再调用 handler |

#### ThreadMemoryStore（存储封装，底层）

| 方法                                                     | 说明                                            |
| -------------------------------------------------------- | ----------------------------------------------- |
| `await query_facts(thread_id)`                         | 读取该 thread 全部 facts（按 create_time 升序） |
| `await save_fact / save_facts_batch(thread_id, items)` | 写入单条 / 批量 facts（按 fact_id 幂等覆盖）    |
| `await prune_facts(thread_id)`                         | LRU 淘汰（超 max_facts 后删除最久未使用）       |
| `await touch_fact(thread_id, fact_id)`                 | 更新 last_used_at（读取时非阻塞触发）           |
| `await clear_thread_memory(thread_id)`                 | 清空该 thread 全部 facts                        |
| `await replace_with_summary(thread_id, summary)`       | 删除全部 facts 后写入单条摘要 fact              |

#### SessionManager 委托接口（对外推荐入口）

| 方法                                | 说明                                                                    |
| ----------------------------------- | ----------------------------------------------------------------------- |
| `await aget_memory_summary()`     | 记忆状态统计（thread_id / checkpoint 消息数 / 长期记忆条数 / 总会话数） |
| `await acompress_memory()`        | 压缩当前线程长期记忆（阻塞 LLM 调用放入`asyncio.to_thread`）          |
| `await aclear_long_term_memory()` | 清空当前线程长期记忆                                                    |

---

## 会话管理（Session）

会话（Session / Thread）是对话隔离的基本单元，每个会话对应一个 `thread_id`（即 session_id），历史消息、工具调用链、执行历史、挂起中断等状态按会话隔离并持久化。早期会话管理内嵌在 `AgentMemory` 中，现已抽离为独立的 `session/` 模块（三层架构）：`SessionRegistry` 负责生命周期、`SessionStore` 负责瞬态状态、`SessionManager` 作为对外门面。

### 架构总览

```
session/
├── context.py          # SessionContext：单会话运行时上下文（session_id + config + checkpointer）
├── store.py            # SessionStore：基于 LangGraph Store 的 per-session 瞬态状态
├── registry.py         # SessionRegistry：会话生命周期管理（生成/查询/删除/消息读取）
├── workspace_store.py  # WorkspaceStore：session_id ↔ workspace_path 映射
└── manager.py          # SessionManager：对外门面 & 会话调度（封装 Agent + Memory）
```

| 组件                | 职责                                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `SessionContext`  | 单个会话的运行时上下文（session_id + LangGraph config + checkpointer）                                              |
| `SessionStore`    | 基于 LangGraph Store 的 per-session 瞬态状态：`execution_history`（执行历史）/ `pending_interrupts`（挂起中断） |
| `SessionRegistry` | 会话生命周期管理（生成/查询/删除/消息读取），桥接 checkpointer 与 Store                                             |
| `WorkspaceStore`  | `session_id ↔ workspace_path` 映射（多会话工作目录隔离）                                                         |
| `SessionManager`  | 对外门面 & 会话调度（封装 Agent + Memory），承接所有流式/并发/记忆调度                                              |

> **设计要点**：`AgentCore` 实例只持有不可变配置 + 共享的编译图 / Store / checkpointer，可安全在多会话间复用；所有会话级可变状态通过 `session_id` 显式隔离（`active_skills` 等放入 `LCAgentState` 随 checkpoint per-thread 持久化）。

### Session ID 生成

`SessionRegistry.generate_session_id()` 生成会话 ID，格式为 `[进程类型-][workflow-工作流名-]thread-8位随机`：

| 示例                              | 说明                                                                          |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `thread-a4d099d2`               | 普通会话                                                                      |
| `workflow-simple-thread-abc123` | 专属工作流会话（`workflow-{名称}-thread-{后缀}`）                           |
| `server-thread-...`             | 带`process_type` 前缀（多进程隔离，server/scheduler/feishu 各进程前缀不同） |

- `is_workflow_session(session_id)` / `workflow_name_of(session_id)`：判断并反解工作流会话。
- `current_session_id` 是 **CLI 单会话语义**的当前指针；**并发场景必须显式传 `session_id`**，不依赖该共享指针。

### 会话生命周期（SessionRegistry API）

`agent.session`（`AgentCore.session` 属性）暴露 `SessionRegistry`：

| 方法                                    | 说明                                                                               |
| --------------------------------------- | ---------------------------------------------------------------------------------- |
| `new_session(workspace_path=None)`    | 开启新会话（原会话保留在数据库），更新 current_session_id                          |
| `new_workflow_session(workflow_name)` | 开启专属工作流会话（ID 含`workflow-{名称}`）                                     |
| `aswitch_session(session_id)`         | 切换到指定会话（恢复历史 + warm workspace 缓存）                                   |
| `adelete_session(session_id)`         | 删除会话：清 checkpoint + Store + workspace 绑定（删当前会话时自动切换到其他会话） |
| `alist_sessions(all_types=False)`     | 列出所有可见会话（checkpoint 存量 ∪ 当前会话）                                    |
| `asummarize(session_id)`              | 返回会话统计（session_id / 消息数 / 总会话数）                                     |
| `aget_messages(session_id)`           | 从 checkpoint 获取该会话所有消息                                                   |
| `aget_short_term(session_id, limit)`  | 取最近 N 条消息转 dict 格式（兼容旧 API）                                          |
| `aexport_session(session_id, fmt)`    | 导出会话为可读文本（`text`）或 Markdown（`markdown`）                          |
| `aclose()`                            | 关闭 checkpointer 持有的 SQLite 连接                                               |

### 工作空间绑定（Workspace）

每个会话可绑定一个工作空间目录（文件与执行类工具被限制在该目录内），由 `WorkspaceStore` 持久化：

| 方法                                      | 说明                                                                  |
| ----------------------------------------- | --------------------------------------------------------------------- |
| `aset_workspace(path, session_id=None)` | 设置/修改会话工作空间（缓存 + DB 双写）                               |
| `aget_workspace(session_id=None)`       | 获取会话工作空间路径（缓存优先，未命中查 DB）                         |
| `aclear_workspace(session_id=None)`     | 清除会话工作空间绑定（`True`=原绑定存在已清除）                     |
| `awarm_workspace(session_id)`           | 进程重启后从 DB 加载 workspace 到缓存，使`get_context()` 同步读命中 |

> 路径校验（`_validate_workspace_path`）：必须为**已存在的目录**的绝对路径；禁止宿主根目录、用户主目录、系统目录（Windows `C:\Windows` / `System32`）。

CLI 命令 `workspace` / `workspace <路径>` / `workspace:clear` 即对应上述 API；`get_context()` 读取 `SessionContext` 供每次图执行注入。

HTTP API（[api/server.py](api/server.py)）同样暴露三个 RESTful 端点，供 Web 前端调用：

| 方法       | 路径                                   | 对应 CLI             | 说明                                                     |
| ---------- | -------------------------------------- | -------------------- | -------------------------------------------------------- |
| `GET`    | `/api/threads/{thread_id}/workspace` | `workspace`        | 查询绑定（`workspace` 字段为 `null` 表示未绑定）     |
| `POST`   | `/api/threads/{thread_id}/workspace` | `workspace <路径>` | 设置/修改绑定，body`{"path": "..."}`；路径非法返回 400 |
| `DELETE` | `/api/threads/{thread_id}/workspace` | `workspace:clear`  | 清除绑定，返回`{"cleared": true/false}`                |

### SessionManager 门面

`AgentCore.session_manager`（懒初始化）是上层流量的统一入口，封装 Agent + Memory：

- **流式接口**：`achat_stream` / `aresume_stream` / `arun_stream`，产出 SSE 事件（`token` / `tool_call` / `tool_result` / `interrupt` / `cancelled` / `error` / `done`）
- **非流式接口**：`achat` / `aresume` / `arun`，收集全部 token 为最终文本
- **会话管理委托**：`new_session` / `new_workflow_session` / `set_current_session` / `current_session_id` / `alist_sessions` / `aswitch_session` / `adelete_session` / `aget_messages` / `aexport_session` / `asummarize`
- **记忆管理委托**：`aget_memory_summary` / `acompress_memory` / `aclear_long_term_memory`
- **执行历史**：`aget_execution_history` / `aclear_history`
- **上下文压缩**：`manually_compact(force, thread_id)`
- **生命周期**：`aclose()`（刷新记忆 buffer + 释放 Agent 资源）

**per-thread 并发锁**（`SessionManager._thread_locks`）：

- 同一 `thread_id` 的请求持有同一把 `asyncio.Lock`，严格串行，保证 checkpoint 读写一致；
- 不同 `thread_id` 各自独立锁，可同时流式对话互不阻塞。

> ⚠️ **初始化顺序**：首次访问 `session_manager` 前必须先调用 `set_memory_manager()` 注入 MemoryManager，否则 SessionManager 将在**无记忆功能**下初始化（重试会抛 `RuntimeError`）。

### 与 CLI 的关系

| 命令                                        | 作用                                                                                   |
| ------------------------------------------- | -------------------------------------------------------------------------------------- |
| `thread`                                  | 方向键菜单切换会话（显示首条用户消息预览 + 消息数；Enter 切换、Ctrl+D 删除、Esc 取消） |
| `thread:new`                              | 开启新会话（原会话保留，可随时切回）                                                   |
| `thread:delete <id>`                      | 删除指定会话（二次确认，不可恢复；删当前会话自动切换到其他会话）                       |
| `export` 或 `export:<thread_id> [路径]` | 导出对话为 Markdown（默认存`exports/`）                                              |
| `workspace <路径>` / `workspace:clear`  | 绑定/清除当前会话工作空间                                                              |

> **存储位置**：会话历史在 `data/checkpoints_async.sqlite` 的 `checkpoints` / `writes` 表（按 `thread_id` 隔离）；工作空间映射在 `session_workspaces` 表；执行历史与挂起中断写入 LangGraph Store（瞬态，随会话删除清理）。详见[记忆系统 → 文件位置](#文件位置)。

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

| 工具               | 文件                                              | 功能                                                                                                                                                                                                                                                                                                                                                     | 参数                                                                                         |
| ------------------ | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `search`         | [tools/search.py](tools/search.py)                 | 联网搜索(Tavily API)                                                                                                                                                                                                                                                                                                                                     | `query`, `num_results`, `search_depth`                                                 |
| `read_file`      | [tools/file_tool.py](tools/file_tool.py)           | 读取文件                                                                                                                                                                                                                                                                                                                                                 | `file_path`                                                                                |
| `write_file`     | [tools/file_tool.py](tools/file_tool.py)           | 写入文件                                                                                                                                                                                                                                                                                                                                                 | `file_path`, `content`, `mode`                                                         |
| `calculate`      | [tools/calculator.py](tools/calculator.py)         | 数学计算                                                                                                                                                                                                                                                                                                                                                 | `expression`                                                                               |
| `run_shell`      | [tools/terminal_tools.py](tools/terminal_tools.py) | 执行 shell 命令                                                                                                                                                                                                                                                                                                                                          | `command`, `cwd`, `timeout`                                                            |
| `run_python`     | [tools/terminal_tools.py](tools/terminal_tools.py) | 执行 Python 脚本文件                                                                                                                                                                                                                                                                                                                                     | `file_path`, `script_args`, `cwd`, `timeout`                                         |
| `run_cmd`        | [tools/terminal_tools.py](tools/terminal_tools.py) | 执行 Shell / .bat / .ps1 脚本文件                                                                                                                                                                                                                                                                                                                        | `file_path`, `script_args`, `cwd`, `timeout`                                         |
| `get_local_time` | [tools/get_local_time.py](tools/get_local_time.py) | 获取本地时间                                                                                                                                                                                                                                                                                                                                             | 无                                                                                           |
| `open_file`      | [tools/open_file.py](tools/open_file.py)           | 用系统默认/指定程序打开文件或文件夹                                                                                                                                                                                                                                                                                                                      | `file_path`, `app_path`                                                                  |
| `open_sqlite`    | [tools/open_file.py](tools/open_file.py)           | 用 DB Browser for SQLite 打开 .sqlite/.db                                                                                                                                                                                                                                                                                                                | `file_path`                                                                                |
| `read_skill`     | [tools/skill_tool.py](tools/skill_tool.py)         | 读取本地技能(SKILL.md)的指引正文                                                                                                                                                                                                                                                                                                                         | `skill_name`(可空)                                                                         |
| `create_tool`    | [tools/create_tools.py](tools/create_tools.py)     | 动态生成工具代码并保存为 .py 文件（默认保存到 tools/ 目录并自动注册到 tools/__init__.py；tool_logic 支持含 f-string/多行字符串的代码，内容行不会被误缩进）。安全边界：工具名须为合法 Python 标识符、路径限制在 tools/ 目录内禁止逃逸、默认禁止覆盖已有文件（`force=True` 可覆盖）、生成源码经 AST 校验禁止导入 os/subprocess/socket 等高风险模块 | `tool_name`, `tool_description`, `args_spec`, `tool_logic`, `tool_path`, `force` |
| `ask_human`      | [cli/human_input.py](cli/human_input.py)           | 暂停 LangGraph 图并请求人工结构化选择                                                                                                                                                                                                                                                                                                                    | `prompt`, `choices`                                                                      |

> `open_sqlite` 会自动查找 DB Browser for SQLite 路径（环境变量 `SQLITE_BROWSER_PATH` → 常见安装位置 → `shutil.which`），找不到则返回下载链接。Linux 下安装 `sqlitebrowser` 包即可使用。

### 2. MCP 工具（MCP Server Tools）

通过 **MCP（Model Context Protocol）** 协议从独立进程加载，支持动态增删。

#### 已注册的 MCP Server

| Server 名称    | 工具数 | 传输方式 | 说明                                                |
| -------------- | ------ | -------- | --------------------------------------------------- |
| `filesystem` | -      | stdio    | 可选官方文件系统服务（需手动添加/启用，需 Node.js） |
| `fetch`      | -      | stdio    | 可选官方网页抓取服务（需手动添加/启用，需 Node.js） |

#### MCP 配置文件

[config/mcp_servers.json](config/mcp_servers.json) 定义所有 MCP Server：

```json
{
    "servers": {}
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

Agent 可自动执行终端命令与文件操作，为防止破坏性操作，内置**命令分类 + 两级路径保护**策略（由 [tools/safety.py](tools/safety.py) 实现）：

#### 命令分类

| 级别                            | 规则                                                                                                                                 | 行为                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| **BLOCKLIST（始终拒绝）** | `format`、`mkfs`、`dd if=`、`shutdown`、`fork bomb`、`:(){`、``curl\|sh``/``wget\|sh``、编码执行的 PowerShell 等灾难性命令 | 直接拦截，返回拒绝错误（不论路径）                     |
| **CONFIRM（需确认）**     | `rm`、`rm -rf`、`sudo`、`chmod`、`mv`、`kill`、`Remove-Item`、脚本执行等危险但可能需要的命令                           | 根据**路径分类**决定：保护级→拒绝，询问级→确认 |

> **重要变更**：`rm -rf`、`Remove-Item -Recurse -Force` 等递归删除命令已从 BLOCKLIST 移至 CONFIRM，通过**路径分类系统**实现精细化保护。

#### 两级路径保护

对于文件操作命令（`rm`、`del`、`mv`、`chmod` 等），系统会提取命令中的路径并分类为两个级别：

| 级别             | 说明                                               | 包含路径                                                                                                          | 行为                                              |
| ---------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **保护级** | 项目核心文件/目录 + 系统关键目录，禁止任何修改操作 | 系统目录（`C:\Windows`、用户主目录）、项目核心（`agent/`、`config/`、`main.py`、`requirements.txt` 等） | CONFIRM 命令 →**拒绝**，普通命令 → 允许   |
| **询问级** | 可操作但需要用户确认的路径                         | `tests/`、`tools/`、`docs/`、`README.md`、`.venv/`、缓存目录等                                          | CONFIRM 命令 →**需确认**，普通命令 → 允许 |

**决策矩阵**：

```
命令类型          保护级路径        询问级路径        普通路径
─────────────────────────────────────────────────────
BLOCKLIST         ❌ deny          ❌ deny          ❌ deny
CONFIRM           ❌ deny          ⚠️ confirm       ⚠️ confirm
普通命令          ✅ allow         ✅ allow         ✅ allow
```

**示例**：

- `rm -rf agent/` → ❌ **拒绝**（保护级路径）
- `rm -rf tests/` → ⚠️ **需确认**（询问级路径）
- `rm -rf /tmp/foo` → ⚠️ **需确认**（普通路径 + 危险命令）
- `ls agent/` → ✅ **允许**（保护级路径 + 普通命令）

#### 配置（config/safety.json）

```json
{
    "mode": "blacklist",
    "confirm_dangerous": true,
    "blacklist": [],
    "whitelist": ["echo", "dir", "ls", "python", "pip", "git", "cat", "type"],
    "path_protection": {
        "enabled": true,
        "protected_paths": [
            "{system_root}",
            "{user_home}",
            "{project_root}",
            "{project_root}/agent",
            "{project_root}/config",
            "..."
        ],
        "confirm_paths": [
            "{project_root}/tests",
            "{project_root}/tools",
            "{project_root}/docs",
            "{project_root}/README.md",
            "..."
        ]
    }
}
```

**字段说明**：

| 字段                                | 说明                                                                 |
| ----------------------------------- | -------------------------------------------------------------------- |
| `mode`                            | `blacklist`(默认) 或 `whitelist`                                 |
| `confirm_dangerous`               | 是否对危险命令交互确认                                               |
| `blacklist`                       | 追加的拒绝正则（与内置 BLOCKLIST 合并）                              |
| `whitelist`                       | 白名单模式下允许的首命令                                             |
| `path_protection.enabled`         | 是否启用路径保护（默认`true`）                                     |
| `path_protection.protected_paths` | 保护级路径列表（支持占位符`{project_root}`、`{system_root}` 等） |
| `path_protection.confirm_paths`   | 询问级路径列表（支持占位符）                                         |

**占位符**：

- `{project_root}` - 项目根目录（`E:\work\LCAgent`）
- `{system_root}` - 系统根目录（Windows: `C:\Windows`）
- `{user_home}` - 用户主目录（`C:\Users\username`）

**路径匹配规则**：

1. **最长匹配原则**：更具体的规则优先生效
   - 例如：`{project_root}` 是保护级，`{project_root}/tests` 是询问级 → `tests/` 目录匹配到询问级
2. **前缀匹配**：规则会匹配其所有子路径
   - 例如：`{project_root}/config` 保护 → `config/safety.json` 也受保护
3. **Windows 大小写不敏感**：`agent/` 和 `AGENT/` 视为同一路径

#### 交互命令

| 命令                                  | 作用                                         |
| ------------------------------------- | -------------------------------------------- |
| `safety`                            | 查看当前安全策略（模式 / 确认开关 / 黑名单） |
| `safety:mode <blacklist\|whitelist>` | 切换模式                                     |
| `safety:confirm <on\|off>`           | 开关危险命令确认                             |

#### 故障排查

**问题 1：合法操作被误拦截**

- **症状**：`rm tests/temp.txt` 被拒绝，但 `tests/` 应该是询问级
- **原因**：可能 `tests/` 没有在 `confirm_paths` 中，或被父路径（如 `{project_root}`）的保护级规则覆盖
- **解决**：
  1. 检查 `config/safety.json` 中的 `path_protection` 配置
  2. 确认 `tests/` 在 `confirm_paths` 中
  3. 确认 `protected_paths` 中没有更具体的规则覆盖（如 `{project_root}/tests` 也在保护级）

**问题 2：路径保护不生效**

- **症状**：删除保护级路径时没有被拒绝
- **原因**：`path_protection.enabled` 可能为 `false`，或配置文件格式错误
- **解决**：
  1. 检查 `config/safety.json` 语法是否正确（JSON 格式）
  2. 确认 `path_protection.enabled: true`
  3. 运行 `safety` 命令查看当前策略是否正确加载

**问题 3：如何临时允许危险操作**

- **方法 1**：将目标路径从 `protected_paths` 移到 `confirm_paths`（需重启程序）
- **方法 2**：使用 `safety:confirm off` 暂时关闭确认（不推荐，风险高）
- **方法 3**：在确认提示时输入 `y` 或 `yes` 允许执行

### 5. 工具调用机制

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

### 6. System Prompt 强化

为确保 LLM 主动调用工具而非拒绝，system prompt 中明确要求：

```
1. 当用户要求创建文件、读写文件、创建目录等操作时，你【必须】调用相应工具
2. 绝对不要回复'我无法访问你的文件系统'、'请你自己保存'之类的话
3. 你确实拥有这些工具的能力，工具会在用户本地执行
4. 创建文件、脚本、文件夹默认位置是 ./tests/
5. 如果用户要保存内容到文件，直接调用 write_file 工具
6. 如果用户要创建目录，直接调用终端工具
7. 测试/运行脚本时直接调用终端工具
8. 危险命令会被安全策略拦截或要求确认
9. 专业任务应优先用 read_skill 读取相关技能指引
10. 需要人工确认、选择或补充信息时，应调用 ask_human 并提供结构化 choices
```

### 7. 扩展工具

#### 方式 1：添加本地工具

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

#### 方式 2：添加 MCP Server

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

#### 方式 3：动态生成工具

除手动编写外，Agent 还可通过内置元工具 [`create_tool`](tools/create_tools.py) **动态生成新的本地工具**：它按统一模板生成 `@tool` 装饰器工具源码，保存为 `.py` 文件，并自动注册到 `tools/__init__.py`。

**参数说明：**

| 参数                 | 类型           | 说明                                                                            |
| -------------------- | -------------- | ------------------------------------------------------------------------------- |
| `tool_name`        | `str`        | 工具函数名，必须匹配`[A-Za-z][A-Za-z0-9_]*` 且不能以下划线开头                |
| `tool_description` | `str`        | 工具文档字符串，说明能力、用途、适用场景                                        |
| `args_spec`        | `str`        | 参数定义说明，分号`;` 分隔，每项格式 `参数名:参数类型=参数说明`             |
| `tool_logic`       | `str`        | 工具主体业务逻辑（函数体内部实现代码，不要写函数定义/装饰器）                   |
| `tool_path`        | `str \| None` | 保存路径，可传目录或`.py` 文件；为空默认保存到 `tools/` 下 `tool_name.py` |
| `force`            | `bool`       | 是否覆盖已存在文件，默认`False` 禁止覆盖                                      |

`args_spec` 示例：`file_path:str=本地文件路径;encoding:str=utf-8文件编码，可选`

**返回结构：**

```python
{
    "success": True,           # 是否生成成功
    "tool_name": "read_markdown_file",
    "source_code": "...",      # 生成的完整源码
    "file_path": "...",        # 保存的文件路径
    "registered": True,        # 是否已自动注册到 tools/__init__.py
    "message": "工具代码已保存到 ...，并已注册到 tools/__init__.py"
}
```

失败时返回 `{"success": False, "error": "工具生成失败：...", ...}`。

**调用示例：**

```
create_tool(
    tool_name="read_markdown_file",
    tool_description="读取本地 Markdown 文件并返回文本内容，用于文档解析",
    args_spec="file_path:str=本地 Markdown 文件路径;encoding:str=utf-8文件编码，可选",
    tool_logic="""with open(file_path, 'r', encoding=encoding) as f:
    result = f.read()"""
)
```

生成后代码自动保存到 `tools/read_markdown_file.py` 并注册到 `tools/__init__.py` 的 `all_tools`，重启后 Agent 即可调用。

**安全边界：**

- 工具名必须是合法 Python 标识符且不能以下划线开头
- 保存路径限制在 `tools/` 目录内，禁止路径逃逸
- 生成源码经 `ast.parse` 语法校验
- 禁止导入高风险模块：`os`、`subprocess`、`socket`、`sys`、`pathlib`、`shutil`、`ctypes`、`importlib`、`requests`、`urllib`
- 默认禁止覆盖已有文件（`force=True` 才可覆盖）

#### 本地工具 vs MCP 工具对比

| 特性       | 本地工具          | MCP 工具                      |
| ---------- | ----------------- | ----------------------------- |
| 定义方式   | `@tool` 装饰器  | `@mcp.tool()` + FastMCP     |
| 运行位置   | Agent 同进程      | 独立子进程                    |
| 加载方式   | import 后直接使用 | 通过 MCP 协议动态加载         |
| 增删       | 需改代码 + 重启   | 改配置 +`mcp:reload` 热加载 |
| 跨语言支持 | 仅 Python         | 任意语言（Node/Go 等）        |
| 适用场景   | 简单、轻量工具    | 复杂服务、第三方集成          |

---

## 可观测性与可靠性

在核心 Agent 之上，项目提供一层**可观测性与可靠性基础设施**，覆盖长上下文压缩、MCP 连接管理、运行时指标、结构化日志、工具超时与统一异常层次。

### 长上下文压缩中间件（Compaction）

[`agent/compaction.py`](agent/compaction.py) 实现了 `LCAgentCompactionMiddleware`，采用**三层压缩策略**，在会话过长时无损释放大量 token：

1. **增量摘要**：已有 `state.summary` + 旧消息 → 更新后的 summary（避免每次全量重做摘要）
2. **工具输出 Prune**：把保留区中过长的历史工具输出替换为占位符（`[工具输出已裁剪 N→M 字符] ...`），工具输出常占 70%+ token
3. **保留近期 N 条原始消息**：不做任何修改，确保近期上下文完整

关键设计：

- 摘要存入 LangGraph `state.summary` 字段，随 **checkpoint 自动持久化**，天然实现 **per-thread 隔离**（每个 thread 拥有独立 summary），彻底消除跨会话污染。
- **安全切割**：不会拆开 `AIMessage(tool_calls)` + `ToolMessage` 配对（切割点落在 `ToolMessage` 上时向前回退到对应的 `AIMessage`）。
- 压缩后用 `RemoveMessage(REMOVE_ALL_MESSAGES)` 先清空 checkpoint 旧消息，再写入 `SystemMessage(摘要) + Pruned 近期消息`，旧消息彻底移除不再占用存储。

触发方式：

| 触发方式 | 入口                                                | 阈值行为                                              |
| -------- | --------------------------------------------------- | ----------------------------------------------------- |
| 自动     | `before_model` / `abefore_model` 中间件         | 消息数 >`max_messages`（默认 50）时触发             |
| 手动     | `AgentCore.manually_compact()` / `compact` 命令 | `force=True` 跳过阈值，仍需消息数 > `keep_recent` |

`CompactionConfig` 关键参数（`agent/compaction.py`）：

| 参数                      | 默认 | 说明                           |
| ------------------------- | ---- | ------------------------------ |
| `max_messages`          | 50   | 触发压缩的消息数阈值           |
| `keep_recent`           | 20   | 保留最近 N 条消息（原样保留）  |
| `max_tool_output_chars` | 200  | 工具输出超过此长度则触发 Prune |
| `tool_prune_preview`    | 100  | Prune 后保留的预览字符数       |

> 可用 `CompactionConfig.from_kwargs(max_context_messages, context_trim_keep)` 从 AgentCore 现有配置参数构建。

### MCP 连接池（MCPPool）

[`tools/mcp_pool.py`](tools/mcp_pool.py) 提供 `MCPPool`，替代旧的 `load_mcp_tools` 全量重载模式：

- **per-server 隔离**：`server-A` 断连不影响 `server-B/C`
- **健康探测**：感知连接状态（`ServerStatus`：`disconnected` / `connecting` / `connected` / `error`）
- **自动重连**：连接断开后可 `reconnect()` 恢复
- **工具缓存 + 连接复用**：连接对象持久化，`reload_server(name)` 只重连单个 server 并重新拉取工具，无需全量重载

```python
pool = MCPPool(config_file)
await pool.initialize()            # 启动时并行连接所有已启用 server（单个失败不阻塞其他）
tools = pool.get_all_tools()       # 聚合所有已连接 server 的工具
await pool.reload_server("name")  # 重连单个 server
infos = pool.get_server_infos()    # 各 server 的 ServerInfo（状态/工具数/最后错误等）
await pool.close()                 # 关闭所有连接
```

### 运行时指标（Metrics）

[`agent/metrics.py`](agent/metrics.py) 的 `MetricsCollector` 挂在 `AgentCore.metrics` 上，**线程安全**，追踪三类核心指标：

| 类别     | 记录方法                                               | 汇总内容                                               |
| -------- | ------------------------------------------------------ | ------------------------------------------------------ |
| LLM 调用 | `record_llm_call` / `extract_and_record_llm_usage` | 次数、prompt/completion/total tokens、按 provider 分组 |
| 工具执行 | `record_tool_call`                                   | 次数、耗时（min/max/avg）、失败/超时次数，按工具名分组 |
| 压缩统计 | `record_compaction`                                  | 触发次数、压缩前后消息数、节省消息数、摘要长度         |

- token 优先从 `AIMessage.response_metadata` 的 `usage_metadata` 提取，缺失时用字符数 `/4` 粗估（`estimate_tokens`）。
- `get_summary()` 返回结构化字典；`reset()` 清空所有指标。
- CLI 用 `metrics` / `metrics:reset` 查看与重置。

### 结构化日志（Logging）

[`agent/logging_config.py`](agent/logging_config.py) 提供 **trace_id / thread_id 上下文注入**，基于 `contextvars` 实现 asyncio 安全传递：

```python
from agent.logging_config import setup_logging, TraceContext

setup_logging(level=logging.INFO)  # 程序入口调用一次（幂等），日志输出到 stderr（不污染 stdout 工具输出）

with TraceContext(trace_id="req-123", thread_id="thread-abc"):
    logger.info("processing")  # 自动附带 [trace:req-123] [thread:thread-abc]
```

日志格式：`2026-08-05 14:30:00 [INFO ] [agent.agent_core] [trace:abc123] [thread:t-1] 消息内容`。未设置上下文时显示 `-` 占位；`setup_logging` 支持可选 `log_file` 落盘。

### 工具超时保护（Tool Timeout）

[`tools/tool_wrapper.py`](tools/tool_wrapper.py) 为本地工具和 MCP 工具叠加统一超时保护，防止 Agent 因工具卡死而永久阻塞：

- 在工具 `_arun` 上叠加 `asyncio.wait_for`（同步工具先 `to_thread` 再加超时）
- **超时后返回 JSON 错误消息**（`{"error": "tool_timeout", ...}`）而非抛异常，让 Agent 能继续推理
- 优先级：`NO_TIMEOUT_TOOLS` > `TOOL_TIMEOUTS`（按工具名覆盖）> 全局 `DEFAULT_TIMEOUT`（60 秒）

默认覆盖：`ask_human` 600s、`schedule_task` 120s、`search` 90s。

### 统一异常层次

[`agent/exceptions.py`](agent/exceptions.py) 定义分层异常，便于上层精准 `catch`：

```
LCAgentError                    ← 所有 LCAgent 异常的基类（含 detail 字段）
├── MCPConnectionError          ← MCP server 连接失败/断连（含 server_name）
├── ToolTimeoutError            ← 工具执行超时（含 tool_name / timeout）
├── CompressError               ← 上下文压缩失败（含 stage）
├── InterruptTimeoutError       ← 中断会话超时（含 thread_id）
└── AgentStateError             ← AgentCore 状态错误（如已关闭后调用）
```

---

## 异步 Public API

`AgentCore` 提供全套异步公开方法（[`agent/agent_core.py`](agent/agent_core.py)），供飞书远程控制、调度器等异步入口调用；CLI 命令层亦已全面迁移到异步 API。

| 方法                                                                                                  | 说明                                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `await arun(task)`                                                                                  | Agent 模式执行任务，返回最终文本                                                                                                                                               |
| `await achat(message)`                                                                              | 普通对话模式，返回最终文本                                                                                                                                                     |
| `await aresume(payload)`                                                                            | 恢复被`ask_human` 中断的会话（`Command(resume=...)`）                                                                                                                      |
| `await arun_structured(task, thread_id=None)` / `await achat_structured(message, thread_id=None)` | 返回`AgentTurnResult`（含 HITL 结构化中断信息）；`thread_id` 显式指定目标会话                                                                                              |
| `await aswitch_llm(llm_client)`                                                                     | 运行时切换 LLM 提供商/模型                                                                                                                                                     |
| `await arebuild_from_team_dir(agent_name, *, task="")`                                              | 按`team/<角色>/` 文件夹名切换主对话 Agent 的角色（读取该目录的 `agent_config.json` + `AGENT.md`，仅提示词变化时不重建 Graph，provider/model 变化时重建 LLM 与 executor） |
| `await areload_mcp_tools()`                                                                         | 通过 MCP 连接池重载工具并按需重建 Graph                                                                                                                                        |
| `await manually_compact(force=False, thread_id=None)`                                               | 手动触发上下文压缩，返回状态更新字典或`None`；`thread_id` 指定目标会话                                                                                                     |
| `await aclose()`                                                                                    | 释放资源（MCP 连接、checkpoint 等）的生命周期收尾                                                                                                                              |

### 并发多会话（Per-thread 并发）

`thread_id` 显式贯穿整个异步调用链（`arun_structured` / `achat_structured` / `aresume_structured` / `astream_chat` / `astream_resume` / `manually_compact`），配合 HTTP 服务端实现**真正的并发多会话**：

| 层                 | 隔离机制                                                                                                                                                        |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 锁                 | 普通对话 / 恢复 / 压缩走**per-thread 锁**（`api/server.py::_thread_lock`），不同会话互不阻塞、同会话严格排队；管理型命令/会话切换仍走全局 `chat_lock` |
| 编译图（executor） | `_executor_for(thread_id)` 为每个线程缓存独立 compiled graph + `SystemMessage`（LRU 上限 `_MAX_THREAD_EXECUTORS=50`），技能提示词互不覆盖                 |
| 中断状态           | `_pending_interrupts` 按 thread_id 记录 HITL 挂起中断，恢复/清理只作用于目标线程，杜绝跨会话串线                                                              |
| config             | `_config_for(thread_id)` 构建含 `configurable.thread_id` 的 LangGraph config，`thread_id=None` 时兼容无参旧调用                                           |

行为要点：

- **同会话串行**：同一 `thread_id` 的请求持有同一把锁，仍严格按到达顺序执行，保证 checkpoint 读写一致。
- **跨会话并发**：不同 `thread_id` 各自持有独立锁，可同时流式对话，互不阻塞。
- **缓存淘汰**：超过 `_MAX_THREAD_EXECUTORS` 时淘汰最久未使用的线程图；运行中的流持有旧 executor 对象引用，不受淘汰影响。

---

## 定时任务调度（Scheduler）

让 Agent 能"定闹钟"——用户说"明天下午3点生成报告"或"每天9点发送日报"，Agent 登记任务后直接回复，后台调度器在时间到达时自动唤起 Agent 执行。

核心思路是**拆分逻辑 A（对话理解）与逻辑 B（时间调度）**，通过 SQLite 解耦：

```
对话阶段（Agent 进程）                   后台调度（独立进程）
──────────────────                      ──────────────────
用户："明天下午3点生成报告"                 调度器轮询 pending + 到期任务
  │                                          │
  ▼                                          ▼
Agent 解析意图，计算 execute_time        claim_task 原子抢占
  │                                          │
  ▼                                          ▼
schedule_task 工具入库                   ThreadPoolExecutor 并发执行
  │                                          │
  ▼                                          ▼
SQLite (status=pending) ◄── 共享库 ───► AgentCore.run()
  │
  ▼
回复"任务已登记"
```

### 工具接口

Agent 在对话中通过以下 `@tool` 函数与调度系统交互：

| 工具                       | 作用                                | 关键参数                                                                                                                                    |
| -------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `schedule_task`          | 登记一次性/周期任务                 | `task_text`（自然语言描述，不要含代码）、`task_type`（`one_time`/`periodic`）、`execute_time`（ISO 8601）、`cron_expr`（5字段） |
| `list_scheduled_tasks`   | 查询任务列表                        | `status`（可选：`pending`/`running`/`done`/`failed`/`cancelled`）                                                               |
| `cancel_scheduled_task`  | 取消 pending 状态的任务             | `task_id`                                                                                                                                 |
| `delete_scheduled_task`  | 删除已完成的单个任务                | `task_id`                                                                                                                                 |
| `cleanup_finished_tasks` | 批量清理 done/failed/cancelled 任务 | 无                                                                                                                                          |

### 快速开始

**1. 初始化工具依赖**

在 `main.py` 启动时调用一次（也可不调，使用默认 DB 路径）：

```python
from tools.scheduler_tool import configure

configure(db_path="data/scheduled_tasks.sqlite")
```

**2. 启动后台调度器**（独立进程，与主对话进程分离）

```bash
python -m scheduler.run
```

**3. 在对话中登记任务**

Agent 会自动处理，无需记忆参数。示例：

```
你: 2分钟后帮我生成一个简介文本放到 tests 目录
助手: 任务已登记，将于 2026-07-29T17:48:57 自动执行。

你: 每天9点整理 tests 目录
助手: 周期任务已登记（cron: 0 9 * * *），将按计划自动执行。
```

### 模块结构

```
scheduler/
├── store.py       # SQLite CRUD + 原子抢占 + 重试
├── executor.py    # agent_factory → AgentCore.acreate/arun/aclose 异步执行桥接
                    # （在单个事件循环内完成构造+执行+释放，适配 AsyncSqliteSaver 绑定创建时 loop 的约束）
├── engine.py      # APScheduler 引擎（轮询一次性 + cron 周期 + 线程池）
└── run.py         # 独立进程入口（make_agent_factory 返回 async factory）
```

> **异步执行**：`make_agent_factory` 是 `async` 工厂，`_run_agent_task` 在同一个事件循环内完成
> `await AgentCore.acreate()` → `await agent.arun()` → `await agent.aclose()`。因为
> `AsyncSqliteSaver` 绑定创建它的事件循环，跨 loop 使用会挂起，因此整个 task 生命周期不可拆到不同线程。

详细技术文档见 [scheduler/README.md](scheduler/README.md)。

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

定义在 [cli/human_input.py](cli/human_input.py)，是一个 `StructuredTool`，LLM 把它当作普通工具调用：

| 参数        | 类型             | 说明                         |
| ----------- | ---------------- | ---------------------------- |
| `prompt`  | `str`          | 展示给用户的提问文本         |
| `choices` | `list[Choice]` | 结构化选项列表（id + label） |

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

| 名称                           | 作用                                                                                                                        |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `AgentTurnResult`            | 一轮图执行结果，`status` 为 `completed` / `interrupted` / `cancelled`；完成时读 `output`，暂停时读 `interrupts` |
| `run_structured(task)`       | ReAct/任务模式入口，返回`AgentTurnResult`，完成后写 checkpoint + 长期记忆                                                 |
| `chat_structured(message)`   | 普通对话入口，返回`AgentTurnResult`，完成后写 checkpoint，不写长期记忆                                                    |
| `resume_structured(payload)` | 对暂停中的图调用`Command(resume=payload)`，继续同一个 checkpoint 线程                                                     |

`AgentTurnResult` 是 frozen dataclass，三种状态的语义：

| 状态            | 触发条件                           | 后续动作                            |
| --------------- | ---------------------------------- | ----------------------------------- |
| `completed`   | LLM 生成最终回复，无 interrupt     | 读`output`，按模式写记忆          |
| `interrupted` | `ask_human` 触发 interrupt       | 读`interrupts`，收集答案后 resume |
| `cancelled`   | 用户在安全护栏确认中拒绝了危险命令 | 读`output`（取消提示），不 resume |

### CLI 编排

[cli/human_input.py](cli/human_input.py) 提供了一组辅助函数，把「invoke → 渲染 → 收集 → resume」封装成循环：

> **全异步**：以下辅助函数均为 `async`（`await agent.arun_structured(...)` 等），CLI 入口
> `main.py` 以 `asyncio.run(main())` 建立**单个长驻事件循环**，REPL、命令分发
> （`dispatch_command`）、HITL 循环全部在该 loop 内完成，不自行创建/关闭临时事件循环。

| 函数                                                   | 作用                                                  |
| ------------------------------------------------------ | ----------------------------------------------------- |
| `run_human_input_loop(agent, message, render, read)` | 启动对话并持续恢复 interrupt，直到本轮完成            |
| `run_structured_until_completion(agent, message)`    | ReAct 模式版本，同上                                  |
| `chat_until_completion(agent, message)`              | 普通对话模式的默认入口                                |
| `complete_human_input_turn(turn, render, read)`      | 循环处理 interrupt 直到`status == "completed"`      |
| `render_human_interrupt(interrupt)`                  | 把 interrupt payload 渲染成终端提示                   |
| `read_human_resume(interrupt)`                       | 用`select_menu` 收集结构化选择，返回 resume payload |

#### 结构化选择的展示与收集

`read_human_resume` 的逻辑：

1. 校验 interrupt 是否为 `human_choice`，否则回退到自由文本输入 `{"text": "..."}`
2. 从 `choices` 提取 `(label, id)` 对，调用 [cli/cli_menu.py](cli/cli_menu.py) 的 `select_menu(prompt, options)` 展示方向键菜单
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

| 机制                   | 触发主体              | 暂停层级                | 恢复方式                    | 中间状态保留           |
| ---------------------- | --------------------- | ----------------------- | --------------------------- | ---------------------- |
| **普通多轮聊天** | 用户主动发问          | 无暂停，每轮独立        | 用户发送下一条消息          | 仅靠 checkpoint 历史   |
| **HITL**         | LLM 调用`ask_human` | 图执行暂停（interrupt） | `Command(resume=payload)` | 保留完整工具调用链     |
| **安全护栏确认** | 工具执行前拦截        | 工具内部抛异常          | 抛异常，整轮取消，不 resume | 工具调用被标记为 error |

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

| 限制                          | 说明                                                                                                     |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| **不跨进程恢复**        | CLI 只负责正在运行进程内的结构化选择与恢复；重启后未完成的人工选择界面不会自动重建                       |
| **重启后可读历史**      | 重启后仍可通过`thread` 菜单恢复历史对话上下文，但图执行不会自动续接                                    |
| **同一 AgentCore 实例** | `resume_structured` 要求在同一个 `AgentCore` 实例中调用，跨实例恢复不支持                            |
| **跨线程拒绝**          | `resume_structured` 校验 `thread_id`，跨线程恢复会抛 `ValueError`                                  |
| **与 cot 无关**         | `cot()` 模式不走 LangGraph，不会触发 interrupt；HITL 仅对 `run_structured`/`chat_structured` 生效  |
| **测试覆盖**            | [tests/test_human_input.py](tests/test_human_input.py) 覆盖了 interrupt、resume、并行选择、线程隔离等场景 |

---

## 交互命令参考

| 命令                                        | 说明                                                                         |
| ------------------------------------------- | ---------------------------------------------------------------------------- |
| `react:任务`                              | Agent 模式，自动调用工具，打印步骤，存长期记忆                               |
| `cot:任务`                                | 链式思考模式，纯推理不调用工具                                               |
| `switch:提供商名`                         | 运行时切换 LLM 提供商（如`switch:deepseek`）                               |
| `model`                                   | 方向键选择切换当前提供商的模型                                               |
| `model:<name>`                            | 直接切换模型（如`model:glm-4-flash`）                                      |
| `help`                                    | 查看完整命令说明                                                             |
| `info`                                    | 查看当前模型和记忆状态(含 thread_id、会话数)                                 |
| `tools`                                   | 查看可用工具列表（含 MCP 工具）                                              |
| `clear [long\|short\|all]`                  | 清理记忆（默认 long）                                                        |
| `compress` 或 `压缩`                    | 压缩长期记忆（LLM 摘要后替换原内容）                                         |
| `compact`                                 | 手动压缩当前会话上下文（增量摘要 + 工具输出 Prune，`force=True` 跳过阈值） |
| `metrics` 或 `metrics:status`           | 查看运行时指标（LLM 调用 / 工具执行 / 压缩统计）                             |
| `metrics:reset`                           | 重置所有运行时指标                                                           |
| `log`                                     | 查看当前日志级别（方向键选择切换）                                           |
| `log:<级别>`                              | 直接切换日志级别（debug\|info\|warning\|error\|critical）                    |
| `thread`                                  | 方向键选择切换会话(显示消息数预览);`Enter` 切换, `Ctrl+D` 删除高亮会话   |
| `thread:new`                              | 开启新会话(原会话保留)                                                       |
| `thread:delete <id>`                      | 删除指定会话(二次确认,不可恢复)                                              |
| `mcp`                                     | 查看 MCP Server 状态                                                         |
| `mcp:reload`                              | 重新加载 MCP 工具                                                            |
| `mcp:add <name> ...`                      | 添加 MCP Server                                                              |
| `mcp:remove <name>`                       | 删除 MCP Server                                                              |
| `mcp:toggle <name> <on\|off>`              | 启用/禁用 MCP Server                                                         |
| `skill` 或 `skills`                     | 查看所有本地可用技能                                                         |
| `skill:<name>`                            | 将某技能加载进当前会话(注入 system prompt)                                   |
| `skill:<name> <任务>`                     | 加载技能并立即以 Agent 模式执行该任务(如`skill:git-commit 提交README`)     |
| `skill:clear`                             | 清空手动加载的技能                                                           |
| `role` 或 `roles`                       | 方向键选择切换团队角色(扫描`team/` 下的可用角色)                           |
| `role:<name>`                             | 直接切换到指定团队角色(如`role:manager`)                                   |
| `role:<name> <任务>`                      | 切换角色并立即以 Agent 模式执行该任务                                        |
| `safety`                                  | 查看当前安全策略                                                             |
| `safety:mode <blacklist\|whitelist>`       | 切换安全模式                                                                 |
| `safety:confirm <on\|off>`                 | 开关危险命令确认                                                             |
| `workflow`                                | 列出可用的多 Agent 工作流                                                    |
| `workflow:<name> <任务>`                  | 运行指定工作流（如`workflow:simple 帮我分析项目结构`）                     |
| `workspace`                               | 查看当前会话绑定的工作空间目录                                               |
| `workspace <路径>`                        | 为当前会话绑定/修改工作空间（文件与执行类工具将被限制在该目录内）            |
| `workspace:clear`                         | 清除当前会话的工作空间绑定                                                   |
| `workspace:help`                          | 显示 workspace 命令帮助                                                      |
| `export` 或 `export:<thread_id> [路径]` | 导出对话为 Markdown(默认存`exports/`)                                      |
| `json:<任务>`                             | 让 Agent 以 JSON 对象返回结果并解析展示                                      |
| `quit` / `exit`                         | 退出                                                                         |
| 其他输入                                    | 普通对话模式（也支持工具调用，但不打印步骤）                                 |

---

## 多 Agent 工作流

项目支持基于 LangGraph 的多 Agent 团队协作，采用**监督者模式**（Supervisor Pattern）编排。

### 架构(simple)

```
用户任务
    ↓
Manager (拆解任务,生成计划)
    ↓
Worker (执行子任务)
    ↓
Terminator (汇总结果,返回最终答案)
```

### 团队角色

| Agent                     | 职责           | 工具能力                   | 配置                 |
| ------------------------- | -------------- | -------------------------- | -------------------- |
| **ManagerAgent**    | 任务拆解与规划 | 纯文本推理(无工具)         | `team/manager/`    |
| **WorkerAgent**     | 执行具体子任务 | 工具模式(注入全部本地工具) | `team/worker/`     |
| **TerminatorAgent** | 汇总结果并返回 | 纯文本推理(无工具)         | `team/terminator/` |

**轻量设计**：团队 Agent 继承 `TeamAgent` 轻量基类,不继承 `AgentCore`。相比完整智能体:

- **无会话记忆/checkpoint**:TeamAgent 自身单轮执行、不管理历史(但工作流会在外层注入当前会话记忆,见「记忆注入机制」;跨轮次记忆由工作流图 checkpointer 承载,见「异步化与跨轮次压缩」)
- **按需工具注入**:Manager/Terminator 纯 LLM 推理,Worker 注入工具列表后用 `create_agent` 构建轻量 ReAct 循环
- **快速构建**:不加载 MCP Server、不扫描技能目录、不创建 SQLite checkpointer
- **能力边界清晰**:规划/汇总角色不暴露危险工具(如 `run_shell`),Worker 才拥有工具执行能力
- **自带 LLM 配置**:每个 agent 的 `agent_config.json` 里配置 `provider` + `model`,TeamAgent 内部创建 LLMClient
- **可定制 LLM 采样参数**:子类可通过类属性或 `__init__` 参数覆盖 `temperature`/`max_tokens`(如 WorkerAgent 用 `temperature=0.3` 提升执行确定性、`max_tokens=4096` 放宽输出上限)

### 异步化与跨轮次压缩

工作流节点已全面异步化,并具备技能注入与跨轮次记忆压缩能力(全部实现在 `graph/common.py`,TeamAgent 零改动):

- **异步节点执行**:`simple.py` / `pipline.py` 的四个业务节点(`summarize`/`manager_plan`/`worker_exec`/`terminator_final`)全部为 `async`。节点经 `ainvoke_team_agent()` 执行团队 Agent——优先调用 `agent.ainvoke()`(若未来 TeamAgent 提供异步接口);否则用 `asyncio.to_thread()` 包装同步 `invoke()`,避免阻塞事件循环,多 Agent 并行不互相阻塞。
- **技能注入(SkillInjector)**:`build_simple_workflow` 接受 `skills_dir` / `auto_match_skills` 参数,构建时创建 `SkillInjector`(复用 `tools.skills.SkillManager` 的确定性打分匹配 + 指引块渲染)。节点渲染 prompt 后调用 `inject_into_prompt()` 把命中技能(`match_skills(task)`)的指引块追加到 prompt 末尾,已含技能块时跳过(防重复)。
- **跨轮次记忆压缩**:`arun_simple_workflow` / `arun_pipline_workflow` 接受 `thread_id` 与 `max_history_chars` 参数。当图编译时注入 checkpointer 且传入 `thread_id`,运行前 `_aget_previous_workflow_summary()` 从 checkpoint 读取上一轮状态(`task`/`plan`/`worker_result`/`final_answer`),超长截断为摘要(默认 6000 字符)拼入 `raw_context`(`【上一轮工作流记录】` 块),实现多轮运行间的上下文延续与压缩。无 checkpointer / 无历史 / 读取失败时静默降级,不影响运行。

### 状态隔离机制

**设计原则**：工作流状态通过外层 `WorkflowState` 显式传递(`plan`/`worker_result`/`final_answer`),每次运行用独立 `thread_id`,不同运行互不干扰。

### 运行进度跟踪

工作流运行期间可实时感知节点执行进度（CLI 打印 + Web 前端节点高亮）：

- **节点级回调**：`arun_simple_workflow` 接受可选 `on_node_start` / `on_node_end` 回调（接收节点名）。内部通过 LangGraph 的 `config["callbacks"]` 注入 `NodeTrackingHandler`（位于 `graph/common.py`），利用节点执行时 `metadata["langgraph_node"]` 字段识别业务节点（哨兵节点与内部 agent 子图会被过滤），在节点开始/结束/异常时触发回调。不传回调时零额外开销。
- **CLI 场景**：`run_workflow` 把节点状态打印到终端（`▸ 节点开始: manager_plan` / `✓ 节点完成: manager_plan`）。
- **Web 场景**：`CommandContext.workflow_event_cb` 把结构化事件（`workflow_node` / `workflow_status`）经 `/api/chat` 的 SSE 流实时推送；服务端将管理型命令的 `dispatch_command` 放到后台线程执行、输出经 `asyncio.Queue` 实时转发，前端 `WorkflowView` 据此高亮节点卡片与流程图。

### 记忆注入机制

工作流把当前 CLI 会话的记忆注入任务上下文,并在结束后写回,形成记忆闭环:

1. **记忆提取**:`run_workflow` 调用 `build_memory_context` 从当前会话提取短期记忆（checkpoint）与长期记忆（`MemoryManager.recall_text`，经 `session_manager.memory` 访问）,拼装为文本(超 `MAX_RAW_CONTEXT_CHARS` 自动截断)。
2. **Manager 总结分发**:工作流首个节点 `summarize` 调用 `ManagerAgent.summarize_context` 把原始记忆提炼成上下文摘要,存入 `WorkflowState.context_summary`。下游仅 `manager_plan` 与 `terminator_final` 节点注入该摘要,`worker_exec` 不注入(靠 plan 承接记忆)。
3. **写回闭环**:工作流结束后,`_record_workflow_result` 把 `workflow:<name> <task>`(HumanMessage)与 `final_answer`(AIMessage)写入当前会话 checkpoint。

### 工作流提示词外置

工作流各节点/记忆提炼的提示词不再硬编码在代码里,而是由各角色 `team/*/AGENT.md` 的 `## workflow:<名称>` 小节驱动(与角色系统提示词同文件)。改 prompt 只改 md,无需动代码:

| 小节                              | 所在文件                     | 用途                                               |
| --------------------------------- | ---------------------------- | -------------------------------------------------- |
| `## workflow:manager_plan`      | `team/manager/AGENT.md`    | Manager 制定执行计划的用户消息模板(内含记忆摘要段) |
| `## workflow:summarize_context` | `team/manager/AGENT.md`    | `summarize_context` 的 system prompt             |
| `## workflow:worker_exec`       | `team/worker/AGENT.md`     | Worker 执行子任务的用户消息模板                    |
| `## workflow:terminator_final`  | `team/terminator/AGENT.md` | Terminator 汇总结果的用户消息模板(内含记忆摘要段)  |

模板用 `{task}`/`{plan}`/`{worker_result}`/`{context_summary}` 占位,运行时以 `TeamAgent.render_template` 做字符串替换(即使模板含 JSON 花括号也不会报错)。各节点在需要时经 `TeamAgent.get_template(name)` 加载对应小节(与系统提示词共用 __init__ 的一次解析缓存,不重复读文件),缺失回退各角色类的 `default_templates` 默认模板。加载/解析逻辑见 `team/base.py`;角色系统提示词在 `TeamAgent.__init__` 自动经 `parse_prompt_sections` 从 `prompt_file` 解析并剥离工作流小节,避免模板混入 system prompt(显式传入 `system_prompt` 时优先)。

### 使用方式

#### 1. CLI 命令

```bash
# 列出可用工作流
你: workflow

# 运行 simple 工作流
你: workflow:simple 帮我分析 LCAgent 项目的目录结构
```

输出示例：

```
构建工作流: simple
初始化团队 Agent(Manager/Worker/Terminator)...
工作流 simple 构建完成

执行任务: 帮我分析 LCAgent 项目的目录结构
--------------------------------------------------
▸ 节点开始: summarize
✓ 节点完成: summarize
▸ 节点开始: manager_plan
✓ 节点完成: manager_plan
▸ 节点开始: worker_exec
✓ 节点完成: worker_exec
▸ 节点开始: terminator_final
✓ 节点完成: terminator_final

==================================================
工作流执行完成
==================================================

项目包含以下主要模块：
- agent/: Agent 核心与配置
- team/: 多 Agent 团队
- graph/: 工作流编排
- tools/: 本地工具 + MCP
...
```

#### 2. 编程接口

```python
import asyncio
from graph.registry import build_workflow
from graph.simple import arun_simple_workflow

# 方式1: 构建并运行(异步接口)
graph, agents = build_workflow("simple")
result = asyncio.run(arun_simple_workflow(graph, "帮我分析项目结构"))
print(result["final_answer"])

# 可选: 手动传入记忆上下文(经 Manager 总结后注入 plan/final 节点)
result = asyncio.run(arun_simple_workflow(
    graph, "帮我分析项目结构",
    raw_context="用户: 之前聊过项目背景",
))

# 可选: 节点进度回调(节点开始/结束时触发,接收节点名,用于进度跟踪)
result = asyncio.run(arun_simple_workflow(
    graph,
    "帮我分析项目结构",
    on_node_start=lambda node: print(f"节点开始: {node}"),
    on_node_end=lambda node: print(f"节点完成: {node}"),
))

# 可选: 跨轮次记忆压缩(需 checkpointer 编译的图 + 显式 thread_id,
# 同一 thread 多轮运行,第二轮自动注入【上一轮工作流记录】)
from langgraph.checkpoint.memory import MemorySaver
graph2, agents2 = build_workflow("simple", checkpointer=MemorySaver())
r1 = asyncio.run(arun_simple_workflow(graph2, "第一轮任务", thread_id="wf-1"))
r2 = asyncio.run(arun_simple_workflow(graph2, "第二轮任务", thread_id="wf-1"))

# 方式2: 通过 CLI 层封装(带打印提示)
from cli.commands.workflow import run_workflow
from cli.commands.types import CommandContext

# 需要构造 CommandContext(简化示例,实际使用中从 main.py 获取)
result = run_workflow(context, "simple", "帮我分析项目结构")
```

### 扩展工作流

#### 1. 添加新 Agent（team 添加 agent）

在 `team/<role>/` 下创建 `__init__.py` 无需,只需 3 个文件:

```python
# team/my_agent/my_agent.py
from graph.registry import register_agent
from team.base import TeamAgent
from tools import all_tools

@register_agent("my_agent", "team/my_agent/agent_config.json", tools=all_tools)
class MyAgent(TeamAgent):
    temperature = 0.3
    max_tokens = 4096
    default_templates = {"my_node": "模板..."}
```

```
# team/my_agent/AGENT.md        — 角色系统提示词 + ## workflow:小节
# team/my_agent/agent_config.json — LLM 配置(provider/model/prompt_file 等)
```

然后在 `team/__init__.py` 中添加导入即可,`@register_agent` 装饰器会在模块加载时自动注册。

#### 2. 添加新工作流（register_workflow 注册）

所有工作流统一通过 `graph.registry.register_workflow` 注册（唯一入口）,仅调用时机不同:

**方式 A — 模块自注册（推荐,内置工作流采用）:**

在 `graph/` 下新建工作流文件,实现 `build_xxx_workflow(agents)` 构建函数与 `run_xxx_workflow` 运行器,文件末尾调用 `register_workflow` 完成自注册（参照 `graph/simple.py` 与 `graph/pipline.py`）:

```python
# graph/my_workflow.py
from graph.registry import register_workflow

def build_my_workflow(agents: dict) -> StateGraph:
    my_agent = agents["my_agent"]
    builder = StateGraph(MyWorkflowState)
    builder.add_node("step1", lambda state: step1_node(state, my_agent))
    builder.add_edge(START, "step1")
    builder.add_edge("step1", END)
    return builder.compile()

# 文件末尾自注册:模块被 import 时写入全局注册表
register_workflow(
    name="my_flow",
    builder=build_my_workflow,
    runner=run_my_workflow,          # 可选,缺失时回退到 arun_simple_workflow
    roles=["my_agent"],              # 可选,声明依赖角色(仅构建这些角色)
    description="我的自定义工作流",  # 可选,CLI 列表展示用
)
```

`graph/registry.py` 底部 `_load_builtin_workflows()` 在 registry 首次 import 时加载 `graph.simple` / `graph.pipline`,触发其自注册;新增内置工作流时在该函数中补充 import 即可。

**方式 B — 动态注册（运行时添加,无需改源码,适合插件式/条件式工作流）:**

```python
from graph.registry import register_workflow

register_workflow(
    name="my_flow",
    builder=build_my_workflow,
    runner=run_my_workflow,          # 可选,缺失时回退到 arun_simple_workflow
    roles=["my_agent"],              # 可选,声明依赖角色(仅构建这些角色)
    description="我的自定义工作流",  # 可选,CLI 列表展示用
)
```

#### register_workflow 参数说明

| 参数            | 必填 | 说明                                                                             |
| --------------- | ---- | -------------------------------------------------------------------------------- |
| `name`        | 是   | 工作流名称(CLI 以`workflow:<name> <任务>` 调用)                                |
| `builder`     | 是   | 构建函数`build_xxx(agents: dict) -> StateGraph`                                |
| `runner`      | 否   | 运行函数`run_xxx(graph, task, ...) -> dict`,缺失时回退 `run_simple_workflow` |
| `roles`       | 否   | 声明依赖角色列表,仅构建这些角色;缺失时构建全部已注册角色                         |
| `description` | 否   | 工作流描述,CLI`workflow` 列表展示用                                            |

CLI 会自动识别新工作流：`workflow:my_flow <任务>`。

---

## 运行示例

### 1. 启动与基础对话

运行 `python main.py`，选择提供商后进入交互界面：

```
==================================================
  LangChain Agent (基于LangChain框架)
==================================================

选择 LLM 提供商
  (↑↓ 选择, Enter 确认, Esc 取消)
    [✓] zhipu      (智谱AI)
        qwen       (通义千问)
  ❯   [✓] deepseek   (DeepSeek)
        kimi       (Kimi (Moonshot))
----------------------------------------

[MCP] 已加载 0 个工具

Agent 已就绪！
当前提供商: DeepSeek
当前模型:   deepseek-chat

本地工具: search, read_file, write_file, calculate, run_shell, ...
MCP工具:
```

### 2. 三种执行模式

**ReAct 模式**（自动调用工具，打印中间步骤，存长期记忆）：

```
你: react:计算 (123 + 456) * 2
--- 步骤 1 ---
工具: calculate
输入: {'expression': '(123 + 456) * 2'}
结果: 1158
最终答案: (123 + 456) * 2 = 1158
```

**CoT 推理模式**（纯推理，不调用工具）：

```
你: cot:分析Python和Java的区别
最终答案: Python和Java的区别在于...
```

**普通对话**（自动判断是否调用工具，不打印步骤）：

```
你: 帮我创建一个叫 test 的文件夹
助手: 已为你创建文件夹 test，路径：D:\work\LangChainAgent\test
```

### 3. 模型与状态管理

切换模型：

```
你: model
当前提供商 [DeepSeek] 可用模型:
  (↑↓ 选择, Enter 确认, Esc 取消)
    deepseek-chat
  ❯ deepseek-reasoner
----------------------------------------
已切换模型: deepseek-reasoner (提供商: DeepSeek)
```

查看当前状态：

```
你: info
当前提供商: DeepSeek
当前模型:   deepseek-reasoner
API地址:    https://api.deepseek.com

--- 记忆状态 ---
当前会话:   thread-a4d099d2
Checkpoint: sqlite → D:\work\LangChainAgent\data\checkpoints_async.sqlite
已存消息:   8 条
长期记忆:   2 条
总会话数:   1
```

### 4. 会话管理

```
你: thread:new
已开启新会话: thread-c7e8f1a3
原会话 thread-a4d099d2 已保留,可用 'thread' 切回

你: thread
选择会话 (共 2 个,↑↓ 选择,Enter 切换)
  (↑↓ 选择, Enter 确认, Esc 取消)
    thread-a4d099d2  [8 条消息]
  ❯ thread-c7e8f1a3  [0 条消息] (当前)
----------------------------------------
```

### 5. Human-in-the-loop（HITL）

Agent 调用 `ask_human` 暂停图执行，等待人工选择后继续：

```
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
```

### 6. 对话导出

```
你: export
已导出当前会话到 exports/thread-a4d099d2.md

你: export:thread-b1dc3b2a 我的导出/backup.md
已导出会话 thread-b1dc3b2a 到 我的导出/backup.md
```

由 [session/registry.py](session/registry.py) 的 `SessionRegistry.aexport_session()` 实现，将 checkpoint 中的对话渲染为可读 Markdown。

### 7. JSON 模式

```
你: json:帮我规划一个Python项目的目录结构，输出JSON
{
  "project": "my_project",
  "structure": [
    {"name": "src", "type": "dir", "children": [
      {"name": "__init__.py", "type": "file"},
      {"name": "main.py", "type": "file"}
    ]},
    {"name": "tests", "type": "dir"},
    {"name": "README.md", "type": "file"}
  ]
}
```

Agent 按要求**只输出一个合法 JSON 对象**（不含 ``` 标记与解释文字），执行后自动用 `LLMClient.extract_json` 解析并美化打印；解析失败则回退显示原始输出。适用于需要结构化返回（如生成配置、报表、API 响应）的场景。

### 8. MCP 管理

```
你: mcp
MCP Servers:
------------------------------------------------------------
  [✓启用] fetch (stdio)
           npx -y @modelcontextprotocol/server-fetch
------------------------------------------------------------
已加载 MCP 工具数: 0
```

### 9. 退出

```
你: quit
再见!
```

> **重启程序后**：用 `thread` 方向键菜单选择历史会话，恢复之前的对话上下文。

---

## 代码使用示例

```python
import asyncio
from memory import MemoryContext
from agent import AgentCore
from agent.llm_client import LLMClient

async def main() -> None:
    # 创建客户端
    llm = LLMClient(provider="deepseek", config_file="config/llm_config.json")

    # 三层架构：先创建 MemoryContext（记忆基础设施），再创建 AgentCore（纯执行内核）
    memory_ctx = await MemoryContext.acreate(
        checkpoint_file="data/checkpoints_async.sqlite",  # Checkpoint + 长期记忆 Store（同一 SQLite 文件）
        llm_getter=lambda: llm,                            # 供 LLM 抽取/压缩记忆使用（支持热切换）
        short_term_size=10,
        buffer_delay_seconds=20,                           # 记忆防抖窗口（秒）
        max_buffer_messages=30,                            # 单 thread 防抖 buffer 上限
        max_facts_per_thread=50,                           # 单 thread 最大 fact 条数（LRU 淘汰）
        recall_limit=10,                                   # 召回默认条数上限
    )

    agent = await AgentCore.acreate(
        llm_client=llm,
        name="LCAgent",                                    # Agent 名称（默认 LCAgent）
        max_iterations=15,
        mcp_config_file="config/mcp_servers.json",
        enable_mcp=True,
        skills_dir=".agents/skills",
        auto_match_skills=True,
        max_context_messages=0,                           # 0=关闭长上下文裁剪
        context_trim_keep=12,
        checkpointer=memory_ctx.checkpointer,             # ← checkpoint 持久化
        store=memory_ctx.store,                           # ← 长期记忆 Store
        extra_middleware=[memory_ctx.read_middleware],    # ← 长期记忆读取中间件
        initial_thread_id=memory_ctx.thread_id,
        async_conn=memory_ctx.async_conn,
    )
    agent.set_memory_manager(memory_ctx.memory_manager)   # ← 注入 MemoryManager（SessionManager 懒初始化时接收）

# 内置变量：agent.name 与 agent.llm（LLM 是 Agent 的内置变量）
    print(agent.name)   # -> LCAgent
    print(agent.llm.get_info())  # 直接通过 agent.llm 访问当前 LLM 客户端

    # 普通对话（自动写 checkpoint；事件驱动沉淀长期记忆，经防抖 + LLM 抽取后入库）
    response = await agent.achat("帮我创建一个叫 test 的文件夹")

    # Agent模式（自动调用工具，打印步骤，写 checkpoint；用户消息标记 important，DONE 事件标记 is_important）
    result = await agent.arun("计算 123 * 456 并把结果写入 result.txt")

    # CoT模式（纯推理，不调用工具，不写 checkpoint，不沉淀长期记忆）
    result = agent.cot("分析机器学习的应用场景")

    # Human-in-the-loop 结构化入口：achat_structured/arun_structured 返回 AgentTurnResult
    turn = await agent.achat_structured("需要人工选择时请先问我")
    while turn.is_interrupted:
        # 单个 interrupt 可直接 resume；并行多个 interrupt 要按 interrupt.id 提交映射
        if len(turn.interrupts) == 1:
            resume_body = {"choice_id": "approve"}
        else:
            resume_body = {
                interrupt.id: {"choice_id": "approve"}
                for interrupt in turn.interrupts
            }
        # CLI 中由 select_menu 生成选择；Esc 会向 Agent 发送 {"cancelled": True}
        turn = await agent.aresume_structured(resume_body)
    print(turn.output)

    # 切换LLM提供商
    new_llm = LLMClient(provider="qwen", config_file="config/llm_config.json")
    await agent.aswitch_llm(new_llm)

    # 切换模型(同一提供商内)
    llm.switch_model("glm-4-flash")    # 切到 glm-4-flash
    print(llm.list_models())           # 查看当前提供商的可用模型
    await agent.aswitch_llm(llm)       # 重建 Agent 以使用新模型

    # 会话管理（已迁移至 agent.session，异步接口）
    agent.session.new_session()                            # 开启新会话
    await agent.session.aswitch_session("thread-abc123")   # 切换到已有会话
    await agent.session.adelete_session("thread-xxx")      # 删除指定会话
    print(await agent.session.alist_sessions())            # 列出所有会话
    print(await agent.session.aexport_session(fmt="markdown"))  # 导出当前会话为 Markdown 文本

    # 记忆管理（经 SessionManager 委托 MemoryManager）
    await agent.session_manager.aclear_long_term_memory()   # 清空当前线程长期记忆
    print(await agent.session.asummarize())    # 查看会话统计(含 session_id、消息数)

    # 压缩长期记忆（LLM 摘要后替换为单条 fact）
    result = await agent.session_manager.acompress_memory()
    print(f"压缩率: {1 - result['compressed_chars']/result['original_chars']:.1%}")

    # MCP 工具管理
    await agent.areload_mcp_tools()  # 重新加载 MCP 工具
    print(agent.get_available_tools())  # 查看所有工具

    # 技能管理
    print(agent.list_skills())       # 列出所有本地技能
    await agent.aload_skill("git-commit")  # 手动加载技能到当前会话
    await agent.aclear_skills()      # 清空手动加载的技能
    agent.set_auto_match(False)      # 关闭任务自动匹配

    await agent.aclose()             # 释放异步资源
    await memory_ctx.aclose()        # 刷新记忆 buffer、关闭中间件、释放 SQLite 连接

asyncio.run(main())
```

## 运行时配置

项目所有外置配置均位于 `config/` 目录下，每个配置文件有对应的 `.example` 模板（不含真实密钥），适合纳入版本控制。

### 1. `agent_config.json` — Agent 运行时参数

原先硬编码在 `main.py` 的运行时参数已外置到此文件，由 [agent/config.py](agent/config.py) 的 `load_agent_config` 加载并与默认值合并（缺省键不报错）。

| 键                        | 类型 | 默认值                      | 说明                                                          |
| ------------------------- | ---- | --------------------------- | ------------------------------------------------------------- |
| `name`                  | str  | `LCAgent`                 | Agent 名称（可通过`agent.name` 访问）                       |
| `max_iterations`        | int  | 15                          | 单次`invoke` 最大推理步数（即 `recursion_limit`）         |
| `skills_dir`            | str  | `.agents/skills`          | 技能目录（相对项目根或绝对路径）                              |
| `auto_match_skills`     | bool | true                        | 任务自动匹配并注入相关技能                                    |
| `enable_mcp`            | bool | true                        | 是否加载 MCP 工具                                             |
| `memory_size`           | int  | 10                          | 短期消息窗口大小（传递给`SessionRegistry.aget_short_term`） |
| `verbose`               | bool | true                        | 是否打印详细过程                                              |
| `mcp_config_file`       | str  | `config/mcp_servers.json` | MCP 配置文件（相对项目根或绝对路径）                          |
| `max_context_messages`  | int  | 0                           | 长上下文裁剪阈值（0 = 关闭）                                  |
| `context_trim_keep`     | int  | 12                          | 裁剪时保留的最近消息条数                                      |
| `max_execution_history` | int  | 100                         | 执行历史最大条数                                              |
| `agent_prompt_file`     | str  | `agent/AGENT.md`          | Agent 核心提示词文件路径（相对项目根或绝对路径）              |
| `tool_timeout`          | int  | 120                         | 工具调用超时（秒）                                            |

**Memory 层配置**（同文件同名键透传，默认值见 [memory/config.py](memory/config.py)）：

| 键                              | 类型 | 默认值 | 说明                                   |
| ------------------------------- | ---- | ------ | -------------------------------------- |
| `memory_buffer_delay_seconds` | int  | 20     | 记忆写入防抖延迟（秒）                 |
| `memory_max_buffer_messages`  | int  | 30     | 防抖 buffer 最大消息数（超出强制刷新） |
| `memory_max_facts_per_thread` | int  | 50     | 单线程最大 fact 条数（超出 LRU 淘汰）  |
| `memory_recall_limit`         | int  | 10     | 召回长期记忆时的默认条数上限           |
| `session_enable_memory`       | bool | true   | SessionManager 是否启用长期记忆处理    |

修改后重启 `main.py` 即可生效。

#### Agent 核心提示词（`agent/AGENT.md`）

Agent 的核心系统提示词（行为规则）已从 `agent_config.json` 中拆分到独立的 [agent/AGENT.md](agent/AGENT.md) 文件，便于单独维护和版本控制。

加载优先级：

1. **`agent/AGENT.md`**（优先，由 `agent_prompt_file` 指定路径）
2. **内置默认提示词**（fallback，当文件不存在或为空时使用）

自定义 Agent 行为规则时，直接编辑 `agent/AGENT.md` 即可，无需修改代码或 JSON 配置。

> **注意**：`agent_prompt_file` 指定的 AGENT.md 同时也承载工作流提示词模板（`## workflow:*` 小节）。构建 TeamAgent 时（`team/factory.py`）只需传 `prompt_file`，`TeamAgent.__init__` 会自动经 `parse_prompt_sections` 剥离这些小节并解析出 system prompt，避免模板内容混入（详见「工作流提示词外置」）。

#### 长上下文裁剪（Long-Context Trimming）

当某个会话的消息数超过 `max_context_messages` 时，Agent 会自动：

1. 用 LLM 将较早的消息压缩成一份中文摘要；
2. 开启**新会话**，并把摘要注入后续 system prompt（保留上下文精华）；
3. 仅保留最近 `context_trim_keep` 条消息，从而避免撞上 LLM 上下文窗口。

> 触发时会在终端打印提示（含新旧 `thread_id`）。默认 `max_context_messages=0`（关闭），需要时在 `agent/agent_config.json` 中设一个合理值（如 60）即可开启。

### 2. `llm_config.json` — LLM 服务商配置

定义多个 LLM 服务商的接入信息，由 [LLMClient](agent/llm_client.py) 按名称引用加载。

```json
{
  "providers": {
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
    "api_key": "tvly-..."
  }
}
```

| 字段                        | 类型     | 说明                                                    |
| --------------------------- | -------- | ------------------------------------------------------- |
| `providers`               | object   | 服务商字典，键为唯一标识（如`deepseek`），值为配置项  |
| `providers.{id}.name`     | string   | 显示名称                                                |
| `providers.{id}.base_url` | string   | OpenAI 兼容 API 地址                                    |
| `providers.{id}.env_key`  | string   | 环境变量名（`api_key` 为空时回退读取该变量）          |
| `providers.{id}.model`    | string   | 默认模型                                                |
| `providers.{id}.models`   | string[] | 可选模型列表（交互式切换使用，如`chat`/`reasoner`） |
| `providers.{id}.api_key`  | string   | API 密钥（留空则从`env_key` 读取）                    |
| `tavily`                  | object   | 联网搜索配置；`api_key` 值或 `env_key` 字段名       |

> **安全提醒**：`.example` 文件中不含真实密钥，提交代码前请确保 `llm_config.json` 在 `.gitignore` 中。

### 3. `mcp_servers.json` — MCP 服务器配置

定义 MCP（Model Context Protocol）服务器，由 `agent_config.json` 的 `mcp_config_file` 指向。

```json
{
  "servers": {}
}
```

| 字段                         | 类型        | 说明                         |
| ---------------------------- | ----------- | ---------------------------- |
| `servers`                  | object      | 服务器字典，键为服务器名     |
| `servers.{name}.transport` | `"stdio"` | 传输协议（目前仅支持 stdio） |
| `servers.{name}.command`   | string      | 启动命令                     |
| `servers.{name}.args`      | string[]    | 命令参数                     |
| `servers.{name}.enabled`   | bool        | 是否默认启用                 |

运行时可通过 `mcp` 交互命令查看/开关服务器。

### 4. `safety.json` — 安全护栏

Agent 执行本地命令时的安全检查策略，由 [tools/safety.py](tools/safety.py) 加载。

```json
{
  "mode": "blacklist",
  "confirm_dangerous": true,
  "blacklist": [],
  "whitelist": ["echo", "dir", "ls", "python", "pip", "git", "cat", "type"],
  "path_protection": {
    "enabled": true,
    "protected_paths": [
      "{system_root}",
      "{user_home}",
      "{project_root}",
      "{project_root}/agent",
      "{project_root}/config",
      "..."
    ],
    "confirm_paths": [
      "{project_root}/tests",
      "{project_root}/tools",
      "{project_root}/docs",
      "..."
    ]
  }
}
```

| 键                                  | 类型                              | 默认值          | 说明                                                                                        |
| ----------------------------------- | --------------------------------- | --------------- | ------------------------------------------------------------------------------------------- |
| `mode`                            | `"blacklist"` / `"whitelist"` | `"blacklist"` | `blacklist` 默认放行（仅拦截匹配项）；`whitelist` 仅放行白名单命令                      |
| `confirm_dangerous`               | bool                              | `true`        | 是否对危险命令交互确认（如`rm`、`sudo`、`chmod` 等）                                  |
| `blacklist`                       | string[]                          | `[]`          | 自定义禁止命令列表（正则表达式，与内置 BLOCKLIST 合并）                                     |
| `whitelist`                       | string[]                          | —              | 白名单模式下允许的首命令列表                                                                |
| `path_protection.enabled`         | bool                              | `true`        | 是否启用两级路径保护                                                                        |
| `path_protection.protected_paths` | string[]                          | —              | 保护级路径（禁止任何 CONFIRM 命令修改），支持占位符`{project_root}`、`{system_root}` 等 |
| `path_protection.confirm_paths`   | string[]                          | —              | 询问级路径（允许 CONFIRM 命令但需确认），支持占位符                                         |

**两级保护策略**：

- **BLOCKLIST**：`format`、`mkfs`、`fork bomb`、`curl|sh` 等灾难性命令始终拒绝
- **CONFIRM + 路径分类**：
  - 保护级路径（如 `agent/`、`config/`、`main.py`）→ 拒绝危险操作
  - 询问级路径（如 `tests/`、`docs/`、`README.md`）→ 需用户确认
  - 普通路径 → 需用户确认

详见 [工具系统 &gt; 安全护栏](#4-安全护栏safety) 章节。运行时可通过 `safety` 交互命令查看/修改。

### 5. `remote_control.json` — 远程控制（飞书）

飞书机器人远程控制配置。

```json
{
  "feishu": {
    "app_id": "cli_xxxxxxxxxxxxxx",
    "app_secret": "xxxxxxxxxxxxxxxxxxxx",
    "allow_open_id": ["ou_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]
  },
  "agent": {
    "provider": ""
  }
}
```

| 键                       | 类型     | 说明                                   |
| ------------------------ | -------- | -------------------------------------- |
| `feishu.app_id`        | string   | 飞书应用 App ID                        |
| `feishu.app_secret`    | string   | 飞书应用 App Secret                    |
| `feishu.allow_open_id` | string[] | 允许远程控制的飞书用户 Open ID 列表    |
| `agent.provider`       | string   | 远程控制使用的服务商标识（空则用默认） |

### 6. `scheduler_config.json` — 定时任务调度

定时任务调度服务的运行配置，由 [scheduler/run.py](scheduler/run.py) 加载。

```json
{
  "db_path": "data/scheduled_tasks.sqlite",
  "poll_interval": 30,
  "timezone": "Asia/Shanghai",
  "max_retries": 3,
  "max_workers": 5,
  "provider": null,
  "blocking": true
}
```

| 键                | 类型         | 默认值                          | 说明                                               |
| ----------------- | ------------ | ------------------------------- | -------------------------------------------------- |
| `db_path`       | string       | `data/scheduled_tasks.sqlite` | 任务数据库路径（相对项目根）                       |
| `poll_interval` | int          | 30                              | 轮询间隔（秒），调度器每隔此时间检查是否有到期任务 |
| `timezone`      | string       | `Asia/Shanghai`               | 任务时区，影响 cron 表达式解析                     |
| `max_retries`   | int          | 3                               | 任务执行失败最大重试次数                           |
| `max_workers`   | int          | 5                               | 并发执行任务的最大工作线程数                       |
| `provider`      | string\|null | null                            | 调度器使用的 LLM 服务商标识（null = 默认）         |
| `blocking`      | bool         | true                            | 是否阻塞主进程（`false` 时调度器在后台运行）     |

---

## 测试

项目提供一套**离线**单元测试（无需 API Key、不联网），覆盖核心逻辑；另含 1 个**在线连通性测试**（真实调用各提供商 API，需配置密钥）。

### 离线单元测试

| 测试文件                                 | 覆盖内容                                                                                |
| ---------------------------------------- | --------------------------------------------------------------------------------------- |
| `tests/test_config.py`                 | 运行时配置：默认值合并、路径解析                                                        |
| `tests/test_config_templates.py`       | 配置模板验证：.example 文件完整性检查                                                   |
| `tests/test_safety.py`                 | 安全护栏：黑名单拒绝、白名单放行、危险命令确认、路径保护                                |
| `tests/test_skills.py`                 | `SkillManager`：列出/读取/匹配(中→英别名)/渲染技能                                   |
| `tests/test_search.py`                 | `search` 工具：无 Key 降级、Tavily 返回结构(mock)                                     |
| `tests/test_cli_commands.py`           | CLI 命令分发：路由优先级、状态变更和各领域处理器                                        |
| `tests/test_human_input.py`            | LangGraph HITL：interrupt、恢复、并行选择和线程隔离                                     |
| `tests/test_terminal.py`               | 终端工具：输出截断、护栏拒绝、安全执行(mock subprocess)                                 |
| `tests/test_calculator.py`             | 计算器工具：表达式求值、错误处理                                                        |
| `tests/test_memory.py`                 | memory/ 包`AgentMemory`：checkpointer + Store 基础设施的初始化、SQLite/acreate/aclose |
| `tests/test_agent_core_regressions.py` | Agent 核心回归：HITL 恢复、会话隔离、技能匹配、长上下文裁剪等                           |
| `tests/test_scheduler.py`              | 定时任务调度：TaskStore CRUD、原子抢占、重试逻辑                                        |
| `tests/test_api.py`                    | API Server：端点路由、流式聊天、命令执行（FastAPI TestClient）                          |
| `tests/test_message_utils.py`          | LLM 异常信息提取：429/5xx/鉴权/未知错误的中文提示                                       |
| `tests/test_llm_client.py`             | 瞬时错误自动重试：should_retry 判定与重试行为                                           |
| `tests/test_compaction.py`             | 长上下文压缩中间件：增量摘要、工具输出 Prune、安全切割                                  |
| `tests/test_create_tool.py`            | `create_tool` 动态生成工具代码并自动注册到 `tools/__init__.py`                      |
| `tests/test_graph_rebuild.py`          | Graph 重建：MCP 工具变化触发重建、技能变化不重建                                        |
| `tests/test_mcp_pool.py`               | `MCPPool` 连接池：连接管理、健康探测、重连（mock 注入）                               |
| `tests/test_threads_preview.py`        | 会话菜单预览：首条用户消息提取与截断                                                    |
| `tests/test_workflow.py`               | 监督者模式工作流编排：Manager→Worker→Terminator 模板                                  |
| `tests/test_metrics.py`                | `MetricsCollector`：LLM/工具/压缩指标记录、token 提取与汇总                           |
| `tests/test_tool_wrapper.py`           | 工具超时包装：超时返回 JSON、按工具名覆盖、无限等待排除                                 |
| `tests/test_logging_config.py`         | 结构化日志：trace_id/thread_id 上下文注入、TraceContext 恢复                            |
| `tests/test_exceptions_and_close.py`   | 异常层次与生命周期：`LCAgentError` 子类、`aclose()` 资源释放                        |

### 在线连通性测试

| 测试文件                          | 覆盖内容                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------ |
| `tests/test_provider_models.py` | 真实调用各提供商`chat/completions`，校验配置的模型当前是否可用（需 API Key） |

> 该文件不在默认离线测试集内，通常**单独运行**（见下文 `--provider` 用法）。失败一般表示某个模型在服务商端不可用或密钥无效，并非代码问题。

### 运行测试

**运行全部测试：**

```bash
# uv 方式（推荐，自动使用 .venv）
uv run pytest tests/ -v

# 直接方式
# Linux / macOS
.venv/bin/pytest

# Windows
.\.venv\Scripts\pytest.exe
```

**运行单个测试文件：**

```bash
# Linux / macOS
.venv/bin/pytest tests/test_safety.py

# Windows
.\.venv\Scripts\pytest.exe tests/test_safety.py
```

**运行单个测试函数：**

```bash
# Linux / macOS
.venv/bin/pytest tests/test_safety.py::test_blacklist_blocks_commands

# Windows
.\.venv\Scripts\pytest.exe tests/test_safety.py::test_blacklist_blocks_commands
```

**显示详细输出（-v）和打印信息（-s）：**

```bash
# Linux / macOS
.venv/bin/pytest -v -s tests/test_memory.py

# Windows
.\.venv\Scripts\pytest.exe -v -s tests/test_memory.py
```

**运行特定模式匹配的测试：**

```bash
# 运行所有包含 "safety" 的测试
pytest -k safety

# 运行所有包含 "memory" 或 "thread" 的测试
pytest -k "memory or thread"
```

**只检测指定提供商（在线连通性测试）：**

```bash
uv run pytest tests/test_provider_models.py --provider kimi -v
```

> 不传 `--provider` 时检测 `config/llm_config.json` 中配置的**全部**提供商；传入未知提供商会直接报 UsageError。

测试配置见 `pytest.ini`（`pythonpath = .` 保证 `import tools`/`import agent` 可用；`testpaths = tests` 指定默认收集目录；`addopts = -q` 默认静默模式），`conftest.py` 提供安全配置缓存隔离的 autouse fixture。

### 静态检查（ruff）

```bash
# uv 方式（推荐）
uv run ruff check .
```

项目在 `pyproject.toml` 中固化了两项约束，保证结果可复现：

- **锁版本**：dev 依赖固定 `ruff>=0.16,<0.17`，避免升级引入新默认规则导致检查结果漂移。
- **默认规则集**：不显式 `select`，沿用 ruff 0.16 默认规则集（仅配置 `line-length = 100`、`target-version = "py314"`、排除目录）。默认规则已包含 `UP`/`B`/`SIM`/`S`/`DTZ` 等质量规则；显式全选会额外激活大量中文标点/魔法数噪音规则（如 `RUF002`、`PLR2004`），故不采用。

**有意忽略的规则：**

| 规则                               | 原因                                                                                                                    |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `BLE001`（盲捕获 `Exception`） | agent 工具层 / API 边界 / 调度作业需捕获一切异常并转成错误信息返回给 LLM/用户或记日志，属有意设计；收窄会让错误处理退化 |

涉及宽异常捕获的代码不写 `# noqa`，由上述项目级 `ignore = ["BLE001"]` 统一裁决。

---

## 技术栈

- Python 3.14+（`requires-python = ">=3.14"`）
- LangChain 1.x（`langchain`, `langchain-core`, `langchain-openai`）
- LangGraph 1.x（`create_react_agent`、自定义 `AgentMiddleware` 压缩中间件）
- LangGraph Checkpoint（`langgraph-checkpoint-sqlite`，同步/异步 SQLite 持久化）
- langchain-mcp-adapters 0.3+（MCP 工具适配）
- mcp 1.9+
- tenacity 9.x（LLM 瞬时错误自动重试）
- OpenAI SDK（用于 OpenAI 兼容接口）
- Tavily Python SDK（联网搜索）
- pytest（离线单元测试）
