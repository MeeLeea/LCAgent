# Planner 子程序（后台方案设计）

> 目录名 `planner` 为项目约定路径（与 `subagent/planner` 保持一致）。

让主 Agent 面对**复杂需求**时"派活"：调用一个工具启动独立后台进程，由子程序**一次 LLM 调用**
产出完整方案文档。子程序是**一次性任务**，不维护会话记忆、不写 checkpoint —— 唤醒、设计、
落盘、退出。

```
主 Agent 进程（subagent/planner/tool.py）              后台子进程（subagent.planner.worker）
──────────────────────────────────               ──────────────────────────────────
start_planner_task(requirement, context)
  │  preflight 校验 provider/model/API Key
  │  写 task.json（status=queued）
  ▼
spawn_worker → detached Popen ──────────────────►  claim（独占 worker.lock）
  │  毫秒级返回 task_id                              │
  ▼                                                  ▼
query_planner_task(task_id, wait_seconds≤30)     system(AGENT.md) + user(需求/上下文/结构)
  │  读 task.json + reap 死进程                     │
  │                                                  ▼
  │                                              LLMClient.ainvoke()
  │                                                  │
  │                                                  ▼
  │                                              plan.md（仅 status=done 可读）
  ▼                                                  │
query_planner_task 返回 plan ◄────────────────────────┘
```

## 一、为什么不复用 AgentCore

| 方案 | 结论 |
| --- | --- |
| 复用 `AgentCore` | ❌ 会带 checkpointer、长期记忆、SessionManager、完整工具集；一次性子任务不需要这些 |
| 复用 `LLMClient` | ✅ 只复用模型调用与 provider/model 配置，零会话状态 |
| 复用 `TeamAgent` | ❌ 为了提示词/技能注入引入 graph 依赖与 agent 抽象，属于过度设计 |

因此 `worker.py` 直接 `LLMClient(...).get_chat_model().ainvoke([...])`；`AGENT.md` 只承担
系统提示词 + 一个用户模板，由 `load_prompt_sections()` 用约 30 行解析。

## 二、目录结构

```
subagent/planner/
├── config.py       # PlannerConfig（provider/model/预算/超时/输出目录）
├── config.json     # 可选本地配置（被 .gitignore 忽略；缺失时用内置默认值）
├── AGENT.md        # 系统提示词 + 单个 ## workflow:design 模板
├── store.py        # 文件系统任务存储（原子写 + 独占认领 + PID 回收）
├── launcher.py     # detached 启动 + 跨平台 PID 存活检测 + preflight
├── worker.py       # 子进程入口：一次 LLM 调用 → plan.md
├── tool.py         # 3 个 async @tool（主 Agent 调用入口，由 tools/__init__.py 注册）
└── README.md

data/subagent/planner/<task_id>/   # 运行时产物（自动创建）
├── task.json       # 任务状态（唯一事实源，原子替换）
├── worker.lock     # 认领锁（worker 生命周期内存在）
├── cancel.flag     # 取消标记
├── worker.log      # 子进程 stdout/stderr
├── prompt.md       # 本次实际发送的 system + user prompt（审计用，非记忆）
└── plan.md         # 最终方案（仅 done 时存在）
```

## 三、主 Agent 工具接口

3 个工具全部是 **async**（`wait_seconds` 用 `asyncio.sleep`，不阻塞共享事件循环），
已注册到 `tools/__init__.py` 的 `all_tools`。

| 工具 | 作用 | 关键行为 |
| --- | --- | --- |
| `start_planner_task(requirement, context="")` | 启动后台方案设计 | **非阻塞**，毫秒级返回 `task_id`；preflight 失败时不建任务、不拉进程 |
| `query_planner_task(task_id, wait_seconds=0, max_chars=12000)` | 查询进度 / 读取结果 | `wait_seconds` clamp `[0,30]`；终态立即返回；`done` 时直接带上 `plan` |
| `cancel_planner_task(task_id)` | 请求取消 | `queued` 立即取消；`running` 在当前 LLM 调用结束后停止 |

> **防编造约束已写入每个工具的 docstring**：`done != true` 之前，模型不得编造、推测或
> 总结方案内容；任务失败时如实告知用户失败原因。

### 典型调用序列

```
1. start_planner_task(requirement="为现有 CLI 增加插件机制并给出落地方案")
   → {"success": true, "task_id": "pl-20261102-143012-ab12cd34", "status": "queued", ...}

2. query_planner_task(task_id, wait_seconds=30)
   → {"success": true, "done": false, "status": "running", ...}   # 继续轮询

3. query_planner_task(task_id, wait_seconds=30)
   → {"success": true, "done": true, "status": "done", "plan": "# ...", "chars": 8231}
```

## 四、任务状态机

```
queued ──claim──► running ──LLM 调用成功──► done
  │                  │
  │                  ├── 异常/空输出/超时 ──► failed
  │                  └── cancel.flag ──────► cancelled
  └── cancel（尚未启动）───────────────────► cancelled
```

- **独占认领**：`worker.lock` 用 `O_CREAT|O_EXCL` 原子创建，只有抢到锁的 worker 继续。
- **僵尸回收**：状态工具对 `running` 任务调用 `reap()`：PID 已死 → `failed`。
  Worker 崩溃不会留下永久 `running` 的任务。
- **取消**：`cancel.flag` 由 worker 在 LLM 调用轮询中检查；取消后**不写 `plan.md`**，
  因此不可能把半成品当成最终方案返回。
- **终态保证**：worker 整段 `try/except BaseException/finally`，异常必写 `failed` + `error`。

## 五、一次 LLM 调用的输入

系统提示词来自 `AGENT.md` 的 `## workflow:design` 之前部分；用户模板来自该小节之后部分，
只支持两个占位符：

| 占位符 | 内容 |
| --- | --- |
| `{requirement}` | 用户需求正文 |
| `{context}` | 已知上下文；为空时替换为「（无）」 |

输出固定为一份自包含方案文档（背景/范围/假设/总览/候选方案对比/模块接口/关键流程/
实施计划/验收标准/风险缓解），最后一行 `<!-- END OF PLAN -->`。

## 六、配置（subagent/planner/config.json）

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `provider` | `null` | `null` 时自动选择第一个已配置 API Key 的提供商 |
| `model` | `null` | `null` 时使用 provider 的默认模型 |
| `temperature` / `max_tokens` | `0.4` / `8192` | LLM 采样参数 |
| `timeout_seconds` | `900` | 单次 LLM 调用总超时 |
| `output_root` | `data/subagent/planner` | 任务产物根目录；可用环境变量 `LCAGENT_PLANNER_ROOT` 覆盖 |
| `agent_prompt_file` | `subagent/planner/AGENT.md` | 提示词文件 |

> `provider` / `model` / API Key 在 `start_planner_task` 阶段由 `launcher.preflight` 同步校验：
> 缺 Key 会直接返回 `success:false`，不会创建一个永远排队的僵尸任务。
>
> `config.json` 是**可选本地配置**：仓库 `.gitignore` 的 `*.json` 规则会忽略它，缺失时直接使用
> `config.py` 内置默认值（与上表一致），因此新克隆的仓库无需该文件即可运行。

## 七、手动运行与调试

```bash
# 手动跑一个已登记的任务（通常由 launcher 自动调用）
uv run python -m subagent.planner.worker --task-id pl-20261102-143012-ab12cd34

# 查看任务状态
uv run python -c "from subagent.planner.store import PlannerStore; import json; s=PlannerStore(); print(json.dumps(s.get('<task_id>'), ensure_ascii=False, indent=2))"

# 查看子进程日志 / 实际 prompt
type data\subagent\planner\<task_id>\worker.log
type data\subagent\planner\<task_id>\prompt.md
```

## 八、已知限制

- **磁盘增长**：任务目录不会自动清理。定期删除 `data/subagent/planner/` 下的历史任务。
- **取消不是 kill**：`cancel_planner_task` 只是写标记，当前 LLM 调用结束后才生效。
- **单机部署**：文件系统存储 + PID 检测假设同一台机器。
- **一次调用**：方案深度受单次输出预算限制；需要多轮打磨时优先提高 `max_tokens` 或换更强模型。

## 九、测试

```bash
uv run pytest tests/subagent -v
uv run ruff check subagent tools tests/subagent
```

覆盖：原子认领独占、PID 回收、取消语义、异常/空输出/超时终态落盘、3 工具契约与防编造门禁、
`AGENT.md` 模板解析、PID 存活检测（真实子进程）。
