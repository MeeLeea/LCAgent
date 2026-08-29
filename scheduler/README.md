# 定时任务调度子系统

让 Agent 能"定闹钟"——用户在对话中说"明天下午3点生成报告"或"每天9点发送日报"，Agent 登记任务后立即结束对话，后台调度器在时间到达时自动唤起 Agent 执行。

核心思路是把"理解任务"和"盯时间"拆成两条独立路径：Agent 只负责解析意图并登记，调度器负责在时间到达时执行。两者通过 SQLite 数据库解耦，互不阻塞。

## 架构

```
对话阶段（逻辑 A）                         后台调度（逻辑 B）
──────────────────                       ──────────────────
用户："明天下午3点生成报告"                 调度器进程（独立运行）
  │                                        │
  ▼                                        ▼
Agent 理解意图，算好 execute_time      轮询 pending + 到期任务    cron 触发器
  │                                        │           │
  ▼                                        ▼           ▼
schedule_task 工具入库                claim_task 抢占      周期任务触发
  │                                        │           │
  ▼                                        ▼           ▼
SQLite (status=pending) ◄─── 共享数据库 ────► 线程池 (ThreadPoolExecutor)
  │                                                    │
  ▼                                                    ▼
回复"任务已登记"                                    execute_task() 路由
                                                     │
                                          ┌───────────┴───────────┐
                                          │                       │
                                   workflow: 前缀              普通任务
                                          │                       │
                                          ▼                       ▼
                                   run_workflow_by_name     AgentCore.run()
                                   (多 Agent 协作)           → done / failed
                                          │
                                          ▼
                                     done / failed
```

- **逻辑 A（对话阶段）**：Agent 解析用户意图，算出 `execute_time` 或 `cron_expr`，调用 `schedule_task` 工具写入 SQLite，然后直接回复"任务已登记"。Agent 不等待、不 sleep。
- **逻辑 B（后台调度）**：独立的调度器进程周期性查询数据库中到期的一次性任务，通过原子抢占后提交到线程池并发执行。周期任务由 APScheduler 的 CronTrigger 在精确时间点触发。executor 根据任务文本自动路由：以 `workflow:` 开头的任务调用多 Agent 工作流执行，其余通过 `agent_factory` 创建新的 AgentCore 实例，经 SessionManager.arun() 执行（三层架构：Agent 执行 → Session 调度 → Memory 沉淀）。

## 模块结构

```
scheduler/
├── __init__.py      # 包导出（TaskStore, execute_task, SchedulerEngine）
├── store.py         # SQLite 持久化层（CRUD + 原子抢占 + 重试逻辑）
├── executor.py      # 执行桥接（agent_factory → SessionManager.arun()）
├── engine.py        # APScheduler 调度引擎（轮询 + cron + 线程池）
└── run.py           # 独立进程入口（读 config/scheduler_config.json）

tools/
└── scheduler_tool.py  # Agent 对话工具（@tool 装饰）

config/
├── scheduler_config.json         # 运行时配置
└── scheduler_config.json.example # 配置模板
```

### 各模块职责

| 模块 | 职责 |
|------|------|
| `store.py` | SQLite 任务存储，线程安全。提供 CRUD、到期查询、原子抢占（`claim_task`）、失败重试回退 |
| `executor.py` | 桥接函数 `execute_task(task, agent_factory)`，自动判断任务类型：以 `workflow:` 开头的路由到多 Agent 工作流（`run_workflow_by_name`），其余调用 `agent_factory()` 创建 Agent 实例后通过 `session_manager.arun(task_text)` 执行（三层架构），捕获所有异常 |
| `engine.py` | 调度核心。一次性任务通过 `IntervalTrigger` 轮询，周期任务通过 `CronTrigger` 精确触发。到期任务提交到 `ThreadPoolExecutor` 并发执行 |
| `run.py` | 命令行入口。加载配置文件 → 构造 `agent_factory` → 预加载 `team` 模块触发 Agent/工作流注册 → 启动 `SchedulerEngine`（阻塞模式） |
| `scheduler_tool.py` | 三个 `@tool` 函数供 Agent 在对话中调用：`schedule_task`、`list_scheduled_tasks`、`cancel_scheduled_task` |

## 快速开始

### 1. 配置文件

编辑 `config/scheduler_config.json`（如不存在，从 `scheduler_config.json.example` 复制）：

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

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `db_path` | SQLite 数据库路径，相对路径锚定项目根 | `data/scheduled_tasks.sqlite` |
| `poll_interval` | 一次性任务轮询间隔（秒） | `30` |
| `timezone` | cron 触发器时区 | `Asia/Shanghai` |
| `max_retries` | 单次任务最大重试次数 | `3` |
| `max_workers` | 任务并发执行线程池大小 | `5` |
| `provider` | LLM 提供商名称，`null` 时自动选择第一个已配置 API Key 的提供商 | `null` |
| `blocking` | 是否使用阻塞调度模式 | `true` |

### 2. 注册 Agent 工具

在 `tools/__init__.py` 中注册调度工具：

```python
from .scheduler_tool import schedule_task, list_scheduled_tasks, cancel_scheduled_task

all_tools = [
    # ... 现有工具
    schedule_task,
    list_scheduled_tasks,
    cancel_scheduled_task,
]
```

### 3. 初始化依赖

在 `main.py` 启动时注入 TaskStore（如不注入，工具会使用默认 DB 路径懒初始化）：

```python
from tools.scheduler_tool import configure

configure(db_path="data/scheduled_tasks.sqlite")
```

### 4. 启动调度器

调度器作为独立进程运行，与 Agent 对话进程分离：

```bash
python -m scheduler.run
```

启动后会看到：

```
[Scheduler] 使用配置文件: .../config/scheduler_config.json
============================================================
  定时任务调度器 (Scheduler Engine)
============================================================

[Scheduler] 使用 LLM 提供商: zhipu
[Scheduler] LLM 已就绪: zhipu / glm-4-flash
[Scheduler] 已注册 Agent: cmodeler, manager, reviewer, spec_analyst, terminator, verifier, worker
[Scheduler] 可用工作流: pipline, simple, systemc_cmodel
[Scheduler] 数据库: .../data/scheduled_tasks.sqlite
[Scheduler] 任务执行线程池已创建（max_workers=5）
[Scheduler] 已注册一次性任务轮询（间隔 30s）
[Scheduler] 调度引擎已启动
[Scheduler] 阻塞模式运行中，按 Ctrl+C 停止...
```

## 使用方法

### 在对话中登记任务

Agent 在对话中调用 `schedule_task` 工具来登记任务。用户不需要知道工具参数，Agent 会自动解析用户意图并计算时间。

**一次性任务**：用户说"明天下午3点生成报告"，Agent 调用：

```python
schedule_task(
    task_text="生成报告",
    task_type="one_time",
    execute_time="2026-07-30T15:00:00",
)
```

**周期任务**：用户说"每天9点发送日报到飞书群"，Agent 调用：

```python
schedule_task(
    task_text="发送日报到飞书群",
    task_type="periodic",
    cron_expr="0 9 * * *",
)
```

工具返回 JSON：

```json
{
  "success": true,
  "task_id": 1,
  "task_type": "one_time",
  "execute_time": "2026-07-30T15:00:00",
  "task_text": "生成报告",
  "message": "一次性任务已登记（ID: 1），将在 2026-07-30T15:00:00 自动执行。"
}
```

### 查询任务列表

```python
list_scheduled_tasks(status="pending")
```

返回所有 `pending` 状态的任务。`status` 可选值：`pending` / `running` / `done` / `failed` / `cancelled`，留空返回全部。

### 取消任务

```python
cancel_scheduled_task(task_id=3)
```

只能取消 `pending` 状态的任务。周期任务取消后会同时从调度器移除 cron job。

### 工作流任务（多 Agent 协作）

调度器支持将任务路由到多 Agent 工作流执行。只需在 `task_text` 前加 `workflow:` 前缀，executor 会自动识别并调用对应工作流，而非创建单个 AgentCore 实例。

**格式**：`workflow:<工作流名称> <任务描述>`

启动时 `run.py` 会预加载 `team` 模块，触发所有 `@register_agent` 装饰器注册 Agent 角色，并加载内置工作流。当前可用工作流：

| 工作流 | 说明 | 角色 |
|--------|------|------|
| `simple` | 监督者模式（Manager 拆解 → Worker 执行 → Terminator 汇总） | manager, worker, terminator |
| `pipline` | 同 simple，监督者流水线 | manager, worker, terminator |
| `systemc_cmodel` | SystemC C-Model 开发（规格分析 → 编码 → 验证 → 审查） | spec_analyst, cmodeler, verifier, reviewer |

登记工作流任务与普通任务完全一样，区别只在 `task_text` 内容：

```python
schedule_task(
    task_text="workflow:systemc_cmodel 为同步FIFO编写C-Model",
    task_type="one_time",
    execute_time="2026-08-04T10:00:00",
)
```

到期后 executor 检测到 `workflow:` 前缀，解析出工作流名 `systemc_cmodel` 和任务 `为同步FIFO编写C-Model`，调用 `graph.registry.run_workflow_by_name()` 构建并运行工作流。执行过程中各节点开始/结束会打印进度日志：

```
[Scheduler] 任务 #5 → 工作流: systemc_cmodel
[Scheduler]   任务内容: 为同步FIFO编写C-Model
[Scheduler]   ▸ 节点开始: spec_analyst
[Scheduler]   ✓ 节点完成: spec_analyst
[Scheduler]   ▸ 节点开始: cmodeler
[Scheduler]   ✓ 节点完成: cmodeler
[Scheduler]   ▸ 节点开始: verifier
[Scheduler]   ✓ 节点完成: verifier
[Scheduler]   ▸ 节点开始: reviewer
[Scheduler]   ✓ 节点完成: reviewer
[Scheduler]   工作流 systemc_cmodel 执行完成
```

工作流任务也支持周期调度，例如每天定时运行 SystemC 开发流程：

```python
schedule_task(
    task_text="workflow:systemc_cmodel 审查今天的 RTL 变更并更新 C-Model",
    task_type="periodic",
    cron_expr="0 20 * * 1-5",
)
```

## 使用案例

### 案例 1：定时生成报告

> 用户：明天上午 9 点帮我生成一份本周项目进展报告

Agent 理解意图后调用 `get_local_time` 获取当前时间，计算出明天的日期，然后调用 `schedule_task`：

```python
schedule_task(
    task_text="生成一份本周项目进展报告",
    task_type="one_time",
    execute_time="2026-07-30T09:00:00",
)
```

Agent 回复用户："任务已登记，将在明天上午 9 点自动执行。"对话结束。

第二天 9:00，调度器轮询到该任务到期，`claim_task` 抢占后提交到线程池，创建新的 AgentCore 实例执行"生成一份本周项目进展报告"。

### 案例 2：每日日报

> 用户：每天下午 6 点把今天的工作总结发到飞书群

Agent 调用：

```python
schedule_task(
    task_text="把今天的工作总结发到飞书群",
    task_type="periodic",
    cron_expr="0 18 * * *",
)
```

调度器启动时（或登记时若引擎已运行）将此任务注册为 APScheduler 的 cron job，每天 18:00 自动触发执行。

### 案例 3：工作日定时检查

> 用户：工作日每天早上 8:30 检查邮箱并汇总未读邮件

```python
schedule_task(
    task_text="检查邮箱并汇总未读邮件",
    task_type="periodic",
    cron_expr="30 8 * * 1-5",
)
```

`1-5` 表示周一到周五。周末不触发。

### 案例 4：多任务并发

同一时间点有多个到期任务时，线程池会并发执行。例如用户登记了三个都在 10:00 的一次性任务：

```
[Scheduler] 发现 3 个到期的一次性任务，提交线程池并发执行
[Scheduler] 任务执行线程池已创建（max_workers=5）
  → worker-1: 执行任务 #1（生成报告）
  → worker-2: 执行任务 #2（发送通知）  
  → worker-3: 执行任务 #3（清理临时文件）
```

三个任务在各自的 worker 线程中并行执行，互不阻塞。`max_workers=5` 控制最大并发数，设为 1 时退化为串行。

### 案例 5：定时执行 SystemC 工作流

> 用户：明天上午 10 点启动 SystemC C-Model 开发流程，为 AXI 总线适配器编写模型

Agent 调用：

```python
schedule_task(
    task_text="workflow:systemc_cmodel 为AXI总线适配器编写C-Model",
    task_type="one_time",
    execute_time="2026-08-04T10:00:00",
)
```

到期后 executor 识别 `workflow:` 前缀，路由到 `systemc_cmodel` 工作流。四个专业 Agent 依次协作：`spec_analyst` 分析规格 → `cmodeler` 编写 C-Model → `verifier` 验证功能 → `reviewer` 审查代码质量，最终返回完整结果。

## 任务生命周期

```
        ┌──────────┐
        │ pending  │ ◄── 创建任务 / 重试回退
        └────┬─────┘
             │ claim_task()
             ▼
        ┌──────────┐
        │ running  │
        └────┬─────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
 ┌─────────┐  ┌─────────┐
 │  done   │  │ failed  │ ◄── 重试次数耗尽
 └─────────┘  └─────────┘

 ┌──────────┐
 │cancelled │ ◄── cancel_task()（仅 pending 可取消）
 └──────────┘
```

- 一次性任务执行成功标记 `done`，失败时若 `retry_count < max_retries` 则回退 `pending` 等下次轮询重新拾取，重试耗尽标记 `failed`
- 周期任务每次执行后保持 `pending`（因为要重复跑），执行结果临时存在 `result` 字段
- `claim_task` 通过 `UPDATE ... WHERE status='pending'` 原子操作保证即使部署多个调度器实例，同一条任务也只会被一个实例抢到执行权

## cron 表达式参考

5 字段 Unix 格式：`分 时 日 月 周`

| 表达式 | 含义 |
|--------|------|
| `0 9 * * *` | 每天 9:00 |
| `30 9 * * *` | 每天 9:30 |
| `0 9 * * 1` | 每周一 9:00 |
| `0 9 * * 1-5` | 工作日 9:00 |
| `0 0 1 * *` | 每月 1 号 0:00 |
| `30 * * * *` | 每小时第 30 分钟 |
| `30 17 * * 1-5` | 工作日 17:30 |
| `0 */2 * * *` | 每 2 小时 |

## 配置优先级

`run.py` 的配置值来源（从高到低）：

1. `config/scheduler_config.json` — 运行时配置文件
2. 环境变量 `AGENT_LLM_PROVIDER` — 仅影响 LLM 提供商选择
3. 内置默认值 `SCHEDULER_DEFAULTS` — `run.py` 顶部的字典

配置文件路径写死在 `run.py` 的 `SCHEDULER_CONFIG_FILE` 常量中。文件不存在时使用全部默认值，不报错。

## 编程式接入

除通过 `run.py` 启动独立进程外，也可以在代码中嵌入调度引擎：

```python
from scheduler import TaskStore, SchedulerEngine
from llm.llm_client import LLMClient
from agent import AgentCore

# 1. 准备 agent_factory
llm = LLMClient(provider="zhipu", config_file="config/llm_config.json")

def agent_factory():
    return AgentCore(llm_client=llm, name="LCAgent", ...)

# 2. 创建引擎（非阻塞模式，嵌入主进程后台线程）
store = TaskStore("data/scheduled_tasks.sqlite")
engine = SchedulerEngine(
    task_store=store,
    agent_factory=agent_factory,
    poll_interval=30,
    blocking=False,          # 后台线程运行
    timezone="Asia/Shanghai",
    max_workers=5,
)

# 3. 启动
engine.start()

# 4. 运行时注册周期任务
task = store.create_task(
    task_type="periodic",
    task_text="每小时检查一次系统状态",
    cron_expr="0 * * * *",
)
store_created_task = store.get_task(task)
engine.register_periodic_task(store_created_task)

# 5. 停止
engine.stop()
```

## 数据库表结构

`data/scheduled_tasks.sqlite` 中的 `scheduled_tasks` 表：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | INTEGER | 自增主键 |
| `task_type` | TEXT | `one_time` 或 `periodic` |
| `execute_time` | TEXT | ISO 8601 时间，一次性任务必填 |
| `cron_expr` | TEXT | 5 字段 cron 表达式，周期任务必填 |
| `task_text` | TEXT | 实际要执行的任务描述 |
| `status` | TEXT | `pending` / `running` / `done` / `failed` / `cancelled` |
| `created_at` | TEXT | 创建时间 |
| `executed_at` | TEXT | 开始执行时间 |
| `result` | TEXT | 执行结果或错误信息 |
| `retry_count` | INTEGER | 已重试次数 |
| `max_retries` | INTEGER | 最大重试次数，默认 3 |

启用了 WAL 模式以提升并发读写性能。可用 DB Browser for SQLite 直接查看任务状态。
