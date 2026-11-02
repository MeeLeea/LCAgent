# graph/ — LangGraph 工作流编排层

本目录只做一件事：**把多个 Agent 角色编排成一张 LangGraph 图**，并注册到全局工作流注册表，供 CLI（`workflow:<name> <任务>`）、API、调度器按名调度。

节点内部"怎么思考、怎么调工具、怎么渲染提示词"**不在本目录**——全部下沉到 `team/` 各角色类。graph 层只保留：**节点表（NodeSpec）+ 边 + 条件路由 + 注册**。

---

## 目录

- [1. 三层职责划分](#1-三层职责划分)
- [2. 目录结构](#2-目录结构)
- [3. graph/common 构件速查](#3-graphcommon-构件速查)
- [4. 一张工作流由哪四要素构成](#4-一张工作流由哪四要素构成)
- [5. 工作流 State：动态组合而非手写 TypedDict](#5-工作流-state动态组合而非手写-typeddict)
- [6. 新建一个工作流（分步）](#6-新建一个工作流分步)
- [7. 运行一张工作流](#7-运行一张工作流)
- [8. 陷阱清单](#8-陷阱清单)
- [9. 参考](#9-参考)
- [10. 案例一：simple.py —— 线性监督者模式](#10-案例一simplepy--线性监督者模式)
- [11. 案例二：rtl_graph.py —— 带条件路由的多轮迭代流水线](#11-案例二rtl_graphpy--带条件路由的多轮迭代流水线)

---

## 1. 三层职责划分

```
┌─────────────────────────────────────────────────────────────┐
│ graph/            编排层：节点表 + 边 + 条件路由 + 注册        │
│                   simple.py / rtl_graph.py                   │
├─────────────────────────────────────────────────────────────┤
│ graph/common/     构图基建：State schema 组合、NodeSpec 声明  │
│                   式注册、注册表、通用运行器、节点进度回调      │
├─────────────────────────────────────────────────────────────┤
│ team/             节点实现层：各角色类持有全部节点            │
│                   LLM 节点 = create_llm_node 类属性闭包       │
│                   代码节点 = 角色实例方法（bound method）     │
└─────────────────────────────────────────────────────────────┘
```

**为什么这样拆**：同一角色（如 `ManagerAgent.summarize_context_node`）会被 `simple` 与 `rtl_graph` 两张图复用；图之间只应差在"节点怎么连"，不该复制节点实现。因此 graph 文件里看不到任何 prompt 拼接或 LLM 调用。

---

## 2. 目录结构

```
graph/                        # 命名空间包（无 __init__.py，PEP 420 隐式包）
├── README.md                 # 本文件
├── simple.py                 # 监督者模式：Manager → Worker → Terminator（4 节点，线性）
├── rtl_graph.py              # RTL 芯片设计流水线（13 节点 + 2 处条件路由 + 迭代环）
└── common/                   # 工作流共享组件（唯一被两图共用的构图基建）
    ├── __init__.py           # 统一再导出（from graph.common import X），末尾触发 _load_builtin_workflows()
    ├── builder_utils.py      # build_graph_scaffold：SkillInjector + StateGraph 一步创建
    ├── node_spec.py          # NodeSpec / register_nodes：声明式节点注册
    ├── schema.py             # compose_state_schema：按角色 state_fields 动态组合 State schema
    ├── registry.py           # WORKFLOWS / AGENT_REGISTRY / register_workflow / build_workflow / arun_workflow_by_name
    ├── workflow_runner.py    # arun_compiled_workflow：统一 ainvoke 逻辑（记忆 / workspace / 进度回调）
    ├── node_tracking.py      # NodeTrackingHandler：节点级进度回调（含 TOKEN 级流式）
    └── compaction_utils.py   # _build_compaction_middleware：手动压缩路径用
```

> 新增文件即自动注册：`common/registry.py::_load_builtin_workflows()` 在 `graph.common` 首次 import 时用 `pkgutil.iter_modules` 扫描 `graph/` 下所有 `.py`（跳过 `__init__` 与子包），逐个 `importlib.import_module` 触发模块末尾的 `register_workflow`。**因此新增工作流不需要改注册表代码。**

---

## 3. graph/common 构件速查

| 构件 | 位置 | 作用 |
| --- | --- | --- |
| `build_graph_scaffold(agents, *, skills_dir, auto_match_skills, extra_channels=None)` | `builder_utils.py` | 一步返回 `(builder, injector)`：构造 `SkillInjector` + `StateGraph(compose_state_schema(...))` |
| `compose_state_schema(agents, extra_channels=None)` | `schema.py` | 按 `type(agent).state_fields` 动态拼 `TypedDict`，通道冲突抛 `ValueError` |
| `NodeSpec(name, fn, role)` | `node_spec.py` | 把"节点名 + 节点函数 + 绑定哪个角色"声明成数据 |
| `register_nodes(builder, agents, injector, specs)` | `node_spec.py` | 批量 `partial` 绑定 + `add_node`；自动识别独立函数与 bound method 两种节点形态 |
| `register_workflow(name, builder, roles=None, description="", initial_fields=None)`（`runner` 形参已废弃） | `registry.py` | 写入 `WORKFLOWS` 注册表（唯一注册入口）；建议全部用关键字实参 |
| `register_agent(name, tools, mcp_tools, mcp_all)` | `registry.py` | 角色类装饰器，写入 `AGENT_REGISTRY` |
| `build_workflow(name, checkpointer=None)` | `registry.py` | 构建角色实例 → 调 `builder(agents, checkpointer=...)` → 返回 `(graph, agents)` |
| `arun_workflow_by_name(workflow_name, task, ...)` | `registry.py` | 按名构建 + 运行（自动透传注册条目里的 `initial_fields`） |
| `arun_compiled_workflow(graph, task, initial_fields=None, ...)` | `workflow_runner.py` | 唯一运行实现：跨轮次记忆 + 长期记忆召回 + workspace 透传 + 节点进度回调 |
| `build_tracking_callbacks(graph, on_node_start, on_node_end, on_node_error, on_token)` | `node_tracking.py` | 构造进度回调 handler 列表（不传回调时零开销） |

---

## 4. 一张工作流由哪四要素构成

1. **角色**：`@register_agent("name")` 装饰的 `TeamAgent` 子类，类上声明 `state_fields`（自己产出哪些 state 通道）。
2. **节点表**：`list[NodeSpec]`，每个节点指到"某角色的某个函数"。
3. **边**：`builder.add_edge(...)` 直线边 + `builder.add_conditional_edges(...)` 条件路由。
4. **注册**：模块末尾 `register_workflow(...)`，声明 `roles` 与 `initial_fields`。

节点函数只有两种合法形态：

| 形态 | 签名 | NodeSpec 里怎么写 | 绑定方式 |
| --- | --- | --- | --- |
| 独立函数（`create_llm_node` 闭包，作为**类属性**） | `async def fn(state, agent, injector=None, config=None)` | `NodeSpec("n", ArchiAgent.plan_node, role="architect")` | partial 补绑 `agent=agents[role]` + `injector` |
| 角色实例方法（bound method） | `async def node(self, state, injector=None, config=None)` | `NodeSpec("n", agents["rtl_designer"].xxx_node, role="rtl_designer")` | 只补绑 `injector`（self 已绑定） |

> 两类节点最终都经 `graph.ainvoke` 收到 LangGraph 注入的 `config`（含 `configurable.workspace_path` / `configurable.thread_id`）。

---

## 5. 工作流 State：动态组合而非手写 TypedDict

不再手写 `WorkflowState` / `RTLGraphState`。`compose_state_schema(agents, extra_channels)` 按确定性顺序（`sorted(agents)`）合并三部分：

| 来源 | 通道 |
| --- | --- |
| 底座（`_BASE_CHANNELS`，所有图共有） | `task: str` / `raw_context: str` / `messages: Annotated[list[AnyMessage], add_messages]` / `active_skills: list[str]` |
| 各角色 `state_fields` ClassVar | manager→`context_summary`/`plan`；worker→`worker_result`；terminator→`final_answer`；architect→5 个 `arch_*`；rtl_designer→`design_spec`/`rtl_code`/`output_files`；rtl_verification→`verification_plan`/`verification_report` |
| `extra_channels`（工作流级编排字段） | rtl 的 `round`/`max_rounds`/`final_answer`/`file_check_passed`/`sim_check_passed`/`sim_status`/`sim_log_path` |

规则：

- 底座**不含** `summary`（工作流级概述已移除）与 `context_summary`（归 `ManagerAgent.state_fields`）。
- **通道名冲突即抛 `ValueError`**（底座 vs 角色 / 角色 vs 角色 / 与 `extra_channels`），不做静默覆盖。
- `messages` 用 `add_messages` reducer，每个业务节点产出追加一条 `AIMessage`；压缩统一由 `LCAgentCompactionMiddleware.before_model` 在模型调用前自动触发，**节点级压缩已移除**（`register_nodes` 不再接受 `compaction_mw`）。

---

## 6. 新建一个工作流（分步）

### Step 1：确认角色已就绪

角色类需带 `@register_agent("name", tools=...)` 且声明 `state_fields`：

```python
# team/my_role/my_role.py
from typing import ClassVar

from graph.common import register_agent
from team.base import TeamAgent
from team.node_factory import create_llm_node

@register_agent("my_role", tools=[])
class MyRoleAgent(TeamAgent):
    # 本角色往 state 写入的通道，compose_state_schema 据此拼 schema
    state_fields: ClassVar[dict[str, type]] = {"draft": str, "review_note": str}

    # LLM 节点：类属性闭包（图里用 MyRoleAgent.draft_node 引用）
    draft_node = create_llm_node(
        template_name="draft",
        output_field="draft",
        template_vars_fn=lambda s: {"task": s["task"]},
    )

    # 代码节点：实例方法（图里用 agents["my_role"].check_node 引用）
    async def check_node(self, state, injector=None, config=None):
        return {"review_note": "..."}
```

并在 `team/__init__.py`（或你的工作流模块顶部）确保它被 import，`@register_agent` 才会执行。

### Step 2：写工作流模块 `graph/my_flow.py`

```python
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from graph.common import NodeSpec, register_nodes, register_workflow
from graph.common.builder_utils import build_graph_scaffold

from team.my_role.my_role import MyRoleAgent  # 触发 @register_agent 注册


def build_my_flow(agents: dict, checkpointer=None, skills_dir: str | None = None,
                  auto_match_skills: bool = True) -> StateGraph:
    builder, injector = build_graph_scaffold(
        agents,
        skills_dir=skills_dir,
        auto_match_skills=auto_match_skills,
        extra_channels={"round": int, "max_rounds": int},   # 无额外通道就传 None
    )

    register_nodes(builder, agents, injector, [
        NodeSpec("draft", MyRoleAgent.draft_node, role="my_role"),        # 类属性
        NodeSpec("check", agents["my_role"].check_node, role="my_role"),  # bound method
    ])

    builder.add_edge(START, "draft")
    builder.add_edge("draft", "check")
    builder.add_edge("check", END)

    return builder.compile(checkpointer=checkpointer)


register_workflow(
    "my_flow",
    builder=build_my_flow,
    roles=["my_role"],
    description="我的自定义工作流",
    initial_fields={"draft": "", "review_note": "", "round": 0, "max_rounds": 3},
)
```

### Step 3：什么都不用改，直接跑

`_load_builtin_workflows()` 会自动 import `graph/my_flow.py` 并完成注册：

```bash
你: workflow                 # 应能看到 my_flow
你: workflow:my_flow <任务>
```

### Step 4：补测试

`tests/graph/` 下按既有风格添加（拓扑断言 + 节点通道断言 + `initial_fields` 子集校验）。

---

## 7. 运行一张工作流

### 7.1 CLI

```
你: workflow                                  # 列出所有已注册工作流（名称 + 描述）
你: workflow:simple 帮我分析项目结构
你: workflow:rtl_graph 设计一个 8 位加法器
```

执行期间按节点打印 `▸ 节点开始: xxx` / `✓ 节点完成: xxx`（`arun_compiled_workflow` 的 `on_node_start` / `on_node_end` 回调）。

### 7.2 编程接口

```python
import asyncio
from graph.common import arun_compiled_workflow, arun_workflow_by_name, build_workflow

# 方式 A：按名调度（推荐，initial_fields 自动从注册条目透传）
result = asyncio.run(arun_workflow_by_name("simple", "帮我分析项目结构"))
print(result["final_answer"])

# 方式 B：先构建再运行
graph, agents = build_workflow("simple")               # 无持久化
result = asyncio.run(arun_compiled_workflow(
    graph, "帮我分析项目结构",
    raw_context="",                                     # 额外记忆文本（可空）
    workspace_path=r"D:\work\some_ws",                  # 注入 configurable.workspace_path
    on_node_start=lambda ev: print("start", ev.node),
    on_node_end=lambda ev: print("end", ev.node),
))

# 方式 C：跨轮次续跑（需 checkpointer + 固定 thread_id）
from langgraph.checkpoint.memory import MemorySaver
graph, agents = build_workflow("simple", checkpointer=MemorySaver())
r1 = asyncio.run(arun_compiled_workflow(graph, "第一轮", thread_id="wf-1"))
r2 = asyncio.run(arun_compiled_workflow(graph, "第二轮", thread_id="wf-1"))
```

`arun_compiled_workflow` 统一承担：

| 能力 | 说明 |
| --- | --- |
| `initial_fields` 子集守卫 | 键不在图的 State schema 内即抛 `ValueError`（LangGraph 对未知输入键**静默丢弃**，拼错字段名不会报错，故加守卫；自省不可用时 fail-open） |
| 跨轮次记忆 | 读 `checkpoint` 上一轮的 `task`/`plan`/`worker_result`/`final_answer`，压成 `【上一轮工作流记录】` 拼进 `raw_context` |
| 长期记忆 | 传 `memory=` / `memory_thread_id=` 时召回文本注入，结束后把 `final_answer` 作为 DONE 事件沉淀 |
| `active_skills` | **仅非空时**写入初始 state，避免清空 checkpoint 中已持久化的 per-thread 技能 |
| workspace | `workspace_path` 非 None 时写入 `configurable`，供 `WorkspaceSecurityMW` 解析路径 |
| 进度回调 | 传了任一 `on_node_*` 才挂 `NodeTrackingHandler`，否则零开销 |

---

## 8. 陷阱清单

| 陷阱 | 后果 | 正解 |
| --- | --- | --- |
| 用 `lambda` 绑定 agent | `lambda` 返回未 await 的 coroutine，LangGraph 抛 `InvalidUpdateError` | 用 `register_nodes` / `functools.partial`（保留 coroutine 特征） |
| 经**实例**引用类属性节点：`agents["architect"].plan_node` | 描述符协议把 `self` 绑到 `state` 参数位，调用错乱 | 类属性节点一律经**类**引用：`ArchiAgent.plan_node` |
| 节点返回值 / `initial_fields` 写了未声明的 state 字段名 | LangGraph 静默丢弃该字段，症状是"值永远为空" | 依赖 `_guard_initial_fields_subset` 的 `ValueError` 提前暴露 |
| 两个角色声明同名 `state_fields`，或与 `extra_channels` 撞名 | `compose_state_schema` 抛 `ValueError` | 通道名全局唯一；角色产出归角色、流程控制归 `extra_channels` |
| 条件路由只写 `add_conditional_edges` 不写映射表 | 运行时报未知目标节点 | 显式给 `{路由返回值: 目标节点}` 映射 |
| 新增工作流却忘记让角色模块被 import | `build_workflow` 抛 `KeyError: 缺少角色` | 在 `team/__init__.py` 或工作流模块顶部 from-import 角色类 |
| 期望 builder 收到 `skills_dir` / `max_rounds` 等参数 | `build_workflow` 只传 `agents` 与 `checkpointer`，其余参数走默认值 | 需要非默认值时直接调 `builder(agents, checkpointer=..., max_rounds=N)`，或把默认值写进 `initial_fields` |
| 在节点里自己再做消息压缩 | 与 `LCAgentCompactionMiddleware.before_model` 双重压缩 | 节点级压缩已移除，压缩交给中间件 |
| 终态节点没有写 `final_answer` | 跨轮次记忆与 DONE 事件内容为空 | 所有 `→ END` 路径都要经过写 `final_answer` 的交付节点（如 `designer_output`） |

---

## 9. 参考

- 顶层项目说明：`../README.md`（「多 Agent 工作流」「扩展工作流」章节）
- 角色与节点实现：`../team/`（各角色目录下 `AGENT.md` 写角色提示词与 `## workflow:<名称>` 节点模板）
- 共享校验纯函数（非角色）：`../team/checker/checker.py`
- 测试：`../tests/graph/`
  - `test_compose_state_schema.py` / `test_register_nodes.py` — 构图基建（schema 通道冲突、节点绑定形态）
  - `test_workflow.py` / `test_rtl_graph.py` — 两张图的拓扑与节点通道
  - `test_workflow_memory.py` / `test_workflow_workspace.py` / `test_workflow_runner_active_skills.py` — 运行器能力
  - `test_workflow_interrupt_integration.py` / `test_workflow_failure_recovery_integration.py` — 中断与失败恢复

---

## 10. 案例一：simple.py —— 线性监督者模式

### 10.1 拓扑

```
START → summarize → manager_plan → worker_exec → terminator_final → END
```

| 节点 | 实现 | 产出通道 | 角色 |
| --- | --- | --- | --- |
| `summarize` | `ManagerAgent.summarize_context_node`（实例方法，与 rtl_graph 共用） | `context_summary` | manager |
| `manager_plan` | `ManagerAgent.plan_node`（类属性） | `plan` | manager |
| `worker_exec` | `WorkerAgent.exec_node`（类属性） | `worker_result` | worker |
| `terminator_final` | `TerminatorAgent.final_node`（类属性） | `final_answer` | terminator |

### 10.2 构建函数

```python
# 文件顶部（此处省略）
# from langgraph.graph import END, START, StateGraph
# from graph.common import NodeSpec, register_nodes, register_workflow
# from graph.common.builder_utils import build_graph_scaffold
# from team import ManagerAgent, TerminatorAgent, WorkerAgent

def build_simple_workflow(
    agents: dict,
    checkpointer=None,
    skills_dir: str | None = None,
    auto_match_skills: bool = True,
) -> StateGraph:
    manager = agents["manager"]

    # ① 基建：技能注入器 + State schema（simple 无工作流级额外通道）
    builder, injector = build_graph_scaffold(
        agents, skills_dir=skills_dir, auto_match_skills=auto_match_skills,
    )

    # ② 节点表
    register_nodes(builder, agents, injector, [
        NodeSpec("summarize", manager.summarize_context_node, role="manager"),
        NodeSpec("manager_plan", ManagerAgent.plan_node, role="manager"),
        NodeSpec("worker_exec", WorkerAgent.exec_node, role="worker"),
        NodeSpec("terminator_final", TerminatorAgent.final_node, role="terminator"),
    ])

    # ③ 边
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "manager_plan")
    builder.add_edge("manager_plan", "worker_exec")
    builder.add_edge("worker_exec", "terminator_final")
    builder.add_edge("terminator_final", END)

    # ④ 编译
    return builder.compile(checkpointer=checkpointer)
```

### 10.3 注册

```python
register_workflow(
    "simple",
    builder=build_simple_workflow,
    roles=["manager", "worker", "terminator"],
    description="监督者模式工作流(Manager 拆解→Worker 执行→Terminator 汇总)",
    initial_fields={
        "context_summary": "", "plan": "", "worker_result": "", "final_answer": "",
    },
)
```

**要点**：

- `summarize` 走实例方法分支（`manager.summarize_context_node`），其余走"类属性引用"分支。两者由 `register_nodes` 用 `inspect.ismethod` 自动区分，无需手工标记。
- `worker_exec` 节点的 `config` 会透传到 Worker 的工具链，使工具调用受 `WorkspaceSecurityMW` 约束（`configurable.workspace_path`）。
- `initial_fields` 里的键必须都在 State schema 中声明，否则 `arun_compiled_workflow` 抛 `ValueError`（见 §7.2）。

---

## 11. 案例二：rtl_graph.py —— 带条件路由的多轮迭代流水线

### 11.1 拓扑

```
START
  → summarize                            # Manager 提炼记忆上下文
  → architect_plan                       # Architect 五阶段
  → architect_design
  → architect_analyze
  → architect_review
  → architect_spec
  → designer_spec                        # Designer 规格梳理 + Filelist
  → verification_plan                    # Verification 验证计划
  → designer_verilog                     # ┌─ RTL 编码（迭代环入口）
  → designer_file_check                  # │  文件存在且非空？
      ├─ 通过 ─────────────→ verification_check
      ├─ 未达 max_rounds ──→ designer_verilog（重做）
      └─ 达 max_rounds ────→ designer_output
  → verification_check                   # Testbench / 验证报告
  → sim_exec_check                       # Vivado 仿真 + 覆盖率
      ├─ 通过 ─────────────→ designer_output
      ├─ 未达 max_rounds ──→ designer_verilog（带反馈重新设计）
      └─ 达 max_rounds ────→ designer_output
  → designer_output（零 LLM 机械拼装）
  → END
```

### 11.2 13 个节点

| 节点 | 实现 | 产出通道 | 角色 |
| --- | --- | --- | --- |
| `summarize` | `manager.summarize_context_node`（实例方法） | `context_summary` | manager |
| `architect_plan` | `ArchiAgent.plan_node`（类属性） | `arch_plan` | architect |
| `architect_design` | `ArchiAgent.design_node` | `arch_design` | architect |
| `architect_analyze` | `ArchiAgent.analyze_node` | `arch_analysis` | architect |
| `architect_review` | `ArchiAgent.review_node` | `arch_review` | architect |
| `architect_spec` | `ArchiAgent.spec_node` | `arch_spec` | architect |
| `designer_spec` | `DesignerAgent.spec_node` | `design_spec` | rtl_designer |
| `verification_plan` | `VerificationAgent.plan_node` | `verification_plan` | rtl_verification |
| `designer_verilog` | `DesignerAgent.verilog_node` | `rtl_code` | rtl_designer |
| `designer_file_check` | `rtl_designer.designer_file_check_node`（实例方法，代码节点） | `file_check_passed` | rtl_designer |
| `verification_check` | `VerificationAgent.check_node` | `verification_report` | rtl_verification |
| `sim_exec_check` | `rtl_verification.sim_exec_check_node`（实例方法，跑 Vivado） | `sim_check_passed` / `sim_status` / `sim_log_path` | rtl_verification |
| `designer_output` | `rtl_designer.designer_output_node`（实例方法，零 LLM） | `final_answer` / `output_files` | rtl_designer |

### 11.3 工作流级额外通道

simple 不需要额外通道，rtl 需要一组"控制流 + 校验"字段，经 `extra_channels` 传给 `compose_state_schema`：

```python
builder, injector = build_graph_scaffold(
    agents,
    skills_dir=skills_dir,
    auto_match_skills=auto_match_skills,
    extra_channels={
        "round": int,
        "max_rounds": int,
        "final_answer": str,
        "file_check_passed": bool,
        "sim_check_passed": bool,
        "sim_status": str,
        "sim_log_path": str,
    },
)
```

### 11.4 条件路由

```python
def route_after_file_check(state) -> str:
    if state.get("file_check_passed"):
        return "verification_check"
    if state.get("round", 0) >= state.get("max_rounds", 3):
        return "designer_output"
    return "designer_verilog"

def route_after_sim_check(state) -> str:
    if state.get("sim_check_passed"):
        return "designer_output"
    if state.get("round", 0) >= state.get("max_rounds", 3):
        return "designer_output"
    return "designer_verilog"
```

注册时必须以 `{返回值: 目标节点}` 映射表显式声明每条分支，否则 LangGraph 报未知目标：

```python
builder.add_conditional_edges(
    "designer_file_check", route_after_file_check,
    {"verification_check": "verification_check",
     "designer_output": "designer_output",
     "designer_verilog": "designer_verilog"},
)
```

**双出口设计**：两条终态路径（仿真通过 / 达 `max_rounds` 上限）都先经 `designer_output` 再 `→ END`，保证 `final_answer` 非空——跨轮次记忆沉淀与 DONE 事件内容都依赖它。`designer_output` 不调 LLM、不渲染模板、不做 IO，只按固定小节机械拼装 state 字段（空字段整段省略，全空则回退 `task`）。

### 11.5 其他公开符号

| 符号 | 说明 |
| --- | --- |
| `verification_passed(report: str) -> bool` | 从验证报告文本解析结论：优先匹配"验证结论: PASS/FAIL"标记行；无标记时回退全文扫描（含 PASS 且不含 FAIL 视为通过）。当前图的路由依据 `sim_check_passed` 通道，本函数保留供外部复用与测试（`tests/graph/test_rtl_graph.py`）。 |
| `build_rtl_graph_workflow(agents, checkpointer=None, skills_dir=None, auto_match_skills=True, max_rounds=3)` | 构建函数。注意 `max_rounds` 在 `extra_channels` 里只是**通道类型声明**，实际初始值由 `register_workflow` 的 `initial_fields["max_rounds"] = 3` 提供——`build_workflow` 不传该参数，改上限须同步改这两处。 |

### 11.6 角色注册的隐藏依赖

`team/__init__.py` **不**导入 `rtl_designer` / `rtl_verification`，因此 rtl_graph 模块顶部必须 from-import 三个角色类——既为引用类属性节点，也为了触发它们的 `@register_agent` 完成注册：

```python
from team.architect.Architect import ArchiAgent
from team.rtl_designer.rtl_designer import DesignerAgent
from team.rtl_verification.rtl_verification import VerificationAgent
```

缺少这一步时 `build_workflow("rtl_graph")` 会抛 `KeyError: 工作流 'rtl_graph' 缺少角色: [...]`。
