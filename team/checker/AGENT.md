# Checker

> **注意：本目录不是工作流角色。** 本文件仅为说明文档，不会被
> `team/team_agents.json` / 角色扫描 / `TeamAgent` 加载；checker 目录
> 也不含 `agent_config.json`、无 `@register_agent` 注册。

承载 RTL 芯片设计流水线中**与具体角色无关的校验工具函数**（filelist 解析、覆盖率包含判定、同步写文件），供 designer / verification 两侧节点复用。

## 职责

1. `checker.py` 提供纯函数工具，不持有 LLM、不注册为工作流角色、无 `state_fields`。
2. 被 `graph/rtl_graph.py` 的校验类节点（designer_file_check / sim_exec_check / designer_verilog 的产出解析）import 使用。
3. 未来若校验逻辑扩展为独立的 LLM 角色（如自动审查 RTL 文件质量），需新增正式角色目录（含 `team_agents.json` 条目 + `AGENT.md` 提示词 + `@register_agent` 注册），而非复用本目录；当前阶段仅作为共享工具库存在。

## 约束

- 本目录不 import `graph/` 或 `team/` 的任何模块，保持零依赖（仅标准库），避免引入循环导入。
