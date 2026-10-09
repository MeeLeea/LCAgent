# Checker

承载 RTL 芯片设计流水线中**与具体角色无关的校验工具函数**（filelist 解析、覆盖率包含判定、同步写文件），供 designer / verification 两侧节点复用。

## 职责

1. `checker.py` 提供纯函数工具，不持有 LLM、不注册为工作流角色、无 `state_fields`。
2. 被 `graph/rtl_graph.py` 的校验类节点（designer_file_check / sim_exec_check / designer_verilog 的产出解析）import 使用。
3. 未来若校验逻辑扩展为独立的 LLM 角色（如自动审查 RTL 文件质量），可在本目录注册为正式角色；当前阶段仅作为共享工具库存在。

## 约束

- 本目录不 import `graph/` 或 `team/` 的任何模块，保持零依赖（仅标准库），避免引入循环导入。
