"""子程序公共工具包（stdlib-only）。

存放 ``planner`` 与 ``executor`` 两个后台子程序共享的、只依赖标准库的轻量工具：

- :mod:`subagent.common.fsx` — 原子写入 / 文件名校验 / 任务 ID 生成
- :mod:`subagent.common.proc` — PID 存活探测 / 分离进程启动
- :mod:`subagent.common.paths` — 路径归一化 / 工作区包含关系校验 / 拒绝清单

设计约束（与 ``subagent.planner`` 的导入纪律一致）：
本包**禁止**在模块级导入 ``langchain`` / ``langgraph`` / ``agent`` / ``team`` /
``graph`` / ``midware`` / ``tools`` 等项目重依赖，保证后台子进程（worker）的
导入链保持轻量。所有共享件的实现均为标准库级别。
"""
