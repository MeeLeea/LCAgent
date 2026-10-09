"""checker — RTL 工作流的文件校验共享工具。

承载原 ``graph/rtl_graph.py`` 中被多角色节点复用的纯函数:
- ``parse_filelist``: 解析 filelist(.f) 得到源文件相对路径列表
  (designer_verilog 节点解析产出清单、sim_exec_check 覆盖率兜底共用)
- ``coverage_inclusion_ok``: syn_filelist ⊆ sim_filelist 的集合包含判定
  (sim_exec_check 节点的覆盖率兜底检查)
- ``write_text_file``: 同步写文本文件(供 asyncio.to_thread 在线程外调用,
  sim_exec_check 写仿真日志用)

本模块不依赖 graph/team 任何一侧,可被他包安全 import(无循环)。
"""
from __future__ import annotations

import os
from typing import Optional

__all__ = ["coverage_inclusion_ok", "parse_filelist", "write_text_file"]


def parse_filelist(workspace: Optional[str], rel_path: str) -> list[str]:  # noqa: UP045
    """解析 filelist(.f),返回其中的源文件相对路径列表。

    跳过空行、`#`/`//` 注释、`+incdir+`/`-f`/`+` 开头的指令行;仅保留
    以 .v/.sv/.vhd/.vhdl 结尾的 token(取行内首个 token,忽略行内注释与多余参数)。
    """
    abs_p = rel_path if os.path.isabs(rel_path) else os.path.join(workspace, rel_path) if workspace else rel_path
    if not os.path.exists(abs_p):
        return []
    files: list[str] = []
    with open(abs_p, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(("#", "//")):
                continue
            if line.startswith(("+", "-")):
                continue
            token = line.split()[0]
            if token.endswith((".v", ".sv", ".vhd", ".vhdl")):
                files.append(token)
    return files


def coverage_inclusion_ok(workspace: Optional[str]) -> bool:  # noqa: UP045
    """覆盖率兜底:scripts/syn_filelist.f 的全部 src 文件必须被 scripts/sim_filelist.f 包含。

    使用集合包含判定(对 filelist 语法稳定,比解析 LLM 生成的 tcl 可靠)。
    """
    syn = parse_filelist(workspace, "scripts/syn_filelist.f")
    sim = parse_filelist(workspace, "scripts/sim_filelist.f")
    if not syn:
        return False
    syn_set = {os.path.normpath(p) for p in syn}
    sim_set = {os.path.normpath(p) for p in sim}
    return syn_set.issubset(sim_set)


def write_text_file(path: str, text: str) -> None:
    """同步写文本文件(供 asyncio.to_thread 在事件循环外调用,避免阻塞)。"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
