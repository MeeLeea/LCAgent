"""文件系统任务存储的公共工具（原子写入 / 校验 / 任务 ID）。

全部实现只依赖标准库；语义与 ``subagent.planner.store`` 中的同名私有实现保持
一致，供 ``ExecutorStore`` 复用（planner 自身的实现保持原样不动）。
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any

# 追加在截断文本末尾的提示（与 planner 的语义一致）
TRUNCATION_SUFFIX: str = "\n\n...[已截断]"


def now_iso() -> str:
    """返回带本地时区信息的 ISO 时间戳（秒级精度）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_task_id(prefix: str) -> str:
    """生成任务 ID，格式 ``<prefix>-YYYYmmdd-HHMMSS-<8 位十六进制>``。"""
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


def is_safe_component(value: str) -> bool:
    """判断 ``value`` 是否为安全的单层文件名/目录名片段（拒绝 ``..`` 与路径分隔符）。"""
    if not value or value in (".", "..") or ".." in value:
        return False
    for sep in (os.sep, os.altsep, "/", "\\"):
        if sep and sep in value:
            return False
    return True


def validate_bare_name(name: str) -> None:
    """校验 ``name`` 为裸文件名；非法时抛出 ``ValueError``。"""
    if not is_safe_component(name):
        raise ValueError(f"非法文件名: {name!r}")


def atomic_write_json(path: str, data: dict[str, Any]) -> None:
    """原子写入 JSON 文件：同目录临时文件 + ``os.replace``。"""
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_text(path: str, text: str) -> None:
    """原子写入文本文件：同目录临时文件 + ``os.replace``。"""
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_task_json(path: str) -> dict[str, Any] | None:
    """读取 JSON 文件；文件缺失、非文件或 JSON 损坏时返回 ``None``。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


def truncate_text(text: str, max_chars: int) -> str:
    """把文本截断到 ``max_chars``（``0`` 表示不截断），超长时追加截断提示。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_SUFFIX


__all__ = [
    "TRUNCATION_SUFFIX",
    "atomic_write_json",
    "atomic_write_text",
    "is_safe_component",
    "load_task_json",
    "new_task_id",
    "now_iso",
    "truncate_text",
    "validate_bare_name",
]
