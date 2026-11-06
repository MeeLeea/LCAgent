"""Planner 启动器（launcher）的离线单元测试。

覆盖范围：
    - pid_is_alive：非法 PID（None/0/负数）判定
    - pid_is_alive：真实子进程存活 -> 退出 -> 高频轮询不抛异常（句柄不泄漏）
    - preflight：默认 provider 选取与显式 provider/model 覆盖
    - preflight：缺 API Key / 未知 provider / 提示词缺失或为空时抛 ValueError
    - spawn_worker：非法任务 ID 抛 ValueError，任务目录不存在抛 LauncherError

全部用例离线运行，不启动真实 worker、不访问网络。
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from subagent.common.config import PlannerConfig
from subagent.planner import launcher
from subagent.planner.launcher import LauncherError, pid_is_alive, preflight, spawn_worker

# ================================================================ Fixtures ===


@pytest.fixture(autouse=True)
def _reset_launcher_state() -> Any:
    """每个测试后重置 launcher 模块级缓存的 kernel32 句柄。"""
    yield
    launcher._kernel32 = None


def _make_prompt_file(tmp_path: Path, content: str = "# Planner 提示词\n") -> str:
    """创建非空提示词文件并返回路径。"""
    path = tmp_path / "AGENT.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


def _fake_providers() -> dict[str, dict[str, Any]]:
    """构造一个带 key/model 的假 provider 定义。"""
    return {
        "fake": {
            "name": "Fake",
            "api_key": "k",
            "model": "m1",
            "env_key": "FAKE_KEY",
        }
    }


# ================================================================ pid_is_alive ===


def test_pid_is_alive_invalid_pids() -> None:
    """None、0、负数一律视为不存活。"""
    assert pid_is_alive(None) is False
    assert pid_is_alive(0) is False
    assert pid_is_alive(-1) is False


def test_pid_is_alive_real_subprocess_lifecycle() -> None:
    """真实子进程：运行期存活，退出后不存活，高频轮询不抛异常。"""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"])
    try:
        assert pid_is_alive(proc.pid) is True
        proc.wait(timeout=15)
        assert pid_is_alive(proc.pid) is False
        for _ in range(200):
            assert pid_is_alive(proc.pid) is False
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=15)


# ================================================================ preflight ===


def test_preflight_default_provider_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未显式指定时选取首个可用 provider，并使用其默认 model。"""
    monkeypatch.setattr(
        "llm.llm_client.load_providers", lambda config_file=None: _fake_providers()
    )
    config = dataclasses.replace(PlannerConfig(), agent_prompt_file=_make_prompt_file(tmp_path))
    assert preflight(config) == ("fake", "m1")


def test_preflight_explicit_provider_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """显式 provider/model 覆盖默认值。"""
    monkeypatch.setattr(
        "llm.llm_client.load_providers", lambda config_file=None: _fake_providers()
    )
    config = dataclasses.replace(
        PlannerConfig(),
        provider="fake",
        model="m2",
        agent_prompt_file=_make_prompt_file(tmp_path),
    )
    assert preflight(config) == ("fake", "m2")


def test_preflight_missing_api_key_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """provider 无 api_key 且环境变量未设置时抛 ValueError。"""
    providers: dict[str, dict[str, Any]] = {
        "fake": {"name": "Fake", "api_key": "", "model": "m1", "env_key": "FAKE_KEY"}
    }
    monkeypatch.setattr("llm.llm_client.load_providers", lambda config_file=None: providers)
    monkeypatch.delenv("FAKE_KEY", raising=False)
    config = dataclasses.replace(PlannerConfig(), agent_prompt_file=_make_prompt_file(tmp_path))
    with pytest.raises(ValueError):
        preflight(config)


def test_preflight_unknown_provider_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """显式指定的 provider 不存在时抛 ValueError。"""
    monkeypatch.setattr(
        "llm.llm_client.load_providers", lambda config_file=None: _fake_providers()
    )
    config = dataclasses.replace(
        PlannerConfig(),
        provider="nope",
        agent_prompt_file=_make_prompt_file(tmp_path),
    )
    with pytest.raises(ValueError):
        preflight(config)


def test_preflight_missing_or_empty_prompt_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """提示词文件缺失或为空时抛 ValueError。"""
    monkeypatch.setattr(
        "llm.llm_client.load_providers", lambda config_file=None: _fake_providers()
    )
    missing = dataclasses.replace(
        PlannerConfig(), agent_prompt_file=str(tmp_path / "missing.md")
    )
    with pytest.raises(ValueError):
        preflight(missing)

    empty_path = tmp_path / "empty.md"
    empty_path.write_text("", encoding="utf-8")
    empty = dataclasses.replace(PlannerConfig(), agent_prompt_file=str(empty_path))
    with pytest.raises(ValueError):
        preflight(empty)


# ================================================================ spawn_worker ===


def test_spawn_rejects_unsafe_task_id(tmp_path: Path) -> None:
    """非法任务 ID 抛 ValueError。"""
    for bad_task_id in ("../x", "a/b"):
        with pytest.raises(ValueError):
            spawn_worker(bad_task_id, root=str(tmp_path), config=PlannerConfig())


def test_spawn_missing_task_dir_raises(tmp_path: Path) -> None:
    """任务目录不存在时抛 LauncherError（不真正启动 worker）。"""
    with pytest.raises(LauncherError):
        spawn_worker("pl-missing", root=str(tmp_path), config=PlannerConfig())
