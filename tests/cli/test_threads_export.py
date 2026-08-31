"""export_thread 命令的单元测试。

覆盖:
- 参数解析(thread_id / fmt / path 三种 token 的归位)
- path NameError bug 修复(默认路径按 fmt 生成正确后缀)
- aexport_session 以正确 fmt 参数调用
- 自定义输出路径写入
- 空会话提示

全部离线,不发起任何 LLM/网络调用。
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cli.commands.threads import export_thread
from cli.commands.types import CommandContext


@dataclass
class FakeSession:
    """模拟 SessionRegistry 的导出接口。"""

    current_session_id: str = "thread-1"
    calls: list[tuple[str | None, str]] = field(default_factory=list)
    return_text: str = "【用户】\n你好"

    async def aexport_session(
        self, session_id: str | None = None, fmt: str = "text"
    ) -> str:
        self.calls.append((session_id, fmt))
        return self.return_text


@dataclass
class FakeAgent:
    """仅暴露 export_thread 所需的 session 属性。"""

    session: FakeSession = field(default_factory=FakeSession)


@dataclass
class StubSafety:
    """safety_backend 占位,export_thread 不使用但 CommandContext 必填。"""

    def load_config(self) -> dict[str, Any]:
        return {}

    def save_config(self, config: dict[str, Any]) -> bool:
        return True


def _make_context(
    tmp_path: Path, session: FakeSession
) -> tuple[CommandContext, list[str]]:
    """构造最小可用的 CommandContext,返回 context 与打印捕获列表。"""
    printed: list[str] = []
    ctx = CommandContext(
        agent=FakeAgent(session=session),
        base_dir=str(tmp_path),
        config_file="",
        mcp_config_file="",
        print_fn=printed.append,
        input_fn=lambda _p="": "",
        select_menu=lambda *a, **k: None,
        create_llm=lambda _p: None,
        list_providers=dict,
        run_structured_until_completion=lambda _a, _t: "",
        chat_until_completion=lambda _a, _t: "",
        safety_backend=StubSafety(),
    )
    return ctx, printed


def test_export_current_session_text(tmp_path: Path) -> None:
    # Given: 当前会话 thread-1。
    session = FakeSession()
    ctx, printed = _make_context(tmp_path, session)
    # When: 输入 export(无参数)。
    asyncio.run(export_thread(ctx, "export"))
    # Then: 以当前会话 + text 调用 aexport_session,默认生成 .txt 文件。
    assert session.calls == [(None, "text")]
    assert (tmp_path / "exports" / "thread-1.txt").exists()
    assert any("格式: text" in line for line in printed)


def test_export_specified_thread(tmp_path: Path) -> None:
    # Given: 任意会话。
    session = FakeSession()
    ctx, _printed = _make_context(tmp_path, session)
    # When: 指定会话 ID 导出。
    asyncio.run(export_thread(ctx, "export:abc-123"))
    # Then: 以该 ID + text 调用,文件名为 abc-123.txt。
    assert session.calls == [("abc-123", "text")]
    assert (tmp_path / "exports" / "abc-123.txt").exists()


def test_export_markdown_format_current_session(tmp_path: Path) -> None:
    # Given: 当前会话 thread-1。
    session = FakeSession()
    ctx, printed = _make_context(tmp_path, session)
    # When: 仅指定 markdown 格式。
    asyncio.run(export_thread(ctx, "export markdown"))
    # Then: fmt=markdown,生成 .md 文件。
    assert session.calls == [(None, "markdown")]
    assert (tmp_path / "exports" / "thread-1.md").exists()
    assert any("格式: markdown" in line for line in printed)


def test_export_specified_thread_with_format(tmp_path: Path) -> None:
    # Given: 任意会话。
    session = FakeSession()
    ctx, _printed = _make_context(tmp_path, session)
    # When: 指定会话 + markdown。
    asyncio.run(export_thread(ctx, "export:abc markdown"))
    # Then: 以 abc + markdown 调用,生成 abc.md。
    assert session.calls == [("abc", "markdown")]
    assert (tmp_path / "exports" / "abc.md").exists()


def test_export_with_custom_path(tmp_path: Path) -> None:
    # Given: 自定义输出路径(绝对路径,含路径分隔符)。
    session = FakeSession()
    ctx, printed = _make_context(tmp_path, session)
    custom = str(tmp_path / "custom.md")
    # When: 指定会话 + markdown + 自定义路径。
    asyncio.run(export_thread(ctx, f"export:abc markdown {custom}"))
    # Then: 写入自定义路径,不使用默认 exports 目录。
    assert session.calls == [("abc", "markdown")]
    assert os.path.isfile(custom)
    assert any("custom.md" in line for line in printed)


def test_export_md_alias_resolves_to_markdown(tmp_path: Path) -> None:
    # Given: 当前会话。
    session = FakeSession()
    ctx, _printed = _make_context(tmp_path, session)
    # When: 使用 md 缩写作为格式关键字。
    asyncio.run(export_thread(ctx, "export md"))
    # Then: 识别为 markdown,生成 .md 文件。
    assert session.calls == [(None, "markdown")]
    assert (tmp_path / "exports" / "thread-1.md").exists()


def test_export_empty_session_prints_hint(tmp_path: Path) -> None:
    # Given: 会话无可导出消息。
    session = FakeSession(return_text="   ")
    ctx, printed = _make_context(tmp_path, session)
    # When: 尝试导出。
    asyncio.run(export_thread(ctx, "export"))
    # Then: 提示无可导出消息,且不生成文件。
    assert any("没有可导出的消息" in line for line in printed)
    assert not (tmp_path / "exports").exists() or not list(
        (tmp_path / "exports").iterdir()
    )


def test_export_no_nameerror_on_default_path(tmp_path: Path) -> None:
    """回归测试:修复前 `if not path` 会抛 NameError。

    任何成功导出的路径都应被正确生成,不再引用未定义变量。
    """
    session = FakeSession()
    ctx, _printed = _make_context(tmp_path, session)
    # When/Then: 不抛 NameError,正常生成文件。
    asyncio.run(export_thread(ctx, "export"))
    assert (tmp_path / "exports" / "thread-1.txt").exists()
