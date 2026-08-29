"""mcp_all 特性(加载全部 MCP 工具)的单元测试。

覆盖两个层次:
1. load_all_mcp_tools_sync 单元: 全量加载、去重、静默降级、空配置
2. register_agent(mcp_all=True) 装配: _build 内 mcp_all 分支、
   与 mcp_tools 互斥 warning、加载失败降级、默认 False 不影响现有行为

不依赖真实 MCP server: 通过 patch load_mcp_tools / load_all_mcp_tools_sync
返回伪造工具列表,验证装配逻辑契约。

现有按名加载路径的测试见 test_mcp_loader.py;
真实 MCP 集成测试见 test_mcp_filesystem.py(标记 slow)。
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from graph.registry import AGENT_REGISTRY, build_workflow, register_agent
from team.base import TeamAgent


@dataclass
class FakeTool:
    """最小工具桩: 只携带 name 属性,供装配测试使用。

    不继承 StructuredTool/BaseTool,避免 pydantic args_schema 校验开销;
    _build 与 load_all_mcp_tools_sync 只读 .name 字段,不调 .invoke(),
    FakeTool 完全满足测试契约。
    """

    name: str


def _make_fake_tool(name: str) -> FakeTool:
    """构造一个最小工具桩。"""
    return FakeTool(name=name)


# ==================== load_all_mcp_tools_sync 单元测试 ====================


class TestLoadAllMcpToolsSync:
    """load_all_mcp_tools_sync 同步全量加载逻辑。

    策略: 只 patch load_mcp_tools(异步函数),让真实 _run_async 执行它,
    避免 patch _run_async 导致未 await 的 coroutine 警告。
    """

    def test_returns_all_loaded_tools(self):
        """load_mcp_tools 返回多个工具时,全量透传(不过滤)。"""
        fake_tools = [
            _make_fake_tool("write_file"),
            _make_fake_tool("read_file"),
            _make_fake_tool("list_directory"),
        ]

        async def _fake_load(config_file=None):
            return fake_tools

        with patch("tools.mcp_loader.load_mcp_tools", new=_fake_load):
            from tools.mcp_loader import load_all_mcp_tools_sync

            result = load_all_mcp_tools_sync()

        names = [t.name for t in result]
        assert set(names) == {"write_file", "read_file", "list_directory"}
        assert len(result) == 3

    def test_dedup_same_name_tools(self):
        """多个 server 提供同名工具时,按先到者保留,丢弃后到者。"""
        fake_tools = [
            _make_fake_tool("write_file"),  # filesystem 先到
            _make_fake_tool("write_file"),  # 另一 server 同名,丢弃
        ]

        async def _fake_load(config_file=None):
            return fake_tools

        with patch("tools.mcp_loader.load_mcp_tools", new=_fake_load):
            from tools.mcp_loader import load_all_mcp_tools_sync

            result = load_all_mcp_tools_sync()

        assert len(result) == 1
        assert result[0].name == "write_file"

    def test_empty_when_no_enabled_server(self):
        """load_mcp_tools 返回空(无 enabled server)时,透传空列表。"""
        async def _fake_load(config_file=None):
            return []

        with patch("tools.mcp_loader.load_mcp_tools", new=_fake_load):
            from tools.mcp_loader import load_all_mcp_tools_sync

            result = load_all_mcp_tools_sync()

        assert result == []

    def test_silently_degrade_on_exception(self):
        """load_mcp_tools 抛异常时,WARNING + 返回空列表,不向上抛。

        模拟 server 连接异常经 _run_async 传播到 load_all_mcp_tools_sync,
        被 try/except 捕获并降级。不 patch _run_async,避免生成未 await
        的真实 load_mcp_tools coroutine。
        """
        async def _boom_load(config_file=None):
            raise RuntimeError("mcp server down")

        with patch("tools.mcp_loader.load_mcp_tools", new=_boom_load):
            from tools.mcp_loader import load_all_mcp_tools_sync

            result = load_all_mcp_tools_sync()

        assert result == []

    def test_no_dedup_when_names_unique(self):
        """无同名冲突时,去重逻辑不影响结果(全保留)。"""
        fake_tools = [
            _make_fake_tool("write_file"),
            _make_fake_tool("search"),
            _make_fake_tool("code_graph"),
        ]

        async def _fake_load(config_file=None):
            return fake_tools

        with patch("tools.mcp_loader.load_mcp_tools", new=_fake_load):
            from tools.mcp_loader import load_all_mcp_tools_sync

            result = load_all_mcp_tools_sync()

        assert len(result) == 3


# ==================== register_agent(mcp_all=True) 装配测试 ====================


def _register_fake_role(
    role_name: str,
    config_file: str = "team/architect/agent_config.json",
    *,
    tools=None,
    mcp_tools=None,
    mcp_all=False,
):
    """注册一个 fake 角色到 AGENT_REGISTRY,返回类对象。

    用 try/finally 由调用方负责清理,避免污染全局注册表。
    """
    @register_agent(role_name, config_file, tools=tools, mcp_tools=mcp_tools, mcp_all=mcp_all)
    class FakeAgent(TeamAgent):
        pass

    return FakeAgent


def _patch_workflow_spec(reg_mod, workflow_name: str, roles: list[str]):
    """patch registry._get_workflow_spec 与 WORKFLOWS,返回伪 spec 避免真实图编译。

    调用方负责在 finally 里恢复 reg_mod._get_workflow_spec 与 reg_mod.WORKFLOWS。
    """
    original_get_spec = reg_mod._get_workflow_spec

    def _fake_get_spec(name):
        return {
            "builder": lambda agents, checkpointer=None: type(
                "FakeGraph", (), {"get_graph": lambda self: type(
                    "G", (), {"nodes": []}
                )()}
            )(),
            "runner": None,
            "roles": roles,
            "description": "test",
        }

    reg_mod._get_workflow_spec = _fake_get_spec
    reg_mod.WORKFLOWS[workflow_name] = {
        "builder": _fake_get_spec(workflow_name)["builder"],
        "runner": None,
        "roles": roles,
        "description": "test",
    }
    return original_get_spec


def _patch_build_team_agent(team_mod, captured: dict):
    """patch team.build_team_agent,捕获 tools 参数并返回最小实例。

    调用方负责在 finally 里恢复 team_mod.build_team_agent。
    """
    original_build = team_mod.build_team_agent

    def _spy_build(agent_class, config_file, base_dir, tools=None, **kwargs):
        captured[agent_class.__name__] = tools
        inst = object.__new__(agent_class)
        inst.tools = tools or []
        inst.agent_executor = None
        return inst

    team_mod.build_team_agent = _spy_build
    return original_build


def test_register_agent_mcp_all_passthrough():
    """装饰器 mcp_all 参数原样透传到注册表(默认 False,显式 True 时存 True)。"""
    # 显式声明 mcp_all=True
    _register_fake_role("fake_mcp_all_role", mcp_all=True)
    try:
        spec = AGENT_REGISTRY["fake_mcp_all_role"]
        assert spec["mcp_all"] is True
    finally:
        AGENT_REGISTRY.pop("fake_mcp_all_role", None)

    # 不传 mcp_all 时默认 False(向后兼容)
    _register_fake_role("fake_mcp_all_default")
    try:
        assert AGENT_REGISTRY["fake_mcp_all_default"]["mcp_all"] is False
    finally:
        AGENT_REGISTRY.pop("fake_mcp_all_default", None)


def test_build_workflow_mcp_all_injection_distinct_classes():
    """mcp_all=True 加载全部;无 mcp_all 角色不受影响。

    用不同类名避免 captured 字典键冲突。
    """
    fake_tools = [_make_fake_tool("write_file"), _make_fake_tool("read_file")]
    captured: dict = {}

    # 用不同类名,让 captured 字典能区分
    @register_agent(
        "fake_all_role",
        "team/architect/agent_config.json",
        mcp_all=True,
    )
    class FakeAllAgent(TeamAgent):
        pass

    @register_agent(
        "fake_none_role",
        "team/manager/agent_config.json",
    )
    class FakeNoneAgent(TeamAgent):
        pass

    try:
        import team as team_mod
        from graph import registry as reg_mod

        original_build = _patch_build_team_agent(team_mod, captured)
        original_get_spec = _patch_workflow_spec(
            reg_mod, "__test_mcp_all_distinct__",
            ["fake_all_role", "fake_none_role"],
        )
        try:
            with patch(
                "tools.mcp_loader.load_all_mcp_tools_sync",
                return_value=fake_tools,
            ):
                build_workflow("__test_mcp_all_distinct__", checkpointer=None)

            # mcp_all=True 角色: 含全部 2 个工具
            assert captured.get("FakeAllAgent") is not None
            assert len(captured["FakeAllAgent"]) == 2
            # 无 mcp_all 角色: tools 为 None
            assert captured.get("FakeNoneAgent") is None
        finally:
            team_mod.build_team_agent = original_build
            reg_mod._get_workflow_spec = original_get_spec
            reg_mod.WORKFLOWS.pop("__test_mcp_all_distinct__", None)
    finally:
        AGENT_REGISTRY.pop("fake_all_role", None)
        AGENT_REGISTRY.pop("fake_none_role", None)


def test_build_workflow_mcp_all_and_mcp_tools_mutex_warning(caplog):
    """mcp_all=True 与 mcp_tools 同时声明时,mcp_all 优先 + warning。

    验证:
    - warning 文案含 mcp_all 优先提示
    - 只调 load_all_mcp_tools_sync,不调 load_mcp_tools_by_name_sync
    - tools 含全部工具(非仅 mcp_tools 列出的子集)
    """
    fake_all_tools = [
        _make_fake_tool("write_file"),
        _make_fake_tool("read_file"),
        _make_fake_tool("search"),
    ]
    captured: dict = {}

    @register_agent(
        "fake_mutex_role",
        "team/architect/agent_config.json",
        mcp_tools=["write_file"],
        mcp_all=True,
    )
    class FakeMutexAgent(TeamAgent):
        pass

    try:
        import team as team_mod
        from graph import registry as reg_mod

        original_build = _patch_build_team_agent(team_mod, captured)
        original_get_spec = _patch_workflow_spec(
            reg_mod, "__test_mutex__", ["fake_mutex_role"],
        )
        try:
            with patch(
                "tools.mcp_loader.load_all_mcp_tools_sync",
                return_value=fake_all_tools,
            ) as mock_all, patch(
                "tools.mcp_loader.load_mcp_tools_by_name_sync",
                return_value=[_make_fake_tool("write_file")],
            ) as mock_by_name:
                build_workflow("__test_mutex__", checkpointer=None)

                # 只调全量,不调按名
                assert mock_all.called
                assert not mock_by_name.called

            # warning 含互斥提示
            warnings = [r.message for r in caplog.records if "mcp_all" in r.message]
            assert any("mcp_all 优先" in w for w in warnings)

            # tools 含全部 3 个(非仅 write_file)
            assert captured.get("FakeMutexAgent") is not None
            assert len(captured["FakeMutexAgent"]) == 3
        finally:
            team_mod.build_team_agent = original_build
            reg_mod._get_workflow_spec = original_get_spec
            reg_mod.WORKFLOWS.pop("__test_mutex__", None)
    finally:
        AGENT_REGISTRY.pop("fake_mutex_role", None)


def test_build_workflow_mcp_all_degrade_on_empty(caplog):
    """mcp_all 加载失败(返回空)时,角色降级为纯文本模式(tools=None)。"""
    captured: dict = {}

    @register_agent(
        "fake_degrade_role",
        "team/architect/agent_config.json",
        mcp_all=True,
    )
    class FakeDegradeAgent(TeamAgent):
        pass

    try:
        import team as team_mod
        from graph import registry as reg_mod

        original_build = _patch_build_team_agent(team_mod, captured)
        original_get_spec = _patch_workflow_spec(
            reg_mod, "__test_degrade__", ["fake_degrade_role"],
        )
        try:
            with patch(
                "tools.mcp_loader.load_all_mcp_tools_sync",
                return_value=[],
            ):
                build_workflow("__test_degrade__", checkpointer=None)

            # 加载失败 → tools 为 None(or None 生效)
            assert captured.get("FakeDegradeAgent") is None
            # 降级 warning
            warnings = [
                r.message for r in caplog.records
                if "降级为纯文本模式" in r.message
            ]
            assert len(warnings) >= 1
        finally:
            team_mod.build_team_agent = original_build
            reg_mod._get_workflow_spec = original_get_spec
            reg_mod.WORKFLOWS.pop("__test_degrade__", None)
    finally:
        AGENT_REGISTRY.pop("fake_degrade_role", None)


def test_mcp_all_false_preserves_existing_behavior():
    """mcp_all=False(默认)时,_build 走原 mcp_tools 分支,不影响现有行为。

    回归测试: 现有 worker/architect 等角色不声明 mcp_all,
    装配路径应与改动前完全一致(走 load_mcp_tools_by_name_sync)。
    """
    fake_write_file = _make_fake_tool("write_file")
    captured: dict = {}

    @register_agent(
        "fake_regression_role",
        "team/architect/agent_config.json",
        mcp_tools=["write_file"],
        # 不传 mcp_all,默认 False
    )
    class FakeRegressionAgent(TeamAgent):
        pass

    try:
        import team as team_mod
        from graph import registry as reg_mod

        original_build = _patch_build_team_agent(team_mod, captured)
        original_get_spec = _patch_workflow_spec(
            reg_mod, "__test_regression__", ["fake_regression_role"],
        )
        try:
            # mcp_all=False 时应走 load_mcp_tools_by_name_sync,不调 load_all
            with patch(
                "tools.mcp_loader.load_mcp_tools_by_name_sync",
                return_value=[fake_write_file],
            ) as mock_by_name, patch(
                "tools.mcp_loader.load_all_mcp_tools_sync",
                return_value=[],
            ) as mock_all:
                build_workflow("__test_regression__", checkpointer=None)

                # 走按名路径,不走全量
                assert mock_by_name.called
                assert not mock_all.called

            # tools 含 write_file(按名加载的子集)
            assert captured.get("FakeRegressionAgent") is not None
            assert len(captured["FakeRegressionAgent"]) == 1
            assert captured["FakeRegressionAgent"][0].name == "write_file"
        finally:
            team_mod.build_team_agent = original_build
            reg_mod._get_workflow_spec = original_get_spec
            reg_mod.WORKFLOWS.pop("__test_regression__", None)
    finally:
        AGENT_REGISTRY.pop("fake_regression_role", None)
