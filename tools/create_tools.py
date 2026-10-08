import ast
import io
import os
import re
import textwrap
import tokenize
from typing import Final

from langchain.tools import tool

# 默认工具存放目录：create_tools.py 的同级目录
DEFAULT_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认工具测试存放目录：项目根目录下的 tests/tools
DEFAULT_TEST_DIR = os.path.join(os.path.dirname(DEFAULT_TOOL_DIR), "tests", "tools")

# args_spec 声明的参数类型 → 生成单元测试时的示例入参源码（未命中的类型回退字符串 "test"）
_SAMPLE_VALUES: Final[dict[str, str]] = {
    "str": '"test"',
    "int": "1",
    "float": "1.0",
    "bool": "True",
    "list": "[1, 2, 3]",
    "dict": '{"key": "value"}',
}

# Python 3.12+ (PEP 701) 引入了 f-string 专用 token 类型；
# Python 3.10/3.11 中 f-string 被当作普通 STRING 处理，getattr 安全回退。
_STRING_TOKEN_TYPES: Final[frozenset[int]] = frozenset(
    t for t in (
        tokenize.STRING,
        getattr(tokenize, "FSTRING_START", None),
        getattr(tokenize, "FSTRING_END", None),
        getattr(tokenize, "FSTRING_MIDDLE", None),
    )
    if t is not None
)
DANGEROUS_IMPORTS: Final[frozenset[str]] = frozenset(
    {
        "ctypes",
        "importlib",
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "urllib",
    }
)
TOOL_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def _build_error_result(tool_name: str, error: str) -> dict[str, str | bool | None]:
    return {
        "success": False,
        "error": f"工具生成失败：{error}",
        "source_code": None,
        "tool_name": tool_name,
        "file_path": None,
        "registered": False,
    }


def _validate_tool_name(tool_name: str) -> str | None:
    if not TOOL_NAME_PATTERN.fullmatch(tool_name):
        return "工具名必须是合法 Python 标识符，且不能以下划线开头"
    return None


def _build_test_error_result(tool_name: str, error: str) -> dict[str, str | bool | None]:
    return {
        "success": False,
        "error": f"工具测试生成失败：{error}",
        "test_source_code": None,
        "tool_name": tool_name,
        "test_file_path": None,
    }


def _is_within_dir(path: str, root: str) -> bool:
    real_root = os.path.realpath(root)
    target = os.path.realpath(path)
    try:
        return os.path.commonpath([real_root, target]) == real_root
    except ValueError:
        return False


def _resolve_tool_path(tool_name: str, tool_path: str | None) -> str:
    if not tool_path:
        return os.path.join(DEFAULT_TOOL_DIR, f"{tool_name}.py")
    if os.path.isdir(tool_path):
        return os.path.join(tool_path, f"{tool_name}.py")
    return tool_path


def _resolve_test_path(tool_name: str, test_path: str | None) -> str:
    if not test_path:
        return os.path.join(DEFAULT_TEST_DIR, f"test_{tool_name}.py")
    if os.path.isdir(test_path):
        return os.path.join(test_path, f"test_{tool_name}.py")
    return test_path


def _parse_args_spec(args_spec: str) -> list[tuple[str, str, str]]:
    """
    解析 args_spec 为 (参数名, 参数类型, 参数说明) 列表。

    每项格式为 `参数名:参数类型=参数说明`，项之间以分号分隔；格式非法时抛 ValueError，
    由调用方统一转成错误结果返回。
    """
    specs: list[tuple[str, str, str]] = []
    for item in (x.strip() for x in args_spec.split(";")):
        if not item:
            continue
        name_type, desc = item.split("=", maxsplit=1)
        param_name, param_type = name_type.split(":")
        specs.append((param_name.strip(), param_type.strip(), desc.strip()))
    return specs


def _find_disallowed_import(source_code: str) -> str | None:
    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=names):
                for alias in names:
                    root_name = alias.name.split(".", maxsplit=1)[0]
                    if root_name in DANGEROUS_IMPORTS:
                        return alias.name
            case ast.ImportFrom(module=module):
                if module is None:
                    continue
                root_name = module.split(".", maxsplit=1)[0]
                if root_name in DANGEROUS_IMPORTS:
                    return module
            case _:
                continue
    return None


def _sync_tools_init(tool_name: str, init_path: str) -> bool:
    """
    将新工具注册到 tools/__init__.py：导入 + all_tools + __all__。
    返回是否发生了修改；幂等，重复调用不会重复插入。
    """
    if not os.path.exists(init_path):
        return False

    with open(init_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    changed = False

    # 1. 导入：插入到本地工具导入块末尾（all_tools 定义之前）
    import_line = f"from .{tool_name} import {tool_name}"
    if import_line not in lines:
        insert_idx = -1
        for i, line in enumerate(lines):
            if line.startswith("from ."):
                insert_idx = i
            elif "all_tools" in line and insert_idx >= 0:
                break
        if insert_idx >= 0:
            lines.insert(insert_idx + 1, import_line)
            changed = True

    # 2. all_tools 列表：插入到列表结尾（闭括号 ] 之前）
    all_tools_entry = f"    {tool_name},"
    if all_tools_entry not in lines:
        start = next((i for i, l in enumerate(lines) if "all_tools" in l and "[" in l), -1)
        end = next((i for i in range(start + 1, len(lines)) if "]" in lines[i]), -1)
        if start >= 0 and end >= 0:
            lines.insert(end, all_tools_entry)
            changed = True

    # 3. __all__ 列表：同样插入到结尾
    all_entry = f"    '{tool_name}',"
    if all_entry not in lines:
        start = next((i for i, l in enumerate(lines) if "__all__" in l and "[" in l), -1)
        end = next((i for i in range(start + 1, len(lines)) if "]" in lines[i]), -1)
        if start >= 0 and end >= 0:
            lines.insert(end, all_entry)
            changed = True

    if changed:
        with open(init_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return changed


def _multiline_string_rows(body: str) -> set[int]:
    """
    找出处于多行字符串（含三引号 f-string）内部的物理行号（从 1 起始）。

    这些行的内容属于字符串字面量，重新缩进会改变运行时值，必须保持原样。
    使用 tokenize 解析，避免手动扫描引号配对时的转义、嵌套等边界问题。
    """
    rows: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(body).readline)
        for token in tokens:
            if token.type not in _STRING_TOKEN_TYPES:
                continue
            start_row, end_row = token.start[0], token.end[0]
            if end_row > start_row:
                # 起始行仍是代码行（写的是 result = f"""），需要缩进；
                # 中间与结尾行是字符串内容/闭合引号，跳过。
                rows.update(range(start_row + 1, end_row + 1))
    except (IndentationError, tokenize.TokenError):
        # 语法不完整时无法安全分词，退回按行整体缩进，交由后续 ast.parse 拦截非法代码。
        pass
    return rows


def _indent_body(body: str, prefix: str = "        ") -> str:
    """
    给工具逻辑整体增加一级缩进（默认 8 空格，对应 try 块内部）。

    先 dedent 归一化用户传入的相对缩进，再对非字符串内容行统一加前缀，
    避免多行 f-string / 三引号字符串的内容行被误加空格而改变运行时值。
    """
    if not body.strip():
        return ""
    body = textwrap.dedent(body).strip("\n")
    protected_rows = _multiline_string_rows(body)
    lines = body.split("\n")
    result = []
    for index, line in enumerate(lines, start=1):
        if line.strip() and index not in protected_rows:
            result.append(prefix + line)
        else:
            result.append(line)
    return "\n".join(result)


@tool
def create_tool(
    tool_name: str,
    tool_description: str,
    args_spec: str,
    tool_logic: str,
    tool_path: str | None = None,
    force: bool = False,
    with_test: bool = True,
) -> dict[str, str | bool | None]:
    """
    动态生成Langchain标准@tool装饰器工具源码。
    使用统一规范模板输出可直接运行的Python工具代码，遵循项目统一返回结构。
    生成后的代码可以直接写入py文件，导入到Agent工具集中使用。
    默认顺带生成对应的pytest单元测试，保存到 tests/tools/test_<tool_name>.py。

    Args:
        tool_name: 工具函数名，仅小写字母、下划线，例如 "read_markdown_file"
        tool_description: 工具文档字符串，说明能力、用途、适用场景，清晰列出工具的边界条件
        args_spec: 参数定义说明，每个参数格式：参数名:参数类型=参数说明
                   示例："file_path:str=本地文件路径;encoding:str=utf-8文件编码，可选"
        tool_logic: 工具主体业务逻辑（函数内部实现代码，不要写函数定义、装饰器）
        tool_path: 工具存放的路径，可传目录或.py文件路径；
                   为空时默认保存到 create_tools.py 同级目录（tools/）下 tool_name.py
        force: 是否覆盖已存在文件，默认禁止覆盖
        with_test: 是否顺带生成单元测试（默认生成到 tests/tools/ 下）

    Returns:
        字典包含生成的完整源码、状态，成功可直接写入文件运行
    """
    try:
        invalid_tool_name = _validate_tool_name(tool_name)
        if invalid_tool_name is not None:
            return _build_error_result(tool_name, invalid_tool_name)

        # 模板：使用占位符 + str.replace 拼接，而不是 str.format。
        # 这样模板内的花括号无需转义，用户 tool_logic 中的 f-string 花括号也不会被误解析。
        template = '''from langchain_core.tools import tool
from typing import Dict, Any

@tool
def @TOOL_NAME@(@PARAMS@) -> Dict[str, Any]:
    """
    @TOOL_DESCRIPTION@

    @ARGS_DOCSTRING@
    """
    try:
@INNER_LOGIC@
        return {
            "success": True,
            "message": "执行成功",
            "result": result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result": None
        }
'''

        # 解析参数
        param_specs = _parse_args_spec(args_spec)
        param_str = ", ".join(f"{name}: {param_type}" for name, param_type, _ in param_specs)
        doc_args = [f"{name}: {desc}" for name, _, desc in param_specs]
        args_doc = "\n    Args:\n        " + "\n        ".join(doc_args) if doc_args else ""

        # 工具逻辑整体缩进到 try 块内部（保留相对缩进，且不破坏多行字符串内容行）
        inner_logic = _indent_body(tool_logic)

        source_code = template
        for placeholder, value in {
            "@TOOL_NAME@": tool_name,
            "@PARAMS@": param_str,
            "@TOOL_DESCRIPTION@": tool_description.strip(),
            "@ARGS_DOCSTRING@": args_doc.strip(),
            "@INNER_LOGIC@": inner_logic,
        }.items():
            source_code = source_code.replace(placeholder, value)

        # 校验生成的源码语法，不合法则不写入文件
        ast.parse(source_code)

        disallowed_import = _find_disallowed_import(source_code)
        if disallowed_import is not None:
            return _build_error_result(tool_name, f"禁止导入高风险模块：{disallowed_import}")

        # 解析保存路径并限制在 tools 目录内
        tool_path = _resolve_tool_path(tool_name, tool_path)

        abs_path = os.path.abspath(tool_path)
        if not _is_within_dir(abs_path, DEFAULT_TOOL_DIR):
            return _build_error_result(tool_name, "路径逃逸被禁止")

        if os.path.exists(abs_path) and not force:
            return _build_error_result(tool_name, f"目标文件已存在：{abs_path}")

        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(source_code)

        # 同步注册到 tools/__init__.py：仅当工具文件保存在 tools 包目录下才生效
        message = f"工具代码已保存到 {abs_path}"
        registered = False
        if os.path.normcase(os.path.dirname(abs_path)) == os.path.normcase(DEFAULT_TOOL_DIR):
            try:
                registered = _sync_tools_init(tool_name, os.path.join(DEFAULT_TOOL_DIR, "__init__.py"))
                if registered:
                    message += "，并已注册到 tools/__init__.py"
            except (IndexError, OSError, ValueError) as err:
                message += f"，但注册到 tools/__init__.py 失败：{err!s}"

        # 顺带生成该工具的单元测试（失败不影响工具本身的生成结果）
        test_file_path: str | None = None
        if with_test:
            test_result = creat_tool_tests(
                tool_name=tool_name,
                tool_description=tool_description,
                args_spec=args_spec,
                tool_path=abs_path,
                force=force,
            )
            if test_result["success"]:
                test_file_path = test_result["test_file_path"]
                message += f"，并已生成单元测试 {test_file_path}"
            else:
                message += f"，但生成单元测试失败：{test_result['error']}"

        return {
            "success": True,
            "tool_name": tool_name,
            "source_code": source_code,
            "file_path": abs_path,
            "registered": registered,
            "test_file_path": test_file_path,
            "message": message
        }

    except (IndexError, KeyError, OSError, SyntaxError, TypeError, ValueError) as err:
        return _build_error_result(tool_name, str(err))


# 测试模板：同样使用占位符 + str.replace 拼接，模板内的花括号无需转义。
# 生成的是“结构冒烟测试”：断言工具对象、参数签名与统一返回结构，
# 不假设业务逻辑的具体输出（示例入参无法通过类型校验时自动跳过冒烟用例）。
TEST_TEMPLATE: Final[str] = '''"""@TOOL_NAME@ 工具单元测试（由 tools/create_tools.py 自动生成，可自由补充用例）。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool
from pydantic import ValidationError

TOOL_FILE = Path(@TOOL_FILE@)
TOOL_NAME = "@TOOL_NAME@"
TOOL_DESCRIPTION = @TOOL_DESCRIPTION@
EXPECTED_PARAMS = @EXPECTED_PARAMS@
SAMPLE_ARGS = @SAMPLE_ARGS@


@pytest.fixture(scope="module")
def generated_tool() -> BaseTool:
    """从生成的工具文件加载 @TOOL_NAME@ 工具对象。"""
    assert TOOL_FILE.is_file(), f"工具文件不存在：{TOOL_FILE}"
    spec = importlib.util.spec_from_file_location("generated_@TOOL_NAME@", TOOL_FILE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, TOOL_NAME)


def test_@TOOL_NAME@_is_langchain_tool(generated_tool: BaseTool) -> None:
    """工具是 LangChain 工具对象，且名称与生成时一致。"""
    assert isinstance(generated_tool, BaseTool)
    assert generated_tool.name == TOOL_NAME


def test_@TOOL_NAME@_exposes_declared_params(generated_tool: BaseTool) -> None:
    """工具入参与 args_spec 声明一致。"""
    assert set(generated_tool.args) == set(EXPECTED_PARAMS)


def test_@TOOL_NAME@_source_follows_project_template() -> None:
    """生成源码遵循项目统一模板：@tool 装饰器 + 统一返回结构 + 异常兜底。"""
    source = TOOL_FILE.read_text(encoding="utf-8")
    assert "@tool" in source
    assert '"success": True' in source
    assert '"success": False' in source
    assert '"error": str(e)' in source


def test_@TOOL_NAME@_returns_structured_result(generated_tool, monkeypatch, tmp_path) -> None:
    """调用工具返回统一结构且不向调用方抛异常（在临时目录内执行，避免污染工作区）。"""
    monkeypatch.chdir(tmp_path)
    try:
        out = generated_tool.invoke(dict(SAMPLE_ARGS))
    except ValidationError as err:
        pytest.skip(f"示例入参不满足参数类型约束，请补充真实用例：{err!s}")
    assert isinstance(out, dict)
    assert "success" in out
    if out["success"]:
        assert "result" in out
    else:
        assert out["error"]
'''


def _sample_literal(param_type: str) -> str:
    """
    按 args_spec 声明的类型生成示例入参源码。

    未识别的类型回退字符串 "test"；回退值若与真实类型约束不符，
    生成的冒烟用例会捕获 ValidationError 并跳过，不会误报失败。
    """
    normalized = param_type.strip().lower()
    exact = _SAMPLE_VALUES.get(normalized)
    if exact is not None:
        return exact
    if normalized.startswith(("list", "sequence", "set", "tuple")):
        inner = normalized.partition("[")[2].rstrip("]").strip() or "int"
        element = _SAMPLE_VALUES.get(inner, '"test"')
        return f"[{element}]"
    return '"test"'


def _build_sample_args_source(param_specs: list[tuple[str, str, str]]) -> str:
    if not param_specs:
        return "{}"
    items = ",\n    ".join(
        f"{name!r}: {_sample_literal(param_type)}" for name, param_type, _ in param_specs
    )
    return "{\n    " + items + ",\n}"


def creat_tool_tests(
    tool_name: str,
    tool_description: str,
    args_spec: str,
    tool_path: str | None = None,
    test_path: str | None = None,
    force: bool = False,
) -> dict[str, str | bool | None]:
    """
    为已生成的工具生成 pytest 单元测试，默认保存到 tests/tools/test_<tool_name>.py。

    生成内容为结构冒烟测试：工具对象类型与名称、args_spec 声明的入参、
    源码是否遵循统一模板、调用后是否返回统一结构（success/result 或 error）。
    业务断言需人工在生成的文件中补充。create_tool 会默认调用本函数。

    Args:
        tool_name: 工具函数名，需与 create_tool 生成的函数名一致
        tool_description: 工具描述，写入测试文件头部说明
        args_spec: 参数定义说明，格式同 create_tool，用于生成入参断言与示例入参
        tool_path: 工具文件路径，可传目录或.py文件路径；
                   为空时默认取 create_tools.py 同级目录（tools/）下 tool_name.py
        test_path: 测试文件存放路径，可传目录或.py文件路径；
                   为空时默认保存到 tests/tools/ 下 test_<tool_name>.py
        force: 是否覆盖已存在的测试文件，默认禁止覆盖

    Returns:
        字典包含生成的测试源码、保存路径与状态
    """
    try:
        invalid_tool_name = _validate_tool_name(tool_name)
        if invalid_tool_name is not None:
            return _build_test_error_result(tool_name, invalid_tool_name)

        param_specs = _parse_args_spec(args_spec)
        tool_file = os.path.abspath(_resolve_tool_path(tool_name, tool_path))

        # 解析测试保存路径并限制在 tests/tools 目录内
        abs_test_path = os.path.abspath(_resolve_test_path(tool_name, test_path))
        if not _is_within_dir(abs_test_path, DEFAULT_TEST_DIR):
            return _build_test_error_result(tool_name, "测试路径逃逸被禁止")

        if os.path.exists(abs_test_path) and not force:
            return _build_test_error_result(tool_name, f"目标测试文件已存在：{abs_test_path}")

        source_code = TEST_TEMPLATE
        for placeholder, value in {
            "@TOOL_NAME@": tool_name,
            "@TOOL_DESCRIPTION@": repr(tool_description.strip()),
            "@TOOL_FILE@": repr(tool_file),
            "@EXPECTED_PARAMS@": repr([name for name, _, _ in param_specs]),
            "@SAMPLE_ARGS@": _build_sample_args_source(param_specs),
        }.items():
            source_code = source_code.replace(placeholder, value)

        # 校验生成的测试源码语法，不合法则不写入文件
        ast.parse(source_code)

        parent = os.path.dirname(abs_test_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(abs_test_path, "w", encoding="utf-8") as f:
            f.write(source_code)

        return {
            "success": True,
            "tool_name": tool_name,
            "test_source_code": source_code,
            "test_file_path": abs_test_path,
            "message": f"工具测试已保存到 {abs_test_path}",
        }

    except (IndexError, KeyError, OSError, SyntaxError, TypeError, ValueError) as err:
        return _build_test_error_result(tool_name, str(err))
