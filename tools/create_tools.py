import ast
import io
import os
import textwrap
import tokenize
from typing import Any, Dict, Optional

from langchain.tools import tool

# 默认工具存放目录：create_tools.py 的同级目录
DEFAULT_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))


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
            if token.type not in (tokenize.STRING, tokenize.FSTRING_START, tokenize.FSTRING_END, tokenize.FSTRING_MIDDLE):
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
def create_tool(tool_name: str, tool_description: str, args_spec: str, tool_logic: str, tool_path: Optional[str] = None) -> Dict[str, Any]:
    """
    动态生成Langchain标准@tool装饰器工具源码。
    使用统一规范模板输出可直接运行的Python工具代码，遵循项目统一返回结构。
    生成后的代码可以直接写入py文件，导入到Agent工具集中使用。

    Args:
        tool_name: 工具函数名，仅小写字母、下划线，例如 "read_markdown_file"
        tool_description: 工具文档字符串，说明能力、用途、适用场景
        args_spec: 参数定义说明，每个参数格式：参数名:参数类型=参数说明
                   示例："file_path:str=本地文件路径;encoding:str=utf-8文件编码，可选"
        tool_logic: 工具主体业务逻辑（函数内部实现代码，不要写函数定义、装饰器）
        tool_path: 工具存放的路径，可传目录或.py文件路径；
                   为空时默认保存到 create_tools.py 同级目录（tools/）下 tool_name.py

    Returns:
        字典包含生成的完整源码、状态，成功可直接写入文件运行
    """
    try:
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
        param_lines = []
        doc_args = []
        arg_items = [x.strip() for x in args_spec.split(";") if x.strip()]
        for item in arg_items:
            name_type, desc = item.split("=", maxsplit=1)
            param_name, param_type = name_type.split(":")
            param_lines.append(f"{param_name}: {param_type}")
            doc_args.append(f"{param_name}: {desc}")

        param_str = ", ".join(param_lines)
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

        # 解析保存路径：为空 -> 默认同级目录；已有目录 -> 拼接 tool_name.py；否则视为完整文件路径
        if not tool_path:
            tool_path = os.path.join(DEFAULT_TOOL_DIR, f"{tool_name}.py")
        elif os.path.isdir(tool_path):
            tool_path = os.path.join(tool_path, f"{tool_name}.py")

        abs_path = os.path.abspath(tool_path)
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
            except Exception as err:
                message += f"，但注册到 tools/__init__.py 失败：{str(err)}"

        return {
            "success": True,
            "tool_name": tool_name,
            "source_code": source_code,
            "file_path": abs_path,
            "registered": registered,
            "message": message
        }

    except Exception as err:
        return {
            "success": False,
            "error": f"工具生成失败：{str(err)}",
            "source_code": None,
            "tool_name": tool_name,
            "file_path": None,
            "registered": False
        }