"""
文件读写工具 - 使用LangChain @tool装饰器
"""
import os
from typing import Any

from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> dict[str, Any]:
    """
    读取文件内容工具。根据给定的文件路径读取文件内容。

    Args:
        file_path: 要读取的文件路径

    Returns:
        包含文件内容的字典，成功时包含content字段
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"文件不存在: {file_path}",
                "content": None
            }

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            "success": True,
            "file_path": file_path,
            "content": content,
            "size": len(content),
            "message": f"成功读取文件，共 {len(content)} 个字符"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "content": None
        }


@tool
def write_file(file_path: str, content: str, mode: str = 'w') -> dict[str, Any]:
    """
    写入文件内容工具。将内容写入指定文件路径。

    Args:
        file_path: 文件路径
        content: 要写入的内容
        mode: 写入模式，'w'为覆盖，'a'为追加，默认'w'

    Returns:
        操作结果字典
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content)

        return {
            "success": True,
            "file_path": file_path,
            "size": len(content),
            "mode": mode,
            "message": f"成功写入文件，共 {len(content)} 个字符"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
