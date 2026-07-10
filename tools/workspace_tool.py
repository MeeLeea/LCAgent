"""
工作目录管理 MCP Server
基于 FastMCP，通过 stdio 传输提供文件夹创建与管理工具

启动方式:
    python workspace_tool.py

注册到 mcp_servers.json:
    {
        "workspace": {
            "transport": "stdio",
            "command": "python",
            "args": ["tools/workspace_tool.py"],
            "enabled": true
        }
    }
"""
import os
import shutil
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP


# 创建 MCP Server 实例
mcp = FastMCP("workspace-server")


@mcp.tool()
def create_workspace(folder_name: str, parent_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    创建文件夹并获取其作为工作目录

    Args:
        folder_name: 要创建的文件夹名称
        parent_dir: 父目录路径，默认为当前工作目录

    Returns:
        包含工作目录信息的字典
    """
    try:
        if parent_dir is None:
            parent_dir = os.getcwd()

        workspace_path = os.path.join(parent_dir, folder_name)
        abs_workspace_path = os.path.abspath(workspace_path)

        os.makedirs(abs_workspace_path, exist_ok=True)

        if not os.path.isdir(abs_workspace_path):
            return {
                "success": False,
                "error": f"无法创建目录: {abs_workspace_path}",
                "workspace_path": None
            }

        return {
            "success": True,
            "workspace_path": abs_workspace_path,
            "folder_name": folder_name,
            "parent_dir": os.path.abspath(parent_dir),
            "message": f"成功创建工作目录: {abs_workspace_path}"
        }

    except ValueError as e:
        return {"success": False, "error": f"路径无效: {str(e)}", "workspace_path": None}
    except PermissionError:
        return {"success": False, "error": "权限不足，无法创建目录", "workspace_path": None}
    except Exception as e:
        return {"success": False, "error": str(e), "workspace_path": None}


@mcp.tool()
def get_current_workspace() -> Dict[str, Any]:
    """
    获取当前工作目录

    Returns:
        包含当前工作目录信息的字典
    """
    try:
        abs_path = os.path.abspath(os.getcwd())
        return {
            "success": True,
            "workspace_path": abs_path,
            "message": f"当前工作目录: {abs_path}"
        }
    except Exception as e:
        return {"success": False, "error": str(e), "workspace_path": None}


@mcp.tool()
def list_directory(path: Optional[str] = None) -> Dict[str, Any]:
    """
    列出指定目录下的所有文件和文件夹

    Args:
        path: 目标目录路径，默认为当前工作目录

    Returns:
        包含目录条目列表的字典
    """
    try:
        target = path if path else os.getcwd()
        abs_path = os.path.abspath(target)

        if not os.path.exists(abs_path):
            return {"success": False, "error": f"路径不存在: {abs_path}", "entries": []}
        if not os.path.isdir(abs_path):
            return {"success": False, "error": f"不是目录: {abs_path}", "entries": []}

        entries: List[Dict[str, Any]] = []
        for name in sorted(os.listdir(abs_path)):
            full = os.path.join(abs_path, name)
            entries.append({
                "name": name,
                "type": "directory" if os.path.isdir(full) else "file",
                "size": os.path.getsize(full) if os.path.isfile(full) else None
            })

        return {
            "success": True,
            "path": abs_path,
            "entries": entries,
            "count": len(entries),
            "message": f"找到 {len(entries)} 个条目"
        }
    except PermissionError:
        return {"success": False, "error": "权限不足", "entries": []}
    except Exception as e:
        return {"success": False, "error": str(e), "entries": []}


@mcp.tool()
def delete_workspace(folder_path: str, recursive: bool = True) -> Dict[str, Any]:
    """
    删除文件夹

    Args:
        folder_path: 要删除的文件夹路径
        recursive: 是否递归删除非空文件夹，默认 True

    Returns:
        操作结果字典
    """
    try:
        abs_path = os.path.abspath(folder_path)

        if not os.path.exists(abs_path):
            return {"success": False, "error": f"路径不存在: {abs_path}"}
        if not os.path.isdir(abs_path):
            return {"success": False, "error": f"不是目录: {abs_path}"}

        # 安全检查：禁止删除根目录或盘符根
        parent = os.path.dirname(abs_path)
        if abs_path == parent or abs_path.endswith(":\\"):
            return {"success": False, "error": "禁止删除根目录"}

        if recursive:
            shutil.rmtree(abs_path)
        else:
            os.rmdir(abs_path)  # 仅能删除空目录

        return {
            "success": True,
            "deleted_path": abs_path,
            "message": f"已删除: {abs_path}"
        }
    except PermissionError:
        return {"success": False, "error": "权限不足"}
    except OSError as e:
        return {"success": False, "error": f"删除失败: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def move_workspace(src_path: str, dest_path: str) -> Dict[str, Any]:
    """
    移动或重命名文件夹

    Args:
        src_path: 源文件夹路径
        dest_path: 目标路径

    Returns:
        操作结果字典
    """
    try:
        abs_src = os.path.abspath(src_path)
        abs_dest = os.path.abspath(dest_path)

        if not os.path.exists(abs_src):
            return {"success": False, "error": f"源路径不存在: {abs_src}"}
        if not os.path.isdir(abs_src):
            return {"success": False, "error": f"源路径不是目录: {abs_src}"}

        shutil.move(abs_src, abs_dest)

        return {
            "success": True,
            "src_path": abs_src,
            "dest_path": abs_dest,
            "message": f"已移动: {abs_src} -> {abs_dest}"
        }
    except PermissionError:
        return {"success": False, "error": "权限不足"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def copy_workspace(src_path: str, dest_path: str) -> Dict[str, Any]:
    """
    复制文件夹（含子目录和文件）

    Args:
        src_path: 源文件夹路径
        dest_path: 目标路径

    Returns:
        操作结果字典
    """
    try:
        abs_src = os.path.abspath(src_path)
        abs_dest = os.path.abspath(dest_path)

        if not os.path.exists(abs_src):
            return {"success": False, "error": f"源路径不存在: {abs_src}"}
        if not os.path.isdir(abs_src):
            return {"success": False, "error": f"源路径不是目录: {abs_src}"}

        shutil.copytree(abs_src, abs_dest)

        return {
            "success": True,
            "src_path": abs_src,
            "dest_path": abs_dest,
            "message": f"已复制: {abs_src} -> {abs_dest}"
        }
    except PermissionError:
        return {"success": False, "error": "权限不足"}
    except FileExistsError:
        return {"success": False, "error": f"目标路径已存在: {abs_dest}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    # 以 stdio 模式启动 MCP Server
    mcp.run()
