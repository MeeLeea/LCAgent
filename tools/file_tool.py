"""文件工具模块

文件读写工具由 MCP-Filesystem server 提供（read_file/write_file/list_directory/
create_directory/move_file/search_files/get_file_info 等）。

本模块不再定义本地文件工具，避免与 MCP 加载的同名工具冲突。

工作空间隔离由 agent/workspace_mw.py 的 WorkspaceSecurityMW
在工具调用前拦截实现：
- 从 config.configurable 读取当前会话 workspace_path
- 将 LLM 传入的相对路径解析为 workspace 内绝对路径
- commonpath 校验防止路径逃逸
- 注入解析后的绝对路径到工具 args
"""
