"""
Agent核心调度模块 - 基于LangChain 1.x + LangGraph
使用 langgraph.prebuilt.create_react_agent 实现工具调用，支持ReAct式推理
支持动态加载本地工具 + MCP Server工具
"""
import asyncio
from typing import Dict, Any, List, Optional
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from llm_client import LLMClient
from .memory import AgentMemory
from tools import all_tools as local_tools
from tools.mcp_loader import load_mcp_tools, DEFAULT_CONFIG_FILE


class AgentCore:
    """基于LangChain 1.x 的Agent核心调度器"""

    def __init__(
        self,
        llm_client: LLMClient,
        memory_size: int = 10,
        long_term_memory_file: Optional[str] = None,
        max_iterations: int = 25,
        verbose: bool = True,
        mcp_config_file: Optional[str] = None,
        enable_mcp: bool = True
    ):
        """
        初始化Agent核心

        Args:
            llm_client: LLM客户端实例
            memory_size: 短期记忆容量
            long_term_memory_file: 长期记忆文件路径
            max_iterations: Agent最大迭代次数(langgraph recursion_limit)
            verbose: 是否打印详细执行过程
            mcp_config_file: MCP servers 配置文件路径
            enable_mcp: 是否启用 MCP 工具加载
        """
        self.llm = llm_client
        self.memory = AgentMemory(
            short_term_size=memory_size,
            long_term_file=long_term_memory_file
        )
        self.max_iterations = max_iterations
        self.verbose = verbose

        # 本地工具
        self.local_tools: List[BaseTool] = list(local_tools)
        # MCP 工具(从 MCP Server 加载)
        self.mcp_tools: List[BaseTool] = []
        # 合并后的完整工具列表
        self.tools: List[BaseTool] = list(self.local_tools)

        # MCP 配置
        self.mcp_config_file = mcp_config_file or DEFAULT_CONFIG_FILE
        self.enable_mcp = enable_mcp

        # 启动时加载 MCP 工具
        if enable_mcp:
            self.reload_mcp_tools()

        # 创建Agent
        self.agent_executor = self._create_agent_executor()

        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []

    def reload_mcp_tools(self) -> int:
        """
        重新加载 MCP 工具(同步入口)

        Returns:
            加载到的 MCP 工具数量
        """
        try:
            count = asyncio.run(self._async_load_mcp_tools())
            # 合并工具列表
            self.tools = list(self.local_tools) + list(self.mcp_tools)
            # 重建Agent
            if hasattr(self, "agent_executor"):
                self.agent_executor = self._create_agent_executor()
            return count
        except Exception as e:
            print(f"[MCP] 重新加载失败: {e}")
            return 0

    async def _async_load_mcp_tools(self) -> int:
        """异步加载 MCP 工具"""
        tools = await load_mcp_tools(self.mcp_config_file)
        self.mcp_tools = tools
        if tools and self.verbose:
            print(f"[MCP] 已加载 {len(tools)} 个工具: {', '.join(t.name for t in tools)}")
        elif self.verbose:
            print("[MCP] 未加载到任何工具(可能配置为空或服务器未启用)")
        return len(tools)

    def _create_agent_executor(self):
        """创建LangGraph ReAct Agent"""
        chat_model = self.llm.get_chat_model()

        # create_react_agent 直接返回可调用的agent
        # prompt 参数作为系统提示词
        agent = create_react_agent(
            model=chat_model,
            tools=self.tools,
            prompt=self._get_system_prompt(),
            # recursion_limit=self.max_iterations
        )
        return agent

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return (
            "你是一个智能助手，配备了多种工具（包括文件读写、目录管理、搜索、计算等）。\n"
            "\n"
            "【重要规则】\n"
            "1. 当用户要求创建文件、读写文件、创建目录、搜索信息等操作时，你【必须】调用相应工具来完成，"
            "绝对不要回复'我无法访问你的文件系统'、'我没有权限'、'请你自己保存'之类的话。\n"
            "2. 你确实拥有这些工具的能力，工具会在用户本地执行，可以真正创建和修改文件。\n"
            "3. 如果用户要保存内容到文件，直接调用 write_file 工具，不要把内容贴出来让用户自己保存。\n"
            "4. 如果用户要创建目录，直接调用 create_workspace 工具，不要让用户手动操作，创建文件夹默认位置是：'当前工作目录/test/'。\n"
            "5. 只有纯知识性问答（不需要操作文件/搜索/计算）才直接回答，不调用工具。\n"
            "6. 调用工具时，如果一次任务需要多步操作（例如先创建目录再写文件），请依次调用多个工具。\n"
            "7. 如果用户要求跑一下或者测试一下，直接调用相应工具，不要让用户手动操作。\n"
            "8. 如果用户要求生成工具,直接在tools目录下创建，并使用@tool装饰器，不修改__init__.py文件。\n"
            "9. 如果用户要求获取本地时间，直接调用 get_local_time 工具，不要自己计算。\n"
            "\n"
            "请用中文回答。"
        )

    def run(self, task: str) -> str:
        """
        使用Agent执行任务（自动决定是否调用工具）

        Args:
            task: 任务描述

        Returns:
            执行结果
        """
        print(f"\n{'='*50}")
        print(f"开始执行任务: {task}")
        print(f"{'='*50}\n")

        try:
            # 构建消息列表：历史消息 + 当前任务
            messages = list(self.memory.get_langchain_messages())
            messages.append(HumanMessage(content=task))

            # 执行Agent
            result = self.agent_executor.invoke({"messages": messages})

            # 解析结果
            output = ""
            result_messages = result.get("messages", [])

            step_count = 0
            for msg in result_messages:
                # 跳过我们传入的历史消息
                if msg in messages:
                    continue

                if isinstance(msg, AIMessage):
                    # 检查是否有工具调用
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            step_count += 1
                            if self.verbose:
                                print(f"\n--- 步骤 {step_count} ---")
                                print(f"工具: {tc.get('name', 'unknown')}")
                                print(f"输入: {tc.get('args', {})}")
                            self.execution_history.append({
                                "step": step_count,
                                "tool": tc.get("name"),
                                "input": tc.get("args"),
                                "observation": ""
                            })
                    # 普通文本回复作为最终输出
                    if msg.content and not tool_calls:
                        output = msg.content
                elif hasattr(msg, "content") and hasattr(msg, "tool_call_id"):
                    # ToolMessage - 工具返回结果
                    if self.execution_history and step_count > 0:
                        self.execution_history[-1]["observation"] = str(msg.content)[:500]
                    if self.verbose and step_count > 0:
                        print(f"结果: {str(msg.content)[:200]}...")

            # 如果没有捕获到输出，取最后一条AI消息
            if not output:
                for msg in reversed(result_messages):
                    if isinstance(msg, AIMessage) and msg.content:
                        output = msg.content
                        break

            print(f"\n最终答案: {output}")

            # 存入记忆
            self.memory.add("user", task)
            self.memory.add("assistant", output, {"important": True})

            return output

        except Exception as e:
            error_msg = f"任务执行失败: {str(e)}"
            print(f"\n错误: {error_msg}")
            return error_msg

    def chat(self, message: str) -> str:
        """
        普通对话模式（也通过Agent执行，自动判断是否调用工具）

        与 run() 的区别：不打印步骤详情，不强制存入长期记忆。

        Args:
            message: 用户消息

        Returns:
            助手回复
        """
        try:
            # 构建消息列表：历史消息 + 当前任务
            messages = list(self.memory.get_langchain_messages())
            messages.append(HumanMessage(content=message))

            # 静默执行（保存原 verbose 状态）
            original_verbose = self.verbose
            self.verbose = False

            result = self.agent_executor.invoke({"messages": messages})

            # 恢复 verbose
            self.verbose = original_verbose

            # 取最后一条 AI 消息作为输出
            output = ""
            result_messages = result.get("messages", [])
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    output = msg.content
                    break

            # 存入短期记忆（不存长期）
            self.memory.add("user", message)
            self.memory.add("assistant", output)
            return output

        except Exception as e:
            # 兜底：Agent 执行失败时降级为纯 LLM 对话
            history = self.memory.get_short_term()
            self.memory.add("user", message)
            response = self.llm.chat_with_history(
                user_input=message,
                history=history,
                system_prompt="你是一个有帮助的AI助手，请用中文回答。"
            )
            self.memory.add("assistant", response)
            return response

    def cot(self, task: str) -> str:
        """
        CoT链式思考模式（不调用工具，纯推理）

        Args:
            task: 任务描述

        Returns:
            推理结果
        """
        print(f"\n{'='*50}")
        print(f"链式思考模式: {task}")
        print(f"{'='*50}\n")

        system_prompt = (
            "你是一个智能助手，使用链式思考(Chain of Thought)来解决问题。\n"
            "请一步步分析问题，最后给出结论。\n"
            "请用中文回答。"
        )

        context = self.memory.get_all_context()
        response = self.llm.chat_with_history(
            user_input=task,
            history=context,
            system_prompt=system_prompt,
            temperature=0.7
        )

        print(f"\n最终答案: {response}")

        # 存入记忆
        self.memory.add("user", task)
        self.memory.add("assistant", response, {"important": True})

        return response

    def switch_llm(self, llm_client: LLMClient):
        """
        切换LLM提供商

        Args:
            llm_client: 新的LLM客户端实例
        """
        self.llm = llm_client
        # 重新创建Agent
        self.agent_executor = self._create_agent_executor()

    def get_available_tools(self) -> List[str]:
        """获取可用工具名称列表"""
        return [t.name for t in self.tools]

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history

    def clear_history(self):
        """清空执行历史"""
        self.execution_history.clear()

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        return self.memory.summarize()

    def compress_memory(self) -> Dict[str, Any]:
        """
        压缩长期记忆

        将 memory.json 中所有长期记忆发送给 LLM，生成摘要后替换原内容。
        这样可以在保留关键信息的同时大幅减少 token 占用。

        Returns:
            {
                "success": bool,
                "original_count": int,      # 原记忆条数
                "original_chars": int,      # 原字符数
                "compressed_chars": int,    # 压缩后字符数
                "summary": str,             # 摘要内容
                "error": str (失败时)
            }
        """
        from datetime import datetime

        if not self.memory.long_term_memory:
            return {
                "success": False,
                "error": "没有长期记忆可压缩"
            }

        # 1. 拼接所有长期记忆为文本
        history_lines = []
        original_chars = 0
        for idx, item in enumerate(self.memory.long_term_memory, 1):
            role = item.get("role", "unknown")
            content = item.get("content", "")
            ts = item.get("timestamp", "")
            history_lines.append(f"[{idx}] ({ts}) {role}: {content}")
            original_chars += len(content)

        history_text = "\n\n".join(history_lines)

        # 2. 构建 LLM 请求
        system_prompt = (
            "你是一个记忆压缩助手。请将以下历史对话记录压缩成一份简洁的摘要，要求：\n"
            "1. 保留所有关键信息、用户意图、重要决策和事实\n"
            "2. 去除重复和冗余内容\n"
            "3. 按主题分条目组织，使用 '- ' 开头\n"
            "4. 保持事实准确，不要添加推测内容\n"
            "5. 用中文输出"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"以下是历史对话记录，请压缩成摘要:\n\n{history_text}"}
        ]

        # 3. 调用 LLM 生成摘要
        try:
            summary = self.llm.chat(messages)
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM 调用失败: {e}"
            }

        # 4. 用摘要替换原长期记忆
        original_count = len(self.memory.long_term_memory)
        compressed_chars = len(summary)

        self.memory.long_term_memory = [{
            "role": "system",
            "content": f"[历史记忆摘要 {datetime.now().isoformat()}]\n{summary}",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "important": True,
                "type": "summary",
                "original_count": original_count,
                "original_chars": original_chars,
                "compressed_chars": compressed_chars
            }
        }]

        # 5. 保存回 memory.json
        self.memory._save_long_term_memory()

        return {
            "success": True,
            "original_count": original_count,
            "original_chars": original_chars,
            "compressed_chars": compressed_chars,
            "summary": summary
        }
