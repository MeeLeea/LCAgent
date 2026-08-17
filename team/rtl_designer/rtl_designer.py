"""
Designer Agent - 数字芯片 RTL 设计工程师,负责规格梳理、模块拆分、Filelist 生成与可综合 RTL 编码
"""
from typing import ClassVar

from graph.registry import register_agent
from team.base import PromptInjector, TeamAgent


@register_agent("rtl_designer", "team/rtl_designer/agent_config.json", tools=None)
class DesignerAgent(TeamAgent):
    """
    设计师 Agent,负责 RTL 设计、综合、布局布线等

    继承 TeamAgent 轻量基类,纯文本推理模式(不使用工具);各工作流方法
    渲染对应 `## workflow:*` 小节 → 可选技能注入 → LLM 调用,同步方法由
    节点层经 asyncio.to_thread 异步执行。
    """

    # 工作流节点提示词的默认模板(仅 AGENT.md 缺失或未定义小节时兜底)
    default_templates: ClassVar[dict[str, str]] = {
        "spec_design": (
            "请根据以下任务完成 RTL 设计前的规格梳理与工程规划:\n\n"
            "{task}\n\n"
            "按输出强制规则输出环节 1-5,严格遵循顺序,不要省略环节:\n"
            "1. 需求与约束梳理:明确功能、时钟频率、位宽、协议、复位类型、跨时钟域、带宽、异常处理需求\n"
            "2. 项目目录层级树(tree格式)\n"
            "3. 模块接口定义表:信号名｜方向｜位宽｜时钟域｜描述\n"
            "4. 核心设计思路:数据流、状态机、握手、CDC方案、复位策略\n"
            "5. Filelist:syn_filelist.f(综合用,仅可综合RTL不含tb)与 sim_filelist.f(仿真用,RTL + tb + include)\n\n"
            "spec 信息不足时,主动列出待确认参数清单,不擅自修改架构Spec。"
        ),
        "verilog_design": (
            "请根据以下任务与上下文输出可综合的 SystemVerilog RTL 源码:\n\n"
            "{task}\n\n"
            "按输出强制规则输出环节 6-10,严格遵循顺序,不要省略环节:\n"
            "6. RTL源码:按模块拆分输出每个模块完整代码,参数化设计;端口严格对齐接口表;"
            "关键逻辑增加注释;禁止仿真语法混入;规避隐式锁存\n"
            "7. Mermaid框图/状态机(需要时输出)\n"
            "8. 风险点清单:时序隐患、锁存风险、死锁、边界漏洞\n"
            "9. 验证测试点Checklist\n"
            "10. 待确认参数清单\n\n"
            "RTL 代码规范:使用 SystemVerilog;parameter 参数化禁止硬编码位宽;时钟复位端口放最前;"
            "状态机用 enum;always_ff 时序逻辑 / always_comb 组合逻辑且完备 else 避免锁存;"
            "跨时钟域信号必须显式同步,禁止直接跨域赋值。"
        ),
    }

    def _run_rtl_design_task(
        self,
        template_name: str,
        task: str,
        injector: PromptInjector | None,
    ) -> str:
        """
        RTL 设计工作流方法通用执行体:渲染模板 → 可选技能注入 → LLM 调用

        作为所有 RTL 设计节点的通用执行体,通过 template_name 动态分发到
        对应的 `## workflow:{name}` 小节模板(spec_design / verilog_design 等)。

        Args:
            template_name: 工作流小节名(对应 AGENT.md 的 `## workflow:{name}`)
            task: 用户任务
            injector: 技能注入器;为 None 时跳过技能注入

        Returns:
            LLM 生成结果文本
        """
        template = self.get_template(template_name)
        prompt = self.render_template(template, task=task)
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, task)
        return self.invoke(prompt)

    def spec_design_task(self, task: str, injector: PromptInjector | None = None) -> str:
        """
        规格梳理与工程规划:需求约束梳理、模块拆分、目录树、接口表、设计思路与 Filelist

        供工作流 spec_design 节点调用:对应 template_name="spec_design",渲染该模板
        → 可选技能注入 → LLM 生成(输出 syn_filelist.f 与 sim_filelist.f,严格遵循输出
        强制规则)。为同步方法,节点层经 asyncio.to_thread 异步执行。

        Args:
            task: 用户原始任务(模块需求/spec/接口)
            injector: 技能注入器;为 None 时跳过技能注入

        Returns:
            规格梳理与工程规划结果文本
        """
        return self._run_rtl_design_task("spec_design", task, injector)

    def verilog_design_task(self, task: str, injector: PromptInjector | None = None) -> str:
        """
        输出可综合 RTL 源码:按模块拆分、参数化设计,附风险清单与验证测试点

        供工作流 verilog_design 节点调用:对应 template_name="verilog_design",渲染该模板
        → 可选技能注入 → LLM 生成可综合 SystemVerilog 代码(端口严格对齐接口表,禁止仿真
        语法混入)。为同步方法,节点层经 asyncio.to_thread 异步执行。

        Args:
            task: 用户原始任务(模块需求/spec/接口)
            injector: 技能注入器;为 None 时跳过技能注入

        Returns:
            RTL 源码与配套输出结果文本
        """
        return self._run_rtl_design_task("verilog_design", task, injector)

    # ============ 异步版(供 rtl_graph 节点直接 await 调用) ============

    async def _arun_rtl_design_task_async(
        self,
        template_name: str,
        task: str,
        injector: PromptInjector | None,
    ) -> str:
        """RTL 设计工作流方法异步通用执行体:模板渲染/技能注入同步,LLM 调用异步流式"""
        template = self.get_template(template_name)
        prompt = self.render_template(template, task=task)
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, task)
        return await self.ainvoke(prompt)

    async def aspec_design_task(self, task: str, injector: PromptInjector | None = None) -> str:
        """异步版 spec_design_task(供 spec_design 节点调用)"""
        return await self._arun_rtl_design_task_async("spec_design", task, injector)

    async def averilog_design_task(self, task: str, injector: PromptInjector | None = None) -> str:
        """异步版 verilog_design_task(供 verilog_design 节点调用)"""
        return await self._arun_rtl_design_task_async("verilog_design", task, injector)

