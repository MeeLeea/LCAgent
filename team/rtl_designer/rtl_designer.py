"""
Designer Agent - 数字芯片 RTL 设计工程师,负责规格梳理、模块拆分、Filelist 生成与可综合 RTL 编码
"""
from collections.abc import Sequence
from typing import ClassVar

from graph.registry import register_agent
from team.base import PromptInjector, TeamAgent


@register_agent("rtl_designer", "team/rtl_designer/agent_config.json", tools=None)
class DesignerAgent(TeamAgent):
    """
    设计师 Agent,负责 RTL 设计、综合、布局布线等

    继承 TeamAgent 轻量基类,纯文本推理模式(不使用工具);各工作流方法
    渲染对应 `## workflow:*` 小节 → 可选技能注入 → 异步 LLM 调用(TOKEN 级流式)。
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

    # ============ 异步版(供 rtl_graph 节点直接 await 调用) ============

    async def _arun_rtl_design_task_async(
        self,
        template_name: str,
        task: str,
        injector: PromptInjector | None,
        config: dict | None = None,
        active_names: Sequence[str] = (),
    ) -> str:
        """RTL 设计工作流方法异步通用执行体:模板渲染/技能注入同步,LLM 调用异步流式

        active_names 由节点函数从 state["active_skills"] 取值传入。
        """
        template = self.get_template(template_name)
        prompt = self.render_template(template, task=task)
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, task, active_names)
        return await self.ainvoke(prompt, config)

    async def aspec_design_task(self, task: str, injector: PromptInjector | None = None, config: dict | None = None, active_names: Sequence[str] = ()) -> str:
        """异步版 spec_design_task(供 spec_design 节点调用)"""
        return await self._arun_rtl_design_task_async("spec_design", task, injector, config, active_names)

    async def averilog_design_task(self, task: str, injector: PromptInjector | None = None, config: dict | None = None, active_names: Sequence[str] = ()) -> str:
        """异步版 verilog_design_task(供 verilog_design 节点调用)"""
        return await self._arun_rtl_design_task_async("verilog_design", task, injector, config, active_names)

