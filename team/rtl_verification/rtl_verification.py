"""
Verification Agent - 数字芯片 RTL 验证工程师,负责验证需求梳理、验证计划、Testbench/UVM 开发与 Vivado Xsim 仿真
"""
from collections.abc import Sequence
from typing import ClassVar

from graph.registry import register_agent
from team.base import PromptInjector, TeamAgent
from tools.human_confirmation import request_user_confirmation


@register_agent(
    "rtl_verification",
    "team/rtl_verification/agent_config.json",
    tools=[request_user_confirmation],
    mcp_tools=["write_file","edit_file","list_directory","read_file","delete_file","create_directory","delete_directory"],
)
class VerificationAgent(TeamAgent):
    """
    验证师 Agent,负责 RTL 验证、测试用例设计、覆盖率分析与 bug 定位

    继承 TeamAgent 轻量基类,可选工具调用能力(声明 mcp_tools=["write_file"],
    由 build_workflow 装配期同步拉取;加载失败时降级为纯文本模式);各工作流方法
    渲染对应 `## workflow:*` 小节 → 可选技能注入(fixed_skills 始终合并
    vivado-2025.2) → 异步 LLM 调用(TOKEN 级流式)。
    """

    # 验证环境固定依赖 Vivado Xsim(AGENT.md 环境约束:当前仿真环境优先使用
    # Vivado Xsim),始终注入 vivado-2025.2 技能指引,不依赖任务关键词自动匹配,
    # 保证 Vivado 工程创建 / 仿真运行 / dump 波形的 TCL 规范始终可被遵循。
    # 经 TeamAgent.fixed_skills 类属性由 build_skill_block 统一合并注入,
    # 无需特殊函数、不走配置穿透、aclear_skills 只清 state 不影响此处。
    fixed_skills: ClassVar[list[str]] = ["vivado-2025.2"]

    # 工作流节点提示词的默认模板(仅 AGENT.md 缺失或未定义小节时兜底)
    default_templates: ClassVar[dict[str, str]] = {
        "spec_design": (
            "请根据以下任务完成 RTL 验证前的需求梳理与验证方案规划:\n\n"
            "{task}\n\n"
            "严格按以下顺序输出环节 1-4,不要省略:\n"
            "1. 验证需求梳理:从 spec/RTL/接口表中提取功能点、接口信号、时钟复位、关键约束(协议、时序、带宽、异常处理)\n"
            "2. 验证计划Checklist:正常场景 / 边界场景 / 异常错误场景,每条测试点标注验证方法与优先级\n"
            "3. 文件规划:基于前端 filelist(syn_filelist.f / sim_filelist.f)补充 tb/uvm 目录结构;仿真文件绝不混入综合filelist,严格区分综合RTL / 仿真TB / UVM环境\n"
            "4. Vivado仿真Tcl脚本:创建工程、加载文件、+incdir、编译选项、仿真、dump波形,适配 Xsim\n\n"
            "环境约束:优先适配 Vivado Xsim,不输出 VCS 专属语法;UVM 仅用 Vivado 支持版本,超范围标注【Xsim不支持,需更换仿真器】。\n"
            "spec 信息不足时调用 request_user_confirmation 批量征询(Vivado版本、是否启用UVM、时钟频率、复位行为、异常场景),每项含 id/question/choices;无则跳过。"
        ),
        "verilog_design": (
            "请根据以下任务与上下文输出验证 Testbench / UVM 框架代码:\n\n"
            "{task}\n\n"
            "严格按以下顺序输出环节 5-9,不要省略:\n"
            "5. Testbench / UVM框架代码:小模块输出定向 SV tb(可在 Vivado 直接跑);复杂IP输出 UVM 组件框架(sequence/driver/monitor/refmodel/scoreboard),仅用 Vivado 支持语法;严格区分可综合RTL / 仿真tb / uvm环境\n"
            "6. 功能 covergroup 覆盖率定义(适配 Xsim)\n"
            "7. 关键断言 property:协议、握手、死锁检测\n"
            "8. 潜在风险与 bug 预判清单:协议错误、死锁、CDC问题、复位错位,给出复现条件\n"
            "9. 待确认项:若存在需用户拍板点(spec歧义、Vivado版本、是否启用UVM、仿真时间、dump波形需求),调用 request_user_confirmation 批量征询(每项含 id/question/choices);无则跳过\n\n"
            "编码与输出约束:可综合 RTL 逻辑不在 tb 中重复实现,参考模型尽量行为级;AXI/AXIS 重点校验 valid/ready 握手、burst 长度、last、error 响应、复位期间信号行为;跨时钟域模块覆盖不同时钟相位与复位错位场景。\n"
            "完成后将工程项目文件、Tcl 脚本保存在输出目录中。"
        ),
    }

    async def _arun_rtl_design_task_async(
        self,
        template_name: str,
        task: str,
        injector: PromptInjector | None,
        config: dict | None = None,
        active_names: Sequence[str] = (),
    ) -> str:
        """RTL 验证工作流方法异步通用执行体:模板渲染/技能注入同步,LLM 调用异步流式

        vivado-2025.2 技能指引经 self.fixed_skills 类属性由 build_skill_block
        统一合并注入(注入器走 self.inject_into_prompt 时合并;即使外部 injector
        传入,self.fixed_skills 在 self.build_skill_block 内仍生效)。
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

