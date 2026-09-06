# -*- coding: utf-8 -*-
"""
文件所在目录：app/core/agent/orchestrator.py
Agent 编排器：根据意图与模式选择 ReAct 或 Plan-and-Execute，串联记忆、工具、追踪与专属技能集。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set

# 💡 修复点 1：全面拥抱项目统一的 loguru 确保日志可见
from loguru import logger

from app.core.agent.planner import PlannerAgent
from app.core.agent.react_agent import AgentResult, ReActAgent
# 整合skills
from app.core.skill.manager import SkillManager
from app.infrastructure.trace import Tracer
from app.core.memory.manager import MemoryManager
from app.core.tools.registry import ToolRegistry, build_tool_call_budget
from app.core.tools.base import tool_to_function_call_definition
from app.llm_model_router.model_router import ModelRouter

# Langfuse 可观测性：@observe 在未配置密钥时自动退化为 no-op（零侵入）
from langfuse import observe as langfuse_observe

OrchestrationMode = Literal["react", "plan_execute"]

# ─── 🎯 SKILLS 渐进式披露系统级核心提示词 ───
SKILLS_SYSTEM_PROMPT = """## Skills System

You have access to a specialized skills library to handle complex workflows and domain-specific tasks.

{skills_locations}{skills_load_warnings}
**Available Skills:**
{skills_list}

**How to Use Skills (Progressive Disclosure):**
To prevent context bloat, you only see the brief abstracts above. When a task matches a skill, you MUST fetch its full details before executing:

1. **Identify Relevance**: Check if the user's goal matches any skill description listed above.
2. **Read Full Instructions**: Use `file_read_tool` with the exact 'Source File' path. (The tool defaults to 2000 lines, which is enough to read the full file).
3. **Strictly Follow Workflows**: Follow the precise workflows, configurations, or script paths defined inside that file. Do not guess parameters.

Remember: Always read the corresponding skill file first if a relevant skill exists for the task!"""

# ---------------------------------------------------------------------------
# 依赖抽象
class OrchestratorConfig:
    def get(self, key: str, default: Any=None): ...

@dataclass
class IntentContext:
    intent: str = "general"
    confidence: float = 1.0
    slots: Dict[str, Any] = field(default_factory=dict)
    preferred_mode: Optional[OrchestrationMode] = None
    allowed_tools: Optional[List[str]] = None


@dataclass
class AgentResponse:
    answer: str
    mode_used: OrchestrationMode
    success: bool
    trace_id: str
    intent: IntentContext
    steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    degraded: bool = False


class AgentOrchestrator:
    """智能体核心编排器。

    统一调度大模型的推理循环，封装了记忆落库、工具输出标准化规约、以及多模式降级熔断保护机制。
    """

    def __init__(
            self,
            config: OrchestratorConfig,
            model_router: ModelRouter,
            memory_manager: MemoryManager,
            tool_registry: ToolRegistry,
            tracer: Tracer,
            skill_manager: SkillManager,
    ) -> None:
        """初始化编排器，注入全局核心基础设施单例。"""
        self._config: OrchestratorConfig = config
        self._model_router: ModelRouter = model_router
        self._memory: MemoryManager = memory_manager
        self._tools: ToolRegistry = tool_registry
        self._tracer: Tracer = tracer
        self._skill_manager: SkillManager = skill_manager
        self._fallback_on_plan_failure: bool = bool(config.get("fallback_react_on_plan_failure", True))
        # 【t3 优化】后台任务强引用登记表：防止 asyncio.create_task 产生的协程被 GC 提前回收
        self._background_tasks: Set[asyncio.Task] = set()

    def _spawn_background_task(self, coroutine: Any) -> asyncio.Task:
        """以安全后台任务方式调度协程：强引用防 GC 回收 + 异常统一记录（不阻塞主流程响应）。"""
        background_task: asyncio.Task = asyncio.create_task(coroutine)
        self._background_tasks.add(background_task)
        background_task.add_done_callback(self._on_background_task_done)
        return background_task

    def _on_background_task_done(self, finished_task: asyncio.Task) -> None:
        """后台任务完成回调：移出强引用登记表，异常仅告警不抛出。"""
        self._background_tasks.discard(finished_task)
        if finished_task.cancelled():
            return
        task_exception: Optional[BaseException] = finished_task.exception()
        if task_exception is not None:
            logger.warning("后台任务执行异常（已忽略，不影响主流程响应）: {}", task_exception)

    def _resolve_skill_tool_gating(
        self,
        intent: Optional[IntentContext],
        user_input: str,
    ) -> List[str]:
        """【技能感知】解析本 Session 号令工具的技能工具集。

        优先级：
          1) Pipeline 改写阶段 LLM 选定的技能名（intent.slots 的
             selected_skills_final / suggested_skills / eligible_skill_names），
             逐个解析其元数据 allowed-tools 去重合并，与已注册工具交集后号令。
          2) 无声明技能时，降级回退 resolve_relevant_skill 轻量关键词匹配
             命中单一技能。

        Returns:
            与注册工具交集后的号令工具名列表；未命中任何技能返回空列表，
            调用方 _tool_names 会回退 intent.allowed_tools。
        """
        registered_tool_names: List[str] = self._tools.list_tool_names()

        def _collect_declared_skill_names() -> List[str]:
            """从 intent.slots 提取 Pipeline 声明的技能名（按优先级去重）。"""
            if intent is None:
                return []
            raw_slots: Any = getattr(intent, "slots", None) or {}
            ordered_candidates: List[str] = []
            for key in (
                "selected_skills_final",
                "suggested_skills",
                "eligible_skill_names",
            ):
                value = raw_slots.get(key) or []
                if isinstance(value, str):
                    value = [value]
                for skill_name in value or []:
                    cleaned_name: str = str(skill_name).strip()
                    if cleaned_name and cleaned_name not in ordered_candidates:
                        ordered_candidates.append(cleaned_name)
            return ordered_candidates

        try:
            available_skill_map: Dict[str, Any] = getattr(
                self._skill_manager.state, "available_skills", {}
            ) or {}
            declared_skill_names: List[str] = _collect_declared_skill_names()
            if declared_skill_names:
                gated_tools: List[str] = []
                resolved_names: List[str] = []
                for skill_name in declared_skill_names:
                    skill_meta: Any = available_skill_map.get(skill_name)
                    if not skill_meta:
                        logger.warning(
                            "【技能感知】Pipeline 声明的技能 [{}] 未在技能树中找到，跳过号令。",
                            skill_name,
                        )
                        continue
                    resolved_names.append(skill_name)
                    for tool_name in (
                        list(skill_meta.get("allowed_tools") or [])
                        if isinstance(skill_meta, dict)
                        else []
                    ):
                        if (
                            tool_name in registered_tool_names
                            and tool_name not in gated_tools
                        ):
                            gated_tools.append(tool_name)
                if resolved_names:
                    logger.info(
                        "【技能感知】命中 Pipeline 声明技能 {}，已号令工具集（与注册交集）: {}",
                        resolved_names,
                        gated_tools,
                    )
                    return gated_tools
                logger.info(
                    "【技能感知】Pipeline 声明技能 {} 均未在技能树中命中，降级关键词匹配。",
                    declared_skill_names,
                )

            # 降级：关键词匹配命中单一技能
            matched_skill_metadata: Any = self._skill_manager.resolve_relevant_skill(
                user_input
            )
            matched_skill_tools: List[str] = (
                list(matched_skill_metadata.get("allowed_tools") or [])
                if isinstance(matched_skill_metadata, dict)
                else []
            )
            if matched_skill_tools:
                skill_gated: List[str] = [
                    tool_name
                    for tool_name in matched_skill_tools
                    if tool_name in registered_tool_names
                ]
                logger.info(
                    "【方案A】命中技能 [{}]，已号令工具集（与注册交集）: {}",
                    matched_skill_metadata.get("name", "?"),
                    skill_gated,
                )
                return skill_gated
        except Exception as skill_gate_error:
            logger.warning(
                "【技能感知】技能工具号令解析失败（降级为 Pipeline 原始工具集）: {}",
                skill_gate_error,
            )
        return []

    def _tool_names(self, intent: IntentContext) -> List[str]:
        """根据当前的意图上下文与基础设施硬性边界，计算并返回当前可用的全量工具名称列表。
        """
        all_registered_names: List[str] = self._tools.list_tool_names()
        logger.info("当前注册中心已就绪的底层工具箱列表: {}", all_registered_names)

        active_skill_tools: List[str] = getattr(self, "_active_skill_tools", None) or []
        allowed_names: List[str]
        if active_skill_tools:
            allowed_names = [name for name in active_skill_tools if name in all_registered_names]
            logger.info("【Skill 号令】命中高级技能，按技能 allowed-tools 重排工具集: {}", allowed_names)
        elif intent.allowed_tools:
            allowed_names = [name for name in intent.allowed_tools if name in all_registered_names]
        else:
            allowed_names = list(all_registered_names)

        # 硬性保障原子级基础设施支撑工具在高级技能激活时不被误杀或过滤
        infrastructure_tools: List[str] = ["file_read_tool", "file_list_tool", "file_grep_tool"]

        for infra_name in infrastructure_tools:
            if infra_name in all_registered_names and infra_name not in allowed_names:
                allowed_names.append(infra_name)
                logger.info("【架构对齐】检测到高级技能上下文，已自动安全注入原子基础设施工具: {}", infra_name)

        # 联网搜索双梯队联动：第一梯队 web_search 被放行时，同步注入第二梯队备选
        # tavily_web_search（是否调用由工具 description/SYSTEM_PROMPT 提示词约束：
        # 仅当豆包搜索不可用/熔断/失败时才允许使用）
        if (
            "web_search" in allowed_names
            and "tavily_web_search" in all_registered_names
            and "tavily_web_search" not in allowed_names
        ):
            allowed_names.append("tavily_web_search")
            logger.info("【梯队联动】第一梯队 web_search 可用，已同步注入第二梯队备选 tavily_web_search")

        return allowed_names

    @langfuse_observe(name="AgentOrchestrator.run", as_type="agent", capture_input=False, capture_output=False)
    async def run(
            self,
            user_input: str,
            session_id: str,
            mode: str = "react",
            intent: Optional[IntentContext] = None,
            precomputed_memory: Optional[Any] = None,
    ) -> AgentResponse:
        """执行 Agent 编排：集成联合记忆、全量扩展工具、追踪与高级技能管理。

        去除了所有陈旧的硬编码 RAG 直接调用，将多路检索、图谱检索的调度权完全归还给底层具体的驱动智能体。
        """
        intent_context: IntentContext = intent or IntentContext()
        if intent_context.preferred_mode:
            mode = intent_context.preferred_mode

        # 【方案A】本次请求起始清空技能工具号令位（避免上个请求残留串扰本 Session 候选）
        self._active_skill_tools = None

        current_trace_id: str = self._tracer.new_trace_id()
        execution_span: Any = self._tracer.start_span(
            name="orchestrator.run",
            trace_id=current_trace_id,
            attributes={"session_id": session_id, "mode": mode, "intent_classify_resolver": intent_context.intent},
        )

        retrieved_memory_context: Dict[str, Any] = {}
        skills_instruction_prompt: str = ""

        # 1+2. 并行预热：技能树刷新 与 记忆联合检索 相互独立，同时发起后
        #       准备墙钟由串行 sum 降为 max(技能扫描, 记忆装载)。长期召回内部
        #       已 asyncio.to_thread（embedding + Milvus），不阻塞事件循环。
        async def _refresh_skill_tree() -> None:
            try:
                await self._skill_manager.scan_and_refresh_skills()

                # 提取技能列表纯文本
                skills_lines = []
                for name, meta in self._skill_manager.state.available_skills.items():
                    skills_lines.append(f"- **{name}**: {meta['description']} (Source File: `{meta['file_path']}`)")

                if skills_lines:
                    nonlocal skills_instruction_prompt
                    # 正确渲染 SKILLS_SYSTEM_PROMPT 模板
                    skills_instruction_prompt = SKILLS_SYSTEM_PROMPT.format(
                        skills_locations="",
                        skills_load_warnings="",
                        skills_list="\n".join(skills_lines)
                    )
                    logger.info("【编排层】高级技能树动态刷新完成，成功挂载渐进式披露操作规约。")
            except Exception as skill_exception:
                logger.warning("外部高级技能树扫描刷新时发生非致命异常: {}", skill_exception)
        async def _fetch_memory_context() -> Any:
            """拉取联合记忆上下文；支持外部并行预取（precomputed_memory）直通。"""
            if precomputed_memory is not None:
                logger.info("【编排层】命中外部并行预取的联合记忆上下文，跳过内部重复检索。")
                return precomputed_memory
            return await self._memory.get_context(session_id, user_input, limit=6)

        skill_refresh_task: asyncio.Task = asyncio.create_task(_refresh_skill_tree())
        memory_fetch_task: asyncio.Task = asyncio.create_task(_fetch_memory_context())

        try:
            raw_memory_context: Any = await memory_fetch_task
            # MemoryContext 是 Pydantic BaseModel（字段名 short_term_messages /
            # long_term_items），不是 dict：统一用 getattr 读取，避免在空列表时
            # 走到 .get() 触发 "'MemoryContext' object has no attribute 'get'"。
            short_term_history: List[Any] = getattr(
                raw_memory_context, "short_term_messages", None
            ) or getattr(raw_memory_context, "short_term", None) or []
            long_term_snippets: List[str] = getattr(
                raw_memory_context, "long_term_items", None
            ) or getattr(raw_memory_context, "long_term", None) or []

            # 🔴 断点1 修复：同时写两套键，保证下游 Planner / ReAct 两种读取方式都命中
            # 下划线键（"short_term" / "long_term"）：供 PlannerAgent.plan()、ReActAgent.run_react_agent()
            #                       内部 .get("short_term") / .get("long_term") 直接命中
            # 带空格键（"short term memory" / "long term memory"）：保留向后兼容，供本文件
            #                       _run_react() 的 formatted_memory_extension 继续读取
            retrieved_memory_context["short_term"] = short_term_history
            retrieved_memory_context["long_term"] = long_term_snippets
            retrieved_memory_context["short term memory"] = short_term_history
            retrieved_memory_context["long term memory"] = long_term_snippets
            logger.info("【编排层】联合记忆提取成功。短期对话历史: {} 条, 跨会话长期记忆: {} 条", len(short_term_history),
                        len(long_term_snippets))
        except Exception as memory_exception:
            logger.warning("联合记忆上下文拉取失败，启用空记忆模块降级运行: {}", memory_exception)
            self._tracer.log_event(current_trace_id, "memory.context_error", {"error": str(memory_exception)})
        finally:
            await skill_refresh_task

        # 【方案A】技能树已刷新，解析与当前 query 最相关的技能，用其 allowed-tools
        # 号令本 Session 的工具候选（读取 config 开关，命中技能且其工具与已注册有交集才生效）。
        #
        # 【技能感知升级】优先采纳 Pipeline 改写阶段 LLM 选定的技能名
        #  （intent.slots.selected_skills_final / suggested_skills），逐个解析其
        #  allowed-tools 合并号令；若 Pipeline 未声明任何技能，再降级回退到
        #  resolve_relevant_skill 的轻量关键词匹配。
        if self._config.get("enable_skill_tool_gating", True):
            self._active_skill_tools = self._resolve_skill_tool_gating(
                intent_context, user_input
            )

        self._tracer.log_event(
            current_trace_id,
            "orchestrator.start",
            {"user_input_len": len(user_input), "mode": mode},
        )

        executed_steps: List[Dict[str, Any]] = []
        is_degraded_execution: bool = False
        agent_execution_result: Optional[AgentResult] = None
        actual_orchestration_mode: OrchestrationMode = "react"

        # 工具调用预算：单次 Agent 请求作用域（plan 失败降级 react 时共享同一份额度，
        # 防止降级路径绕过总硬上限）。供 LLM 动态规划（react/plan/replan 提示词注入）与熔断共用。
        call_budget: Any = build_tool_call_budget(
            config=self._config,
            tool_registry=self._tools,
            tool_names=self._tool_names(intent_context),
        )
        logger.info(
            "本次 Agent 请求工具预算初始化完成：每工具上限 {}，总硬上限 {}",
            call_budget.per_tool_limits, call_budget.total_budget,
        )

        # 3. 路由并驱动具体的智能体核心（全权委派，无硬编码 RAG 调用）
        try:
            if mode == "plan_execute":
                actual_orchestration_mode = "plan_execute"
                agent_execution_result = await self._run_plan_execute(
                    user_input=user_input,
                    session_id=session_id,
                    intent=intent_context,
                    trace_id=current_trace_id,
                    steps=executed_steps,
                    memory_context=retrieved_memory_context,
                    skills_prompt=skills_instruction_prompt,
                    call_budget=call_budget,
                )
                # 规划执行模式如果失败，根据策略安全切换至 ReAct 熔断兜底模式
                if not agent_execution_result.success and self._fallback_on_plan_failure:
                    self._tracer.log_event(current_trace_id, "orchestrator.fallback", {"to": "react"})
                    is_degraded_execution = True
                    actual_orchestration_mode = "react"
                    agent_execution_result = await self._run_react(
                        user_input=user_input,
                        session_id=session_id,
                        intent=intent_context,
                        trace_id=current_trace_id,
                        steps=executed_steps,
                        memory_context=retrieved_memory_context,
                        suffix="[规划失败降级] ",
                        skills_prompt=skills_instruction_prompt,
                        call_budget=call_budget,
                    )
            else:
                agent_execution_result = await self._run_react(
                    user_input=user_input,
                    session_id=session_id,
                    intent=intent_context,
                    trace_id=current_trace_id,
                    steps=executed_steps,
                    memory_context=retrieved_memory_context,
                    skills_prompt=skills_instruction_prompt,
                    call_budget=call_budget,
                )

            final_answer_text: str = (agent_execution_result.final_answer if agent_execution_result else "") or ""
            is_execution_success: bool = bool(agent_execution_result and agent_execution_result.success)

            # 4. 执行成功，驱动持久化双向记忆沉淀闭环
            # 【t3 优化】短期历史保留同步写入（Redis 低延迟，且必须在本轮响应前落库，
            # 保证下一轮对话立即可见）；长期记忆沉淀（embedding API + Milvus 写入，
            # 高延迟）改为后台异步任务，不再阻塞 Agent 出结果的主链路。
            if is_execution_success and final_answer_text:
                try:
                    await self._memory.append_turn(session_id, "user", user_input)
                    await self._memory.append_turn(
                        session_id=session_id,
                        role="assistant",
                        content=final_answer_text,
                        metadata={"mode": actual_orchestration_mode, "trace_id": current_trace_id}
                    )
                except Exception as sync_memory_error:
                    logger.warning("智能体短期对话历史录入时发生异常: {}", sync_memory_error)
                # 沉淀至长期知识向量存储中（后台异步，异常由 _on_background_task_done 统一记录）
                if hasattr(self._memory, "_ltm") and hasattr(self._memory._ltm, "store"):
                    structured_long_term_record: str = f"用户问题: {user_input} \n智能体回答: {final_answer_text}"
                    self._spawn_background_task(self._memory._ltm.store(
                        session_id=session_id,
                        content=structured_long_term_record,
                        metadata={"mode": actual_orchestration_mode, "trace_id": current_trace_id},
                    ))
                    logger.info("已将当前对话对提交为长期向量记忆后台沉淀任务（不阻塞响应主链路）。")

            # （reflection 反思层已按需求整体剥离，不再触发独立审视与质量门判定；
            #  is_degraded_execution 仅由执行层降级路径决定）

            orchestration_response = AgentResponse(
                answer=final_answer_text,
                mode_used=actual_orchestration_mode,
                success=is_execution_success,
                trace_id=current_trace_id,
                intent=intent_context,
                steps=executed_steps,
                error=agent_execution_result.error if agent_execution_result and not is_execution_success else None,
                degraded=is_degraded_execution,
            )
            self._tracer.end_span(execution_span, error=None)
            return orchestration_response

        except Exception as system_uncaught_exception:
            logger.exception("智能体编排器内核发生未捕获的严重异常危机")
            self._tracer.end_span(execution_span, error=system_uncaught_exception)
            return AgentResponse(
                answer="",
                mode_used="react",
                success=False,
                trace_id=current_trace_id,
                intent=intent_context,
                steps=executed_steps,
                error=str(system_uncaught_exception),
                degraded=is_degraded_execution,
            )

    # _run_plan_execute跟_run_react是完全对立的两个分支
    async def _run_react(
            self,
            user_input: str,
            session_id: str,
            intent: IntentContext,
            trace_id: str,
            steps: List[Dict[str, Any]],
            memory_context: Dict[str, Any],
            suffix: str = "",
            skills_prompt: str = "",
            call_budget: Optional[Any] = None,
    ) -> AgentResult:
        """运行 ReAct（Thought -> Action -> Observation）交互推理循环。

        【核心重写点】：通过内建标准化工具调用代理，全面拉平新扩展工具箱（RAG检索、图谱、待办事项）的
        输出差异，保证复杂的结构化字典或异常信息被完美转换为符合模型闭环消费的标准字符串格式。
        """
        configured_max_steps: int = int(self._config.get("react_max_steps", 10))

        async def step_tracing_callback(step_record: Dict[str, Any]) -> None:
            wrapped_payload: Dict[str, Any] = {"ts": time.time(), **step_record}
            steps.append(wrapped_payload)
            self._tracer.log_event(trace_id, "react.step", wrapped_payload)

        # 1. 组装对话历史与事实记忆片段的文本提示词芯片
        # 🔴 断点1 兼容：优先读 "short_term" 下划线键（新版写法），回退到带空格键（旧版写法兜底）
        history_prompt_chips: List[str] = []
        recent_short_term_history: List[Any] = (
            memory_context.get("short_term") or memory_context.get("short_term_memory") or []
        )
        if recent_short_term_history:
            history_prompt_chips.append("\n【当前会话近期历史对话上下文（按时间顺序由远及近）】:")
            for history_message in recent_short_term_history:
                message_role: Any = getattr(history_message, "role", None) or (
                    history_message.get("role") if isinstance(history_message, dict) else "user")
                role_string_key: str = message_role.value if hasattr(message_role, "value") else str(message_role)
                message_content: str = getattr(history_message, "content", None) or (
                    history_message.get("content") if isinstance(history_message, dict) else "")
                history_prompt_chips.append(f" - {role_string_key}: {message_content}")

        # 🔴 断点1 兼容：优先读 "long_term" 下划线键（新版），回退带空格键（旧版兜底）
        recalled_long_term_facts: List[str] = (
            memory_context.get("long_term") or memory_context.get("long_term_memory") or []
        )
        if recalled_long_term_facts:
            history_prompt_chips.append("\n【从本地私有向量库中检索出的历史跨会话长期事实背景（供答题参考）】:")
            for fact_snippet in recalled_long_term_facts:
                history_prompt_chips.append(f" - {fact_snippet}")

        formatted_memory_extension: str = "\n".join(history_prompt_chips) if history_prompt_chips else ""

        # 🟡 断点3 修复：Pipeline 的 first_tool_hint 单独结构化引导（不再混在 slots Python repr 里）
        # 注意：这里用 intent.allowed_tools（Pipeline 阶段已计算好并和 REGISTERED 做过交集）
        #       做合法性校验，而不是 target_active_tool_names（后者在 L377 才初始化，避免 NameError）。
        pipeline_first_tool_hint_line: str = ""
        first_tool_hint_value: Any = intent.slots.get("first_tool_hint")
        allowed_tool_names_for_hint: List[str] = list(intent.allowed_tools or [])
        if (isinstance(first_tool_hint_value, str)
                and first_tool_hint_value.strip()
                and (not allowed_tool_names_for_hint
                     or first_tool_hint_value.strip() in allowed_tool_names_for_hint)):
            pipeline_first_tool_hint_line = (
                "【Pipeline 决策层冷启动提示】"
                f"Pipeline 阶段分析推荐的第一步工具是：{first_tool_hint_value.strip()}，"
                "建议优先考虑从它切入（若实际推理判断不合理，可自由切换到其他匹配工具）。\n"
            )
        system_base_guideline: str = (
            suffix
            + pipeline_first_tool_hint_line
            + f"当前识别意图：{intent.intent}，定位槽位：{intent.slots}\n{formatted_memory_extension}"
        )

        # 2. 挂载最高执行纲领技能树提示词
        if skills_prompt:
            complete_extra_system_prompt: str = f"{skills_prompt}\n\n{system_base_guideline}"
        else:
            complete_extra_system_prompt = system_base_guideline

        # 3. 提取全量可用工具名称，并动态解析导出工具库的精确参数 Schema（防止大模型盲猜字段名）
        target_active_tool_names: List[str] = self._tool_names(intent)

        # 【改进点 2 · KB 命中引导】意图路由硬约束：Pipeline 已识别出 KB TOP1 节点的
        # 专用集合（slots.top_kb_node.collection_names，经 raw_slots → merged_slots 直通），
        # 将其注入 System Prompt，强制 ReAct 调用 rag_knowledge_search 时携带
        # collection_names 定向检索，避免向量检索在全库"乱撞"。
        kb_node_slot: Dict[str, Any] = intent.slots.get("top_kb_node") or {}
        kb_collection_names: List[str] = [
            str(name).strip()
            for name in (kb_node_slot.get("collection_names") or [])
            if str(name).strip()
        ]
        if kb_collection_names and "rag_knowledge_search" in target_active_tool_names:
            kb_collection_hint: str = (
                "【意图路由硬约束】用户意图明确指向知识库集合："
                f"{', '.join(kb_collection_names)}。"
                "当你决定调用 `rag_knowledge_search` 时，**必须**将 `collection_names` 参数设置为 "
                f"{kb_collection_names}，严禁查询其他无关集合。"
                "如果其他工具（如 web_search）更适合，可正常使用。"
            )
            complete_extra_system_prompt = (
                f"{complete_extra_system_prompt}\n\n{kb_collection_hint}"
                if complete_extra_system_prompt
                else kb_collection_hint
            )
            logger.info(
                "ReAct 已注入 KB 集合定向检索硬约束: collection_names={}", kb_collection_names
            )

        extracted_tool_schemas: Dict[str, str] = {}
        # 原生 Function Calling 工具池：与文本 catalog 同步构建，供 ReAct 主循环以 tools= 透传
        fc_tool_definitions: List[Dict[str, Any]] = []

        # 建立健壮的工具反射读取机制，优先读取原生 get_tool，否则尝试反射内部受保护字典容器
        tools_source_container: Dict[str, Any] = getattr(self._tools, "_tools", {}) if not hasattr(self._tools,
                                                                                                   "get_tool") else {}

        for tool_name in target_active_tool_names:
            target_tool_instance: Optional[Any] = None
            if hasattr(self._tools, "get_tool"):
                target_tool_instance = self._tools.get_tool(tool_name)
            else:
                target_tool_instance = tools_source_container.get(tool_name)

            if target_tool_instance and hasattr(target_tool_instance, "schema_parameters"):
                # FC 工具定义：参数 JSON Schema 由工具自身 schema_parameters() 提供
                fc_tool_definitions.append(
                    tool_to_function_call_definition(target_tool_instance)
                )

            if target_tool_instance and hasattr(target_tool_instance, "parameters"):
                schema_lines_builder: List[str] = [f"功能描述: {target_tool_instance.description}"]
                # 特殊兼容带有特化结构（如 write_todos 嵌套复写）的 Schema 生成方法
                if hasattr(target_tool_instance, "schema_parameters") and callable(
                        target_tool_instance.schema_parameters):
                    try:
                        custom_schema: Dict[str, Any] = target_tool_instance.schema_parameters()
                        schema_lines_builder.append(
                            f"输入JSON格式规约Schema: {json.dumps(custom_schema, ensure_ascii=False)}")
                    except Exception as schema_error:
                        logger.warning("导出高级工具复写 Schema 失败，退化为基础描述。原因: {}", schema_error)
                else:
                    # 基础扁平化工具参数反射提取
                    parameter_details_list: List[str] = []
                    for single_param in target_tool_instance.parameters:
                        is_required_mark: str = "必填" if getattr(single_param, "required", False) else "可选"
                        parameter_details_list.append(
                            f"  - {single_param.name} ({single_param.type}, {is_required_mark}): {single_param.description}")
                    if parameter_details_list:
                        schema_lines_builder.append("可接受输入参数列表:\n" + "\n".join(parameter_details_list))

                # 针对多路检索与任务规划工具，动态附加工具专用的操作守则，完成系统提示词纵向深度融合
                if hasattr(target_tool_instance, "SYSTEM_PROMPT"):
                    schema_lines_builder.append(f"【此工具专属调用守则】：\n{target_tool_instance.SYSTEM_PROMPT}")

                extracted_tool_schemas[tool_name] = "\n".join(schema_lines_builder)

        # 4. 【核心闭环设计】：构造本地闭环工具调用转换拦截代理
        class StandardizedToolInvokerProxy:
            """拦截底层具体工具执行结果，统一将 dict、JSON 及异常拉平转换为标准文本形态的 Observation。"""

            def __init__(self, core_registry: ToolRegistry) -> None:
                self._core_registry: ToolRegistry = core_registry

            async def invoke(self, name: str, arguments: Dict[str, Any]) -> str:
                raw_tool_output: Any = await self._core_registry.invoke(name, arguments)
                if raw_tool_output is None:
                    return "【系统提示】工具执行完毕，未返回任何有效可视化数据。"
                if isinstance(raw_tool_output, dict):
                    # 如果工具内部返回了结构化字典（例如 write_todos 响应），统一序列化为易读的标准 JSON 文本段
                    return json.dumps(raw_tool_output, ensure_ascii=False, indent=2)
                return str(raw_tool_output)

        standardized_invoker_proxy = StandardizedToolInvokerProxy(core_registry=self._tools)

        # 5. 实例化底层的纯净驱动型 ReActAgent 实例，交出执行支配权
        # 【P1① 扁平化改造】：直接注入全局 ModelRouter 单例 + purpose_hint，
        # 链路由「Agent → _LLMAdapter → _PurposeLLMAdapter → ModelRouter」（4 层）
        # 降为「Agent → ModelRouter」（2 层），同时共享全局熔断器状态。
        react_agent_driver = ReActAgent(
            model_router=self._model_router,
            purpose_hint="react",
            tools=standardized_invoker_proxy,  # 注入拉平后的标准化执行代理
            memory=None,
            max_steps=configured_max_steps,
            session_id=session_id,
        )

        reactive_agent_context: Dict[str, Any] = {
            "session_id": session_id,
            "tool_names": target_active_tool_names,
            "tool_schemas": extracted_tool_schemas,  # 动态灌入经过深度提炼拼接参数的 Schema
            "openai_tools": fc_tool_definitions,     # 原生 Function Calling 工具池（tools[]）
            "call_budget": call_budget,              # 工具调用预算（动态规划 + 熔断）
            "tool_limits": call_budget.per_tool_limits if call_budget is not None else {},  # 兼容键
            "max_tool_attempts": (
                int(call_budget.default_per_tool)
                if call_budget is not None
                else int(self._config.get("tool_max_attempts", 3))
            ),
            "trace_callback": step_tracing_callback,
            "memory_context": memory_context,
            "extra_system": complete_extra_system_prompt,
        }

        return await react_agent_driver.run_react_agent(query=user_input, context=reactive_agent_context)
#_run_plan_execute跟——run_react是完全对立的两个分支
    async def _run_plan_execute(
            self,
            user_input: str,
            session_id: str,
            intent: IntentContext,
            trace_id: str,
            steps: List[Dict[str, Any]],
            memory_context: Dict[str, Any],
            skills_prompt: str = "",
            call_budget: Optional[Any] = None,
    ) -> AgentResult:
        """运行基于宏观任务拆解与重规划（Plan-and-Execute）架构的执行驱动器。

        【核心重写点】：在子任务拆解前，将新注入的混合RAG工具、实体图谱工具以及TodoList任务规划工具
        进行动态名称过滤并足额喂给规划器，确保宏观战略阶段能充分考虑并编排多路跨维度检索。
        """
        async def subtask_tracing_callback(subtask_payload: Dict[str, Any]) -> None:
            wrapped_entry: Dict[str, Any] = {"ts": time.time(), **subtask_payload}
            steps.append(wrapped_entry)
            self._tracer.log_event(trace_id, "plan_execute.subtask", wrapped_entry)

        # 确保新加入的检索利器与规划利器完整纳入规划宏观视野
        available_tools_for_planning: List[str] = self._tool_names(intent)
        logger.info("【规划层激活】已将扩展工具集全面呈报至宏观规划器视野: {}", available_tools_for_planning)


        planner_agent_driver = PlannerAgent(
            model_router=self._model_router,
            purpose_hint="planner",
            tools=self._tools,
            memory=None,
            max_replan_attempts=int(self._config.get("max_replan_attempts", 2)),
            call_budget=call_budget,
            enable_empty_result_replan=bool(self._config.get("enable_empty_result_replan", True)),
        )

        # 🟠 断点4 修复：主意图锚点注入 Planner（放在 skills_prompt 最前面，plan() 会直接渲染到 user_message）
        planner_intent_anchor_block: str = ""
        primary_intent_text: str = str(intent.intent or "").strip()
        if primary_intent_text and primary_intent_text.lower() != "general":
            planner_intent_anchor_block = (
                f"## 当前识别用户意图（Pipeline 决策层分析结果）\n{primary_intent_text}\n\n"
                "请围绕以上意图方向进行子任务拆解。\n\n"
            )

        # 🔴 断点2 修复：initial_plan_hint 冷启动宏观步骤建议注入 Planner
        planner_initial_hint_block: str = ""
        initial_plan_hint_value: Any = intent.slots.get("initial_plan_hint")
        if isinstance(initial_plan_hint_value, str) and initial_plan_hint_value.strip():
            planner_initial_hint_block = (
                "## Pipeline 阶段给出的宏观计划参考（冷启动 hint）\n"
                f"{initial_plan_hint_value.strip()}\n\n"
                "以上是前置决策层的参考步骤建议，可直接采纳，也可结合记忆与工具情况进行合理调整，"
                "但请保证最终拆解方向与上述宏观目标保持一致。\n\n"
            )

        # 【改进点 3 · 子问题意图感知】分而治之约束：Pipeline 改写阶段若拆分了多个
        # 子问题（slots.per_sub_questions / sub_intent_scores，由 _merge_slots 透传），
        # 则要求 Planner 为每个子问题规划至少一个独立子任务，而不是把拼接后的
        # 大问题当作一个整体笼统规划。
        sub_question_constraint_block: str = ""
        per_sub_questions_list: List[str] = [
            str(item).strip() for item in (intent.slots.get("per_sub_questions") or []) if str(item).strip()
        ]
        sub_intent_scores_list: List[Dict[str, Any]] = (
            intent.slots.get("sub_intent_scores") or []
        )
        if len(per_sub_questions_list) > 1:
            constraint_lines: List[str] = [
                "## 多子问题拆分约束（Pipeline 决策层分析结果）",
                "用户原问题已被拆解为以下相互独立的子问题，每个子问题应视为一个独立的目标：",
            ]
            for sub_index, sub_question_text in enumerate(per_sub_questions_list):
                sub_scores_entry: Dict[str, Any] = (
                    sub_intent_scores_list[sub_index]
                    if sub_index < len(sub_intent_scores_list) and isinstance(sub_intent_scores_list[sub_index], dict)
                    else {}
                )
                sub_scores_items: List[Dict[str, Any]] = (
                    sub_scores_entry.get("scores") or []
                )
                top_intent_id: str = str(
                    (sub_scores_items[0].get("id") if sub_scores_items else None)
                    or (sub_scores_items[0].get("name") if sub_scores_items else None)
                    or "general"
                )
                constraint_lines.append(
                    f"  子问题 {sub_index + 1}: {sub_question_text}（意图倾向: {top_intent_id}）"
                )
            constraint_lines.append(
                "请为每个子问题规划至少一个对应的子任务（tool / reasoning），"
                "确保所有子问题都被覆盖，不得遗漏。"
            )
            sub_question_constraint_block = "\n".join(constraint_lines) + "\n\n"

        # 合并顺序：意图锚点（最高优先级） → 初始计划 hint → 子问题拆分约束 → 原高级技能树规约 skills_prompt
        merged_planner_skills_prompt: str = (
            planner_intent_anchor_block + planner_initial_hint_block + sub_question_constraint_block + (skills_prompt or "")
        )

        return await planner_agent_driver.run_with_plan(
            query=user_input,
            session_id=session_id,
            tool_names=available_tools_for_planning,
            trace_callback=subtask_tracing_callback,
            memory_context=memory_context,
            skills_block=merged_planner_skills_prompt,  # 已合并：意图锚点 + 初始计划 hint + 高级技能规约
        )
