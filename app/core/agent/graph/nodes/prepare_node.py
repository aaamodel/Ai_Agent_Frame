# -*- coding: utf-8 -*-
"""prepare 节点：原 orchestrator.run 前导段（L283-L388）的平移。

- 技能树刷新（与记忆拉取并行）+ skills_prompt 渲染
- 联合记忆拉取（外部并行预取直通）+ 四键归一
- 技能 gating（Pipeline 声明技能优先，关键词匹配降级）
- 工具白名单（基础设施工具注入 + web_search 梯队联动）
- 工具文本 schema / 原生 FC 定义构建
- ToolCallBudget 初始化 → state 预算账本
- first_tool_hint / KB 集合硬约束 → extra_system_hints
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.agent.graph.deps import get_deps
from app.core.agent.graph.nodes._common import (
    agent_goal_trace_payload,
    normalize_memory_context,
    render_skills_index,
    render_skills_prompt,
    trace_event,
)
from app.core.agent.graph.state import (
    AgentGraphState,
    budget_to_ledger,
)
from app.core.tools.base import tool_to_function_call_definition
from app.core.tools.registry import build_tool_call_budget


# ---------------------------------------------------------------------------
# 技能工具号令（从 orchestrator._resolve_skill_tool_gating 平移，逻辑零改动）
# ---------------------------------------------------------------------------
def _resolve_skill_tool_gating(deps: Any, intent: Dict[str, Any], user_input: str) -> List[str]:
    """解析本 Session 号令工具的技能工具集（见原 orchestrator 同名方法 docstring）。"""
    registered_tool_names: List[str] = deps.tools.list_tool_names()

    def _collect_declared_skill_names() -> List[str]:
        raw_slots: Any = intent.get("slots") or {}
        ordered_candidates: List[str] = []
        for key in ("selected_skills_final", "suggested_skills", "eligible_skill_names"):
            value = raw_slots.get(key) or []
            if isinstance(value, str):
                value = [value]
            for skill_name in value or []:
                cleaned_name: str = str(skill_name).strip()
                if cleaned_name and cleaned_name not in ordered_candidates:
                    ordered_candidates.append(cleaned_name)
        return ordered_candidates

    try:
        available_skill_map: Dict[str, Any] = (
            getattr(deps.skill_manager.state, "available_skills", {}) or {}
        )
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
                    if tool_name in registered_tool_names and tool_name not in gated_tools:
                        gated_tools.append(tool_name)
            if resolved_names:
                logger.info(
                    "【技能感知】命中 Pipeline 声明技能 {}，已号令工具集（与注册交集）: {}",
                    resolved_names, gated_tools,
                )
                return gated_tools
            logger.info(
                "【技能感知】Pipeline 声明技能 {} 均未在技能树中命中，降级关键词匹配。",
                declared_skill_names,
            )

        matched_skill_metadata: Any = deps.skill_manager.resolve_relevant_skill(user_input)
        matched_skill_tools: List[str] = (
            list(matched_skill_metadata.get("allowed_tools") or [])
            if isinstance(matched_skill_metadata, dict)
            else []
        )
        if matched_skill_tools:
            skill_gated: List[str] = [
                tool_name for tool_name in matched_skill_tools
                if tool_name in registered_tool_names
            ]
            logger.info(
                "【方案A】命中技能 [{}]，已号令工具集（与注册交集）: {}",
                matched_skill_metadata.get("name", "?"), skill_gated,
            )
            return skill_gated
    except Exception as skill_gate_error:  # noqa: BLE001
        logger.warning(
            "【技能感知】技能工具号令解析失败（降级为 Pipeline 原始工具集）: {}",
            skill_gate_error,
        )
    return []


def _resolve_tool_names(
    deps: Any, intent: Dict[str, Any], active_skill_tools: Optional[List[str]]
) -> List[str]:
    """计算本次可用工具白名单（从 orchestrator._tool_names 平移，逻辑零改动）。"""
    all_registered_names: List[str] = deps.tools.list_tool_names()
    logger.info("当前注册中心已就绪的底层工具箱列表: {}", all_registered_names)

    if active_skill_tools:
        skill_names: List[str] = [
            name for name in active_skill_tools if name in all_registered_names
        ]
        # 技能只做**收窄**，不做扩充：与意图层白名单取交集。
        # 原实现是"用技能的 allowed-tools 整体替换意图白名单"，会把意图层刻意没给的
        # 工具一起塞进来（实测 T01：意图 6 个 → 替换后 10 个），每轮多带 4~5 份工具
        # schema（≈ +400 token/轮）与相应的选择噪音，纯浪费。交集为空时退回技能清单
        # （防御：正常不会发生，留个兜底避免把工具集清空）。
        intent_names: List[str] = [
            name
            for name in (intent.get("allowed_tools") or [])
            if name in all_registered_names
        ]
        if intent_names:
            narrowed: List[str] = [name for name in skill_names if name in intent_names]
            allowed_names = narrowed or skill_names
        else:
            allowed_names = skill_names
        logger.info(
            "【Skill 号令】命中高级技能，与意图白名单取交集: {}（技能 {} / 意图 {}）",
            allowed_names, skill_names, intent_names,
        )
    elif intent.get("allowed_tools"):
        allowed_names = [
            name for name in intent["allowed_tools"] if name in all_registered_names
        ]
    else:
        allowed_names = list(all_registered_names)

    infrastructure_tools: List[str] = ["file_read_tool", "file_list_tool", "file_grep_tool"]
    for infra_name in infrastructure_tools:
        if infra_name in all_registered_names and infra_name not in allowed_names:
            allowed_names.append(infra_name)
            logger.info("【架构对齐】检测到高级技能上下文，已自动安全注入原子基础设施工具: {}", infra_name)

    # 联网搜索不再有"梯队联动"注入：Tavily 降级通道已不注册、不进白名单，
    # 降级完全发生在 web_search 工具内部（空结果 / 调用异常 → 代码层直接换通道）。
    return allowed_names


def _build_tool_catalog(deps: Any, tool_names: List[str]):
    """构建文本路径工具 schema 文本与原生 FC 工具定义（从 _run_react L599-L645 平移）。"""
    tool_schemas: Dict[str, str] = {}
    fc_tool_definitions: List[Dict[str, Any]] = []

    tools_source_container: Dict[str, Any] = (
        getattr(deps.tools, "_tools", {}) if not hasattr(deps.tools, "get_tool") else {}
    )

    for tool_name in tool_names:
        target_tool_instance: Any = None
        if hasattr(deps.tools, "get_tool"):
            target_tool_instance = deps.tools.get_tool(tool_name)
        else:
            target_tool_instance = tools_source_container.get(tool_name)

        if target_tool_instance and hasattr(target_tool_instance, "schema_parameters"):
            fc_tool_definitions.append(
                tool_to_function_call_definition(target_tool_instance)
            )

        if target_tool_instance and hasattr(target_tool_instance, "parameters"):
            schema_lines: List[str] = [f"功能描述: {target_tool_instance.description}"]
            if (
                hasattr(target_tool_instance, "schema_parameters")
                and callable(target_tool_instance.schema_parameters)
            ):
                try:
                    custom_schema: Dict[str, Any] = target_tool_instance.schema_parameters()
                    schema_lines.append(
                        f"输入JSON格式规约Schema: {json.dumps(custom_schema, ensure_ascii=False)}"
                    )
                except Exception as schema_error:  # noqa: BLE001
                    logger.warning("导出高级工具复写 Schema 失败，退化为基础描述。原因: {}", schema_error)
            else:
                parameter_details: List[str] = []
                for single_param in target_tool_instance.parameters:
                    is_required = "必填" if getattr(single_param, "required", False) else "可选"
                    parameter_details.append(
                        f"  - {single_param.name} ({single_param.type}, {is_required}): "
                        f"{single_param.description}"
                    )
                if parameter_details:
                    schema_lines.append("可接受输入参数列表:\n" + "\n".join(parameter_details))

            if hasattr(target_tool_instance, "SYSTEM_PROMPT"):
                schema_lines.append(f"【此工具专属调用守则】：\n{target_tool_instance.SYSTEM_PROMPT}")

            tool_schemas[tool_name] = "\n".join(schema_lines)

    return tool_schemas, fc_tool_definitions


def _build_extra_hints(intent: Dict[str, Any], active_tool_names: List[str]) -> List[str]:
    """first_tool_hint 冷启动提示 + KB 集合定向硬约束（从 _run_react 平移）。"""
    hints: List[str] = []

    first_tool_hint_value: Any = (intent.get("slots") or {}).get("first_tool_hint")
    allowed_for_hint: List[str] = list(intent.get("allowed_tools") or [])
    if (
        isinstance(first_tool_hint_value, str)
        and first_tool_hint_value.strip()
        and (not allowed_for_hint or first_tool_hint_value.strip() in allowed_for_hint)
    ):
        hints.append(
            "【Pipeline 决策层冷启动提示】"
            f"Pipeline 阶段分析推荐的第一步工具是：{first_tool_hint_value.strip()}，"
            "建议优先考虑从它切入（若实际推理判断不合理，可自由切换到其他匹配工具）。"
        )

    kb_node_slot: Dict[str, Any] = (intent.get("slots") or {}).get("top_kb_node") or {}
    kb_collection_names: List[str] = [
        str(name).strip()
        for name in (kb_node_slot.get("collection_names") or [])
        if str(name).strip()
    ]
    # 动态集合节点携带的工具名决定走哪个引擎：
    # knowledge_graph_search → LightRAG 图谱 workspace；rag_knowledge_search → Milvus
    kb_tool_names: List[str] = [
        str(name).strip()
        for name in (kb_node_slot.get("agent_tool_names") or [])
        if str(name).strip()
    ]
    if kb_collection_names and "knowledge_graph_search" in kb_tool_names:
        if "knowledge_graph_search" in active_tool_names:
            hints.append(
                "【意图路由硬约束】用户意图明确指向知识图谱集合："
                f"{', '.join(kb_collection_names)}。"
                "当你决定调用 `knowledge_graph_search` 时，**必须**将 `collection` 参数"
                f"设置为 {kb_collection_names[0]!r}，严禁查询其他图谱集合。"
                "如果其他工具（如 web_search）更适合，可正常使用。"
            )
            logger.info(
                "ReAct 已注入图谱集合定向检索硬约束: collection={}",
                kb_collection_names,
            )
    elif kb_collection_names and "rag_knowledge_search" in active_tool_names:
        hints.append(
            "【意图路由硬约束】用户意图明确指向知识库集合："
            f"{', '.join(kb_collection_names)}。"
            "当你决定调用 `rag_knowledge_search` 时，**必须**将 `collection_names` 参数设置为 "
            f"{kb_collection_names}，严禁查询其他无关集合。"
            "如果其他工具（如 web_search）更适合，可正常使用。"
        )
        logger.info("ReAct 已注入 KB 集合定向检索硬约束: collection_names={}", kb_collection_names)

    return hints


async def prepare_node(state: AgentGraphState, config: RunnableConfig) -> dict:
    """前导段：技能/记忆并行预热 → gating → 工具集/schema → 预算账本 → 提示词 hints。"""
    deps = get_deps(config)
    trace_id: str = state["trace_id"]
    session_id: str = state["session_id"]
    user_input: str = state["user_input"]
    intent: Dict[str, Any] = state.get("intent") or {}

    # 1. 技能树刷新 与 记忆检索 并行（记忆支持图外并行预取直通）
    # text = 执行侧完整规约；index = 规划侧极简清单（两者用途不同，都要渲染）
    skills_prompt_holder: Dict[str, str] = {"text": "", "index": ""}

    async def _refresh_skill_tree() -> None:
        try:
            await deps.skill_manager.scan_and_refresh_skills()
            skills_prompt_holder["text"] = render_skills_prompt(deps.skill_manager)
            skills_prompt_holder["index"] = render_skills_index(deps.skill_manager)
            if skills_prompt_holder["text"]:
                logger.info("【编排层】高级技能树动态刷新完成，成功挂载渐进式披露操作规约。")
        except Exception as skill_exception:  # noqa: BLE001
            logger.warning("外部高级技能树扫描刷新时发生非致命异常: {}", skill_exception)

    async def _fetch_memory_context() -> Any:
        precomputed = state.get("memory_context")
        if precomputed:
            logger.info("【编排层】命中外部并行预取的联合记忆上下文，跳过内部重复检索。")
            return precomputed
        return await deps.memory.get_context(session_id, user_input, limit=6)

    skill_task = asyncio.create_task(_refresh_skill_tree())
    memory_task = asyncio.create_task(_fetch_memory_context())

    memory_context: Dict[str, Any] = {}
    try:
        raw_memory = await memory_task
        memory_context = normalize_memory_context(raw_memory)
        logger.info(
            "【编排层】联合记忆提取成功。短期对话历史: {} 条, 跨会话长期记忆: {} 条",
            len(memory_context.get("short_term") or []),
            len(memory_context.get("long_term") or []),
        )
    except Exception as memory_exception:  # noqa: BLE001
        logger.warning("联合记忆上下文拉取失败，启用空记忆模块降级运行: {}", memory_exception)
        trace_event(
            deps.tracer, trace_id, "memory.context_error", {"error": str(memory_exception)}
        )
        memory_context = normalize_memory_context(None)
    finally:
        await skill_task

    # 2. 技能 gating（配置开关默认开）
    active_skill_tools: Optional[List[str]] = None
    if deps.cfg("enable_skill_tool_gating", True):
        active_skill_tools = _resolve_skill_tool_gating(deps, intent, user_input) or None

    # 3. 工具白名单 + schema / FC 定义
    active_tool_names = _resolve_tool_names(deps, intent, active_skill_tools)
    tool_schemas, fc_definitions = _build_tool_catalog(deps, active_tool_names)

    # 4. 单次 run 作用域预算（plan/react/replan 共享）→ state 账本
    #    检索类工具的配额是"每个 rewrite 子问题 3 次"，所以这里要把子问题数传进去。
    #    子问题数只认 slots["per_sub_questions"]：该键**仅在实际拆分时存在**，
    #    缺键即按 1 个子问题算（不要改用 pipeline_rewrite_meta.sub_questions_count，
    #    未拆分时它也可能是 1，语义不同）。
    sub_question_count: int = (
        len(
            [
                str(q)
                for q in ((intent.get("slots") or {}).get("per_sub_questions") or [])
                if str(q).strip()
            ]
        )
        or 1
    )
    call_budget = build_tool_call_budget(
        config=deps.config,
        tool_registry=deps.tools,
        tool_names=active_tool_names,
        sub_question_count=sub_question_count,
    )
    logger.info(
        "本次 Agent 请求工具预算初始化完成：每工具上限 {}，总硬上限 {}（rewrite 子问题数 {}）",
        call_budget.per_tool_limits, call_budget.total_budget, sub_question_count,
    )

    # 5. 提示词 hints
    extra_hints = _build_extra_hints(intent, active_tool_names)

    trace_event(
        deps.tracer, trace_id, "orchestrator.start",
        {"user_input_len": len(user_input), "should_plan": bool(state.get("should_plan"))},
    )

    # 本轮目标的留痕（`prepare` 是入口，两条路径必经，因此**一轮只记一次**）。
    # 用途有两个：① 抽查目标质量是否稳定为"交付物形态"而非问题复述；
    # ② 排查"某个注入点其实用的是改写后的问题"——source=fallback 即说明槽位没供上。
    trace_event(deps.tracer, trace_id, "agent.goal", agent_goal_trace_payload(state))

    return {
        "memory_context": memory_context,
        "skills_prompt": skills_prompt_holder["text"],
        "skills_index": skills_prompt_holder.get("index", ""),
        "active_tool_names": active_tool_names,
        "tool_schemas": tool_schemas,
        "fc_tool_definitions": fc_definitions,
        "extra_system_hints": extra_hints,
        "budget": budget_to_ledger(call_budget),
        "max_replan": int(deps.cfg("max_replan_attempts", 2)),
        "max_steps": int(deps.cfg("react_max_steps", 10)),
    }
