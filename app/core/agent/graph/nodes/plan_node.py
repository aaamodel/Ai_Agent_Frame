# -*- coding: utf-8 -*-
"""plan 节点：复用 PlannerAgent.plan()，意图锚点/hint/子问题约束块从旧 _run_plan_execute 迁入。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.agent.graph.deps import get_deps
from app.core.agent.graph.nodes._common import agent_goal_from_state, trace_event
from app.core.agent.graph.state import AgentGraphState, budget_from_ledger
from app.core.agent.planner import PlannerAgent


def _flatten(text: Any) -> str:
    """去掉全部空白，用于"逐字相同"判定（忽略换行与缩进差异）。"""
    return "".join(str(text or "").split())


def _build_planner_skills_block(
    state: AgentGraphState,
    intent_confidence_threshold: float = 0.0,
) -> str:
    """合并顺序：本轮目标 → 意图锚点 → 初始计划 hint → 子问题拆分约束 → 技能清单。

    Args:
        state: 图状态。
        intent_confidence_threshold: 意图锚点的最低置信度门槛。低于该值（含置信度
            缺失或无法解析）时**不注入**意图方向约束——错误意图比没有意图更糟，
            实测一个不该通过的意图标签让 planner 把子任务分配给了知识图谱检索，
            白跑一次工具调用与一次子任务提炼。
    """
    # 本轮目标：作为**最高层**约束放在最前——规划要围绕"最终交付什么"来拆解，
    # 而意图标签只说明"问题属于哪个领域"。缺失（两侧都空）时不注入占位噪音。
    goal_text: str = agent_goal_from_state(state)
    goal_block: str = (
        "## 本轮目标（最终要交付的东西）\n"
        f"{goal_text}\n\n"
        "请让拆解出的每个子任务都服务于该目标。注意本目标描述的是最终交付物，"
        "不是步骤清单——具体步骤由你拆解。\n\n"
    ) if goal_text else ""

    intent: Dict[str, Any] = state.get("intent") or {}
    slots: Dict[str, Any] = intent.get("slots") or {}

    # 主意图锚点（非 general 才注入，且**必须达到置信度门槛**）
    # ⚠️ 置信度缺失或无法解析一律按"不足"处理，不得默认视为达标。
    anchor_block: str = ""
    primary_intent_text: str = str(intent.get("intent") or "").strip()
    raw_confidence: Any = intent.get("confidence")
    try:
        effective_confidence: Optional[float] = (
            float(raw_confidence) if raw_confidence is not None else None
        )
    except (TypeError, ValueError):
        effective_confidence = None
    confidence_ok: bool = (
        effective_confidence is not None
        and effective_confidence >= intent_confidence_threshold
    )
    if (
        primary_intent_text
        and primary_intent_text.lower() != "general"
        and confidence_ok
    ):
        anchor_block = (
            f"## 当前识别用户意图（Pipeline 决策层分析结果）\n{primary_intent_text}\n\n"
            f"（置信度 {effective_confidence:.2f}）\n\n"
            "请围绕以上意图方向进行子任务拆解。\n\n"
        )

    # initial_plan_hint 冷启动宏观步骤建议
    #
    # ⚠️ 与用户问题逐字相同时**整段不注入**：该段的意义是提供 planner 无法自行推导
    # 的步骤语义；内容等于用户原问题时，它既不携带新信息，又会让"复述问题"看起来
    # 像一条计划依据。实测（2026-09）中该段与「当前用户新目标」逐字重复——根因是
    # 步骤提示抽取把整句问题当成了步骤（已在改写层修掉），这里再做一道确定性防御。
    hint_block: str = ""
    initial_plan_hint_value: Any = slots.get("initial_plan_hint")
    hint_text: str = (
        initial_plan_hint_value.strip()
        if isinstance(initial_plan_hint_value, str)
        else ""
    )
    if hint_text and _flatten(hint_text) != _flatten(state.get("user_input")):
        hint_block = (
            "## Pipeline 阶段给出的宏观计划参考（冷启动 hint）\n"
            f"{hint_text}\n\n"
            "以上是前置决策层的参考步骤建议，可直接采纳，也可结合记忆与工具情况进行合理调整，"
            "但请保证最终拆解方向与上述宏观目标保持一致。\n\n"
        )

    # 多子问题拆分约束
    sub_constraint_block: str = ""
    per_sub_questions_list: List[str] = [
        str(item).strip() for item in (slots.get("per_sub_questions") or []) if str(item).strip()
    ]
    sub_intent_scores_list: List[Dict[str, Any]] = slots.get("sub_intent_scores") or []
    if len(per_sub_questions_list) > 1:
        lines: List[str] = [
            "## 多子问题拆分约束（Pipeline 决策层分析结果）",
            "用户原问题已被拆解为以下相互独立的子问题，每个子问题应视为一个独立的目标：",
        ]
        for sub_index, sub_question_text in enumerate(per_sub_questions_list):
            scores_entry: Dict[str, Any] = (
                sub_intent_scores_list[sub_index]
                if sub_index < len(sub_intent_scores_list)
                and isinstance(sub_intent_scores_list[sub_index], dict)
                else {}
            )
            scores_items: List[Dict[str, Any]] = scores_entry.get("scores") or []
            top_intent_id: str = str(
                (scores_items[0].get("id") if scores_items else None)
                or (scores_items[0].get("name") if scores_items else None)
                or "general"
            )
            lines.append(
                f"  子问题 {sub_index + 1}: {sub_question_text}（意图倾向: {top_intent_id}）"
            )
        # ⚠️ 每个子问题 MUST 有自己的子任务——这里**刻意回退**了"可合并"约束。
        # 实测教训：合并后若该次工具调用未能同时覆盖全部子问题，就会有子问题无答案，
        # 而链路当时没有任何机制能发现这种遗漏（模型只会看到"检索成功了"）。
        # 取舍：覆盖可靠性 > 合并省下的 token（预期每次多约 1 次检索 + 1 次提炼）。
        lines.append(
            "规划要求：\n"
            "  1) 请为**每一个**子问题规划至少一个对应的子任务（tool / reasoning）；\n"
            "  2) 禁止以『可由同一次工具调用覆盖』为理由，把多个子问题合并成一个子任务；\n"
            "  3) 每个子任务须对应明确的子问题，最终回复必须覆盖全部子问题，不得遗漏。"
        )
        sub_constraint_block = "\n".join(lines) + "\n\n"

    # 技能清单：规划侧**只给清单**，不给"如何读取 / 如何遵守"的操作指引——那是
    # 执行期的事，同一份指引在 planner 与 executor 各出现一次等于让渐进式披露被
    # 承担两遍。
    #
    # 标题与清单**紧邻**：标题由本函数自己输出，无技能时不输出标题，避免出现
    # "## 可用高级技能" 底下直接跟另一个块标题的空标题现象（实测存在）。
    skills_index: str = str(state.get("skills_index") or "").strip()
    skills_block: str = f"## 可用技能\n{skills_index}\n\n" if skills_index else ""

    return (
        goal_block
        + anchor_block
        + hint_block
        + sub_constraint_block
        + skills_block
    )


async def plan_node(state: AgentGraphState, config: RunnableConfig) -> dict:
    """初始规划：plan() 内部已含 parse retry + fallback 单 reasoning 任务，不抛异常。"""
    deps = get_deps(config)
    trace_id: str = state["trace_id"]
    query: str = state["user_input"]
    memory_context: Dict[str, Any] = state.get("memory_context") or {
        "short_term": [], "long_term": []
    }

    # 预算账本重建（plan 提示词需要 snapshot_lines / 剩余额度；plan 不消耗调用次数）
    call_budget = budget_from_ledger(state.get("budget") or {}) if state.get("budget") else None

    planner = PlannerAgent(
        model_router=deps.model_router,
        purpose_hint="planner",
        tools=deps.tools,
        memory=None,
        max_replan_attempts=int(state.get("max_replan", 2)),
        call_budget=call_budget,
        enable_empty_result_replan=bool(deps.cfg("enable_empty_result_replan", True)),
        # 运行时工具白名单：用于提示词注入工具清单 + tool_name enum + 兜底映射。
        # 不传的话 planner 只能从预算段看到"其余 N 个工具额度充足"，会编造工具名。
        allowed_tool_names=list(state.get("active_tool_names") or []),
    )

    # 意图锚点的最低置信度门槛。
    # ⚠️ 实测（2026-09）：意图聚合给出的 aggregated_confidence 普遍偏高
    #（0.75 / 1.0 / 1.0，连"你好"都是 1.0），因此这道门槛**拦不住**那种"高置信度
    # 但方向错误"的意图标签。它的作用是挡掉真正低置信度的情形；更根本的方向修正
    # 需要另外的信号，不属本变更范围。
    intent_confidence_threshold: float = float(
        deps.cfg("agent_intent_confidence_threshold", 0.5)
    )
    skills_block = _build_planner_skills_block(state, intent_confidence_threshold)
    # 子问题数只认 slots["per_sub_questions"]：该键**仅在实际拆分时存在**
    #（与 prepare_node 计算工具配额、_build_planner_skills_block 判定是否注入
    # 约束块的口径完全一致）。未拆分时为 0 → 解析层不做覆盖校验。
    slots: Dict[str, Any] = (state.get("intent") or {}).get("slots") or {}
    sub_question_count: int = len(
        [item for item in (slots.get("per_sub_questions") or []) if str(item).strip()]
    )
    subtasks = await planner.plan(
        query, memory_context, skills_block=skills_block,
        sub_question_count=sub_question_count,
    )
    plan_dicts: List[Dict[str, Any]] = [asdict(task) for task in subtasks]

    logger.info("【规划层激活】初始计划生成 {} 个子任务: {}",
                len(plan_dicts), [t.get("id") for t in plan_dicts])
    trace_event(
        deps.tracer, trace_id, "plan.generated",
        {"subtask_ids": [t.get("id") for t in plan_dicts]},
    )

    return {
        "plan": plan_dicts,
        "cursor": 0,
        "last_error": None,
        "empty_data_signal": None,
    }
