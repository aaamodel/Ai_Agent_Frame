# -*- coding: utf-8 -*-
"""replan 节点：复用 PlannerAgent.replan()（保留 thinking），空数据换源语义不变。"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.agent.graph.deps import get_deps
from app.core.agent.graph.nodes._common import trace_event
from app.core.agent.graph.state import AgentGraphState, budget_from_ledger
from app.core.agent.planner import (
    PlannerAgent,
    SubTask,
    compact_attempt,
    result_is_ineffective,
)


def _proven_dead_tokens(prior_results: Any) -> List[str]:
    """收集"本轮已尝试且未取得有效数据"的位置 / 查询条件。"""
    tokens: List[str] = []
    for record in prior_results or []:
        if not isinstance(record, dict) or not result_is_ineffective(record):
            continue
        for chunk in re.split(r"[，,；;]", compact_attempt(record)):
            value = chunk.split("=", 1)[-1].strip().strip("[]'\"")
            # 太短的片段会误伤大量正常句子，这里只认有辨识度的
            if len(value) >= 4:
                tokens.append(value)
    return list(dict.fromkeys(tokens))


def strip_contradicting_suggestions(
    error_text: str, prior_results: Any
) -> Tuple[str, List[str]]:
    """移除失败原因中与"已试且未取得有效数据"矛盾的方向建议。

    失败原因（``insufficiency_signal``）是汇总模型**自由生成**的散文，它可能
    建议一个方向，而同一份重规划输入的取数记录已经证明该方向为空。

    实测（2026-09）：第二次重规划的 error 写着"建议检查 /data 目录下是否存在
    其他命名的销售数据文件"，而同一份 payload 的 `results_so_far` 已明确记录
    "在 /data 目录下未找到"。两者直接矛盾且无人拦截，于是又扫了一次 /data。

    Returns:
        (处理后的文本, 被移除的建议列表)
    """
    tokens = _proven_dead_tokens(prior_results)
    if not error_text or not tokens:
        return error_text, []

    kept: List[str] = []
    removed: List[str] = []
    for sentence in re.split(r"(?<=[。；\n])", error_text):
        if any(token in sentence for token in tokens):
            removed.append(sentence.strip())
            continue
        kept.append(sentence)
    if not removed:
        return error_text, []
    return "".join(kept).strip(), removed


async def replan_node(state: AgentGraphState, config: RunnableConfig) -> dict:
    """根据上一轮错误/空数据信号修订计划；replan() 内部含 fallback，不抛异常。"""
    deps = get_deps(config)
    trace_id: str = state["trace_id"]

    call_budget = budget_from_ledger(state.get("budget") or {}) if state.get("budget") else None
    planner = PlannerAgent(
        model_router=deps.model_router,
        purpose_hint="planner",
        tools=deps.tools,
        memory=None,
        max_replan_attempts=int(state.get("max_replan", 2)),
        call_budget=call_budget,
        enable_empty_result_replan=bool(deps.cfg("enable_empty_result_replan", True)),
        # 与 plan_node 同口径：重规划同样要知道真实白名单，否则兜底映射会退回
        # 注册中心全量工具，白名单约束在 replan 环节失效。
        allowed_tool_names=list(state.get("active_tool_names") or []),
    )

    old_plan: List[SubTask] = [SubTask(**item) for item in (state.get("plan") or [])]
    prior_results: List[Dict[str, Any]] = state.get("subtask_results") or []

    # 信号优先级：L3 summarize 缺口判定 > L2 全坏规则信号 > 旧步级留痕
    base_error: Optional[str] = (
        state.get("insufficiency_signal")
        or state.get("empty_data_signal")
        or state.get("last_error")
        or None
    )
    # 去重提示：已失败/无效的工具不允许新计划重复调用。
    # ⚠️ 判据是"是否取得有效数据"，不是 status——实测知识图谱检索未返回数据
    # 但 status 记为 ok，按旧判据会逃过去重。
    failed_tool_names: List[str] = sorted({
        str(r.get("tool_name")) for r in prior_results
        if r.get("tool_name") and result_is_ineffective(r)
    })
    # 矛盾拦截：失败原因里若建议了一个"本轮已证明为空"的方向，必须移除——
    # 否则重规划会照着它再撞一次同一面墙。
    error_text: Optional[str] = None
    removed_suggestions: List[str] = []
    if base_error:
        error_text, removed_suggestions = strip_contradicting_suggestions(
            base_error, prior_results
        )
    if removed_suggestions:
        trace_event(
            deps.tracer, trace_id, "replan.contradiction_stripped",
            {"removed_suggestions": removed_suggestions,
             "reason": "建议方向已被本轮记录证明未取得有效数据"},
        )

    if failed_tool_names:
        dedup_block: str = (
            f"\n【已失效工具去重】以下工具本轮已被证明无效，新计划严禁重复调用："
            f"{failed_tool_names}。"
        )
        error_text = f"{error_text or '上一轮计划取数失败。'}{dedup_block}"

    new_subtasks = await planner.replan(
        old_plan,
        prior_results,
        error_text,
        # 把运行期提取到的结构化事实交给重规划：它要换数据源时不必再猜目录。
        available_assets=list(state.get("extracted_facts") or []),
    )
    new_plan_dicts: List[Dict[str, Any]] = [asdict(task) for task in (new_subtasks or [])]
    replan_attempts: int = int(state.get("replan_attempts", 0)) + 1

    trace_event(
        deps.tracer, trace_id, "orchestrator.replan",
        {"attempt": replan_attempts, "reason": error_text,
         "failed_tools": failed_tool_names,
         "subtask_ids": [t.get("id") for t in new_plan_dicts]},
    )

    if not new_plan_dicts:
        # 与旧 run_with_plan 一致：重规划返回空 → 失败收尾
        logger.error("Planner.replan() 返回空计划，进入失败收尾。")
        return {
            "plan": [],
            "cursor": 0,
            "replan_attempts": replan_attempts,
            "degraded": True,
            "last_error": "重规划返回空计划",
            "empty_data_signal": None,
            "insufficiency_signal": None,
            # 跳过决策是针对**旧计划**做出的，与 cursor=0 同样必须重置：
            # 若被新计划继承，replan 要补取的那一步可能恰好已被跳过 → 死锁。
            "skipped_task_ids": [],
            "early_finish": False,
        }

    logger.info(
        "【重规划】第 {} 次 replan 产出 {} 个子任务（原因: {}）",
        replan_attempts, len(new_plan_dicts), (error_text or "")[:200],
    )
    return {
        "plan": new_plan_dicts,
        "cursor": 0,
        "replan_attempts": replan_attempts,
        "degraded": True,  # 新语义：发生过重规划即标记降级（API 字段保留）
        "last_error": None,
        "empty_data_signal": None,
        "insufficiency_signal": None,  # 已被本次 replan 消费
        "draft_answer": None,
        # 重置跳过记录（同 cursor=0）：旧计划的"跳过/提前收尾"结论对新计划无效，
        # 否则 replan 补取证据时，恰恰需要执行的步骤可能已被标记为跳过。
        "skipped_task_ids": [],
        "early_finish": False,
    }
