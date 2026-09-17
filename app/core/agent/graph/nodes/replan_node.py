# -*- coding: utf-8 -*-
"""replan 节点：复用 PlannerAgent.replan()（保留 thinking），空数据换源语义不变。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.agent.graph.deps import get_deps
from app.core.agent.graph.nodes._common import trace_event
from app.core.agent.graph.state import AgentGraphState, budget_from_ledger
from app.core.agent.planner import PlannerAgent, SubTask


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
    # 去重提示：已失败/无效的工具不允许新计划重复调用
    failed_tool_names: List[str] = sorted({
        str(r.get("tool_name")) for r in prior_results
        if r.get("tool_name") and r.get("status") in {
            "empty_data", "error", "budget_denied", "approval_denied"
        }
    })
    error_text: Optional[str] = base_error
    if failed_tool_names:
        dedup_block: str = (
            f"\n【已失效工具去重】以下工具本轮已被证明无效，新计划严禁重复调用："
            f"{failed_tool_names}。"
        )
        error_text = f"{base_error or '上一轮计划取数失败。'}{dedup_block}"

    new_subtasks = await planner.replan(old_plan, prior_results, error_text)
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
