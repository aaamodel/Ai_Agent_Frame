# -*- coding: utf-8 -*-
"""重规划门控：**单一真源**。

为什么必须抽出来（2026-09 实测事故）：

    收窄重规划时，判定被写成了两份——`summarize_node` 用 `replan_capacity(state)`
    决定"我要写下不足信号、等着被重规划"，而路由 `route_after_summarize` 用的是更
    严的收窄条件。两者一旦不一致就会出现**空答案**：

        summarize: 还有余量 → 写 signal，final_answer="" 、success=False，等路由重规划
        router   : 缺口性质不是方向性错误 → 拒绝重规划 → 直接 persist
        → 用户拿到 answer="" 且 success=False

    实测：plan_execute 模式下跑「我们优先做哪些行业」，2 个子任务、工具全失败，
    最终 answer 为空。这违反 `agent/replan-context` 的要求——"被排除时，系统 MUST
    直接以已有信息给出诚实的部分答案"。

    因此把判定收拢到本模块，**产出方与路由方共用同一个函数**。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.agent.planner import result_is_ineffective
from app.core.agent.graph.state import replan_capacity

#: 同一轮请求内重规划的**无条件**硬上限：配置只能调得更小，不能放宽。
REPLAN_HARD_LIMIT: int = 1

#: 缺口性质（结构化）。只有 off_topic 属于"方向性错误"，才允许触发重规划。
INSUFFICIENCY_KIND_OFF_TOPIC: str = "off_topic"
INSUFFICIENCY_KIND_NO_DATA: str = "no_data"


def normalize_gap_kind(raw: Any) -> str:
    """把模型给出的缺口性质归一化成字面值。

    缺失 / 空值 / 取值不在允许集合内，一律按 **no_data（非方向性错误）** 处理：
    宁可少一次重规划，也不要让步级问题付出整轮重规划的代价。
    """
    value = str(raw or "").strip().lower()
    if value == INSUFFICIENCY_KIND_OFF_TOPIC:
        return INSUFFICIENCY_KIND_OFF_TOPIC
    return INSUFFICIENCY_KIND_NO_DATA


def all_candidates_exhausted(state: Dict[str, Any]) -> bool:
    """本轮所有候选工具是否都已被证明未取得有效数据。

    成立时重规划只能重复已失效的路径，因此必须排除（转诚实收尾）。
    """
    results = state.get("subtask_results") or []
    failed: set = {
        str(record.get("tool_name"))
        for record in results
        if isinstance(record, dict)
        and record.get("tool_name")
        and result_is_ineffective(record)
    }
    allowed: set = {str(name) for name in (state.get("active_tool_names") or [])}
    if not allowed:
        return False
    return allowed.issubset(failed)


def replan_allowed(
    state: Dict[str, Any],
    *,
    signal_present: Optional[bool] = None,
    insufficiency_kind: Optional[str] = None,
) -> bool:
    """收窄后的重规划触发判定（**产出方与路由方共用**）。

    Args:
        state: 图状态。
        signal_present: 是否已判定证据不足。默认（``None``）**从 state 读**——
            路由侧正是这个用法；summarize_node 在**写入信号之前**调用，因此由它
            显式传 ``True``。
        insufficiency_kind: 缺口性质。默认从 state 读。

    必须**同时**满足：① 有证据不足信号且仍有余量；② 缺口性质是方向性错误
    （`off_topic`）；③ 仍有未失效的候选工具；④ 未达硬上限。
    """
    if signal_present is None:
        signal_present = bool(state.get("insufficiency_signal"))
    if not signal_present:
        return False
    if not replan_capacity(state):
        return False

    kind: str = normalize_gap_kind(
        insufficiency_kind if insufficiency_kind is not None
        else state.get("insufficiency_kind")
    )
    if kind != INSUFFICIENCY_KIND_OFF_TOPIC:
        return False

    if all_candidates_exhausted(state):
        return False

    if int(state.get("replan_attempts", 0) or 0) >= REPLAN_HARD_LIMIT:
        return False

    return True


def replan_exclusion_reason(
    state: Dict[str, Any],
    *,
    signal_present: Optional[bool] = None,
    insufficiency_kind: Optional[str] = None,
) -> str:
    """重规划被排除的原因（供留痕；允许重规划时返回空串）。"""
    if signal_present is None:
        signal_present = bool(state.get("insufficiency_signal"))
    if not signal_present:
        return "无证据不足信号"
    if not replan_capacity(state):
        return "次数或预算余量已耗尽"
    kind = normalize_gap_kind(
        insufficiency_kind if insufficiency_kind is not None
        else state.get("insufficiency_kind")
    )
    if kind != INSUFFICIENCY_KIND_OFF_TOPIC:
        return f"缺口性质非方向性错误（{kind}），步级问题不付整轮重规划代价"
    if all_candidates_exhausted(state):
        return "本轮所有候选工具均未取得有效数据，重规划只会重复已失效路径"
    if int(state.get("replan_attempts", 0) or 0) >= REPLAN_HARD_LIMIT:
        return "已达重规划硬上限"
    return ""
