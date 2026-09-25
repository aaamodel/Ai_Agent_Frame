# -*- coding: utf-8 -*-
"""execute 节点：图中唯一的数据面执行点，**单步激活**。

- plan 形态：一次推进 plan[cursor] 一个子任务（tool：FC 强制取参→审批闸门→
  统一工具管线→子任务 LLM 提炼；reasoning：直接 LLM 提炼），cursor+1。
  异常 → last_error；空业务数据 → empty_data_signal，交条件边 replan。
- react 形态（无 plan 自环）：一次一轮 LLM 决策（FC 优先，通道不可用当轮降级
  文本协议），产出一个/多个 ToolCall 或 Final Answer。

预算：每次激活从 state 账本重建临时 ToolCallBudget，结束写回；
工具调用统一走 2.1 的 execute_tool_call；工具本身永不建节点。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.agent.graph.approval import (
    danger_tool_names,
    gate_tool_approval,
    take_resume_approval_arguments,
)
from app.core.agent.graph.deps import get_deps
from app.core.agent.graph.nodes._common import (
    agent_goal_from_state,
    build_extra_system,
    compact_history_lines,
    compact_tool_observations,
    emit_plan_step,
    emit_step,
    long_term_mem_block,
    next_pending_cursor,
    render_plan_ledger,
    short_term_messages,
    trace_event,
    write_back_budget,
)
from app.core.agent.graph.nodes._fc_args import (
    resolve_tool_args_from_hint,
    resolve_tool_args_via_function_call,
    substitute_task_refs,
)
from app.core.agent.step_correction import (
    Candidate,
    EVIDENCE_COVERAGE_THRESHOLD,
    build_candidates,
    build_question_keywords,
    detect_evidence_gap,
    injection_reason,
    remaining_quota,
    render_candidates,
    translate_candidate,
)
from app.core.skill.asset_map import (
    extract_asset_facts,
    merge_facts,
    render_facts_for_prompt,
)
from app.core.agent.graph.state import (
    AgentGraphState,
    budget_from_ledger,
    budget_to_ledger,
    replan_capacity,
)
from app.core.agent.planner import EMPTY_DATASOURCE_REPLAN_PREFIX, _is_empty_data
from app.core.agent.react_agent import (
    GRACEFUL_TIMEOUT_MESSAGE,
    REACT_FC_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT,
    _FCFallbackRequired,
    _is_candidate_exhaustion,
    _mini_json_repair,
    _parse_react_step,
    build_react_user_prompt,
)
from app.query_intent.llm_schemas import (
    SubTaskOutcomeSchema,
    pydantic_to_openai_response_format,
)
from app.core.agent.toolcall import (
    ToolCall,
    ToolResult,
    execute_tool_call,
    observation_has_error_marker,
    tool_call_from_text,
    tool_calls_from_fc,
)
from app.core.agent.evidence import (
    FETCH_TOOL_NAME,
    fetch_tool_definition,
    handle_fetch_evidence,
    ingest_observation,
    is_fetch_call,
    render_fc_messages,
    render_plan_observation,
    render_text_history_lines,
)
from app.core.agent.evidence.fetch import (
    EVIDENCE_RESTORE_TOOL,
    build_plan_restore_text,
    parse_fetch_arguments,
)
from app.core.agent.evidence.models import KIND_CONTENT, KIND_TABLE
from app.core.agent.evidence.view import plan_view_uids
from app.core.agent.evidence.pipeline import extract_query_text
from app.core.trace_to_markdown import trace_to_markdown

# 控制协议（SubTaskOutcomeSchema）的提示词版输出契约。
# ⚠️ 与 planner 同理：GLM-4.7 实测静默忽略 response_format=json_schema，
# 不写死字段名模型就自由发挥（2026-09-23 事故：输出散文 → 控制协议不可解析）。
PLAN_CONTROL_OUTPUT_CONTRACT = """
【输出格式·强制契约】
整个回复只能是一个 JSON 对象，不要输出 JSON 之外的任何文字、Markdown 围栏或思考过程。
字段固定为：
- "conclusion"：字符串，本子任务的最终结论或分析结果；
- "solved"：只能是 "yes"（已拿到所需答案）、"partial"（只拿到部分）或 "no"；
- "next_action"：只能是 "continue"（继续后续子任务）或 "finish"（结论已足以收尾）；
- "skip_task_ids"：要跳过的子任务 id 字符串数组，没有则为 null；
- "selected_alternative_id"：需要换数据源/换工具时，从给定「可选的替代方向」原样照抄的标识，没有则为 null；
- "requested_evidence_uids"：本步某条被省略的工具结果需要查看全文时，填其证据编号字符串数组（最多 3 条），没有则为 null；
- "reason"：字符串，调度判断的简短理由。
示例：{"conclusion":"2026年7月业绩第一的是张三，销售额120万","solved":"yes","next_action":"finish","skip_task_ids":null,"selected_alternative_id":null,"requested_evidence_uids":null,"reason":"排名结果已取得，剩余导出步骤仍需执行时置 continue"}
严禁自造 conclusion/solved/next_action 之外的顶层结构，严禁输出散文。
"""


# ── 证据板辅助（enable_evidence_board 默认关；任何异常回退现状压缩）────────
def _evidence_rounds(meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list((meta or {}).get("rounds") or [])


def _evidence_ingest(
    deps: Any,
    trace_id: str,
    *,
    existing_units: List[Dict[str, Any]],
    meta: Dict[str, Any],
    user_question: str,
    tool_name: str,
    round_idx: int,
    observation: str,
    current_query: str,
    action_input: Any,
    call_id: Optional[str],
) -> tuple:
    """入管一次工具观测并留痕 evidence.round；失败返回 (None, None) 静默降级。"""
    try:
        all_units, new_meta, report = ingest_observation(
            existing_units=existing_units,
            meta=meta,
            tool_name=tool_name,
            round_idx=round_idx,
            observation=observation or "",
            current_query=current_query or "",
            user_question=user_question or "",
            action_input=action_input if isinstance(action_input, dict) else None,
            call_id=call_id,
        )
        trace_event(deps.tracer, trace_id, "evidence.round", report.to_dict())
        return all_units, new_meta
    except Exception as exc:  # noqa: BLE001 - 证据板是增值压缩，绝不允许打断主链路
        logger.warning("证据板入管失败，本轮回退现状压缩: {}", exc)
        trace_event(
            deps.tracer, trace_id, "evidence.fallback",
            {"stage": "ingest", "error": str(exc)[:200]},
        )
        return None, None


def _evidence_fetch(
    deps: Any,
    trace_id: str,
    *,
    state: Dict[str, Any],
    arguments: Any,
    next_round: int,
) -> tuple:
    """拦截 fetch_evidence：纯本地回取，零注册表调用。返回 (obs_text, units|None)。"""
    uid, window, parse_error = parse_fetch_arguments(arguments)
    try:
        if parse_error is not None:
            trace_event(
                deps.tracer, trace_id, "evidence.fetch",
                {"uid": "", "window": 0, "chars": len(parse_error), "hit": False},
            )
            return parse_error, None
        obs_text, units = handle_fetch_evidence(
            state, uid=uid, window=window, round_idx=next_round
        )
        trace_event(
            deps.tracer, trace_id, "evidence.fetch",
            {"uid": uid, "window": window, "chars": len(obs_text), "hit": True},
        )
        return obs_text, units
    except Exception as exc:  # noqa: BLE001
        logger.warning("证据板回取失败，返回确定性错误文本: {}", exc)
        trace_event(
            deps.tracer, trace_id, "evidence.fallback",
            {"stage": "fetch", "error": str(exc)[:200]},
        )
        return f"fetch_evidence 执行失败：{exc}", None


def _evidence_fc_view(
    deps: Any,
    trace_id: str,
    *,
    messages: List[Dict[str, Any]],
    units: List[Dict[str, Any]],
    meta: Dict[str, Any],
    step_idx: int,
) -> List[Dict[str, Any]]:
    try:
        return render_fc_messages(
            messages, units, _evidence_rounds(meta), round_idx=step_idx
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("证据板 FC 视图渲染失败，回退 compact_tool_observations: {}", exc)
        trace_event(
            deps.tracer, trace_id, "evidence.fallback",
            {"stage": "fc_view", "error": str(exc)[:200]},
        )
        return compact_tool_observations(messages)


def _evidence_text_view(
    deps: Any,
    trace_id: str,
    *,
    history_lines: List[str],
    units: List[Dict[str, Any]],
    meta: Dict[str, Any],
    step_idx: int,
) -> List[str]:
    try:
        return render_text_history_lines(
            history_lines, units, _evidence_rounds(meta), round_idx=step_idx
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("证据板文本视图渲染失败，回退 compact_history_lines: {}", exc)
        trace_event(
            deps.tracer, trace_id, "evidence.fallback",
            {"stage": "text_view", "error": str(exc)[:200]},
        )
        return compact_history_lines(history_lines)


async def execute_node(state: AgentGraphState, config: RunnableConfig) -> dict:
    """按 state 形态分流：有 plan 走子任务单步；无 plan 走 react 单轮。"""
    if state.get("plan"):
        return await _execute_plan_step(state, config)
    return await _execute_react_step(state, config)


# ===========================================================================
# plan 形态
# ===========================================================================
# 步级"坏结果"状态：只记账、不中断计划；计划末步由 L2 闸门整体评估
PLAN_BAD_STATUSES = frozenset({"empty_data", "error", "budget_denied", "approval_denied"})
_BAD_STATUS_ZH = {
    "empty_data": "空数据（数据源无相关记录）",
    "error": "执行错误",
    "budget_denied": "工具预算熔断",
    "approval_denied": "人工审批拒绝",
}


def _parse_subtask_outcome(raw: str) -> Optional[Dict[str, Any]]:
    """解析子任务的结构化产出（结论 + 控制指令）。不可解析返回 None。

    ⚠️ 返回 None 时调用方 MUST 降级为「继续执行下一子任务」，并且 MUST NOT
    丢弃已经取回的结论——控制指令是**增值能力，不是主链路依赖**。
    """
    if not raw or not raw.strip():
        return None
    try:
        data: Any = json.loads(_mini_json_repair(raw.strip()))
    except Exception:  # noqa: BLE001 - 三档降级链最差档：原文直接当结论
        return None
    if not isinstance(data, dict):
        return None
    try:
        return SubTaskOutcomeSchema(**data).model_dump()
    except Exception:  # noqa: BLE001 - 字段缺失/取值非法同样按"未解析"处理
        return None


def _apply_step_correction(
    *,
    state: AgentGraphState,
    plan: List[Dict[str, Any]],
    cursor: int,
    skipped_ids: List[str],
    candidates: Sequence[Candidate],
    selected: Any,
    deps: Any,
    trace_id: str,
    subtask_id: Any,
    trigger: Optional[str],
) -> Dict[str, Any]:
    """应用一次就地纠偏；未应用时返回空 dict。

    分工：模型只输出**候选项标识**，工具名 / 参数 / 路径全部在这里构造，并做一次
    白名单复核。标识不在本次候选列表中、翻译失败、或配额耗尽时，一律忽略该指令并
    按既有口径降级为「继续执行下一个子任务」——**本步已产出的结论绝不因此丢弃**。

    作用域闸门：只在 cursor 之后**插入**一条新的纠偏子任务；当前步、已有计划项
    （含其 id）、已跳过记录、提前收尾标记一律不动（对比重规划的"整体替换计划 +
    cursor 归零 + 清空跳过记录"）。
    """
    if not selected or not candidates:
        return {}

    record: Dict[str, Any] = {
        "subtask_id": subtask_id,
        "trigger": trigger or "",
        "selected": str(selected).strip(),
        "remaining_quota": remaining_quota(state),
    }

    def _skip(reason: str) -> Dict[str, Any]:
        record["applied"] = False
        record["reason"] = reason
        # 留痕：被忽略的纠偏指令同样要可观测，便于事后判断是"模型乱选"还是"闸门过紧"
        trace_event(deps.tracer, trace_id, "plan_execute.correction_skipped", record)
        return {}

    if remaining_quota(state) <= 0:
        return _skip("纠偏配额已耗尽")

    # 链式纠偏闸门：纠偏任务（id 形如 ``correction_xxx``）执行后即使仍未解决，
    # 也**不再插入第二层纠偏**——否则会出现 correction_correction_xxx 连锁插步，
    # 且空 hint 的工具候选只能靠 FC 自由组参，层数越深模型越靠臆造参数硬凑
    # （2026-09 实测：二次纠偏给 file_read_tool 编出不存在的"线索管理规范.md"）。
    # 本步结论照常保留，交计划级闸门/最终汇总裁决。
    if str(subtask_id or "").startswith("correction_"):
        return _skip("纠偏任务不再触发二次纠偏（禁止链式插步）")

    action = translate_candidate(
        selected,
        candidates=candidates,
        allowed_tools=state.get("active_tool_names") or [],
        facts=state.get("extracted_facts"),
    )
    if action is None:
        return _skip("标识不在本次候选列表中，或未通过白名单复核")

    # 在 ``cursor + 1`` 处**插入**一条新的 tool 子任务（不再替换下一待执行项）：
    # 本节点收尾时 cursor 恰好推进到 cursor+1 → 插入的纠偏步下一步立即执行，
    # 原下一任务顺延一位、内容不动。cursor 在计划末尾时即追加，因此不再需要
    # "cursor 之后存在待执行子任务"这一前置条件。
    tool_name: Any = action.get("tool_name")
    existing_ids = {str(entry.get("id")) for entry in plan}
    base_id: str = f"correction_{subtask_id}"
    new_id: str = base_id
    suffix: int = 2
    while new_id in existing_ids:
        new_id = f"{base_id}_{suffix}"
        suffix += 1

    new_task: Dict[str, Any] = {
        "id": new_id,
        "title": str(action.get("title") or f"调用 {tool_name}"),
        "action_type": "tool",
        "tool_name": tool_name,
        "tool_args_hint": dict(action.get("action_input") or {}),
        "description": (
            f"执行期就地纠偏：改用 {tool_name}"
            + (f"（触发：{trigger}）" if trigger else "")
        ),
    }

    new_plan: List[Dict[str, Any]] = [dict(entry) for entry in plan]
    new_plan.insert(cursor + 1, new_task)

    record.update({
        "applied": True,
        "inserted": True,
        "insert_at": cursor + 1,
        "target_subtask_id": new_id,
        "target_tool": action.get("tool_name"),
        "target_args": action.get("action_input"),
    })
    trace_event(deps.tracer, trace_id, "plan_execute.correction_applied", record)

    return {
        "plan": new_plan,
        # 留痕与配额计数共用同一条记录
        "step_corrections": list(state.get("step_corrections") or []) + [record],
    }


def _restore_uids_from_hint(task: Dict[str, Any]) -> List[str]:
    """从内部恢复步的 tool_args_hint 取出 uids（插入时写入，零 LLM）。"""
    hint = task.get("tool_args_hint")
    data: Any = hint
    if isinstance(hint, str):
        try:
            data = json.loads(hint)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(data, dict):
        return []
    raw = data.get("uids") or []
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _apply_evidence_restore(
    *,
    state: Dict[str, Any],
    plan: List[Dict[str, Any]],
    cursor: int,
    subtask_id: Any,
    requested: Any,
    unit_dicts: List[Dict[str, Any]],
    deps: Any,
    trace_id: str,
) -> Dict[str, Any]:
    """应用一次 plan 延迟证据回取（D10）；不满足条件时返回空 dict。

    与 _apply_step_correction 同构：只在 cursor+1 插入一个内部恢复步，
    受同一配额与链式闸门约束；恢复步不注册工具、分发前拦截、零外部调用。
    """
    sid = str(subtask_id or "")
    record: Dict[str, Any] = {
        "subtask_id": sid,
        "trigger": "evidence_restore",
        "remaining_quota": remaining_quota(state),
    }

    def _skip(reason: str, rejected: Optional[List[str]] = None) -> Dict[str, Any]:
        record.update({
            "applied": False,
            "reason": reason,
            "requested": [str(x) for x in (requested or [])][:3],
            "accepted": [],
            "rejected": rejected or [],
            "chars": 0,
        })
        trace_event(deps.tracer, trace_id, "evidence.restore", record)
        return {}

    # 链式闸门：纠偏/恢复步自身不再受理证据请求（每子任务最多恢复一次）
    if sid.startswith("correction_"):
        return _skip("恢复/纠偏步不再触发证据回取（每子任务最多一次）")
    if remaining_quota(state) <= 0:
        return _skip("纠偏配额已耗尽")

    raw_uids = requested if isinstance(requested, list) else []
    candidates: List[str] = []
    for item in raw_uids:
        uid = str(item or "").strip()
        if uid and uid not in candidates:
            candidates.append(uid)
    candidates = candidates[:3]
    if not candidates:
        return {}

    unit_dicts = list(unit_dicts or [])
    units_by_uid = {
        str(d.get("uid")): d for d in unit_dicts if d.get("uid")
    }
    _, omitted_uids = plan_view_uids(unit_dicts, round_idx=cursor)

    accepted: List[str] = []
    rejected: List[str] = []
    for uid in candidates:
        unit = units_by_uid.get(uid)
        unit_round = unit.get("round_idx") if unit is not None else None
        if (
            unit is not None
            and unit_round is not None
            and int(unit_round) == cursor
            and str(unit.get("kind") or "") in (KIND_CONTENT, KIND_TABLE)
            and not unit.get("dupe_of")
            and uid in omitted_uids
        ):
            accepted.append(uid)
        else:
            rejected.append(uid)

    if not accepted:
        return _skip("没有属于本步已省略集合的合法编号", rejected=rejected)

    existing_ids = {str(entry.get("id")) for entry in plan}
    base_id = f"correction_evidence_{sid}"
    new_id = base_id
    suffix = 2
    while new_id in existing_ids:
        new_id = f"{base_id}_{suffix}"
        suffix += 1

    new_task: Dict[str, Any] = {
        "id": new_id,
        "title": "回填本步被省略证据原文（内部步，零外部调用）",
        "action_type": "tool",
        "tool_name": EVIDENCE_RESTORE_TOOL,
        "tool_args_hint": json.dumps({"uids": accepted}, ensure_ascii=False),
        "description": (
            "内部恢复步：从本请求已保存观测回填被省略证据原文，"
            "不调用任何外部工具。"
        ),
    }
    new_plan: List[Dict[str, Any]] = [dict(entry) for entry in plan]
    new_plan.insert(cursor + 1, new_task)

    restore_chars = sum(
        len(str(units_by_uid[uid].get("text") or "")) for uid in accepted
    )
    record.update({
        "applied": True,
        "inserted": True,
        "insert_at": cursor + 1,
        "target_subtask_id": new_id,
        "requested": candidates,
        "accepted": accepted,
        "rejected": rejected,
        "chars": restore_chars,
    })
    trace_event(deps.tracer, trace_id, "evidence.restore", record)

    return {
        "plan": new_plan,
        "step_corrections": list(state.get("step_corrections") or []) + [record],
    }


def _apply_subtask_control(
    rec: Dict[str, Any],
    outcome: Optional[Dict[str, Any]],
    plan: List[Dict[str, Any]],
    cursor: int,
    *,
    control_enabled: bool,
    skipped_ids: List[str],
) -> Dict[str, Any]:
    """把控制协议的调度决策折算成 state 更新（跳过列表 / 提前收尾）。

    只允许「跳过」与「提前收尾」两种收缩操作——**不提供新增或修改子任务**，
    这是本变更的范围硬约束，模型即便声明了也只会被忽略并留痕。
    """
    if not control_enabled or not outcome:
        return {}

    valid_ids = {str(entry.get("id")) for entry in plan}
    declared: List[str] = [
        str(item).strip() for item in (outcome.get("skip_task_ids") or []) if str(item).strip()
    ]
    # 只接受计划里真实存在的 id；模型编造的 id 直接忽略（并留痕）
    new_skips: List[str] = [item for item in declared if item in valid_ids]
    ignored: List[str] = [item for item in declared if item not in valid_ids]

    update: Dict[str, Any] = {}
    if str(outcome.get("next_action") or "") == "finish":
        # finish 归一为"跳过剩余全部"：不新增路由分支，游标自然走完
        remaining: List[str] = [str(entry.get("id")) for entry in plan[cursor + 1:]]
        new_skips = list(dict.fromkeys(new_skips + remaining))
        update["early_finish"] = True
    if new_skips:
        update["skipped_task_ids"] = list(dict.fromkeys(skipped_ids + new_skips))

    rec["control_reason"] = str(outcome.get("reason") or "")[:200]
    if ignored:
        rec["control_ignored_ids"] = ignored
    return update


def _plan_level_evidence_gate(
    *,
    state: AgentGraphState,
    plan: List[Dict[str, Any]],
    cursor_after: int,
    all_results: List[Dict[str, Any]],
    budget: Any,
    gate_enabled: bool,
) -> Dict[str, Any]:
    """L2 规则闸门（0 LLM）：本轮计划全部跑完且**所有**工具子任务都坏 → 证据不足。

    - 只统计当前 plan 轮次（``plan_attempt == replan_attempts``）的结果，
      上一轮的成功步骤不参与；
    - 至少一个工具步 ok → 交给 summarize L3 自判（含跑题识别）；
    - 本轮无工具子任务（纯推理 replan）→ 不触发，直接 summarize；
    - 重规划次数/预算余量不满足时不发信号（留给 summarize 走草稿/GRACEFUL）。
    """
    if not gate_enabled or cursor_after < len(plan):
        return {}

    attempt: int = int(state.get("replan_attempts", 0) or 0)
    current_results: List[Dict[str, Any]] = [
        r for r in all_results if int(r.get("plan_attempt", 0) or 0) == attempt
    ]
    tool_steps: List[Dict[str, Any]] = [
        r for r in current_results if r.get("action_type") == "tool"
    ]
    if not tool_steps:
        return {}
    bad_steps: List[Dict[str, Any]] = [
        r for r in tool_steps if r.get("status") in PLAN_BAD_STATUSES
    ]
    if len(bad_steps) < len(tool_steps):
        return {}

    # 用写回后的最新账本判定余量
    capacity_view: Dict[str, Any] = {**state, "budget": budget_to_ledger(budget)}
    if not replan_capacity(capacity_view):
        logger.info(
            "L2 闸门：全部 {} 个工具子任务均失败，但已无 replan 余量，转 summarize。",
            len(tool_steps),
        )
        return {}

    failed_lines: List[str] = []
    seen_tools: set = set()
    for r in bad_steps:
        tool_name = r.get("tool_name") or r.get("subtask_id") or "unknown"
        seen_tools.add(tool_name)
        detail = r.get("error") or r.get("replan_reason") or str(r.get("observation") or "")[:200]
        failed_lines.append(
            f"- 工具 [{tool_name}]：{_BAD_STATUS_ZH.get(str(r.get('status')), r.get('status'))}"
            f"（{str(detail)[:160]}）"
        )
    signal: str = (
        f"{EMPTY_DATASOURCE_REPLAN_PREFIX}: 本轮计划的全部 {len(tool_steps)} 个取数子任务均未获得"
        "可用业务数据：\n" + "\n".join(failed_lines) + "\n"
        f"以下工具已被证明无效，新计划严禁重复调用：{sorted(seen_tools)}；"
        "请改用其他可用数据源/工具重新获取；若确无其他数据源，则生成纯推理子任务基于已有信息作答，"
        "严禁编造占位数据。"
    )
    logger.warning(
        "L2 闸门：本轮工具子任务全部失败（{}），触发计划级换源重规划。", sorted(seen_tools)
    )
    return {"insufficiency_signal": signal}


async def _execute_plan_step(state: AgentGraphState, config: RunnableConfig) -> dict:
    deps = get_deps(config)
    trace_id: str = state["trace_id"]
    query: str = state["user_input"]
    plan: List[Dict[str, Any]] = state["plan"]
    cursor: int = int(state.get("cursor", 0))
    prior_results: List[Dict[str, Any]] = state.get("subtask_results") or []

    update: Dict[str, Any] = {}
    budget = budget_from_ledger(state.get("budget") or {})
    evidence_on: bool = bool(deps.cfg("enable_evidence_board", False))
    # 本步入管结果（整体替换写回 state；None 表示开关关或入管降级）
    evidence_units: Optional[List[Dict[str, Any]]] = None
    evidence_meta: Optional[Dict[str, Any]] = None
    # ── 游标快进：跳过被模型决定跳过的子任务 ──────────────────────────────
    # 与 builder.route_after_execute **共用** next_pending_cursor，避免两处各写一份
    # "怎么算下一个待执行" 的逻辑（历史上工具白名单就因三处同口径而漂移过）。
    skipped_ids: List[str] = list(state.get("skipped_task_ids") or [])
    pending: Optional[int] = next_pending_cursor(plan, cursor, skipped_ids)
    if pending is None:
        # 剩余全部被跳过（即提前收尾）→ 不再执行，交给路由进汇总
        update["cursor"] = len(plan)
        update.update(write_back_budget(budget))
        return update
    cursor = pending

    # 防御：空计划（plan() 的 fallback 保证不会发生，双保险）
    if not plan or cursor >= len(plan):
        update["last_error"] = "planner 返回空计划或游标越界"
        update.update(write_back_budget(budget))
        return update

    task: Dict[str, Any] = plan[cursor]
    rec: Dict[str, Any] = {
        "subtask_id": task.get("id"),
        "title": task.get("title"),
        "action_type": task.get("action_type"),
        "plan_attempt": int(state.get("replan_attempts", 0) or 0),
    }

    # 坏结果（空数据/错误/熔断/审批拒绝）只记账、跳过本步提炼，但 cursor 照常推进
    skip_distill: bool = False
    obs_str: Optional[str] = None

    # 前序子任务精炼上下文
    simplified_context = [
        {
            "subtask_id": r.get("subtask_id"),
            "title": r.get("title"),
            "conclusion": r.get("llm_output", r.get("observation", "")),
        }
        for r in prior_results
    ]
    ctx_str = json.dumps(simplified_context, ensure_ascii=False, indent=2)[:8000]

    # 前序子任务**实际使用的入参**，用于解析 tool_args_hint 里的
    # "<从 task_N 获取的路径>" 占位符（planner 计划时看不到运行结果，
    # 只能写这种依赖声明；不替换就会当成参数值原样传给工具）。
    artifacts_by_task: Dict[str, Any] = {
        str(item.get("subtask_id")): item.get("action_input")
        for item in prior_results
        if item.get("subtask_id") and isinstance(item.get("action_input"), dict)
    }

    try:
        if task.get("action_type") == "tool":
            tool_name: Optional[str] = task.get("tool_name")
            rec["tool_name"] = tool_name
            if not tool_name:
                raise ValueError(
                    f"子任务 [{task.get('id')}] 声明为 tool 类型，但未指定 tool_name"
                )

            # 内部证据恢复步（D10）：分发前拦截，不耗预算、不过注册表、
            # 不重新入管，回填文本即本步观测，随后正常走一次提炼。
            if tool_name == EVIDENCE_RESTORE_TOOL:
                restore_uids = _restore_uids_from_hint(task)
                obs_str, restore_hits = build_plan_restore_text(
                    state, restore_uids
                )
                rec["kind"] = "plan_tool_call"
                rec["action"] = EVIDENCE_RESTORE_TOOL
                rec["action_input"] = {"uids": restore_uids}
                rec["observation"] = obs_str[:8000]
                rec["internal_evidence_restore"] = True
                rec["restore_hits"] = restore_hits
                logger.info(
                    "Planner 内部恢复步 [{}] 回填证据 {} 条（零外部调用）。",
                    task.get("id"), restore_hits,
                )
            # 预算熔断：文本留痕、记坏状态，不再做本步提炼（计划继续推进）
            elif not budget.can_call(tool_name):
                obs_str = budget.deny_text(tool_name)
                rec["budget_denied"] = True
                rec["status"] = "budget_denied"
                rec["error"] = obs_str[:200]
                skip_distill = True
                # ⚠️ loguru 只支持 {} 占位符：写成 %s 会原样打印 "%s" 并丢弃参数
                logger.info(
                    "Planner 子任务 [{}] 工具 [{}] 触发额度熔断，记账后继续后续子任务。",
                    task.get("id"), tool_name,
                )
            else:
                # interrupt 恢复重放：优先复用审批时已确认的入参。
                # LangGraph 恢复时整个节点从头重跑，若再次执行 hint/FC 解析，
                # 非确定性 LLM 可能漂移出与审批卡片**不同的字段**——实测事故
                # （2026-09-23）：审批正文 A，批准后 FC 重跑变成正文 B 落盘，
                # 人工审批在字段层面被架空。runner 已把审批载荷注入 config。
                args: Optional[Dict[str, Any]] = take_resume_approval_arguments(
                    config, tool_name=tool_name, subtask_id=task.get("id"),
                )
                if args is not None:
                    logger.info(
                        "Planner 子任务 [{}] 工具 [{}] 审批恢复重放：复用已审批入参，"
                        "跳过 hint/FC 参数重解析（审批所见=实际执行）。",
                        task.get("id"), tool_name,
                    )

                # 参数解析优先级：planner 的 tool_args_hint（零 LLM）
                #   → FC 强制取参（1 次 LLM）→ 文本解析 + user_query 兜底。
                # 旧顺序是 FC 优先，每个 tool 子任务都白花一次「参数填充器」LLM 调用
                # （实测 769 / 1346 / 1756 tokens 三笔，合计约 3.8k）。
                #
                # ⚠️ 但 hint 直用有一个前提：**先把前序任务占位符替换掉**。
                # 实测故障：hint 里写 "<从 task_2 获取的路径>"，占位符是"非空字符串"
                # 能通过必填校验，于是一路直达工具 → "找不到文件: <从 task_2 获取的路径>"。
                if args is None:
                    resolved_hint: Any = substitute_task_refs(
                        task.get("tool_args_hint"), artifacts_by_task
                    )
                    args = resolve_tool_args_from_hint(
                        deps.tools, tool_name, resolved_hint
                    )
                    # 下游写/审批工具且已有前序子任务结果：即便 hint 必填参数齐全，
                    # 也强制用 FC 结合前序真实结论重组一次参数。
                    # 实测事故（2026-09-23）：planner 计划时看不到运行结果，给导出
                    # 工具的 content 只写了标题性占位"2026年7月销售及负责人产品销量
                    # 排名数据"——非空且通过必填校验，零 LLM 直达工具 → 导出只有
                    # 一行占位语的空报表，还白烧一次人工审批。写操作的代价（审批+
                    # 落盘）远高于一次参数填充 LLM。
                    compose_from_prior: bool = (
                        args is not None
                        and bool(prior_results)
                        and tool_name in danger_tool_names(deps)
                    )
                    if args is None or compose_from_prior:
                        fc_args: Optional[Dict[str, Any]] = (
                            await resolve_tool_args_via_function_call(
                                deps.model_router,
                                deps.tools,
                                tool_name=tool_name,
                                title=str(task.get("title") or ""),
                                description=str(task.get("description") or ""),
                                # 给 FC 的也必须是**替换后**的 hint：否则模型会把占位符原样抄回来
                                tool_args_hint=resolved_hint,
                                query=query,
                                prior_context_str=ctx_str,
                                purpose_hint="planner",
                            )
                        )
                        if args is None:
                            args = fc_args
                        elif isinstance(fc_args, dict) and fc_args:
                            # FC 值（基于真实结论的正文）覆盖 hint；FC 省略的计划期
                            # 已知字段（report_title/file_name 等）保留 hint 值。
                            args = {
                                **args,
                                **{
                                    key: value
                                    for key, value in fc_args.items()
                                    if value not in (None, "", [], {})
                                },
                            }
                if args is None:
                    args = {}
                    hint = task.get("tool_args_hint")
                    if hint:
                        try:
                            parsed = json.loads(hint)
                            args = parsed if isinstance(parsed, dict) else {"hint": str(parsed)}
                        except Exception:  # noqa: BLE001
                            args = {"hint": hint}
                    if "user_query" not in args:
                        args["user_query"] = query

                logger.info(
                    "Planner 子任务 [{}] 调用工具 [{}]，实际参数: {}",
                    task.get("id"), tool_name, args,
                )

                plan_call = ToolCall(
                    tool_name=tool_name,
                    arguments=args,
                    source="fc",
                    raw_arguments=str(args)[:1500],
                    raw={"subtask_id": task.get("id"), "args": args},
                )
                # 记账：把工具名与**真实入参**写进 step record。
                # 之前只 log 不进账，导致 plan 路径的 steps 拿不到参数（评测
                # key_arg_recall 少样本，见 evals/runners/run_tool.py 的说明）。
                rec["kind"] = "plan_tool_call"
                rec["action"] = tool_name
                rec["action_input"] = args

                # 危险工具人审闸门（默认关闭；拒绝时不执行、不耗预算）
                denied_obs: Optional[str] = await gate_tool_approval(
                    deps, plan_call, run_id=state["run_id"], subtask_id=task.get("id")
                )
                if denied_obs is not None:
                    obs_str = denied_obs
                    rec["approval_denied"] = True
                    rec["status"] = "approval_denied"
                    rec["error"] = "人工审批拒绝该工具调用"
                    skip_distill = True
                else:
                    tool_result: ToolResult = await execute_tool_call(
                        plan_call,
                        deps.tools,
                        call_budget=budget,
                        allowed_names=state.get("active_tool_names") or None,
                        model_router=deps.model_router,
                        purpose_hint="planner",
                    )
                    obs_str = tool_result.observation
                    rec["observation"] = obs_str[:8000]
                    # 证据板：ok/error/empty 真实工具观测都入管（审批/熔断的
                    # 合成短文本不属证据，不入管）。
                    if evidence_on:
                        plan_query = " ".join(
                            part for part in (
                                str(task.get("title") or ""),
                                str(task.get("description") or ""),
                                agent_goal_from_state(state),
                                query,
                            ) if part
                        )
                        evidence_units, evidence_meta = _evidence_ingest(
                            deps, trace_id,
                            existing_units=state.get("evidence_units") or [],
                            meta=state.get("evidence_meta") or {},
                            user_question=query,
                            tool_name=tool_name,
                            round_idx=cursor,
                            observation=obs_str,
                            current_query=plan_query,
                            action_input=args,
                            call_id=f"plan_{cursor}_{task.get('id')}",
                        )
                    if tool_result.budget_denied:
                        rec["budget_denied"] = True
                        rec["status"] = "budget_denied"
                        rec["error"] = obs_str[:200]
                        skip_distill = True
                    elif (
                        tool_result.denied
                        or tool_result.status == "error"
                        or observation_has_error_marker(obs_str)
                        or "error" in obs_str.lower()
                    ):
                        # 步级错误：记账后继续后续子任务，不再即时打断整个计划
                        rec["status"] = "error"
                        rec["error"] = obs_str[:200]
                        skip_distill = True
                        logger.warning(
                            "Planner 子任务 [{}] 工具 [{}] 返回错误，记账后继续后续子任务: {}",
                            task.get("id"), tool_name, obs_str[:160],
                        )
                    elif (
                        bool(deps.cfg("enable_empty_result_replan", True))
                        and isinstance(obs_str, str)
                        and _is_empty_data(obs_str)
                    ):
                        rec["status"] = "empty_data"
                        rec["replan_reason"] = "工具未返回可用业务数据"
                        skip_distill = True
                        logger.info(
                            "Planner 子任务 [{}] 工具 [{}] 返回空业务数据，记账后继续后续子任务。",
                            task.get("id"), tool_name,
                        )

        # 子任务 LLM 提炼（仅 ok 步；坏结果步跳过，避免把错误/空数据"提炼"成伪结论）
        if not skip_distill:
            # 计划台账：旧实现只把"已执行步骤的结论文本"给模型，它既看不到计划全貌
            # 也看不到成败状态，因而无从判断"答案够了 / 某步可以跳过"。
            ledger: str = render_plan_ledger(
                plan,
                prior_results,
                cursor=cursor,
                skipped_task_ids=skipped_ids,
                agent_goal=agent_goal_from_state(state),
            )
            # 已提取的结构化事实：让后续子任务**看得见**真实路径，不必再猜目录。
            # 只有提取到东西时才注入，避免给提示词塞空段。
            facts_text: str = render_facts_for_prompt(state.get("extracted_facts"))

            # ── 执行期就地纠偏：候选方向列表（**只在需要时**注入）────────────
            # 候选由**代码**确定性算出：已提取事实 − 已试资产、白名单 − 已用工具。
            # 模型只负责从这个封闭列表里挑一个。本 trace 的病根正是"让模型在没有
            # 候选集的情况下自己发明方向"——它发明了 /data，一个已被同一份输入记录
            # 证否的路径。有了候选集，这个矛盾在构造阶段就被结构性排除。
            question_keywords = build_question_keywords(state)
            evidence_gap, coverage = detect_evidence_gap(question_keywords, obs_str)
            correction_reason = injection_reason(state, evidence_gap=evidence_gap)
            # rec 此刻尚未入账，但它调用的工具/尝试的资产已"在飞"：必须计入差集，
            # 否则当前工具会作为"尚未尝试的候选"被重新选中，导致同方向连锁插步。
            correction_set = build_candidates(state, current_result=rec)

            correction_candidates: List[Candidate] = []
            candidates_text: str = ""
            # 纠偏步不再给候选：链式纠偏已被 _apply_step_correction 硬闸门禁止，
            # 这里不渲染候选列表，避免诱导模型输出必然被忽略的选择。
            correction_chain_blocked: bool = str(
                task.get("id") or ""
            ).startswith("correction_")
            if (
                correction_reason
                and remaining_quota(state) > 0
                and not correction_chain_blocked
            ):
                correction_candidates = correction_set.candidates
                candidates_text = render_candidates(correction_candidates)
                if evidence_gap:
                    logger.info(
                        "子任务 [{}] 证据缺口置位：问题核心词覆盖率={:.2f} < 阈值 {}",
                        task.get("id"), coverage, EVIDENCE_COVERAGE_THRESHOLD,
                    )
            elif correction_reason and correction_chain_blocked:
                logger.info(
                    "纠偏子任务 [{}] 仍未解决，按链式闸门不再注入候选、不再二次纠偏。",
                    task.get("id"),
                )
            if correction_set.excluded:
                # 留痕：存在已提取资产、但白名单里没有能消费它的工具。这同时是
                # "白名单过窄"的观测点（本 trace 即为此情形）。
                trace_event(
                    deps.tracer, trace_id, "plan_execute.candidates_excluded",
                    {"reason": "当前白名单不含可消费该资产的工具",
                     "excluded": correction_set.excluded},
                )

            prompt_content = (
                f"{ledger}\n\n"
                f"原始总问题：{query}\n当前子任务：{task.get('title')}\n"
                f"详细要求：{task.get('description')}\n"
                + (f"{facts_text}\n" if facts_text else "")
                + (f"{candidates_text}\n" if candidates_text else "")
                + f"历史子任务结论：\n{ctx_str}\n"
            )
            if obs_str:
                obs_for_distill = obs_str[:6000]
                if evidence_on and evidence_units is not None:
                    try:
                        obs_for_distill = render_plan_observation(
                            obs_str, evidence_units, round_idx=cursor
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("证据板 plan 视图渲染失败，回退 [:6000]: {}", exc)
                        trace_event(
                            deps.tracer, trace_id, "evidence.fallback",
                            {"stage": "plan_view", "error": str(exc)[:200]},
                        )
                prompt_content += f"\n本步骤工具调用返回的原始数据：\n{obs_for_distill}\n请结合工具数据完成本子任务。"
            else:
                prompt_content += "\n请根据历史上下文推理并完成本子任务。"

            control_enabled: bool = bool(deps.cfg("agent_plan_control_enabled", True))
            system_text: str = (
                "你是高效的子任务执行专家。请根据上下文（及工具数据），"
                "针对当前子任务给出简洁、准确的最终结论或分析结果。"
            )
            if control_enabled:
                system_text += PLAN_CONTROL_OUTPUT_CONTRACT
                system_text += (
                    "\n同时按以下规则填写 JSON 的 next_action / skip_task_ids 字段：\n"
                    "- 若现有结论已足以回答【原始总问题】，把 next_action 置为 finish；\n"
                    "- 若某个尚未执行的子任务、其答案已由其它子任务取得"
                    "（见台账的『是否解决』列），把它的 id 放进 skip_task_ids；\n"
                    "- 若本步取到的信息与问题对不上、需要换数据源或换工具，"
                    "而提示词里给出了「可选的替代方向」，请把其中**一项**的方括号内容"
                    "**原样照抄**进 selected_alternative_id（形如 asset:xxx 或 tool:xxx）；"
                    "不需要换方向时置为 null；\n"
                    "- selected_alternative_id 是候选的资产/工具方向；"
                    "requested_evidence_uids 是索取本步某条工具观测结果的全文"
                    "（填「已省略」清单里的证据编号，可多条）；两者互不影响，"
                    "都不需要时均留空；\n"
                    "- ⚠️ 严禁自行编造标识、工具名、参数、路径或证据编号——"
                    "只能从给定列表/清单里选，编造的会被忽略；也不允许自行新增或修改子任务。"
                )
            subtask_msgs = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt_content},
            ]
            chat_kwargs: Dict[str, Any] = {}
            if control_enabled:
                # 控制指令与结论在**同一次调用**里产出——这是它优于"注册 end 工具"
                # 的关键：后者需要模型额外发起一轮工具调用。
                chat_kwargs["response_format"] = pydantic_to_openai_response_format(
                    SubTaskOutcomeSchema
                )
            resp = await deps.model_router.chat(
                messages=subtask_msgs, purpose_hint="planner", thinking=False,
                temperature=0.3, **chat_kwargs,
            )
            text: str = (getattr(resp, "content", None) or "").strip()
            rec["llm_output"] = text
            rec["status"] = "ok"

            outcome: Optional[Dict[str, Any]] = _parse_subtask_outcome(text)
            if outcome is not None:
                if str(outcome.get("conclusion") or "").strip():
                    rec["llm_output"] = str(outcome["conclusion"]).strip()
                rec["solved"] = str(outcome.get("solved") or "")
                update.update(
                    _apply_subtask_control(
                        rec, outcome, plan, cursor,
                        control_enabled=control_enabled, skipped_ids=skipped_ids,
                    )
                )

                # ── 就地纠偏：模型只给了标识，这里翻译并应用 ──────────────
                correction_update = _apply_step_correction(
                    state=state,
                    plan=plan,
                    cursor=cursor,
                    skipped_ids=skipped_ids,
                    candidates=correction_candidates,
                    selected=outcome.get("selected_alternative_id"),
                    deps=deps,
                    trace_id=trace_id,
                    subtask_id=task.get("id"),
                    trigger=correction_reason,
                )
                if correction_update:
                    update.update(correction_update)
                    # 计划已被就地改写 → 后续的计划级闸门必须基于新计划判定
                    plan = correction_update.get("plan") or plan

                # ── 延迟一步证据回取：独立字段，与纠偏互不影响（D10） ──────
                if evidence_on:
                    restore_update = _apply_evidence_restore(
                        state=state,
                        plan=plan,
                        cursor=cursor,
                        subtask_id=task.get("id"),
                        requested=outcome.get("requested_evidence_uids"),
                        unit_dicts=(
                            evidence_units
                            if evidence_units is not None
                            else list(state.get("evidence_units") or [])
                        ),
                        deps=deps,
                        trace_id=trace_id,
                    )
                    if restore_update:
                        update.update(restore_update)
                        plan = restore_update.get("plan") or plan
            else:
                # 降级：控制指令解析不出来时**保留结论**、按"继续执行"处理。
                # 控制是增值能力，绝不能因为它没解析出来就丢掉已取回的工具数据。
                logger.info(
                    "Planner 子任务 [{}] 控制协议输出不可解析，降级为继续执行（结论保留）",
                    task.get("id"),
                )

    except Exception as exc:  # noqa: BLE001 - 取参/调用异常同样记账继续，交计划级闸门统一决策
        # ⚠️ 人工审批的 GraphInterrupt 必须原样穿透节点上抛给 LangGraph Pregel：
        # 图才能真正暂停（快照写 next + interrupts，前端收到 awaiting_approval）。
        # langgraph 1.2.x 中 GraphInterrupt 的 MRO 是
        # GraphInterrupt → GraphBubbleUp → Exception（不是 BaseException），
        # 会被本宽异常捕获。2026-09-23 事故：审批中断在此被当成步级错误吞掉，
        # 图未暂停、继续跑完走友好降级，危险工具的人工审批永远等不到人。
        # react FC/文本两条路径不经过本宽 except，天然能穿透，无需处理。
        from langgraph.errors import GraphInterrupt
        if isinstance(exc, GraphInterrupt):
            raise
        rec["status"] = "error"
        rec["error"] = str(exc)[:200]
        skip_distill = True
        # ⚠️ 原来是 "%s ... {}" 混用：loguru 下 task_id 会被塞进 {}，%s 原样露出，参数错位
        logger.warning("Planner 子任务 [{}] 执行异常，记账后继续后续子任务: {}", task.get("id"), exc)

    if skip_distill:
        # 执行失败/空数据/熔断/审批拒绝的步骤**没有走结论提炼**，模型并未看到数据，
        # 因此「是否解决」必须由程序强制为否——让模型自评就是在编。
        rec["solved"] = "no"

    # ── 结构化事实提取：把工具返回里的「数据资产地图」确定性搬进状态 ──────
    # ⚠️ 这一步**不经过模型摘要**。摘要会丢掉文档里逐字写明的路径（本 trace 的
    # 病根：摘要只说"涵盖客户线索台账等 Excel 文件"，路径全没了，重规划只能
    # 盲猜目录）。提取器原样保留，模型摘要只负责语义、不再承担事实传递。
    # 提取不到任何结果时保持既有事实不变，绝不中断链路。
    if obs_str:
        new_facts = extract_asset_facts(obs_str)
        if new_facts:
            update["extracted_facts"] = merge_facts(
                state.get("extracted_facts"), new_facts
            )

    # ── 步级统一收尾：记账 + cursor 推进（坏结果不再阻断计划）──────────────
    update["subtask_results"] = [rec]
    update.update(emit_plan_step(rec))
    cursor_after: int = cursor + 1
    update["cursor"] = cursor_after
    update["last_error"] = rec.get("error") if rec.get("status") == "error" else None
    update["empty_data_signal"] = (
        f"{EMPTY_DATASOURCE_REPLAN_PREFIX}: 工具 [{rec.get('tool_name')}] 空数据（留痕）"
        if rec.get("status") == "empty_data" else None
    )
    update.update(write_back_budget(budget))
    if rec.get("status") in PLAN_BAD_STATUSES:
        trace_event(deps.tracer, trace_id, "plan_execute.bad_step", rec)
    else:
        trace_event(deps.tracer, trace_id, "plan_execute.subtask", rec)

    # ── L2 计划级规则闸门：整轮计划跑完且本轮工具步全坏 → 换源 replan ───────
    gate_update: Dict[str, Any] = _plan_level_evidence_gate(
        state=state,
        plan=plan,
        cursor_after=cursor_after,
        all_results=prior_results + [rec],
        budget=budget,
        gate_enabled=bool(deps.cfg("enable_empty_result_replan", True)),
    )
    if gate_update:
        update["insufficiency_signal"] = gate_update["insufficiency_signal"]
        update["last_error"] = None
        trace_event(
            deps.tracer, trace_id, "plan_execute.all_tools_failed",
            {"signal": gate_update["insufficiency_signal"][:500]},
        )
    else:
        update["insufficiency_signal"] = None
    if evidence_units is not None and evidence_meta is not None:
        update["evidence_units"] = evidence_units
        update["evidence_meta"] = evidence_meta

    return update


# ===========================================================================
# react 形态（一次一轮）
# ===========================================================================
async def _execute_react_step(state: AgentGraphState, config: RunnableConfig) -> dict:
    deps = get_deps(config)
    protocol: str = state.get("react_protocol") or ""
    fc_definitions: List[Dict[str, Any]] = state.get("fc_tool_definitions") or []

    # 未初始化：有 FC 工具池优先 FC；FC 通道确认不可用后当轮降级文本
    if protocol == "text" or not fc_definitions:
        return await _react_text_turn(state, config)
    try:
        return await _react_fc_turn(state, config)
    except _FCFallbackRequired as fallback_err:
        logger.warning("ReAct FC 主链路不可用，降级文本 Thought/Action 协议。原因: {}", fallback_err)
        if _is_candidate_exhaustion(fallback_err):
            # 模型候选耗尽/超时：文本兜底只会再超时一次，直接友好降级
            rec = {
                "step": int(state.get("react_step", 0)),
                "phase": "react",
                "kind": "fc_candidate_exhausted",
                "final": True,
            }
            return {
                **emit_step(rec),
                "final_answer": GRACEFUL_TIMEOUT_MESSAGE,
                "success": False,
                "last_error": str(fallback_err),
                "react_protocol": "text",
            }
        # 文本协议从头开始（与旧 run_react_agent 清空 openai_tools 后重启一致）
        return await _react_text_turn(state, config, forced_protocol=True)


def _bootstrap_fc_messages(state: AgentGraphState) -> List[Dict[str, Any]]:
    """FC 路径首轮消息：system（提示词+附加段+长期记忆）→ 短期历史 → user。"""
    memory_context: Dict[str, Any] = state.get("memory_context") or {}
    extra_system = build_extra_system(state)
    system_content: str = (
        REACT_FC_SYSTEM_PROMPT
        + ("\n\n" + extra_system if extra_system else "")
        + f"\n\n## 检索长期事实记忆\n{long_term_mem_block(memory_context)}"
    )
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
    messages.extend(short_term_messages(memory_context))

    user_parts: List[str] = [f"## 用户问题\n{state['user_input']}"]
    skills_prompt: str = state.get("skills_prompt") or ""
    if skills_prompt:
        user_parts.append(f"## 可用高级技能 (渐进式披露)\n{skills_prompt}")
    user_parts.append(
        "请基于以上信息推进任务：需要外部信息或动作时调用工具，信息足够时直接输出面向用户的最终答案。"
    )
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})
    return messages

@trace_to_markdown(output_file="_react_fc_turn.md")
async def _react_fc_turn(state: AgentGraphState, config: RunnableConfig) -> dict:
    """FC 路径单轮：chat_with_tools → Final Answer 或 若干 tool_calls 执行闭环。"""
    deps = get_deps(config)
    trace_id: str = state["trace_id"]
    step_idx: int = int(state.get("react_step", 0))
    fc_tools: List[Dict[str, Any]] = state.get("fc_tool_definitions") or []
    active_names: List[str] = state.get("active_tool_names") or []

    budget = budget_from_ledger(state.get("budget") or {})
    evidence_on: bool = bool(deps.cfg("enable_evidence_board", False))
    evidence_units: List[Dict[str, Any]] = list(state.get("evidence_units") or [])
    evidence_meta: Dict[str, Any] = dict(state.get("evidence_meta") or {})

    persisted: List[Dict[str, Any]] = state.get("react_messages") or []
    messages: List[Dict[str, Any]] = list(persisted)
    new_messages: List[Dict[str, Any]] = []
    if not messages:
        messages = _bootstrap_fc_messages(state)
        new_messages.extend(messages)

    # fetch_evidence 只在证据板开启时注入，且不进工具注册表（循环内本地拦截）。
    send_tools: List[Dict[str, Any]] = list(fc_tools)
    if evidence_on and not any(
        (t.get("function") or {}).get("name") == FETCH_TOOL_NAME for t in send_tools
    ):
        send_tools.append(fetch_tool_definition())

    # 发送视图：证据板开启时按 tool_call_id 属主渲染（确定性选择替代位置型截断）；
    # 关闭/降级时保持现状 compact_tool_observations。state 里始终保留完整原文。
    if evidence_on:
        send_messages: List[Dict[str, Any]] = _evidence_fc_view(
            deps, trace_id,
            messages=messages, units=evidence_units,
            meta=evidence_meta, step_idx=step_idx,
        )
    else:
        send_messages = compact_tool_observations(messages)
    live_prompt: str = budget.live_prompt() if state.get("budget") else ""
    if live_prompt:
        send_messages.append({"role": "system", "content": live_prompt})

    resp = await deps.model_router.chat_with_tools(
        messages=send_messages,
        tools=send_tools,
        tool_choice="auto",
        purpose_hint="react",
        temperature=0.2,
        thinking=False,
    )

    reasoning_txt: str = getattr(resp, "reasoning_content", None) or ""
    tool_calls: List[Dict[str, Any]] = list(getattr(resp, "tool_calls", None) or [])
    step_recs: List[Dict[str, Any]] = []

    if not tool_calls:
        answer: str = (getattr(resp, "content", None) or "").strip()
        if not answer:
            # 空转：注入提示让模型下一轮明确选择，连续 3 轮空转判失败
            empty_turns: int = int(state.get("react_empty_turns", 0)) + 1
            nudge = {
                "role": "user",
                "content": "[系统提示] 你上一轮既没有调用工具，也没有输出最终答案。"
                           "请直接判断：若仍需数据请立即调用合适工具；若已足够请直接输出最终答案文本。",
            }
            messages.append(nudge)
            new_messages.append(nudge)
            last_error = None
            success = False
            final_answer = ""
            if empty_turns > 2:
                last_error = "FC 主链路连续多轮未产出工具调用或最终答案"
                logger.warning(last_error)
            return {
                "react_messages": new_messages,
                "react_empty_turns": empty_turns,
                "react_step": step_idx + 1,
                "react_protocol": "fc",
                "last_error": last_error,
                "success": success,
                "final_answer": final_answer,
                **write_back_budget(budget),
            }

        rec = {
            "step": step_idx,
            "phase": "react",
            "kind": "fc_final",
            "raw_llm": answer[:4000],
            "reasoning": reasoning_txt[:2000],
            "parsed": {"done": True, "final_answer": answer},
            "final": True,
        }
        step_recs.append(rec)
        trace_event(deps.tracer, trace_id, "react.step", rec)
        return {
            "react_messages": new_messages,
            "react_step": step_idx + 1,
            "react_empty_turns": 0,
            "react_protocol": "fc",
            "final_answer": answer,
            "success": True,
            "last_error": None,
            "empty_data_signal": None,
            **emit_step(rec),
            **write_back_budget(budget),
        }

    # 本轮有工具调用：回填 assistant 消息（id 补全）→ 逐工具执行 → role=tool 闭环
    for call_index, tc in enumerate(tool_calls):
        if not tc.get("id"):
            tc["id"] = f"call_{step_idx}_{call_index}"
    assistant_msg = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    messages.append(assistant_msg)
    new_messages.append(assistant_msg)

    for call in tool_calls_from_fc(tool_calls, step_idx):
        is_fetch = evidence_on and is_fetch_call(call)
        if is_fetch:
            # 回取：不进审批闸门、不进工具注册表、不耗工具预算；
            # 强制保留轮标记为下一轮（取回结果在下一激活才被模型看到）。
            fetch_state = {**state, "evidence_units": evidence_units}
            obs_text, fetched_units = _evidence_fetch(
                deps, trace_id,
                state=fetch_state,
                arguments=call.arguments,
                next_round=step_idx + 1,
            )
            if fetched_units is not None:
                evidence_units = fetched_units
            denied_obs: Optional[str] = None
            budget_denied = False
        else:
            denied_obs = await gate_tool_approval(
                deps, call, run_id=state["run_id"]
            )
            if denied_obs is not None:
                obs_text = denied_obs
                budget_denied = False
            else:
                tool_result: ToolResult = await execute_tool_call(
                    call,
                    deps.tools,
                    call_budget=budget,
                    allowed_names=active_names or None,
                    model_router=deps.model_router,
                    purpose_hint="react",
                )
                obs_text = tool_result.observation
                budget_denied = tool_result.budget_denied
                # 真实工具观测入管（审批拒绝的合成短文本不入管）
                if evidence_on and denied_obs is None:
                    ingested_units, ingested_meta = _evidence_ingest(
                        deps, trace_id,
                        existing_units=evidence_units,
                        meta=evidence_meta,
                        user_question=state["user_input"],
                        tool_name=call.tool_name,
                        round_idx=step_idx,
                        observation=obs_text,
                        current_query=extract_query_text(
                            call.arguments, call.tool_name
                        ),
                        action_input=call.arguments,
                        call_id=call.call_id,
                    )
                    if ingested_units is not None:
                        evidence_units = ingested_units
                        evidence_meta = ingested_meta

        rec = {
            "step": step_idx,
            "phase": "react",
            "kind": "fc_fetch_evidence" if is_fetch else "fc_tool_call",
            "tool_call": call.raw,
            "reasoning": reasoning_txt[:2000],
            "parsed": {"action": call.tool_name, "action_input": call.arguments},
            "action": call.tool_name,
            "action_input": call.arguments,
            "observation": obs_text,
        }
        if denied_obs is not None:
            rec["approval_denied"] = True
        if budget_denied:
            rec["budget_denied"] = True
        step_recs.append(rec)
        trace_event(deps.tracer, trace_id, "react.step", rec)

        tool_msg = {"role": "tool", "tool_call_id": call.call_id, "content": obs_text}
        messages.append(tool_msg)
        new_messages.append(tool_msg)

    update = {
        "react_messages": new_messages,
        "react_step": step_idx + 1,
        "react_empty_turns": 0,
        "react_protocol": "fc",
        "last_error": None,
        "empty_data_signal": None,
        "steps": [{"ts": time.time(), **rec} for rec in step_recs],
        **write_back_budget(budget),
    }
    if evidence_on:
        update["evidence_units"] = evidence_units
        update["evidence_meta"] = evidence_meta
    return update


def _tool_catalog_text(tool_names: List[str], tool_schemas: Dict[str, str]) -> str:
    """文本路径工具说明块（从 ReActAgent._tool_catalog_text 平移）。"""
    lines: List[str] = []
    for name in tool_names:
        if name in tool_schemas:
            lines.append(f"### {name}\n{tool_schemas[name]}")
        else:
            lines.append(f"- {name}  （参数信息缺失，请根据常识谨慎填写）")
    return "\n\n".join(lines) if lines else "（无外部工具，请直接 Final Answer）"


async def _react_text_turn(
    state: AgentGraphState,
    config: RunnableConfig,
    *,
    forced_protocol: bool = False,
) -> dict:
    """文本 ReAct 路径单轮：每轮用 history_lines 重建完整 user prompt（与旧循环一致）。"""
    deps = get_deps(config)
    trace_id: str = state["trace_id"]
    step_idx: int = int(state.get("react_step", 0))
    active_names: List[str] = state.get("active_tool_names") or []
    tool_schemas: Dict[str, Any] = state.get("tool_schemas") or {}
    memory_context: Dict[str, Any] = state.get("memory_context") or {}
    extra_system = build_extra_system(state)
    skills_prompt: str = state.get("skills_prompt") or ""
    history_lines: List[str] = list(state.get("react_history_lines") or [])

    budget = budget_from_ledger(state.get("budget") or {})
    evidence_on: bool = bool(deps.cfg("enable_evidence_board", False))
    evidence_units: List[Dict[str, Any]] = list(state.get("evidence_units") or [])
    evidence_meta: Dict[str, Any] = dict(state.get("evidence_meta") or {})

    tool_desc = _tool_catalog_text(active_names, tool_schemas)
    if evidence_on:
        # 文本协议无 function 定义通道：以目录条目形式告知回取能力。
        tool_desc += (
            f"\n\n### {FETCH_TOOL_NAME}\n取回证据板上某条证据的完整原文。"
            "Action Input JSON：{\"uid\": \"e编号（取自 Observation 中的 [eN｜来源:...] 标记）\", "
            "\"window\": 0-5 的整数，可选，同时取回前后相邻片段}"
        )
    # 与 FC 路径同口径的历史收口：证据板开启时 Step 观测段切换属主渲染。
    if evidence_on and history_lines:
        compacted_lines = _evidence_text_view(
            deps, trace_id,
            history_lines=history_lines,
            units=evidence_units,
            meta=evidence_meta,
            step_idx=step_idx,
        )
    else:
        compacted_lines = compact_history_lines(history_lines)
    history_block = "\n".join(compacted_lines) if history_lines else "（尚无）"
    user_prompt = build_react_user_prompt(
        query=state["user_input"],
        tool_descriptions=tool_desc,
        history_block=history_block,
        skills_block=skills_prompt,
    )
    if state.get("budget"):
        live_prompt = budget.live_prompt()
        if live_prompt:
            user_prompt = f"{live_prompt}\n\n{user_prompt}"

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": REACT_SYSTEM_PROMPT
            + ("\n\n" + extra_system if extra_system else "")
            + f"\n\n## 检索长期事实记忆\n{long_term_mem_block(memory_context)}",
        }
    ]
    messages.extend(short_term_messages(memory_context))
    messages.append({"role": "user", "content": user_prompt})

    try:
        resp = await deps.model_router.chat(
            messages=messages, purpose_hint="react", temperature=0.2
        )
        raw: str = (getattr(resp, "content", None) or "").strip()
    except Exception as exc:  # noqa: BLE001 - 与旧文本循环一致：友好降级答复
        err = f"LLM 调用失败: {exc}"
        logger.warning("文本 ReAct LLM 调用异常，返回友好降级答复: {}", err)
        rec = {
            "step": step_idx,
            "phase": "react",
            "kind": "text_llm_error",
            "final": True,
        }
        return {
            **emit_step(rec),
            "react_protocol": "text",
            "final_answer": GRACEFUL_TIMEOUT_MESSAGE,
            "success": False,
            "last_error": err,
        }

    parsed = _parse_react_step(raw)
    rec: Dict[str, Any] = {
        "step": step_idx,
        "phase": "react",
        "raw_llm": raw[:4000],
        "parsed": {k: v for k, v in parsed.items() if k != "raw"},
    }
    new_history: List[str] = []

    # Final Answer
    if parsed.get("done") and parsed.get("final_answer"):
        answer = str(parsed["final_answer"])
        rec["final"] = True
        trace_event(deps.tracer, trace_id, "react.step", rec)
        return {
            **emit_step(rec),
            "react_step": step_idx + 1,
            "react_protocol": "text",
            "final_answer": answer,
            "success": True,
            "last_error": None,
            "empty_data_signal": None,
            **write_back_budget(budget),
        }

    action = parsed.get("action")
    action_input = parsed.get("action_input") or {}

    # L8 FAIL：结构化 Observation 注入，下一轮天然 Retry
    if (
        not action
        or parsed.get("parse_error")
        or parsed.get("semantic_error")
        or parsed.get("business_error")
    ):
        fail_reasons: List[str] = []
        for key in ("parse_error", "semantic_error", "business_error"):
            if parsed.get(key):
                fail_reasons.append(str(parsed[key]))
        if not action and not fail_reasons:
            fail_reasons.append("未解析到 Action，也没有 Final Answer。")
        obs = (
            "【系统校验 FAIL（进入下一轮循环 Retry）】：\n"
            + "\n".join(f"- {reason}" for reason in fail_reasons)
            + "\n\n【建议调整】："
            + "请严格遵守 ReAct 输出格式。你只能选择两条路径之一：\n"
            + "  (1) 调用工具：Thought: ... → Action: <合法工具名> → Action Input: <严格合法 JSON object>\n"
            + "  (2) 直接回答：直接输出 Final Answer: <你的最终答复>\n"
            + f"\n当前步骤的 Thought 内容是：{parsed.get('thought', '')}"
        )
        rec["error"] = fail_reasons
        rec["observation"] = obs[:8000]
        new_history.append(
            f"Step {step_idx + 1}\nThought: {parsed.get('thought', '')}\n"
            f"[系统校验 FAIL，结构化回传以便下一轮 Retry]\nObservation: {obs}\n"
        )
        trace_event(deps.tracer, trace_id, "react.step", rec)
        return {
            **emit_step(rec),
            "react_step": step_idx + 1,
            "react_protocol": "text",
            "react_history_lines": new_history,
            **write_back_budget(budget),
        }

    # 工具调用：审批闸门 → 统一执行管线
    text_call: Optional[ToolCall] = tool_call_from_text(parsed)
    if text_call is None:
        # 理论不可达（action 存在），防御性按 FAIL 处理
        text_call = ToolCall(tool_name=str(action), arguments={}, source="text")

    is_fetch_text = evidence_on and str(action) == FETCH_TOOL_NAME
    if is_fetch_text:
        # 文本协议回取：与 FC 同拦截器，零注册表调用、不耗预算
        fetch_state = {**state, "evidence_units": evidence_units}
        obs_text, fetched_units = _evidence_fetch(
            deps, trace_id,
            state=fetch_state,
            arguments=action_input,
            next_round=step_idx + 1,
        )
        if fetched_units is not None:
            evidence_units = fetched_units
        denied_obs: Optional[str] = None
        tool_result: Optional[ToolResult] = None
    else:
        denied_obs = await gate_tool_approval(
            deps, text_call, run_id=state["run_id"]
        )
        if denied_obs is not None:
            obs_text = denied_obs
            tool_result = None
        else:
            tool_result = await execute_tool_call(
                text_call,
                deps.tools,
                call_budget=budget,
                allowed_names=active_names or None,
                model_router=deps.model_router,
                purpose_hint="react",
            )
            obs_text = tool_result.observation
            if tool_result.denied and not tool_result.budget_denied:
                rec["warn"] = f"工具 [{action}] 不在白名单，本轮按 FAIL 回注 Observation"
            if tool_result.budget_denied:
                rec["budget_denied"] = True
            # 真实工具观测入管（白名单拒绝的 FAIL 观测是系统反馈，仍入管为 status；
            # 审批拒绝的合成短文本不入管）
            if evidence_on and denied_obs is None:
                ingested_units, ingested_meta = _evidence_ingest(
                    deps, trace_id,
                    existing_units=evidence_units,
                    meta=evidence_meta,
                    user_question=state["user_input"],
                    tool_name=str(action),
                    round_idx=step_idx,
                    observation=obs_text,
                    current_query=extract_query_text(action_input, str(action)),
                    action_input=action_input,
                    call_id=f"text_call_{step_idx}",
                )
                if ingested_units is not None:
                    evidence_units = ingested_units
                    evidence_meta = ingested_meta
    if denied_obs is not None:
        rec["approval_denied"] = True

    rec["action"] = action
    rec["action_input"] = action_input
    rec["observation"] = obs_text
    if is_fetch_text:
        rec["kind"] = "text_fetch_evidence"
    new_history.append(
        f"Step {step_idx + 1}\nThought: {parsed.get('thought', '')}\n"
        f"Action: {action}\nObservation: {obs_text}\n"
    )
    trace_event(deps.tracer, trace_id, "react.step", rec)

    update = {
        **emit_step(rec),
        "react_step": step_idx + 1,
        "react_protocol": "text",
        "react_history_lines": new_history,
        "last_error": None,
        "empty_data_signal": None,
        **write_back_budget(budget),
    }
    if evidence_on:
        update["evidence_units"] = evidence_units
        update["evidence_meta"] = evidence_meta
    if forced_protocol:
        # FC 首轮降级：react_protocol 固化为 text（值已是 text，显式留痕）
        update["degraded"] = True
    return update
