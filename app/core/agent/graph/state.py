# -*- coding: utf-8 -*-
"""Agent 状态图的 State 定义、路由常量与（预算账本 / Intent）序列化适配。

设计铁律（见 .trae/documents/state_graph_refactor_plan.md 第四节）：

1. 进入 state 的值必须是 JSON 友好类型（str/int/bool/dict/list/None）；
   registry / model_router / memory / skill / ToolCallBudget 等运行时对象
   **禁止入 state**，它们只存在于 ``RunnableConfig["configurable"]["deps"]``。
2. 步骤 / 观测 / 消息类追加列表使用 ``Annotated[list, operator.add]`` reducer；
   标量字段由节点整体覆盖（节点返回 partial dict，未返回的键保持原值）。
3. dict 类型账本（budget / retry_counts）由节点返回**合并后的完整 dict**，
   不做浅合并，避免跨节点丢计数。
4. 不设 ``paused`` / ``interrupt_info`` 字段：暂停判定只用
   ``aget_state(cfg).next`` + ``snapshot.tasks[].interrupts[]``，
   审批信息只存在于 interrupt payload 与 API 响应层。
"""

from __future__ import annotations

import operator
from dataclasses import asdict, is_dataclass
from typing import Annotated, Any, Dict, List, Optional, TypedDict

# ---------------------------------------------------------------------------
# 节点名 / 路由值常量（builder 与单测共用，避免散落的魔法字符串）
# ---------------------------------------------------------------------------
NODE_PREPARE = "prepare"
NODE_PLAN = "plan"
NODE_EXECUTE = "execute"
NODE_REPLAN = "replan"
NODE_REFLECT = "reflect"
NODE_SUMMARIZE = "summarize"
NODE_PERSIST = "persist"

ROUTE_PLAN = "plan"
ROUTE_EXECUTE = "execute"
ROUTE_REPLAN = "replan"
ROUTE_REFLECT = "reflect"
ROUTE_SUMMARIZE = "summarize"
ROUTE_PERSIST = "persist"

# reflect 质量门在 retry_counts 中的保留键
REFLECT_RETRY_KEY = "__reflect__"


# ---------------------------------------------------------------------------
# State 本体
# ---------------------------------------------------------------------------
class AgentGraphState(TypedDict, total=False):
    """LangGraph 全局状态。``total=False``：节点可只返回需要更新的 partial dict。"""

    # ── 标识与输入（runner 注入，整轮不变）─────────────────────────────
    run_id: str                          # = checkpoint thread_id，f"{session_id}:{uuid4}"
    session_id: str
    trace_id: str
    user_input: str
    intent: Dict[str, Any]               # IntentContext 的可序列化 dict 形态
    should_plan: bool                    # mode_decider/strategy 映射结果
    mode_source: str                     # 路由留痕：strategy/manual/intent

    # ── prepare 节点产出的上下文 ───────────────────────────────────────
    memory_context: Dict[str, Any]       # short_term / long_term（可序列化列表）
    skills_prompt: str                   # 技能渐进式披露规约（**执行侧**用，含读取指引）
    skills_index: str                    # 技能极简清单（**规划侧**用，只含名称+一句话用途）
    extracted_facts: List[Dict[str, Any]]  # 运行期从文档中提取的结构化事实（数据资产名/位置/工具）
    step_corrections: List[Dict[str, Any]]  # 执行期就地纠偏的留痕（同时作为配额已用计数）
    active_tool_names: List[str]         # 号令后的白名单工具
    tool_schemas: Dict[str, str]         # 文本路径的工具 schema 文本
    fc_tool_definitions: List[Dict[str, Any]]  # 原生 FC tools[] 定义
    extra_system_hints: Annotated[List[str], operator.add]
    # 额外系统提示片段（first_tool_hint / KB 集合硬约束 / 降级 suffix 等），execute 按序拼接

    # ── 预算账本（ToolCallBudget 的可序列化镜像，见文件尾适配函数）──────
    budget: Dict[str, Any]

    # ── plan 路径 ──────────────────────────────────────────────────────
    plan: List[Dict[str, Any]]           # SubTask.to_dict 列表
    cursor: int                          # 当前待执行子任务下标
    subtask_results: Annotated[List[Dict[str, Any]], operator.add]
    skipped_task_ids: List[str]
    # 被模型决定跳过的子任务 id（控制协议产出）。
    # ⚠️ 这里**刻意不维护台账状态**：每步的状态由 plan + subtask_results 推导
    #（见 nodes/_common.render_plan_ledger），本字段只承载"无法推导的模型决策"，
    # 避免出现第二份需要同步的真源（漂移后表现为"模型看到的进度"与真实进度不一致）。
    early_finish: bool
    # 模型判定证据已充分、提前收尾。执行上等价为"跳过剩余全部"，
    # 单独存一个布尔只为留痕区分"跳过某几个"与"整体提前收尾"。

    # ── react（无 plan 自环）路径 ──────────────────────────────────────
    react_messages: Annotated[List[Dict[str, Any]], operator.add]
    # FC 协议消息（role/content/tool_calls/tool_call_id 的纯 dict 形态）
    react_history_lines: Annotated[List[str], operator.add]  # 文本路径历史行
    react_protocol: str                  # "" 未初始化 / "fc" / "text"（FC 不可用时降级一次固化）
    react_step: int                      # 已完成的 react 单轮数（自环保护）
    react_empty_turns: int               # FC 链路连续空转（无 tool_call 无答案）轮数

    # ── 证据板（请求级；见 app/core/agent/evidence/）────────────────────
    evidence_units: List[Dict[str, Any]]
    # 归一化后的证据单元（块原文 ≤600 字；完整观测仍在 react_messages/
    # subtask_results 中，ref 坐标指向它们，不另建存储）。
    # 每轮证据板要从全量单元重选/重打分/去重，故由节点**整体替换**返回全量。
    evidence_meta: Dict[str, Any]
    # 整体替换：{next_seq: 下一个单元序号, queries: 累计查询词, rounds: [每轮计数]}

    # ── 共用执行控制 ───────────────────────────────────────────────────
    steps: Annotated[List[Dict[str, Any]], operator.add]  # trace 记录（等价旧 steps 列表）
    retry_counts: Dict[str, int]         # key: subtask_id 或 "react:{step}" -> 节点级重试次数
    replan_attempts: int
    max_replan: int                      # prepare 时从 config 固化
    max_steps: int                       # react_max_steps，prepare 时固化
    last_error: Optional[str]            # 最近一次异常文本（留痕/兜底文案用）
    empty_data_signal: Optional[str]     # 旧字段：单步空数据留痕（不再驱动即时 replan）
    insufficiency_signal: Optional[str]
    # 计划级"证据不足"信号（L2 规则闸门 / L3 summarize 自判写入），驱动换源 replan
    insufficiency_kind: Optional[str]
    # 缺口性质（结构化）：仅 "off_topic"（全部结论跑题＝方向性错误）才允许重规划；
    # 其余取值与缺失一律按非方向性错误处理。依据自由文本推断会导致步级问题付整轮代价。
    draft_answer: Optional[str]          # summarize 判定不足时的草稿（额度耗尽直接用它收尾）
    pending_tool: Optional[Dict[str, Any]]
    # 当前激活步待执行的 ToolCall dict（危险工具 interrupt 前后保持一致；非 paused 标志）
    reflect_failed: bool                 # reflect 质量门是否未通过（reflect 默认关闭）

    # ── 产出 ──────────────────────────────────────────────────────────
    final_answer: str
    success: bool
    degraded: bool                       # 新语义：本轮是否发生过节点级降级/重规划
    mode_used: str                       # react / plan_execute（API 兼容字段）
    route: str                           # 最近一次条件路由决策留痕


# ---------------------------------------------------------------------------
# 初始 state 组装
# ---------------------------------------------------------------------------
def make_initial_state(
    *,
    run_id: str,
    session_id: str,
    trace_id: str,
    user_input: str,
    intent: Any,
    should_plan: bool,
    mode_source: str,
    precomputed_memory: Any = None,
) -> Dict[str, Any]:
    """runner 首次调用时组装的初始 state（precomputed_memory 为 None 时由 prepare 拉取）。"""
    initial: Dict[str, Any] = {
        "run_id": run_id,
        "session_id": session_id,
        "trace_id": trace_id,
        "user_input": user_input,
        "intent": intent_to_dict(intent),
        "should_plan": bool(should_plan),
        "mode_source": mode_source,
        "skills_prompt": "",
        "skills_index": "",
        "extracted_facts": [],
        "step_corrections": [],
        "active_tool_names": [],
        "tool_schemas": {},
        "fc_tool_definitions": [],
        "extra_system_hints": [],
        "budget": {},
        "plan": [],
        "cursor": 0,
        "subtask_results": [],
        "skipped_task_ids": [],
        "early_finish": False,
        "react_messages": [],
        "react_history_lines": [],
        "react_protocol": "",
        "react_step": 0,
        "react_empty_turns": 0,
        "evidence_units": [],
        "evidence_meta": {"next_seq": 1, "rounds": []},
        "steps": [],
        "retry_counts": {},
        "replan_attempts": 0,
        # 与 Settings.max_replan_attempts 同口径：收窄后至多 1 次
        "max_replan": 1,
        "max_steps": 10,
        "last_error": None,
        "empty_data_signal": None,
        "insufficiency_signal": None,
        "insufficiency_kind": None,
        "draft_answer": None,
        "pending_tool": None,
        "reflect_failed": False,
        "final_answer": "",
        "success": False,
        "degraded": False,
        "mode_used": "plan_execute" if should_plan else "react",
        "route": "",
    }
    if precomputed_memory is not None:
        # Pipeline 已并行预取：prepare 节点直接使用，不再重复检索。
        # MemoryContext 是 Pydantic BaseModel，用 model_dump 转可序列化 dict；
        # 普通 dict 原样透传。
        if hasattr(precomputed_memory, "model_dump"):
            initial["memory_context"] = precomputed_memory.model_dump()
        elif is_dataclass(precomputed_memory):
            initial["memory_context"] = asdict(precomputed_memory)
        elif isinstance(precomputed_memory, dict):
            initial["memory_context"] = dict(precomputed_memory)
    return initial


# ---------------------------------------------------------------------------
# IntentContext <-> dict（函数内 import 避免与 orchestrator 形成导入环）
# ---------------------------------------------------------------------------
def intent_to_dict(intent: Any) -> Dict[str, Any]:
    """把 IntentContext（dataclass）序列化为 state 用的纯 dict。"""
    if intent is None:
        return {}
    if isinstance(intent, dict):
        return dict(intent)
    if is_dataclass(intent):
        return asdict(intent)
    # 兜底：按属性摘取
    return {
        "intent": getattr(intent, "intent", "general"),
        "confidence": getattr(intent, "confidence", 1.0),
        "slots": dict(getattr(intent, "slots", {}) or {}),
        "preferred_mode": getattr(intent, "preferred_mode", None),
        "allowed_tools": list(getattr(intent, "allowed_tools", None) or []),
    }


def dict_to_intent(data: Dict[str, Any]):
    """state intent dict 还原为 orchestrator.IntentContext（runner 映射 AgentResponse 用）。"""
    from app.core.agent.orchestrator import IntentContext

    if not data:
        return IntentContext()
    return IntentContext(
        intent=str(data.get("intent", "general")),
        confidence=float(data.get("confidence", 1.0)),
        slots=dict(data.get("slots", {}) or {}),
        preferred_mode=data.get("preferred_mode"),
        allowed_tools=list(data.get("allowed_tools") or []),
    )


# ---------------------------------------------------------------------------
# ToolCallBudget 账本 ↔ 对象适配（registry.py 零改动：只用其公开构造参数 +
# 恢复私有计数字段，跨 checkpoint/resume 保持熔断状态一致）
# ---------------------------------------------------------------------------
def budget_to_ledger(budget: Any) -> Dict[str, Any]:
    """从 ToolCallBudget 抽取静态限额与动态计数为纯 dict 账本（写入 state）。"""
    if budget is None:
        return {}
    return {
        "per_tool_limits": dict(getattr(budget, "per_tool_limits", {}) or {}),
        "default_per_tool": int(getattr(budget, "default_per_tool", 10)),
        "total_budget": int(getattr(budget, "total_budget", 0)),
        "invalid_limit": int(getattr(budget, "invalid_limit", 3)),
        "invalid_consecutive_limit": int(getattr(budget, "invalid_consecutive_limit", 2)),
        "relevance_check_call": int(getattr(budget, "relevance_check_call", 5)),
        "used": dict(getattr(budget, "_used_per_tool", {}) or {}),
        "invalid": dict(getattr(budget, "_invalid_per_tool", {}) or {}),
        "consecutive_invalid": dict(
            getattr(budget, "_consecutive_invalid_per_tool", {}) or {}
        ),
        "locked": dict(getattr(budget, "_locked", {}) or {}),
        "total_used": int(getattr(budget, "_total_used", 0)),
    }


def budget_from_ledger(ledger: Dict[str, Any]) -> Any:
    """用 state 账本重建一个**临时** ToolCallBudget（仅本次节点激活内使用）。

    节点用它做 can_call/consume/deny_text/live_prompt；激活结束后必须再调
    ``budget_to_ledger`` 把计数写回 state。
    """
    from app.core.tools.registry import ToolCallBudget

    ledger = ledger or {}
    budget = ToolCallBudget(
        per_tool_limits=dict(ledger.get("per_tool_limits", {}) or {}),
        default_per_tool=int(ledger.get("default_per_tool", 10)),
        total_budget=int(ledger.get("total_budget", 0)),
        invalid_limit=int(ledger.get("invalid_limit", 3)),
        invalid_consecutive_limit=int(ledger.get("invalid_consecutive_limit", 2)),
        relevance_check_call=int(ledger.get("relevance_check_call", 5)),
    )
    # 恢复动态计数（含已熔断工具），保证 resume 后熔断状态不丢失
    budget._used_per_tool = {k: int(v) for k, v in (ledger.get("used") or {}).items()}
    budget._invalid_per_tool = {k: int(v) for k, v in (ledger.get("invalid") or {}).items()}
    budget._consecutive_invalid_per_tool = {
        k: int(v) for k, v in (ledger.get("consecutive_invalid") or {}).items()
    }
    budget._locked = dict(ledger.get("locked") or {})
    budget._total_used = int(ledger.get("total_used", 0))
    return budget


def budget_remaining(ledger: Dict[str, Any]) -> Optional[int]:
    """剩余工具总调用额度。

    ``total_budget <= 0`` 表示未设全局上限（与 ToolCallBudget.can_call 语义一致），
    返回 ``None`` 表示"无限"；调用方判定余量时应把 None 视为可继续。
    """
    ledger = ledger or {}
    total_budget: int = int(ledger.get("total_budget", 0) or 0)
    if total_budget <= 0:
        return None
    return max(0, total_budget - int(ledger.get("total_used", 0) or 0))


def replan_capacity(state: Dict[str, Any]) -> bool:
    """综合判定：当前是否还允许一次"带工具补救"的 replan（次数与总预算双条件）。"""
    attempts: int = int(state.get("replan_attempts", 0) or 0)
    if attempts >= int(state.get("max_replan", 2) or 0):
        return False
    remaining: Optional[int] = budget_remaining(state.get("budget") or {})
    return remaining is None or remaining > 0
