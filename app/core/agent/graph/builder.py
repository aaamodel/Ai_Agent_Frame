# -*- coding: utf-8 -*-
"""StateGraph 拓扑编译与条件路由纯函数。

拓扑（计划第二节）::

    START → prepare ─┬─ should_plan ─→ plan ─→ execute ◀──────┐
                    └──────────────→ execute ──▲              │
              execute：步级空/错只记账继续；计划跑完按 L2/L3 评估
    replan ─→ execute | summarize
    reflect ─→ execute | summarize
    summarize ─┬─ L3 判定证据不足且有余量 ─→ replan（缺口说明/失败工具去重）
               └─ 否则 ─→ persist → END

路由函数全部为无副作用纯函数（config 只读），可直接单测；
节点是否暂停由 runner 用 checkpoint 快照判定，与路由无关。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.core.agent.graph.deps import get_deps
from app.core.agent.graph.nodes._common import next_pending_cursor
# 重规划门控的**单一真源**：产出方（summarize_node）与路由方共用，避免出现
# "summarize 等着重规划、路由却拒绝"的不一致（那会产出空答案）。
from app.core.agent.graph.replan_gate import (
    REPLAN_HARD_LIMIT,  # noqa: F401 - re-export，供既有引用点使用
    replan_allowed,
    replan_exclusion_reason,
)
from app.core.agent.graph.nodes import (
    execute_node,
    persist_node,
    plan_node,
    prepare_node,
    reflect_node,
    replan_node,
    summarize_node,
)
from app.core.agent.graph.state import (
    NODE_EXECUTE,
    NODE_PLAN,
    NODE_PREPARE,
    NODE_REFLECT,
    NODE_REPLAN,
    NODE_PERSIST,
    NODE_SUMMARIZE,
    ROUTE_EXECUTE,
    ROUTE_PERSIST,
    ROUTE_PLAN,
    ROUTE_REFLECT,
    ROUTE_REPLAN,
    ROUTE_SUMMARIZE,
    AgentGraphState,
    REFLECT_RETRY_KEY,
    replan_capacity,
)


# ---------------------------------------------------------------------------
# 条件路由纯函数
# ---------------------------------------------------------------------------
def route_after_prepare(state: AgentGraphState) -> str:
    """prepare → plan（需规划）/ execute（ReAct 态：跳过 plan 直接自环）。"""
    return ROUTE_PLAN if state.get("should_plan") else ROUTE_EXECUTE


def route_after_execute(
    state: AgentGraphState, config: Optional[RunnableConfig] = None
) -> str:
    """execute 之后的互斥优先级路由（步级失败不再即时 replan）。

    1. 已有 Final Answer（react 终局）→ 质量门（开启时）或汇总；
    2. plan 路径还有下一子任务 → execute 自环（空/错只记账，不阻断计划）；
    3. plan 已跑完：L2 规则闸门判全坏且仍有 replan 余量 → replan，否则 summarize；
    4. react 路径未达步数上限 → execute 自环；
    5. 兜底 → summarize（GRACEFUL 收尾，不把异常抛出图）。
    """
    # 1. 终局答案
    if (state.get("final_answer") or "").strip():
        if _reflect_enabled(config):
            return ROUTE_REFLECT
        return ROUTE_SUMMARIZE

    # 2-3. plan 路径：先把计划跑完（被跳过的不再执行），再做计划级证据评估
    plan = state.get("plan") or []
    if plan:
        cursor: int = int(state.get("cursor", 0))
        # 提前收尾：模型判定证据已充分。执行上等价为"剩余全部被跳过"，
        # 所以这里直接进汇总；**证据闸门仍在 summarize 里**——不足时依旧会
        # replan 补取，因此这不是绕过质量，只是不再执行冗余步骤。
        if state.get("early_finish"):
            return ROUTE_SUMMARIZE
        pending: Optional[int] = next_pending_cursor(
            plan, cursor, state.get("skipped_task_ids") or []
        )
        if pending is not None:
            return ROUTE_EXECUTE
        if _replan_allowed(state):
            return ROUTE_REPLAN
        return ROUTE_SUMMARIZE

    # 4. react 自环（一步未产出 Final Answer 且未达上限）
    react_step: int = int(state.get("react_step", 0))
    max_steps: int = int(state.get("max_steps", 10))
    if react_step < max_steps:
        return ROUTE_EXECUTE

    # 5. 兜底收尾
    return ROUTE_SUMMARIZE


def _replan_allowed(state: AgentGraphState) -> bool:
    """收窄后的重规划触发判定（委托给单一真源 `replan_gate.replan_allowed`）。

    ⚠️ 判定**必须与 `summarize_node` 完全一致**。曾经两处各写一份：summarize
    按"还有余量"就写下不足信号并置空答案，路由按收窄条件拒绝重规划 → 用户拿到
    `answer=""` 且 `success=False`（实测复现：plan_execute 模式下工具全失败）。
    """
    allowed = replan_allowed(state)
    if not allowed and state.get("insufficiency_signal"):
        # 留痕：路由是纯函数、拿不到 tracer，这里落日志。事后可统计有多少请求
        # 被收窄拦下、以及被哪一条规则拦下。
        logger.warning(
            "重规划被排除（{}）：本轮将直接以已有信息诚实收尾。",
            replan_exclusion_reason(state),
        )
    return allowed


def route_after_summarize(state: AgentGraphState) -> str:
    """summarize L3 自判后：仅在**方向性错误**且仍有可用候选时 replan；否则收尾。

    信号由 summarize_node 写入；这里依据结构化的 `insufficiency_kind` 再收窄一次。
    """
    if _replan_allowed(state):
        return ROUTE_REPLAN
    return ROUTE_PERSIST


def route_after_replan(state: AgentGraphState) -> str:
    """replan 后：新计划有效 → execute；replan 失败/返回空 → summarize 失败收尾。"""
    if state.get("last_error") and not (state.get("plan") or []):
        return ROUTE_SUMMARIZE
    if not (state.get("plan") or []):
        return ROUTE_SUMMARIZE
    return ROUTE_EXECUTE


def route_after_reflect(
    state: AgentGraphState, config: Optional[RunnableConfig] = None
) -> str:
    """reflect 后：通过 → summarize；不通过且节点重试未超 → execute；超出 → summarize。"""
    if not state.get("reflect_failed"):
        return ROUTE_SUMMARIZE
    retry_counts: Dict[str, int] = state.get("retry_counts") or {}
    used: int = int(retry_counts.get(REFLECT_RETRY_KEY, 0))
    max_retry: int = _node_retry_max(config)
    if used <= max_retry:
        return ROUTE_EXECUTE
    return ROUTE_SUMMARIZE


def _reflect_enabled(config: Optional[RunnableConfig]) -> bool:
    """从 per-request deps 读 reflect 总开关（默认关，保持线上行为）。"""
    if not config:
        return False
    try:
        return bool(get_deps(config).cfg("agent_reflect_enabled", False))
    except KeyError:
        return False


def _node_retry_max(config: Optional[RunnableConfig]) -> int:
    """从 per-request deps 读节点级重试上限（默认 1）。"""
    if not config:
        return 1
    try:
        return int(get_deps(config).cfg("agent_node_retry_max", 1))
    except KeyError:
        return 1


# ---------------------------------------------------------------------------
# 图构建 / 编译
# ---------------------------------------------------------------------------
def build_agent_graph() -> StateGraph:
    """组装 StateGraph 拓扑（未编译，主要供测试自定义 compile 参数）。"""
    graph: StateGraph = StateGraph(AgentGraphState)

    graph.add_node(NODE_PREPARE, prepare_node)
    graph.add_node(NODE_PLAN, plan_node)
    graph.add_node(NODE_EXECUTE, execute_node)
    graph.add_node(NODE_REPLAN, replan_node)
    graph.add_node(NODE_REFLECT, reflect_node)
    graph.add_node(NODE_SUMMARIZE, summarize_node)
    graph.add_node(NODE_PERSIST, persist_node)

    graph.add_edge(START, NODE_PREPARE)

    graph.add_conditional_edges(
        NODE_PREPARE,
        route_after_prepare,
        {ROUTE_PLAN: NODE_PLAN, ROUTE_EXECUTE: NODE_EXECUTE},
    )
    graph.add_edge(NODE_PLAN, NODE_EXECUTE)

    graph.add_conditional_edges(
        NODE_EXECUTE,
        route_after_execute,
        {
            ROUTE_EXECUTE: NODE_EXECUTE,
            ROUTE_REPLAN: NODE_REPLAN,
            ROUTE_REFLECT: NODE_REFLECT,
            ROUTE_SUMMARIZE: NODE_SUMMARIZE,
        },
    )
    graph.add_conditional_edges(
        NODE_REPLAN,
        route_after_replan,
        {ROUTE_EXECUTE: NODE_EXECUTE, ROUTE_SUMMARIZE: NODE_SUMMARIZE},
    )
    graph.add_conditional_edges(
        NODE_REFLECT,
        route_after_reflect,
        {ROUTE_EXECUTE: NODE_EXECUTE, ROUTE_SUMMARIZE: NODE_SUMMARIZE},
    )

    graph.add_conditional_edges(
        NODE_SUMMARIZE,
        route_after_summarize,
        {ROUTE_REPLAN: NODE_REPLAN, ROUTE_PERSIST: NODE_PERSIST},
    )
    graph.add_edge(NODE_PERSIST, END)
    return graph


def compile_agent_graph(checkpointer: Any = None) -> Any:
    """编译图。生产由 dependencies 单例调用并注入 checkpointer；测试可传 None。"""
    return build_agent_graph().compile(checkpointer=checkpointer)
