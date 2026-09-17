# -*- coding: utf-8 -*-
"""Agent 状态图包（LangGraph StateGraph 重构 2.2）。

- ``state``：``AgentGraphState``（TypedDict + reducer）与预算账本适配。
- ``builder``：StateGraph 编译与条件路由纯函数。
- ``runner``：门面适配（供 ``AgentOrchestrator.run`` 调用）。

运行时依赖（registry / model_router / memory / skill / tracer / config）
只走 ``RunnableConfig["configurable"]["deps"]``，不进入 state、不被 checkpoint 序列化。
"""

from app.core.agent.graph.state import (
    NODE_EXECUTE,
    NODE_PLAN,
    NODE_PREPARE,
    NODE_REFLECT,
    NODE_REPLAN,
    NODE_PERSIST,
    NODE_SUMMARIZE,
    AgentGraphState,
    ROUTE_EXECUTE,
    ROUTE_PLAN,
    ROUTE_REFLECT,
    ROUTE_REPLAN,
    ROUTE_SUMMARIZE,
    budget_from_ledger,
    budget_to_ledger,
    dict_to_intent,
    intent_to_dict,
    make_initial_state,
)

__all__ = [
    "AgentGraphState",
    "NODE_PREPARE",
    "NODE_PLAN",
    "NODE_EXECUTE",
    "NODE_REPLAN",
    "NODE_REFLECT",
    "NODE_SUMMARIZE",
    "NODE_PERSIST",
    "ROUTE_PLAN",
    "ROUTE_EXECUTE",
    "ROUTE_REPLAN",
    "ROUTE_REFLECT",
    "ROUTE_SUMMARIZE",
    "make_initial_state",
    "intent_to_dict",
    "dict_to_intent",
    "budget_to_ledger",
    "budget_from_ledger",
]
