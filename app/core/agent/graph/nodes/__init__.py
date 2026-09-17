# -*- coding: utf-8 -*-
"""状态图控制流节点（7 个）。

prepare  → 技能刷新/gating、工具 schema/FC 定义、预算账本、记忆 chips、提示词组装
plan     → PlannerAgent.plan()（复用现有规划资产）
execute  → 单步激活（plan 子任务 / react 单轮），危险工具 interrupt 闸门
replan   → PlannerAgent.replan()（空数据换源语义）
reflect  → ReflectionAgent 质量门（默认关闭）
summarize→ 子任务结论汇总 / Final Answer 归一
persist  → 短期记忆同步写 + 长期记忆后台沉淀

工具**不是节点**：全部在 execute 内经 ToolRegistry 动态分发。
"""

from app.core.agent.graph.nodes.prepare_node import prepare_node
from app.core.agent.graph.nodes.plan_node import plan_node
from app.core.agent.graph.nodes.execute_node import execute_node
from app.core.agent.graph.nodes.replan_node import replan_node
from app.core.agent.graph.nodes.reflect_node import reflect_node
from app.core.agent.graph.nodes.summarize_node import summarize_node
from app.core.agent.graph.nodes.persist_node import persist_node

__all__ = [
    "prepare_node",
    "plan_node",
    "execute_node",
    "replan_node",
    "reflect_node",
    "summarize_node",
    "persist_node",
]
