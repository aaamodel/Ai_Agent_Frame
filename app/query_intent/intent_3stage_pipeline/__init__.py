"""query_intent.orchestration 子包：Agent 编排模式决策 + Pipeline 总入口。

对外导出主要公共类型，方便外部（app 侧 dependencies / chat 路由）直接
单 import 使用：

    from app.query_intent.orchestration import (
        AgentQueryIntentPipeline,
        ModeDecider,
        OrchestrationModeLiteral,
    )

注意：子包内部（mode_decider / pipeline）对 rewrite/intent_classify_resolver/rag_constant
的引用都使用 query_intent.* 绝对导入路径，避免相对导入导致的包名歧义。
"""

from __future__ import annotations

from app.query_intent.intent_3stage_pipeline.mode_decider import ModeDecider
from app.query_intent.intent_3stage_pipeline.agent_query_intent_pipeline import (
    AgentQueryIntentPipeline,
)
from app.query_intent.intent_dto import OrchestrationModeLiteral

__all__ = [
    "AgentQueryIntentPipeline",
    "ModeDecider",
    "OrchestrationModeLiteral",
]
