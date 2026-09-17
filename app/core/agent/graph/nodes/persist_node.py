# -*- coding: utf-8 -*-
"""persist 节点：记忆双写（从 orchestrator 成功分支 L439-L458 平移）。

- 成功且有 final_answer：短期对话 append_turn(user/assistant) 同步写，当轮可见；
- 长期向量记忆 _ltm.store 后台任务（deps 强引用登记，异常仅告警）；
- 失败/无答案不写；断点 resume 不会重入已完成节点，天然避免重复写。
"""

from __future__ import annotations

from typing import Any, List

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.agent.graph.deps import get_deps
from app.core.agent.graph.nodes._common import trace_event
from app.core.agent.graph.state import AgentGraphState
from app.core.memory.long_term import build_ltm_digest


def _called_tools_from_steps(steps: Any) -> List[str]:
    """从 steps 里提取真实调用过的工具名（兼容 react 与 plan 两种记录形态）。

    作为长期记忆的 sidecar 字段落库，支持"按工具过滤历史记忆"这类结构化召回；
    取不到就返回空列表，绝不影响主链路。
    """
    names: List[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        record: Any = step.get("record") if isinstance(step.get("record"), dict) else step
        if not isinstance(record, dict):
            continue
        name: Any = record.get("action") or record.get("tool_name")
        if name:
            names.append(str(name))
    return list(dict.fromkeys(names))


async def persist_node(state: AgentGraphState, config: RunnableConfig) -> dict:
    deps = get_deps(config)
    trace_id: str = state["trace_id"]
    session_id: str = state["session_id"]
    user_input: str = state["user_input"]
    final_answer: str = state.get("final_answer") or ""
    success: bool = bool(state.get("success"))
    mode_used: str = state.get("mode_used") or "react"

    if not (success and final_answer):
        logger.info("persist 跳过记忆写入（success={}，answer_len={}）", success, len(final_answer))
        return {}

    # 1. 短期历史同步写（Redis 低延迟，必须在响应前落库）
    try:
        await deps.memory.append_turn(session_id, "user", user_input)
        await deps.memory.append_turn(
            session_id=session_id,
            role="assistant",
            content=final_answer,
            metadata={"mode": mode_used, "trace_id": trace_id},
        )
    except Exception as sync_memory_error:  # noqa: BLE001
        logger.warning("智能体短期对话历史录入时发生异常: {}", sync_memory_error)

    # 2. 长期向量记忆后台沉淀（embedding + Milvus 高延迟，不阻塞响应）
    ltm = getattr(deps.memory, "_ltm", None)
    if ltm is not None and hasattr(ltm, "store"):
        # 只沉淀「问题全文 + 规则判定的结论句 + 结构化 sidecar」，不落整段答案：
        # 这条记录会被注入**每一轮** system 提示词，存全文会让单轮 token 随会话数膨胀。
        # 用确定性句级选择而非再调一次 LLM 做摘要——理由见 build_ltm_digest 的 docstring。
        digest = build_ltm_digest(
            user_input,
            final_answer,
            mode=mode_used,
            trace_id=trace_id,
            intent=str((state.get("intent") or {}).get("intent") or "") or None,
            tools=_called_tools_from_steps(state.get("steps")),
        )
        deps.spawn_background_task(
            ltm.store(
                session_id=session_id,
                content=digest.content,
                metadata=digest.metadata,
            )
        )
        logger.info("已将当前对话对提交为长期向量记忆后台沉淀任务（不阻塞响应主链路）。")

    trace_event(deps.tracer, trace_id, "memory.persisted", {"mode": mode_used})
    return {}
