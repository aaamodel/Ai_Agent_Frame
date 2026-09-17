# -*- coding: utf-8 -*-
"""reflect 节点（默认关闭，路由仅在 agent_reflect_enabled=True 时可达）。

包装现有 ReflectionAgent：
- 通过：保持 final_answer，reflect_failed=False；
- 不通过且 react 形态：计数 + 置 reflect_failed，条件边回 execute 重做
  （审查建议作为系统消息注入下一轮 FC，文本路径下轮 prompt 经由 steps 可见）；
- 不通过但 plan 形态（execute 无重做目标）/ 反思自身异常：不回炉，标记 degraded
  后照常 summarize，审查意见写 steps 留痕。
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.core.agent.graph.deps import get_deps
from app.core.agent.graph.nodes._common import emit_step, trace_event
from app.core.agent.graph.state import REFLECT_RETRY_KEY, AgentGraphState
from app.core.agent.reflection import ReflectionAgent


async def reflect_node(state: AgentGraphState, config: RunnableConfig) -> dict:
    deps = get_deps(config)
    trace_id: str = state["trace_id"]

    answer: str = state.get("final_answer") or ""
    if not answer:
        return {"reflect_failed": False}

    # 证据片段：工具观测 + 子任务结论（截断防 prompt 膨胀）
    evidence: List[str] = []
    for step in state.get("steps") or []:
        record = step.get("record", step)
        obs = record.get("observation") if isinstance(record, dict) else None
        if obs:
            evidence.append(str(obs)[:1500])
        if len(evidence) >= 5:
            break

    agent = ReflectionAgent(
        model_router=deps.model_router,
        min_quality_to_pass=int(deps.cfg("agent_reflect_min_score", 60)),
        purpose_hint="reflection",
    )
    report = await agent.reflect(
        user_query=state["user_input"],
        agent_answer=answer,
        evidence_snippets=evidence or None,
    )
    decision = agent.should_retry_or_warn(report)

    trace_event(
        deps.tracer, trace_id, "reflect.done",
        {"quality_score": report.quality_score, "decision": decision,
         "parse_error": report.parse_error},
    )

    # 反思流程自身异常（保守评分）：不回炉，避免无限循环
    if report.parse_error:
        logger.warning("反思阶段异常，跳过回炉直接收尾: {}", report.parse_error)
        return {"reflect_failed": False, "degraded": True}

    needs_retry: bool = bool(decision.get("suggest_retry"))
    if not needs_retry:
        return {"reflect_failed": False}

    review_note = (
        f"【质量审查未通过（score={report.quality_score}）】{report.summary}\n"
        f"改进建议：{'；'.join(report.suggestions) or '请复核事实与完整性'}"
    )

    # plan 形态没有可重做的 execute 目标：带原答案收尾，仅留痕 + degraded
    if state.get("plan"):
        rec = {"step": "reflect", "phase": "reflect", "review": review_note, "final": True}
        return {**emit_step(rec), "reflect_failed": False, "degraded": True}

    # react 形态：计数 + 注入审查建议，回 execute 自环重做
    retry_counts: Dict[str, int] = dict(state.get("retry_counts") or {})
    retry_counts[REFLECT_RETRY_KEY] = int(retry_counts.get(REFLECT_RETRY_KEY, 0)) + 1

    # 告知模型上一版答案的问题（FC 持久化消息；清空 final_answer 以便重做后重新判定）
    nudge_msg = {
        "role": "user",
        "content": (
            f"{review_note}\n请基于已有工具结果修正你的回答，"
            "需要补数据时继续调用工具，完成后重新输出 Final Answer。"
        ),
    }
    rec = {"step": "reflect", "phase": "reflect", "review": review_note}
    logger.info("reflect 未通过（第 {} 次），回 execute 重做。",
                retry_counts[REFLECT_RETRY_KEY])
    return {
        **emit_step(rec),
        "reflect_failed": True,
        "retry_counts": retry_counts,
        "final_answer": "",
        "success": False,
        "degraded": True,
        "react_messages": [nudge_msg] if state.get("react_protocol") == "fc" else [],
    }
