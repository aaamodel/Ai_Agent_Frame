# -*- coding: utf-8 -*-
"""GraphRunner：StateGraph 门面适配层（供 AgentOrchestrator / 审批恢复 API 共用）。

职责：
1. 把旧编排参数映射为图入参（mode/intent → should_plan/mode_source）；
2. 以 ``astream`` 驱动图（首次传初始 state，resume 传 ``Command(resume=...)``）；
3. 暂停判定**只以** checkpointer 快照 ``next`` + ``tasks[].interrupts[]`` 为准，
   最终状态**只从** ``snapshot.values`` 拷贝，不做 chunk 累积；
4. 把快照 values 映射回旧门面契约 ``AgentResponse``（字段全兼容）。

节点需要的每请求依赖（ToolRegistry/Memory/Skill 等）经 RunnableConfig
``configurable["deps"]`` 注入，永不进入 checkpoint（见 deps.py）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger

from app.core.agent.graph.checkpoint import (
    aget_snapshot,
    is_paused_for_approval,
    snapshot_exists,
    snapshot_interrupts,
    snapshot_values,
)
from app.core.agent.graph.deps import GraphDeps, with_deps
from app.core.agent.graph.state import dict_to_intent, make_initial_state

if TYPE_CHECKING:  # 避免与 orchestrator → runner 的顶层循环导入
    from app.core.agent.orchestrator import AgentResponse, IntentContext

MODE_PLAN_EXECUTE = "plan_execute"
MODE_REACT = "react"


@dataclass
class RunOutcome:
    """一次图驱动结果。

    正常完成：``paused=False``、response 为最终响应；
    暂停等审批：``paused=True``、response.success=False 且 awaiting_approval=True
    （携带 approval_payloads，供 SSE 推 awaiting_approval 事件）。
    """

    run_id: str
    response: "Optional[AgentResponse]" = None
    paused: bool = False
    approval_payloads: Optional[List[Dict[str, Any]]] = None


class GraphRunner:
    """编译后图的驱动器（无状态；checkpointer 持有持久化）。"""

    def __init__(self, compiled_graph: Any, checkpointer: Any = None) -> None:
        self._graph: Any = compiled_graph
        self._checkpointer: Any = checkpointer

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    async def run(
        self,
        *,
        deps: GraphDeps,
        user_input: str,
        session_id: str,
        mode: str = MODE_REACT,
        intent: Optional[IntentContext] = None,
        precomputed_memory: Optional[Any] = None,
    ) -> RunOutcome:
        """首次运行一条 Agent 图。"""
        from app.core.agent.orchestrator import AgentResponse, IntentContext

        intent_context: IntentContext = intent or IntentContext()

        # 与旧 orchestrator.run 一致：intent.preferred_mode 覆盖入参 mode
        if intent_context.preferred_mode:
            effective_mode: str = str(intent_context.preferred_mode)
            mode_source: str = "intent"
        else:
            effective_mode = mode
            mode_source = "strategy"
        should_plan: bool = effective_mode == MODE_PLAN_EXECUTE

        trace_id: str = deps.tracer.new_trace_id()
        run_id: str = f"{session_id}:{uuid4().hex}"

        execution_span: Any = deps.tracer.start_span(
            name="orchestrator.run",
            trace_id=trace_id,
            attributes={
                "session_id": session_id,
                "mode": effective_mode,
                "intent_classify_resolver": intent_context.intent,
                "run_id": run_id,
            },
        )
        deps.tracer.log_event(
            trace_id, "orchestrator.start",
            {"user_input_len": len(user_input), "mode": effective_mode, "run_id": run_id},
        )

        initial_state: Dict[str, Any] = make_initial_state(
            run_id=run_id,
            session_id=session_id,
            trace_id=trace_id,
            user_input=user_input,
            intent=intent_context,
            should_plan=should_plan,
            mode_source=mode_source,
            precomputed_memory=precomputed_memory,
        )
        config: Dict[str, Any] = with_deps(run_id, deps)

        try:
            # stream_mode="updates"：每个节点产出一份 partial，天然适合未来接 SSE
            async for _chunk in self._graph.astream(
                initial_state, config=config, stream_mode="updates"
            ):
                pass  # 节点 trace 已由各自埋点写入 Tracer；此处不累积状态
        except Exception as system_uncaught_exception:  # noqa: BLE001 - 门面兜底契约
            logger.exception("状态图执行发生未捕获异常")
            deps.tracer.end_span(execution_span, error=system_uncaught_exception)
            return RunOutcome(
                run_id=run_id,
                response=AgentResponse(
                    answer="",
                    mode_used=MODE_REACT,
                    success=False,
                    trace_id=trace_id,
                    intent=intent_context,
                    steps=[],
                    error=str(system_uncaught_exception),
                    run_id=run_id,
                ),
            )

        outcome: RunOutcome = await self._build_outcome(run_id, config, intent_context)
        deps.tracer.end_span(execution_span, error=None)
        return outcome

    async def resume(
        self,
        *,
        run_id: str,
        deps: GraphDeps,
        approved: bool,
        comment: str = "",
    ) -> RunOutcome:
        """审批后从 interrupt 点恢复（Command(resume=...) 注入闸门决策）。"""
        from langgraph.types import Command

        from app.core.agent.orchestrator import AgentResponse

        config: Dict[str, Any] = with_deps(run_id, deps)
        snapshot: Any = await aget_snapshot(self._graph, run_id)
        if not snapshot_exists(snapshot):
            raise KeyError(f"未找到 run_id={run_id} 的检查点（可能已过期或后端已重启为内存模式）")
        values: Dict[str, Any] = snapshot_values(snapshot)
        trace_id: str = values.get("trace_id", "")
        intent_context: IntentContext = dict_to_intent(values.get("intent") or {})

        span: Any = deps.tracer.start_span(
            name="orchestrator.resume",
            trace_id=trace_id,
            attributes={"run_id": run_id, "approved": approved},
        )
        try:
            resume_value: Dict[str, Any] = {"approved": bool(approved), "comment": str(comment or "")}
            async for _chunk in self._graph.astream(
                Command(resume=resume_value), config=config, stream_mode="updates"
            ):
                pass
        except Exception as resume_exception:  # noqa: BLE001
            logger.exception("审批恢复执行异常")
            deps.tracer.end_span(span, error=resume_exception)
            return RunOutcome(
                run_id=run_id,
                response=AgentResponse(
                    answer="",
                    mode_used=str(values.get("mode_used") or MODE_REACT),
                    success=False,
                    trace_id=trace_id,
                    intent=intent_context,
                    steps=list(values.get("steps") or []),
                    error=str(resume_exception),
                    degraded=bool(values.get("degraded")),
                    run_id=run_id,
                ),
            )

        outcome: RunOutcome = await self._build_outcome(run_id, config, intent_context)
        deps.tracer.end_span(span, error=None)
        return outcome

    # ------------------------------------------------------------------
    async def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """供 GET /agent/runs/{run_id}：快照状态 + interrupt 载荷。"""
        snapshot: Any = await aget_snapshot(self._graph, run_id)
        if not snapshot_exists(snapshot):
            return {"run_id": run_id, "exists": False, "paused": False, "interrupts": []}
        values: Dict[str, Any] = snapshot_values(snapshot)
        return {
            "run_id": run_id,
            "exists": True,
            "paused": is_paused_for_approval(snapshot),
            "next": list(snapshot.next or ()),
            "interrupts": snapshot_interrupts(snapshot),
            "mode_used": values.get("mode_used"),
            "success": values.get("success"),
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    async def _build_outcome(
        self,
        run_id: str,
        config: Dict[str, Any],
        intent_context: IntentContext,
    ) -> RunOutcome:
        """图驱动结束后：快照判停（暂停→审批载荷；否则→AgentResponse）。"""
        snapshot: Any = await aget_snapshot(self._graph, run_id)
        values: Dict[str, Any] = snapshot_values(snapshot) if snapshot_exists(snapshot) else {}

        if snapshot is not None and is_paused_for_approval(snapshot):
            payloads: List[Dict[str, Any]] = snapshot_interrupts(snapshot)
            return RunOutcome(
                run_id=run_id,
                paused=True,
                approval_payloads=payloads,
                response=self._map_response(
                    values, intent_context, awaiting_approval=True, approval_payloads=payloads
                ),
            )

        # 最终状态一律以快照 values 为准（HITL 铁律）
        return RunOutcome(run_id=run_id, response=self._map_response(values, intent_context))

    @staticmethod
    def _map_response(
        values: Dict[str, Any],
        intent_context: IntentContext,
        *,
        awaiting_approval: bool = False,
        approval_payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentResponse:
        """快照 values → 旧门面 AgentResponse（answer/mode_used/steps 等字段契约不变）。"""
        from app.core.agent.orchestrator import AgentResponse

        return AgentResponse(
            answer=str(values.get("final_answer") or ""),
            mode_used=str(values.get("mode_used") or MODE_REACT),  # type: ignore[arg-type]
            success=bool(values.get("success")),
            trace_id=str(values.get("trace_id") or ""),
            intent=intent_context,
            steps=list(values.get("steps") or []),
            error=values.get("last_error") if not values.get("success") else None,
            degraded=bool(values.get("degraded")),
            run_id=str(values.get("run_id") or ""),
            awaiting_approval=awaiting_approval,
            approval_payloads=list(approval_payloads or []),
        )
