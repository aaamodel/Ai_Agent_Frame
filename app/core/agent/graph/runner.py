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
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from loguru import logger

from app.core.agent.graph.checkpoint import (
    aget_snapshot,
    alist_thread_ids,
    checkpointer_backend_name,
    is_paused_for_approval,
    snapshot_exists,
    snapshot_interrupts,
    snapshot_values,
)
from app.core.agent.graph.deps import GraphDeps, with_deps
from app.core.agent.graph.approval import RESUME_APPROVALS_CONFIG_KEY
from app.core.agent.graph.state import dict_to_intent, make_initial_state

# Langfuse 可观测性：审批恢复路由（agent_runs.py）直接调用 GraphRunner.resume，
# 不经过 orchestrator 的 @observe 根；这里为恢复续跑补一条独立根 trace，
# 使续跑期间的 llm.* generation / tool span 仍然聚合（未配置密钥时为 no-op）。
from langfuse import observe as langfuse_observe

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


# ---------------------------------------------------------------------------
# astream updates chunk → 可推送的步骤事件
# ---------------------------------------------------------------------------
# 步骤记录有两种历史形态（见 nodes/_common.py）：
#   - react：{"ts":..., **rec}                        （emit_step）
#   - plan ：{"ts":..., "phase":..., "record": rec}   （emit_plan_step）
# 这里统一摊平成一份给前端的载荷，前端不必知道这两种形态。
_STEP_DETAIL_MAX_CHARS: int = 5000
"""单条步骤详情（工具观测）的推送上限。

5000 而不是更小：工具观测常常就是用户想看的原始证据（检索片段、SQL 结果表），
截太狠等于没推。前端默认只渲染前两行，点「展开」才铺开全文，所以长度本身
不构成视觉负担——只是 SSE 体积，而 5000 字符 × 几步仍在可接受范围。

⚠️ 注意这里**只是推送上限**，不是把观测压缩到 5000——观测本身可能有 8000 字符
（`execute_node` 自己就有 8000 的截断），所以前端拿到的仍可能是被这里截过的。
"""


def _brief(text: Any) -> str:
    """步骤详情截断：工具观测可能是几千字符的表格，超限才截。"""
    s: str = str(text or "").strip()
    if len(s) <= _STEP_DETAIL_MAX_CHARS:
        return s
    return s[:_STEP_DETAIL_MAX_CHARS] + "\n…（内容过长，已截断）"


def _step_tool(inner: Dict[str, Any]) -> Optional[str]:
    """工具名：plan 路径写 ``tool_name``，react 文本协议写 ``action``。"""
    name: Any = inner.get("tool_name") or inner.get("action")
    return str(name) if name else None


def _step_status(inner: Dict[str, Any]) -> str:
    """步骤状态。

    ⚠️ react 记录**没有** status 字段，成败散落在几个布尔标记位里，
    必须在这里统一推导，否则前端只会看到一片 "ok"。
    """
    explicit: Any = inner.get("status")
    if explicit:
        return str(explicit)
    if inner.get("final"):
        # "直接给最终答案"这一轮，不是工具调用
        return "final"
    if inner.get("approval_denied"):
        return "approval_denied"
    if inner.get("budget_denied"):
        return "budget_denied"
    if inner.get("error"):
        return "error"
    return "ok"


def _step_payloads(chunk: Any) -> List[Dict[str, Any]]:
    """从一个 ``stream_mode="updates"`` chunk 里抽取步骤事件载荷。

    chunk 形如 ``{"execute": {"steps": [rec, ...], ...}}``。
    """
    if not isinstance(chunk, dict):
        return []
    payloads: List[Dict[str, Any]] = []
    for node_name, update in chunk.items():
        if not isinstance(update, dict):
            continue
        for rec in (update.get("steps") or []):
            if not isinstance(rec, dict):
                continue
            # plan 路径把真实记录套在 "record" 里
            inner: Dict[str, Any] = (
                rec["record"] if isinstance(rec.get("record"), dict) else rec
            )
            payloads.append({
                "node": str(node_name),
                "tool": _step_tool(inner),
                "title": str(inner.get("title") or inner.get("subtask_id") or ""),
                "status": _step_status(inner),
                "detail": _brief(inner.get("observation") or ""),
            })
    return payloads


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
        """首次运行一条 Agent 图（一次性返回终局；步骤事件在此被丢弃）。

        ⚠️ 与改造前**完全等价**：原先 inline 的 astream 循环搬到了
        :meth:`run_stream`，这里只负责把它抽干并留下终局。
        需要"过程可见"的调用方（SSE）请改用 :meth:`run_stream`。
        """
        outcome: Optional[RunOutcome] = None
        async for event in self.run_stream(
            deps=deps,
            user_input=user_input,
            session_id=session_id,
            mode=mode,
            intent=intent,
            precomputed_memory=precomputed_memory,
        ):
            if event.get("type") == "outcome":
                outcome = event.get("outcome")
        if outcome is None:  # 不应发生：run_stream 契约保证必产出一个终局事件
            raise RuntimeError("run_stream 未产出终局事件")
        return outcome

    async def run_stream(
        self,
        *,
        deps: GraphDeps,
        user_input: str,
        session_id: str,
        mode: str = MODE_REACT,
        intent: Optional[IntentContext] = None,
        precomputed_memory: Optional[Any] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """首次运行一条 Agent 图，**边跑边产出步骤事件**。

        产出两类事件：

        - ``{"type": "step", ...}``：节点产出的每条步骤记录（工具调用 / 子任务结论）
        - ``{"type": "outcome", "outcome": RunOutcome}``：终局（正常完成 / 暂停等审批）

        ⚠️ 步骤事件产生在**节点产出之后**。LangGraph 的 ``updates`` 模式给的是
        节点返回值，所以能推"某步已完成"，推不了"即将调用某个工具"——后者
        要节点内部用 ``get_stream_writer`` 主动推，会侵入 execute 节点，而那里
        的结构化输出与多重闸门（写操作 / 证据 / 重规划）不能冒险改动。
        """
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
            # stream_mode="updates"：每个节点产出一份 partial，天然适合接 SSE
            async for chunk in self._graph.astream(
                initial_state, config=config, stream_mode="updates"
            ):
                # 节点 trace 已由各自埋点写入 Tracer；此处只抽取可推送的步骤事件，
                # 最终状态仍一律以快照为准（HITL 铁律），不做 chunk 累积。
                for payload in _step_payloads(chunk):
                    yield {"type": "step", **payload}
        except Exception as system_uncaught_exception:  # noqa: BLE001 - 门面兜底契约
            logger.exception("状态图执行发生未捕获异常")
            deps.tracer.end_span(execution_span, error=system_uncaught_exception)
            yield {
                "type": "outcome",
                "outcome": RunOutcome(
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
                ),
            }
            return

        outcome: RunOutcome = await self._build_outcome(run_id, config, intent_context)
        deps.tracer.end_span(execution_span, error=None)
        yield {"type": "outcome", "outcome": outcome}

    @langfuse_observe(name="AgentOrchestrator.resume", as_type="agent", capture_input=False, capture_output=False)
    async def resume(
        self,
        *,
        run_id: str,
        deps: GraphDeps,
        approved: bool,
        comment: str = "",
    ) -> RunOutcome:
        """审批后从 interrupt 点恢复（一次性返回终局；步骤事件在此被丢弃）。"""
        outcome: Optional[RunOutcome] = None
        async for event in self.resume_stream(
            run_id=run_id, deps=deps, approved=approved, comment=comment
        ):
            if event.get("type") == "outcome":
                outcome = event.get("outcome")
        if outcome is None:
            raise RuntimeError("resume_stream 未产出终局事件")
        return outcome

    async def resume_stream(
        self,
        *,
        run_id: str,
        deps: GraphDeps,
        approved: bool,
        comment: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        """审批后从 interrupt 点恢复，**边跑边产出步骤事件**（契约同 run_stream）。"""
        from langgraph.types import Command

        from app.core.agent.orchestrator import AgentResponse

        config: Dict[str, Any] = with_deps(run_id, deps)
        snapshot: Any = await aget_snapshot(self._graph, run_id)
        if not snapshot_exists(snapshot):
            raise KeyError(f"未找到 run_id={run_id} 的检查点（可能已过期或后端已重启为内存模式）")
        # 把待处理审批载荷（含**完整**入参，非 2000 字截断预览）注入 config：
        # 节点重放时 plan 路径据此短路参数重解析，杜绝 FC 重跑漂移导致
        # "审批卡片字段 ≠ 实际执行字段"（2026-09-23 实测事故）。
        config.setdefault("configurable", {})[RESUME_APPROVALS_CONFIG_KEY] = (
            snapshot_interrupts(snapshot)
        )
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
            async for chunk in self._graph.astream(
                Command(resume=resume_value), config=config, stream_mode="updates"
            ):
                for payload in _step_payloads(chunk):
                    yield {"type": "step", **payload}
        except Exception as resume_exception:  # noqa: BLE001
            logger.exception("审批恢复执行异常")
            deps.tracer.end_span(span, error=resume_exception)
            yield {
                "type": "outcome",
                "outcome": RunOutcome(
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
                ),
            }
            return

        outcome: RunOutcome = await self._build_outcome(run_id, config, intent_context)
        deps.tracer.end_span(span, error=None)
        yield {"type": "outcome", "outcome": outcome}

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
    async def list_paused_runs(self, *, limit: int = 50) -> Dict[str, Any]:
        """供 GET /agent/approvals/pending：列出全部暂停等待人工审批的 run。

        枚举 checkpointer 中的线程，逐个读最新快照，只保留
        ``next 非空且存在未处理 interrupt`` 的 run。每条带区分度摘要
        （原始问题/会话/trace/模式/计划进度/暂停时间）与审批内容精简版，
        避免"多个图审批内容相同无法区分"。
        """
        backend: str = checkpointer_backend_name(self._checkpointer)
        if self._checkpointer is None:
            return {"backend": "none", "count": 0, "items": []}

        items: List[Dict[str, Any]] = []
        for run_id in await alist_thread_ids(self._checkpointer):
            try:
                snapshot: Any = await aget_snapshot(self._graph, run_id)
            except Exception:  # noqa: BLE001 — 单条损坏不拖垮整个列表
                logger.warning("读取 run 快照失败，已跳过: {}", run_id)
                continue
            if not snapshot_exists(snapshot) or not is_paused_for_approval(snapshot):
                continue

            values: Dict[str, Any] = snapshot_values(snapshot)
            intent: Dict[str, Any] = values.get("intent") or {}
            plan = values.get("plan") or []
            cursor = int(values.get("cursor") or 0)
            created_at = getattr(snapshot, "created_at", None)
            # langgraph 各版本 created_at 类型不一（datetime 或 ISO 字符串），统一成字符串
            paused_at = created_at.isoformat() if hasattr(created_at, "isoformat") else created_at

            # 审批载荷：列表只给预览（完整 arguments 在 GET /agent/runs/{run_id}）
            approvals: List[Dict[str, Any]] = []
            for payload in snapshot_interrupts(snapshot):
                approvals.append({
                    "type": payload.get("type", "tool_approval"),
                    "tool_name": payload.get("tool_name", ""),
                    "arguments_preview": payload.get("arguments_preview", ""),
                    "source": payload.get("source", ""),
                    "subtask_id": payload.get("subtask_id"),
                    "thought": payload.get("thought"),
                })

            session_id = values.get("session_id") or str(run_id).split(":", 1)[0]
            items.append({
                "run_id": run_id,
                "session_id": session_id,
                "trace_id": values.get("trace_id", ""),
                "user_input": values.get("user_input", ""),
                "intent": str(intent.get("intent") or "") if isinstance(intent, dict) else "",
                "mode_used": values.get("mode_used") or MODE_REACT,
                "steps_executed": len(values.get("steps") or []),
                # plan 路径下的进度（react 路径 plan 为空 → 不给该字段）
                "plan_progress": (
                    {"cursor": cursor, "total": len(plan)} if plan else None
                ),
                "next": list(snapshot.next or ()),
                "paused_at": paused_at,
                "approvals": approvals,
            })

        # 最新暂停的排最前（created_at 缺失的沉底）
        items.sort(key=lambda item: item.get("paused_at") or "", reverse=True)
        return {"backend": backend, "count": len(items), "items": items[:limit]}

    async def delete_session_runs(self, session_id: str) -> int:
        """删除某会话在 checkpointer 里的全部 run，返回删除条数（best-effort）。

        为什么必须做：``run_id`` 是 ``f"{session_id}:{uuid}"``，所以一个会话可能
        留下多个 run（每轮对话一个）。只删记忆不删检查点的话，该会话挂起的审批
        仍会被 ``/agent/approvals/pending`` 列出来——用户已经删了会话，却还能看到
        一个点不进去的幽灵待审批项，而且它永远悬着。

        ⚠️ 检查点删除失败**不抛出**：它属于清理性收尾，不该让"会话删除"这个
        动作整体失败（记忆已经清了，那才是主要目的）。失败计入返回值之外仅打日志。
        """
        if self._checkpointer is None:
            return 0

        deleter: Any = getattr(self._checkpointer, "adelete_thread", None)
        if deleter is None:
            # 未知 checkpointer 实现：如实告警，不做假装成功的清理
            logger.warning(
                "当前 checkpointer 不支持 adelete_thread，会话的检查点未清理: {}",
                session_id,
            )
            return 0

        prefix: str = f"{session_id}:"
        removed: int = 0
        try:
            thread_ids: List[Any] = await alist_thread_ids(self._checkpointer)
        except Exception:  # noqa: BLE001 - 枚举失败同样只告警
            logger.warning("枚举检查点线程失败，跳过会话检查点清理: {}", session_id)
            return 0

        for run_id in thread_ids:
            if not str(run_id).startswith(prefix):
                continue
            try:
                await deleter(run_id)
                removed += 1
            except Exception:  # noqa: BLE001 - 单条失败不影响其余
                logger.warning("删除检查点失败，已跳过: {}", run_id)
        return removed

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
