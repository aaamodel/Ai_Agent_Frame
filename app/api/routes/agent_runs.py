# -*- coding: utf-8 -*-
"""Agent 运行态 API（2.2 标准档 HITL）：

- GET  /agent/runs/{run_id}            查询运行快照（是否暂停/下一节点/interrupt 载荷）
- POST /agent/runs/{run_id}/approval   审批决策后从 interrupt 点恢复（SSE 续流）

恢复后的事件协议与 /chat/with_agent 收尾段一致：content 分片 → done；
若续跑中再次命中危险工具，会再次推 awaiting_approval 事件（同一 run_id 继续审批）。
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.api.depends.dependencies import (
    get_agent_config,
    get_agent_graph_runner,
    get_memory_manager,
    get_model_router,
    get_skill_manager,
    get_tool_registry,
)
from app.api.routes.chat import _sse_payload, _stream_final_answer
from app.core.agent.graph.deps import GraphDeps
from app.core.memory.manager import MemoryManager
from app.core.skill.manager import SkillManager
from app.core.tools.registry import ToolRegistry
from app.infrastructure.trace.tracer import Tracer
from app.llm_model_router.model_router import ModelRouter

router = APIRouter(tags=["agent-runs"])

# 与 chat.py 一致的进程级 Tracer（无状态，直接实例化）
_tracer = Tracer()


class ApprovalDecision(BaseModel):
    """审批恢复请求体。"""

    approved: bool = Field(..., description="是否批准本次危险工具调用")
    comment: str = Field(default="", description="审批意见（拒绝时回注给模型）")


@router.get("/agent/runs/{run_id}")
async def get_agent_run(
    run_id: str,
    agent_graph_runner: Any = Depends(get_agent_graph_runner),
) -> Dict[str, Any]:
    """查询 run 快照：paused/next/interrupts/mode_used/success。"""
    status: Dict[str, Any] = await agent_graph_runner.get_run_status(run_id)
    if not status.get("exists"):
        # 快照不存在（TTL 过期/内存后端重启）：404 语义由 status 字段表达，不抛 500
        return status
    return status


@router.post("/agent/runs/{run_id}/approval")
async def resume_agent_run(
    run_id: str,
    decision: ApprovalDecision,
    agent_config: Dict[str, Any] = Depends(get_agent_config),
    agent_graph_runner: Any = Depends(get_agent_graph_runner),
    model_router: ModelRouter = Depends(get_model_router),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    tool_registry: ToolRegistry = Depends(get_tool_registry),
    skill_manager: SkillManager = Depends(get_skill_manager),
) -> StreamingResponse:
    """审批恢复并以 SSE 续流（协议与 /chat/with_agent 的收尾段一致）。"""

    async def _resume_stream() -> Any:
        # 每请求依赖重建（工具 registry 等无状态运行态以 checkpoint state 为准）
        deps = GraphDeps(
            config=agent_config,
            model_router=model_router,
            memory=memory_manager,
            tools=tool_registry,
            skill_manager=skill_manager,
            tracer=_tracer,
        )
        try:
            outcome = await agent_graph_runner.resume(
                run_id=run_id,
                deps=deps,
                approved=decision.approved,
                comment=decision.comment,
            )
        except KeyError as missing_error:
            logger.warning("审批恢复失败（检查点不存在）: {}", missing_error)
            yield _sse_payload({"error": str(missing_error), "code": "run_not_found"})
            return
        except Exception as resume_error:  # noqa: BLE001 - SSE 内不抛 500
            logger.exception("审批恢复执行异常")
            yield _sse_payload({"error": str(resume_error)})
            return

        result = outcome.response

        # 续跑中再次命中危险工具：继续挂起，等待下一轮审批
        if getattr(result, "awaiting_approval", False):
            yield _sse_payload({
                "awaiting_approval": True,
                "run_id": outcome.run_id,
                "trace_id": result.trace_id,
                "approvals": getattr(result, "approval_payloads", []),
                "done": True,
                "status": "awaiting_approval",
            })
            return

        if not result.success:
            graceful_answer: str = (result.answer or "").strip()
            if graceful_answer:
                async for chunk in _stream_final_answer(graceful_answer, result.trace_id):
                    yield chunk
                yield _sse_payload({
                    "done": True,
                    "status": "degraded",
                    "trace_id": result.trace_id,
                    "run_id": outcome.run_id,
                    "steps_executed": len(result.steps),
                    "degraded": True,
                })
                return
            yield _sse_payload({"error": result.error or "审批恢复后执行失败"})
            return

        async for chunk in _stream_final_answer(result.answer, result.trace_id):
            yield chunk
        yield _sse_payload({
            "done": True,
            "status": "degraded" if result.degraded else "success",
            "trace_id": result.trace_id,
            "run_id": outcome.run_id,
            "steps_executed": len(result.steps),
            "degraded": bool(result.degraded),
        })

    return StreamingResponse(
        _resume_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
