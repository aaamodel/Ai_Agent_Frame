# -*- coding: utf-8 -*-
"""对话 API：非流式、流式输出以及 Agent 编排。
所在文件目录：app/api/routes/chat.py"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, List, Optional, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.chat_recognizer.recognizer import IntentRecognizer

from app.llm_model_router.model_router import ModelRouter
from app.llm_model_router.model_router_config import AIModelProperties
from app.infrastructure.trace.tracer import Tracer
from app.infrastructure.database.session import get_async_session
from app.models.agent_schemas import ChatRequest, ChatResponse, MemoryContext, Message  # 保持原有的标准对话模型

# ── Agent 编排基础类型 ──────────────────────────────────────────────
from app.core.agent.orchestrator import AgentOrchestrator, IntentContext
from app.core.memory.manager import MemoryManager
from app.core.skill.manager import SkillManager
from app.core.tools.registry import ToolRegistry

# ── Agent 编排前置 Pipeline（改写 + 意图 + 模式决策）注入 ────────────
from app.api.depends.dependencies import (
    get_memory_manager,
    get_model_router,
    get_pipeline,
    get_skill_manager,
    get_tool_registry,
)
from app.query_intent.intent_data_base import IntentChatMessage
from app.query_intent.intent_3stage_pipeline import AgentQueryIntentPipeline

router = APIRouter(tags=["chat"])

# 框架全局单例组件
_tracer = Tracer()
_intent = IntentRecognizer()

# 【t2/t3 优化】后台异步任务的模块级强引用登记表（防止 asyncio.create_task 产生的
# 协程在无引用时被 GC 提前回收）。任务完成后由回调自动移出并消费异常。
_background_memory_prefetch_tasks: set = set()
_background_ltm_store_tasks: set = set()


def _discard_background_task(finished_task: "asyncio.Task[Any]", *,
                             registry: set,
                             warn_label: str = "后台任务") -> None:
    """后台任务完成回调：移出登记表，异常仅告警不抛出（不影响主流程响应）。"""
    registry.discard(finished_task)
    if finished_task.cancelled():
        return
    task_exception: Optional[BaseException] = finished_task.exception()
    if task_exception is not None:
        logger.warning("{}执行异常（已忽略，不影响主流程响应）: {}", warn_label, task_exception)


def _on_memory_prefetch_done(finished_task: "asyncio.Task[Any]") -> None:
    _discard_background_task(finished_task, registry=_background_memory_prefetch_tasks,
                             warn_label="长期记忆并行预取任务")


def _on_ltm_store_done(finished_task: "asyncio.Task[Any]") -> None:
    _discard_background_task(finished_task, registry=_background_ltm_store_tasks,
                             warn_label="长期记忆后台沉淀任务")


def _extract_relevant_texts(long_items: List[Any]) -> List[str]:
    """将长期记忆召回条目（MemoryItem 或兼容对象）拉平为纯文本片段列表。"""
    relevant_texts: List[str] = []
    for item in long_items:
        if hasattr(item, "content"):
            relevant_texts.append(str(item.content))
        elif hasattr(item, "text"):
            relevant_texts.append(str(item.text))
        else:
            relevant_texts.append(str(item))
    return relevant_texts


def _build_router() -> ModelRouter:
    """根据配置构造模型路由器（单模型兜底：3 个 tier 复用同一模型）。"""
    settings = get_settings()
    logger.info("--- 调试当前实际加载的 Base URL: {}", settings.openai_api_base)
    logger.info("--- 调试当前实际加载的 KEY: {}", settings.openai_api_key)
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY，无法调用模型")
    mid = settings.openai_llm_model
    base = settings.openai_api_base or None
    properties = AIModelProperties.from_dict({
        "providers": {"openai": {"url": base, "api_key": settings.openai_api_key, "endpoints": {}}},
        "chat": {
            "default_model": mid,
            "candidates": [{
                "id": mid,
                "provider": "openai",
                "model": mid,
                "url": base,
                "priority": 0,
                "enabled": True,
                "supports_thinking": True,
            }],
            "default_tier": "standard",
            "deep_thinking_tier": "deep",
            "tiers": {
                "fast": {"candidates": [mid], "timeout_ms": settings.llm_tier_fast_timeout_ms},
                "standard": {"candidates": [mid], "timeout_ms": settings.llm_tier_standard_timeout_ms},
                "deep": {"candidates": [mid], "timeout_ms": settings.llm_tier_deep_timeout_ms},
            },
        },
        "selection": {"failure_threshold": 5, "open_duration_ms": 60_000},
    })
    return ModelRouter(properties)


async def _execute_chat_core(
    session_id: str,
    user_query: str,
    memory_manager: MemoryManager,
    model_router: ModelRouter,
    *,
    skip_intent: bool = True,
    model_preference: Optional[str] = None,
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = 1000,
    trace_id: Optional[str] = None,
    precomputed_relevant: Optional[List[str]] = None,
) -> Any:
    """标准对话核心逻辑（无 FastAPI 依赖注入外壳）。

    负责：1) 意图识别（可选跳过） 2) 长期记忆召回（支持外部预取结果直传）
    3) 合并短期滑动窗口 4) 调用模型 5) 记忆落库（短期 append_turn 同步 +
    长期 _ltm.store 后台异步）。

    【改进点 1 · sys 短路复用】：/chat 端点与 /chat/with_agent 的系统意图
    短路分支共用本函数；后者以 skip_intent=True 跳过重复意图识别，
    节省一次 LLM 调用。

    Returns:
        ModelRouter 的模型响应对象（.content / .model_id / .usage）。
    """
    # 1. 意图识别与澄清（若已由 Pipeline 确认是 sys 意图，则整体跳过）
    intent = None
    clarify = None
    if not skip_intent:
        intent = await _intent.recognize(user_query)
        clarify = await _intent.clarify(user_query, intent)

    # 2. 长期记忆召回：根据当前输入从 Milvus 语义检索相关历史碎片
    #    【t2 优化】若调用方已在 Pipeline 阶段并行预取，则直接复用，避免重复 embedding+Milvus 检索
    if precomputed_relevant is not None:
        relevant_history: List[str] = precomputed_relevant
    else:
        relevant_history = await memory_manager.get_relevant(
            session_id=session_id,
            query=user_query,
            limit=3,
        )

    # 3. 构造大模型消息桶
    messages: List[Dict[str, Any]] = []

    # 召回的长期记忆作为系统背景知识（RAG）注入给大模型
    if relevant_history:
        knowledge_context = "\n".join([f"- {text}" for text in relevant_history])
        messages.append({
            "role": "system",
            "content": f"【系统检索到的历史相关事实，请参考以下背景解答用户疑问】：\n{knowledge_context}"
        })

    # 短期记忆合并：从 Redis 读取本会话更早之前的对话历史
    short_term_history = await memory_manager._stm.get_history(session_id)
    for old_msg in short_term_history:
        messages.append({"role": old_msg.role.value, "content": old_msg.content})

    # 意图澄清（仅在执行了意图识别且置信度不足时注入）
    if clarify and intent and intent.confidence < _intent.confidence_threshold:
        messages.append({"role": "system", "content": f"（系统提示：{clarify}）"})

    # 补入当前用户最新输入
    messages.append({"role": "user", "content": user_query})

    # 4. 调用大模型集群进行推理
    resp = await model_router.chat(
        messages,
        model_preference=model_preference,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 5. 短期与长期记忆持久化（一问一答沉淀，否则下次请求就成失忆状态）
    #    【t3 优化】短期历史同步落库（Redis 低延迟，下一轮立即可见）；
    #    长期向量沉淀（embedding API + Milvus 写入，高延迟）转后台异步任务，
    #    不再阻塞对话响应主链路。
    await memory_manager.append_turn(session_id=session_id, role="user", content=user_query)
    await memory_manager.append_turn(session_id=session_id, role="assistant", content=resp.content)

    async def _store_long_term_in_background() -> None:
        await memory_manager._ltm.store(
            session_id=session_id,
            content=f"用户提问: {user_query} \n系统回复: {resp.content}",
            metadata={"trace_id": trace_id} if trace_id else {"source": "chat_core"},
        )

    ltm_store_task: "asyncio.Task[None]" = asyncio.create_task(_store_long_term_in_background())
    _background_ltm_store_tasks.add(ltm_store_task)
    ltm_store_task.add_done_callback(_on_ltm_store_done)

    return resp


class AgentChatRequest(BaseModel):
    """Agent 智能对话请求架构规约"""
    query: str = Field(..., description="用户输入的原始提问")
    session_id: str = Field(..., description="会话唯一 ID")
    strategy: Optional[str] = Field("auto", description="全局路由编排策略")


@router.post("/chat/with_agent")
async def handle_agent_chat(
    request: AgentChatRequest,
    memory_manager: MemoryManager = Depends(get_memory_manager),
    model_router: ModelRouter = Depends(get_model_router),
    tool_registry: ToolRegistry = Depends(get_tool_registry),
    skill_manager: SkillManager = Depends(get_skill_manager),
    pipeline: AgentQueryIntentPipeline = Depends(get_pipeline),
    db_session: AsyncSession = Depends(get_async_session),
) -> StreamingResponse:
    """Agent 智能对话入口（最终答案 SSE 流式化）。

    保留原有完整编排链路（Pipeline 决策 → 系统意图短路 → ReAct/Plan-Execute
    → 降级兜底）不动，仅在最终答案产出后按字/段以多条 SSE data 事件逐段吐出，
    实现「打字机」流式效果。执行过程（工具调用/规划）仍不可见，只流出最终答案。

    SSE data 事件格式（与 /chat/stream 一致的多行 JSON）：
      data: {"content": "...", "trace_id": "..."}        # 最终答案片段
      data: {"done": true, "status": "success|degraded",
             "session_id": "...", "trace_id": "...",
             "steps_executed": N, "degraded": bool}      # 结束信号 + 元数据
      data: {"error": "..."}                             # 失败信号

    前端收到 "done": true 即视为本轮结束，可据 status/degraded 渲染提示。
    """
    active_session_id: str = request.session_id or f"session_{uuid.uuid4().hex[:12]}"

    return StreamingResponse(
        _agent_stream_generator(
            request=request,
            active_session_id=active_session_id,
            memory_manager=memory_manager,
            model_router=model_router,
            tool_registry=tool_registry,
            skill_manager=skill_manager,
            pipeline=pipeline,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _sse_payload(data: Dict[str, Any]) -> bytes:
    """序列化单条 SSE data 事件（JSON → bytes）。"""
    return b"data: " + json.dumps(data, ensure_ascii=False).encode() + b"\n\n"


async def _stream_final_answer(
    text: str,
    trace_id: str,
    *,
    chunk_size: int = 8,
) -> AsyncIterator[bytes]:
    """将最终答案按字/段切分为多条 SSE chunk 事件，逐段 yield（打字机效果）。

    Args:
        text: 待流式吐出的完整答案。
        trace_id: 用于每段 chunk 标注链路 ID。
        chunk_size: 每次跃出多少个字符（中文无空格，按字符切分最稳）。

    Yields:
        每段的 SSE 字节负载。
    """
    for index in range(0, len(text), chunk_size):
        yield _sse_payload({"content": text[index:index + chunk_size], "trace_id": trace_id})
        await asyncio.sleep(0)  # 让出事件循环，保证多次 SSE 分帧及时 flush


async def _agent_stream_generator(
    request: AgentChatRequest,
    active_session_id: str,
    memory_manager: MemoryManager,
    model_router: ModelRouter,
    tool_registry: ToolRegistry,
    skill_manager: SkillManager,
    pipeline: AgentQueryIntentPipeline,
) -> AsyncIterator[bytes]:
    """Agent 全链路执行 + 最终答案流式吞吐（与旧 handle_agent_chat 逻辑完全等价）。

    执行顺序：
      1. Pipeline 前置决策：用户问题改写 → 意图聚合 → 编排模式判定
         （同步 Pipeline 通过 asyncio.to_thread 包装，保持 query_intent
         代码零 async 侵入）。
      2. 将改写后的问题 + 显式 mode + IntentContext（含 slots/allowed_tools）
         委派给 AgentOrchestrator.run。底层 ReAct/Plan-Execute 引擎据此
         调度工具、动态生成计划，并以标准 AgentResponse 回传最终答案。
      3. G-1 约定：若 request.strategy == "react" / "plan_execute"，则前端
         可强覆盖 Pipeline 的模式决策结果（原始决策仍保留在 slots 便于 debug）。
      4. 最终答案以多条 SSE chunk 事件逐段吐出，随后发 "done" 收尾。
    """
    logger.info("接收到 Agent 编排会话请求. Session ID: {}, 策略: {}", active_session_id, request.strategy)

    registered_tool_snapshot: List[str] = tool_registry.list_tool_names()

    # ------------------------------------------------------------------
    # 1.5 高级技能快照 → Pipeline：改写阶段需要技能名+描述清单让 LLM 挑选排序。
    #     skills 可能存在冷启动延迟（首次请求前未扫描），此处按需触发一次刷新，
    #     保证 Pipeline 注入的技能清单与编排层实际技能树一致。
    # ------------------------------------------------------------------
    try:
        if not getattr(skill_manager.state, "available_skills", None):
            await skill_manager.scan_and_refresh_skills()
        available_skills_snapshot: List[Dict[str, Any]] = [
            {
                "name": str(meta.get("name") or ""),
                "description": str(meta.get("description") or ""),
                "file_path": str(meta.get("file_path") or ""),
                "allowed_tools": list(meta.get("allowed_tools") or []),
            }
            for meta in (
                getattr(skill_manager.state, "available_skills", {}).values()
            )
            if isinstance(meta, dict) and (meta.get("name") or "").strip()
        ]
    except Exception as skill_snapshot_error:
        logger.warning(
            "Pipeline：高级技能快照构建失败，视为无技能继续执行。session_id={}, error={}",
            active_session_id,
            skill_snapshot_error,
        )
        available_skills_snapshot = []

    # ------------------------------------------------------------------
    # 1. 拉取短期对话历史 → IntentChatMessage，服务于改写阶段的上下文参考
    #    【t2 优化】此处只读短期历史（Redis，低延迟）；高延迟的长期向量召回
    #    拆到步骤 2 与 Pipeline 并行执行，改写阶段无需等待。
    # ------------------------------------------------------------------
    rewrite_history_messages: List[IntentChatMessage] = []
    short_term_messages_snapshot: List[Any] = []
    try:
        short_term_messages_snapshot = list(
            await memory_manager._stm.get_history(active_session_id) or []
        )
        for memory_msg in short_term_messages_snapshot:
            role_raw_value = getattr(memory_msg.role, "value", memory_msg.role)
            role_str_value: str = str(role_raw_value).lower()
            if role_str_value == "user":
                rewrite_history_messages.append(IntentChatMessage.user(memory_msg.content))
            elif role_str_value == "assistant":
                rewrite_history_messages.append(IntentChatMessage.assistant(memory_msg.content))
    except Exception as history_error:
        logger.warning(
            "Pipeline：短期对话历史拉取失败，视为空历史继续执行。session_id={}, error={}",
            active_session_id,
            history_error,
        )

    # ------------------------------------------------------------------
    # 2. 同步 Pipeline：改写 → 意图 → 模式决策（to_thread 包装避免阻塞事件循环）
    #    【t2 优化】同时在事件循环侧并行发起长期向量记忆召回（embedding +
    #    Milvus 检索，高延迟），两者执行期完全重叠，记忆检索净耗时趋近于 0；
    #    同一请求内的记忆检索总次数由 2 次（此处 + 编排器内部）降为 1 次。
    # ------------------------------------------------------------------
    long_term_recall_task: "asyncio.Task[Any]" = asyncio.create_task(
        memory_manager._ltm.recall(request.query, active_session_id, top_k=6)
    )
    _background_memory_prefetch_tasks.add(long_term_recall_task)
    long_term_recall_task.add_done_callback(_on_memory_prefetch_done)

    pipeline_output = await asyncio.to_thread(
        pipeline.run,
        request.query,
        registered_tool_snapshot,
        active_session_id,
        rewrite_history_messages,
        available_skills_snapshot,
    )

    # ------------------------------------------------------------------
    # 2.5 【改进点 1 · sys 短路】系统意图命中 → 走标准聊天核心，不启动 Agent
    #     省去 ReAct 循环（LLM 思考 + 工具探测）的 3~5 秒浪费。
    # ------------------------------------------------------------------
    intents_result = pipeline_output.intents_result
    if intents_result.sys_hit_count > 0 and intents_result.aggregated_confidence >= 0.6:
        quick_trace_id = str(uuid.uuid4())
        quick_span = _tracer.start_trace(quick_trace_id, "agent_chat_sys_shortcircuit")
        logger.info(
            "系统意图命中 (sys_hit_count={}, aggregated_confidence={:.2f})，跳过 Agent 循环，"
            "直接调用标准聊天核心。session_id={}",
            intents_result.sys_hit_count,
            intents_result.aggregated_confidence,
            active_session_id,
        )
        try:
            # 【t2 优化】复用 Pipeline 阶段并行预取的长期记忆（此时大概率已完成），
            # 省去标准聊天核心内部的重复 embedding+Milvus 检索
            try:
                sys_precomputed_relevant: Optional[List[str]] = _extract_relevant_texts(
                    list(await long_term_recall_task or [])
                )
            except Exception as sys_recall_error:
                logger.warning("sys 短路分支复用长期记忆预取结果失败，降级为内部自行检索: {}", sys_recall_error)
                sys_precomputed_relevant = None
            quick_resp = await _execute_chat_core(
                session_id=active_session_id,
                user_query=request.query,  # 使用原始问题（sys 闲聊场景改写收益低）
                memory_manager=memory_manager,
                model_router=model_router,
                skip_intent=True,  # Pipeline 已确认 sys 意图，跳过重复识别，节省一次 LLM 调用
                trace_id=quick_trace_id,
                precomputed_relevant=sys_precomputed_relevant,
            )
            _tracer.end_span(quick_span, result={"status": "sys_shortcircuit", "model": quick_resp.model_id})
            async for chunk in _stream_final_answer(quick_resp.content, quick_trace_id):
                yield chunk
            yield _sse_payload({
                "done": True,
                "status": "success",
                "session_id": active_session_id,
                "trace_id": quick_trace_id,
                "steps_executed": 0,
                "degraded": False,
            })
            return
        except Exception as quick_error:
            logger.exception("sys 短路聊天核心失败，回退 Agent 流程: {}", quick_error)
            _tracer.end_span(quick_span, error=str(quick_error))
            # 不直接抛出，继续走下方完整 Agent 编排流程兜底

    # G-1：前端 request.strategy 强覆盖模式（其他值如 "auto" 则走 Pipeline 决策）
    force_override_mode: Optional[str] = None
    raw_strategy_value: str = (request.strategy or "").strip().lower()
    if raw_strategy_value in ("react", "plan_execute"):
        force_override_mode = raw_strategy_value

    final_mode, intent_payload_dict, effective_user_input = pipeline.build_orchestrator_input(
        pipeline_output=pipeline_output,
        force_override_mode=force_override_mode,  # type: ignore[arg-type]
    )
    intent_context_for_orchestrator: IntentContext = IntentContext(**intent_payload_dict)
    logger.info(
        "Pipeline 决策完成：session_id={}, final_mode={}, primary_intent={!r}, "
        "allowed_tools_count={}, effective_input={!r}",
        active_session_id,
        final_mode,
        intent_context_for_orchestrator.intent,
        len(intent_context_for_orchestrator.allowed_tools or []),
        effective_user_input,
    )

    # 【t2 优化】汇聚 Pipeline 期间并行预取的长期记忆，与步骤 1 的短期历史合并为
    # 完整记忆上下文，直传编排器（precomputed_memory），编排器内部不再重复发起
    # embedding+Milvus 检索。
    long_term_items_snapshot: List[Any] = []
    try:
        long_term_items_snapshot = list(await long_term_recall_task or [])
    except Exception as recall_error:
        logger.warning("Pipeline 并行长期记忆召回失败，降级为空长期记忆继续执行: {}", recall_error)
        long_term_items_snapshot = []
    precomputed_memory_for_orchestrator = MemoryContext(
        session_id=active_session_id,
        short_term_messages=short_term_messages_snapshot,
        long_term_items=long_term_items_snapshot,
    )
    logger.info(
        "【t2 优化】长期记忆并行预取完成并与短期历史合并：短期 {} 条，长期 {} 条（编排器内部将跳过重复检索）",
        len(precomputed_memory_for_orchestrator.short_term_messages),
        len(precomputed_memory_for_orchestrator.long_term_items),
    )

    # ------------------------------------------------------------------
    # 3. 初始化 Agent 编排器 + 委派执行（显式传 mode 和 intent）
    # ------------------------------------------------------------------
    agent_orchestrator = AgentOrchestrator(
        config={"default_strategy": request.strategy},
        model_router=model_router,
        memory_manager=memory_manager,
        tool_registry=tool_registry,
        tracer=_tracer,
        skill_manager=skill_manager,
    )

    orchestrator_result = await agent_orchestrator.run(
        user_input=effective_user_input,
        session_id=active_session_id,
        mode=final_mode,
        intent=intent_context_for_orchestrator,
        precomputed_memory=precomputed_memory_for_orchestrator,
    )

    # ------------------------------------------------------------------
    # 4. 状态检查与断言保护
    # ------------------------------------------------------------------
    if not orchestrator_result.success:
        graceful_answer: str = (getattr(orchestrator_result, "answer", "") or "").strip()
        if graceful_answer:
            # 优雅降级：底层 LLM 候选超时/不可用，但已给出可读降级答复（不再是
            # 空 answer），返回 200 + degraded 状态，避免用户端整链 500 硬失败。
            logger.warning(
                "智能体执行降级完成（graceful）：返回友好降级答复。Trace ID: {}",
                orchestrator_result.trace_id,
            )
            async for chunk in _stream_final_answer(graceful_answer, orchestrator_result.trace_id):
                yield chunk
            yield _sse_payload({
                "done": True,
                "status": "degraded",
                "session_id": active_session_id,
                "trace_id": orchestrator_result.trace_id,
                "steps_executed": len(getattr(orchestrator_result, "steps", None) or []),
                "degraded": True,
            })
            return
        logger.error("底层智能体决策树执行失败，准备向上层抛出异常。Trace ID: {}", orchestrator_result.trace_id)
        yield _sse_payload({"error": orchestrator_result.error or "智能体编排器未能成功完成当前意图的闭环推理"})
        return

    # degraded 场景来源：PlanExecute 失败降级 ReAct（reflection 质量门已剥离，
    # 不再产生质量告警）；reflection_quality 为兼容保留的废弃字段，恒为 None。
    degraded_value: bool = bool(getattr(orchestrator_result, "degraded", False))
    async for chunk in _stream_final_answer(orchestrator_result.answer, orchestrator_result.trace_id):
        yield chunk
    yield _sse_payload({
        "done": True,
        "status": "success",
        "session_id": active_session_id,
        "trace_id": orchestrator_result.trace_id,
        "steps_executed": len(orchestrator_result.steps),
        "degraded": degraded_value,
    })
    return






# ==========================================
# 2. 原有标准对话接口 (保持不动)
# ==========================================
class EnhancedChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="当前轮次的对话消息")
    session_id: Optional[str] = Field(None, description="会话唯一 ID，不传则自动建立新会话")
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000


@router.post("/chat", response_model=ChatResponse)
async def chat(
        request: EnhancedChatRequest,
        memory_manager: MemoryManager = Depends(get_memory_manager),  # 👈 优雅注入记忆大管家
) -> ChatResponse:
    """非流式对话：集成意图识别、长期记忆召回、短期记忆滑动沉淀与链路追踪。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat")

    # 确定锁定的会话 ID
    current_session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"

    try:
        user_text = request.messages[-1].content if request.messages else ""

        router_llm = _build_router()

        # 复用标准对话核心（skip_intent=False：本端点需要意图识别做澄清）
        resp = await _execute_chat_core(
            session_id=current_session_id,
            user_query=user_text,
            memory_manager=memory_manager,
            model_router=router_llm,
            skip_intent=False,
            model_preference=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            trace_id=trace_id,
        )

        _tracer.end_span(span, result={"model": resp.model_id, "usage": resp.usage})

        # 将生成的/锁定的 session_id 包含在 trace 或扩展字段中返回给前端
        return ChatResponse(
            id=current_session_id,  # 此处复用 id 字段传递 session_id，也可以扩展专属字段
            model=resp.model_id,
            content=resp.content,
            trace_id=trace_id,
            usage=resp.usage,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat 失败: {}", exc)
        _tracer.end_span(span, error=str(exc))
        raise HTTPException(status_code=500, detail=f"对话失败: {exc!s}") from exc


async def _stream_generator(request: ChatRequest, trace_id: str) -> AsyncIterator[bytes]:
    """SSE 风格流：每行 data: {json}\n\n。

    【接入说明 · repo_map B3】：本端点对 chunk 级流式语义敏感（逐字节 yield 给前端），
    而 ModelRouter 的 AsyncModelRoutingExecutor 需要在"单个请求完整成功/失败"
    时才进行熔断状态回写（mark_success / mark_failure），流式 chunk 与回写时机
    存在天然冲突——强行接入会导致：① chunk 要走 Executor 的 for 循环包装，
    引入深层异步生成器嵌套；② 流结束后的 permit 释放与 chunk 生成器关闭
    竞争导致 permit 泄漏。

    因此 **/chat/stream 端点保留独立流式客户端（0 适配器，最扁平化）**，
    暂不纳入 ModelRouter 熔断降级体系：
      - 固定使用 settings.openai_api_key + settings.openai_api_base +
        request.model or settings.openai_llm_model（与 ModelRouter 的
        default 首选模型一致）；
      - 若将来需要流式也支持多模型降级，需新增独立的 AsyncStreamRoutingExecutor，
        在流 finally 子句中统一释放 permit；本版本不做。
    """
    settings = get_settings()
    if not settings.openai_api_key:
        yield b"data: " + json.dumps({"error": "未配置 API Key"}, ensure_ascii=False).encode() + b"\n\n"
        return

    client = AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base or None,
    )
    messages = [m.model_dump() for m in request.messages]
    model = request.model or settings.openai_llm_model

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                payload = {"content": delta, "trace_id": trace_id}
                yield b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"
        yield b"data: " + json.dumps({"done": True}, ensure_ascii=False).encode() + b"\n\n"
    except Exception as exc:
        logger.exception("chat_stream 失败: {}", exc)
        yield b"data: " + json.dumps({"error": str(exc)}, ensure_ascii=False).encode() + b"\n\n"


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """流式输出（Server-Sent Events 兼容格式）。"""
    trace_id = str(uuid.uuid4())
    span = _tracer.start_trace(trace_id, "chat_stream")
    _tracer.end_span(span, result={"mode": "stream"})

    return StreamingResponse(
        _stream_generator(request, trace_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Trace-Id": trace_id,
        },
    )