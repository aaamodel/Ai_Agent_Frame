# -*- coding: utf-8 -*-
"""AsyncOpenAI 兼容客户端 + Chat Completions 异步 Caller 实现。

为重写版 ``AsyncModelRoutingExecutor`` 提供可插拔的 OpenAI 协议调用器，
同时将返回值结构化为 ``AsyncOpenAICallResult``，供外层兼容壳
（``infrastructure.llm.model_router.ModelRouter.chat``）转化为旧的
``LLMResponse``。
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from openai import APIError, AsyncOpenAI, RateLimitError

from app.llm_model_router.model_router_config import (
    AIModelProperties,
    STYLE_DASHSCOPE,
    STYLE_DEEPSEEK,
    STYLE_ZHIPU,
)
from app.llm_model_router.model_router_enums import ModelProvider, ModelTarget

# 旁观通道：调用器在被观测时改用流式收流，边收边把增量推给通道，
# **但对外仍返回完整结果**（见 specs/2026-09-20-llm-token-streaming-design.md §4.2）。
from app.core.agent.stream_sink import emit as _emit_delta
from app.core.agent.stream_sink import has_sink as _has_sink

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路由内部控制参数：这些字段只服务于 ModelRouter / Selector / Executor 内部
# 决策（tier 路由、降级、日志），**绝对不允许**出现在 OpenAI SDK 的
# chat.completions.create(**params) 里。作为双保险，在 SDK 边界层做最后一次过滤。
# ---------------------------------------------------------------------------
_ROUTER_CONTROL_KWARGS = frozenset({
    "purpose_hint",
    "tier_override",
    "model_preference",
    "preferred_model_id",
    "preferred_model",
    "model_id",
    "capability",
})


# ---------------------------------------------------------------------------
# 统一调用结果（供 AsyncModelRoutingExecutor 返回 → 上层 LLMResponse 转换）
# ---------------------------------------------------------------------------
@dataclass
class AsyncOpenAICallResult:
    """异步 OpenAI 调用结果的内部结构。"""

    content: str
    """模型返回的纯文本内容（choice.message.content，Function Calling 时通常为空）。"""

    model_id: str
    """实际命中的模型 ID（与 target.id 一致，或 provider 返回的重写 model）。"""

    usage: Optional[Dict[str, Any]]
    """用量字典 {prompt_tokens, completion_tokens, total_tokens}。"""

    raw: Optional[Dict[str, Any]]
    """完整响应对象的 dict 形态（调用 model_dump，失败时为 None）。"""

    tool_calls: Optional[List[Dict[str, Any]]] = None
    """Function Calling 解析结果列表；每项结构：
    {id, type, function: {name, arguments}}（arguments 为 JSON 字符串）。
    非工具调用场景为 None。"""

    reasoning_content: Optional[str] = None
    """思考模型（如 qwen/deepseek 推理模式）返回的思维链文本，未开启时为 None。"""


# ---------------------------------------------------------------------------
# 客户端工厂：根据 ModelTarget（candidate + provider config）建 AsyncOpenAI
# ---------------------------------------------------------------------------
def async_build_openai_client(
    candidate: AIModelProperties.ModelCandidate,
    provider: Optional[AIModelProperties.ProviderConfig],
    timeout: Optional[float] = None,
) -> Optional[AsyncOpenAI]:
    """按候选 + provider 配置构造 AsyncOpenAI。失败/缺失返回 None（由 Executor skip）。

    规则：
      - 如果 provider.api_key 缺失但 candidate.url 是本地 Ollama 等不需要 key 的
        NOOP 来源：给占位 api_key，否则直接判定为缺配置。
      - timeout_ms（来自 tier）可映射为 httpx 级别的 timeout。但**单次预算按请求传递**
        （见 ``async_openai_chat_caller(request_timeout=...)``）：同一模型可能出现在
        多个档位且各档预算不同（如 glm-4.7 同时在 FAST 与 STANDARD），故不在构建时固定。
      - 构建时显式关闭 SDK 隐式重试（``max_retries=0``）：重试改由 Executor 逐次控制，
        避免"多次尝试共用一份候选级超时预算"。
    """
    # 1) 优先使用 candidate 级别的 url / api_key，其次 provider 级别
    #    （多厂商接入时每个候选可挂不同账户/网关，candidate 级必须能覆盖 provider 级）
    api_key: Optional[str] = candidate.api_key
    if not api_key and provider is not None:
        api_key = provider.api_key
    # candidate 级覆盖（如某模型单独挂到另一个账户下）
    if candidate.url:
        base_url: Optional[str] = candidate.url
    elif provider is not None:
        base_url = provider.url
    else:
        base_url = None

    if not api_key:
        # NOOP：用户显式表明该 provider 不需要密钥（本地/Ollama 代理已鉴权）
        if ModelProvider.NOOP.matches(candidate.provider):
            api_key = "noop"
        else:
            logger.warning(
                "OpenAI 兼容客户端缺少 api_key，跳过该候选: provider=%s model=%s",
                candidate.provider,
                candidate.model,
            )
            return None

    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    # ⚠️ 关闭 SDK 的隐式重试。它的两宗罪（实测）：
    #   1) 把"多次尝试"藏在一次调用里，与候选级超时**共用同一份预算**——
    #      日志表现为 `Retrying request ... in 0.47s` 紧接 `tier-timeout (50000ms)`，
    #      第二次尝试注定被掐掉，看起来像"模型算得慢"；
    #   2) **吞掉首次失败的真实原因**，排查时只剩一行 SDK 重试日志。
    #    重试改由 Executor 显式控制：逐次独立预算 + 逐次记录耗时与失败类型。
    kwargs["max_retries"] = 0
    try:
        return AsyncOpenAI(**kwargs)
    except Exception as exc:  # pragma: no cover - 初始化异常极少见
        logger.exception(
            "AsyncOpenAI 客户端初始化失败: provider=%s model=%s err=%s",
            candidate.provider,
            candidate.model,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# Langfuse 观测：为底层 LLM 调用补充 generation 富化（可开关、永不阻断真实调用）
# ---------------------------------------------------------------------------
from langfuse import get_client as langfuse_get_client

from app.infrastructure.trace.langfuse import is_langfuse_enabled


def _enrich_langfuse_generation(generation: Any, resp: Any, model: str) -> None:
    """把真实调用结果回写到 Langfuse generation（usage/output）。

    任何异常只降级为 debug 日志，绝不影响上层已拿到真实响应的流程。
    """
    try:
        usage_details: Dict[str, int] = {}
        resp_usage = getattr(resp, "usage", None)
        if resp_usage is not None:
            usage_details = {
                "input": int(getattr(resp_usage, "prompt_tokens", 0) or 0),
                "output": int(getattr(resp_usage, "completion_tokens", 0) or 0),
                "total": int(getattr(resp_usage, "total_tokens", 0) or 0),
            }
        content: str = ""
        choices = getattr(resp, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content") or "" if message is not None else ""
        update_kwargs: Dict[str, Any] = {"output": {"content": content}}
        if usage_details:
            update_kwargs["usage_details"] = usage_details
        update_kwargs["model"] = model
        generation.update(**update_kwargs)
    except Exception:  # pragma: no cover - 观测富化永不阻塞主流程
        logger.debug("Langfuse generation 富化失败，忽略。", exc_info=False)


async def _chat_completion_with_langfuse(
    client: AsyncOpenAI,
    params: Dict[str, Any],
    *,
    model: str,
    provider: str,
) -> Any:
    """带 Langfuse generation 观测的底层 LLM 调用。

    设计原则：
      - 未启用 Langfuse 或观测层任何异常 → 直接原样发起真实 LLM 调用，零副作用。
      - 真实 LLM 调用只发生一次；观测层异常使用一次性兜底重发，但绝不吞掉
        APIError / RateLimitError（交由上层 Executor 做熔断降级）。
    """
    if not is_langfuse_enabled():
        return await client.chat.completions.create(**params)

    try:
        langfuse = langfuse_get_client()
        context_manager = langfuse.start_as_current_observation(
            name=f"llm.{provider}",
            as_type="generation",
            input={
                "model": params.get("model") or model,
                "messages": params.get("messages") or [],
                "temperature": params.get("temperature"),
                "max_tokens": params.get("max_tokens"),
                "tools": params.get("tools"),
            },
            model=params.get("model") or model,
            end_on_exit=False,
        )
    except Exception:  # pragma: no cover - 观测层初始化失败不阻断调用
        return await client.chat.completions.create(**params)

    if context_manager is None:
        return await client.chat.completions.create(**params)

    try:
        with context_manager as generation:
            resp = await client.chat.completions.create(**params)
            _enrich_langfuse_generation(generation, resp, model)
        return resp
    except (APIError, RateLimitError):
        raise
    except Exception:  # pragma: no cover - langfuse with 块自身异常，兜底重发
        return await client.chat.completions.create(**params)


# ---------------------------------------------------------------------------
# 厂商方言翻译：thinking 开关 / response_format 能力降级
# ---------------------------------------------------------------------------
def _apply_thinking_dialect(
    candidate: AIModelProperties.ModelCandidate,
    thinking: Optional[bool],
    extra_body: Dict[str, Any],
) -> None:
    """把统一的 ``thinking: bool`` 翻译为目标厂商的实际请求参数（原地写 extra_body）。

    各厂商协议（2026-09 官方文档/实测）：
      - dashscope（百炼）: ``enable_thinking: true/false``
      - deepseek          : ``thinking: {"type": "enabled"/"disabled"}``
      - zhipu（智谱）     : ``thinking: {"type": "enabled"/"disabled"}``
                            但 GLM-5.3 等强制思考模型禁用 disabled → 跳过注入
      - openai            : 无该参数，不注入
    """
    if thinking is None:
        return
    style = candidate.api_style
    if style == STYLE_DASHSCOPE:
        extra_body["enable_thinking"] = bool(thinking)
    elif style in (STYLE_DEEPSEEK, STYLE_ZHIPU):
        if thinking:
            extra_body["thinking"] = {"type": "enabled"}
        else:
            if not candidate.thinking_can_disable:
                # 强制思考模型（如 GLM-5.3）：发 disabled 会直接 400，保持服务端默认
                logger.info(
                    "模型 %s 不支持关闭思考（thinking_can_disable=False），跳过 disabled 注入",
                    candidate.id,
                )
                return
            extra_body["thinking"] = {"type": "disabled"}
    # openai 或其他标准协议：不注入任何思考参数


# 公开别名：其他 LLM 入口（如知识图谱抽取）复用**同一套**思考方言翻译，
# 避免各自硬编码某一家厂商的参数名（历史问题：写死 enable_thinking，只有百炼成立）。
apply_thinking_dialect = _apply_thinking_dialect


def _strictify_schema_node(node: Any) -> Any:
    """递归把 JSON Schema 补成 OpenAI/GLM 严格模式要求的形状。

    严格模式硬性约束：
      1. 每个 type=object 节点必须 ``additionalProperties: false``；
      2. properties 里的所有字段必须出现在 required 中
         （GLM 官方明确：未列入 required 的字段不会出现在输出里）。
    递归覆盖嵌套 properties / array items / anyOf, oneOf, allOf / $defs。
    """
    if isinstance(node, dict):
        typ = node.get("type")
        if typ == "object" and isinstance(node.get("properties"), dict):
            node["additionalProperties"] = False
            required = node.get("required")
            required = list(required) if isinstance(required, list) else []
            for key in node["properties"].keys():
                if key not in required:
                    required.append(key)
            node["required"] = required
            node["properties"] = {k: _strictify_schema_node(v) for k, v in node["properties"].items()}
        for key in ("items", "additionalItems", "contains"):
            if key in node:
                node[key] = _strictify_schema_node(node[key])
        for key in ("anyOf", "oneOf", "allOf"):
            if isinstance(node.get(key), list):
                node[key] = [_strictify_schema_node(item) for item in node[key]]
        if isinstance(node.get("$defs"), dict):
            node["$defs"] = {k: _strictify_schema_node(v) for k, v in node["$defs"].items()}
    elif isinstance(node, list):
        return [_strictify_schema_node(item) for item in node]
    return node


def _normalize_json_schema(fmt: Dict[str, Any]) -> Dict[str, Any]:
    """对 json_schema response_format 做严格模式归一化（深拷贝，不污染调用方字典）。"""
    normalized = copy.deepcopy(fmt)
    js = normalized.get("json_schema")
    if isinstance(js, dict):
        js["strict"] = True
        if isinstance(js.get("schema"), dict):
            _strictify_schema_node(js["schema"])
    return normalized


def _resolve_response_format(
    candidate: AIModelProperties.ModelCandidate,
    response_format: Any,
) -> Any:
    """按候选能力对 response_format 做透传（严格归一化）/降级/剥离。

    - json_schema 且候选支持：强制严格模式（补 additionalProperties:false 与全字段 required），
      GLM-4.7+/OpenAI/百炼严格模式均按此校验；
    - 降级链：json_schema → json_object → 剥离（业务侧有 JSON 容错解析兜底）。
    注意：json_object 模式下百炼/DeepSeek 均要求消息体出现 "json" 字样，
    项目各结构化环节的 system prompt 已包含 JSON 说明。
    """
    if response_format is None:
        return None
    fmt = response_format if isinstance(response_format, dict) else None
    fmt_type = (fmt or {}).get("type")

    if fmt_type == "json_schema":
        if candidate.json_schema_supported():
            return _normalize_json_schema(fmt)
        if candidate.json_object_supported():
            logger.warning(
                "模型 %s(%s) 不支持 json_schema，自动降级为 json_object",
                candidate.id, candidate.api_style,
            )
            return {"type": "json_object"}
        logger.warning(
            "模型 %s(%s) 不支持任何结构化 response_format，已剥离（依赖 prompt + 容错解析）",
            candidate.id, candidate.api_style,
        )
        return None
    if fmt_type == "json_object" and not candidate.json_object_supported():
        logger.warning(
            "模型 %s(%s) 不支持 json_object，已剥离 response_format",
            candidate.id, candidate.api_style,
        )
        return None
    return response_format


# ---------------------------------------------------------------------------
# Chat Caller：由 AsyncModelRoutingExecutor.execute_with_candidate_fallback 内部直接调用
#             （原以 caller= 参数注入，现已内联去泛型）
# ---------------------------------------------------------------------------
async def async_openai_chat_caller(
    client: AsyncOpenAI,
    target: ModelTarget,
    *,
    messages: Sequence[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    thinking: Optional[bool] = None,
    response_format: Optional[Any] = None,
    tools: Optional[Any] = None,
    tool_choice: Optional[Any] = None,
    stream: bool = False,
    request_timeout: Optional[float] = None,
    **extra_kwargs: Any,
) -> AsyncOpenAICallResult:
    """异步调用 AsyncOpenAI.chat.completions.create。

    说明：
      - ``messages`` 接受 Sequence[Dict[str, Any]]（或 list[dict]），兼容
        orchestrator 的 Sequence[Dict[str,str]] 以及 chat.py 的 list[dict] 两种入参。
      - ``thinking`` 为路由层统一布尔信号，由 ``_apply_thinking_dialect`` 按候选
        的 ``api_style`` 翻译成各厂商协议（百炼 enable_thinking /
        DeepSeek·智谱 thinking.type / OpenAI 不注入）。
      - ``response_format`` 由 ``_resolve_response_format`` 按候选能力自动降级
        （json_schema → json_object → 剥离），避免不支持的厂商直接 400。
      - 所有 Provider 异常（APIError/RateLimitError）都向上抛出，由 Executor 统一
        做熔断计数 + 降级。
    """
    # 只有"有旁观观测者"时才开流；否则保持本调用器的历史契约（非流式）。
    #
    # ⚠️ 判定**不参考传入的 stream 参数**：该参数在本调用器里一直被强制关闭
    #    （原实现 `if stream: stream = False`），没有任何上游真的传它。
    #    以它为门禁会让流式永远不触发。真正的门禁是"此刻有没有人在听"。
    #
    # ⚠️ 即便开了流，对外仍返回完整结果 —— 上游的重试/熔断/候选降级/业务解析
    #    全部不受影响，因为它们的调用方式一个字都没变。
    stream_enabled: bool = _has_sink()
    stream = False

    # 1) 组装参数
    params: Dict[str, Any] = {
        "model": target.candidate.model or target.id,
        "messages": [dict(m) for m in messages],  # 转 list[dict]
        "temperature": temperature,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    # 结构化输出：按候选厂商能力透传/降级/剥离
    resolved_rf = _resolve_response_format(target.candidate, response_format)
    if resolved_rf is not None:
        params["response_format"] = resolved_rf
    if tools is not None:
        params["tools"] = tools
    if tool_choice is not None:
        params["tool_choice"] = tool_choice
    # 单次尝试预算：按**请求**传递（同一模型可能跨多个档位、各档预算不同，不能固化
    # 在客户端上）。底层 httpx 会在自己的超时内报错，从而使"提供方慢"与"被我们主动
    # 掐断"在异常类型上可区分。
    if request_timeout is not None:
        params["timeout"] = request_timeout

    # thinking 统一信号 → 各厂商方言（写入 extra_body）
    if thinking is not None:
        extra_body = dict(extra_kwargs.pop("extra_body", None) or {})
        _apply_thinking_dialect(target.candidate, thinking, extra_body)
        if extra_body:
            params["extra_body"] = extra_body
    if extra_kwargs:
        # ⚠️ 兜底过滤：任何残留在 extra_kwargs 里的路由内部控制参数都必须丢弃，
        #    防止误传到 SDK（例如 purpose_hint）触发 unexpected keyword argument。
        sanitized = {
            k: v for k, v in extra_kwargs.items()
            if k not in _ROUTER_CONTROL_KWARGS
        }
        if sanitized:
            params.update(sanitized)

    # 2) 发起调用（带 Langfuse generation 观测；未启用或观测层异常时原样调用）
    if stream_enabled:
        # 旁路观测模式：改走流式收流 + 增量推送，返回值仍是完整结果。
        return await _streaming_chat_call(
            client=client, params={**params, "stream": True}, target=target,
        )

    try:
        resp = await _chat_completion_with_langfuse(
            client,
            params,
            model=target.id,
            provider=target.candidate.provider,
        )
    except (APIError, RateLimitError):
        raise
    except Exception:
        logger.exception(
            "OpenAI 兼容接口调用异常: provider=%s modelId=%s",
            target.candidate.provider,
            target.id,
        )
        raise

    # 3) 标准化结果
    return _standardize_completion(resp, target)


def _standardize_completion(resp: Any, target: ModelTarget) -> AsyncOpenAICallResult:
    """把 OpenAI 的 ChatCompletion 响应标准化为 ``AsyncOpenAICallResult``。

    ⚠️ 这是**非流式路径与"探测回落"共用**的标准化入口：两条路径必须给出完全
    一致的字段，否则"厂商不支持流式"时用户拿到的结果与原来不同。
    本函数是从原 ``async_openai_chat_caller`` 内联块**原样搬移**而来。
    """
    choice = resp.choices[0] if getattr(resp, "choices", None) else None
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None
    if choice is not None:
        message = getattr(choice, "message", None)
        if message is not None:
            # 同 493 行：``getattr`` 必须带默认值。这里只被 ``is not None`` 守着，
            # 属性缺失照样抛 AttributeError；``or ""`` 只能兜住"值为 None"，
            # 兜不住"没有这个属性"。
            content = getattr(message, "content", None) or ""
            # 结构化 tool_call 不是字符串 content，但 orchestrator / pipeline
            # 目前都走纯文本契约，所以取 content 即可；若为空尝试 reasoning_content
            if not content and hasattr(message, "reasoning_content"):
                content = getattr(message, "reasoning_content") or ""

            # Function Calling：message.tool_calls → 结构化 dict 列表（保序透传）
            raw_tool_calls = getattr(message, "tool_calls", None)
            if raw_tool_calls:
                tool_calls = []
                for raw_call in raw_tool_calls:
                    tool_calls.append({
                        "id": getattr(raw_call, "id", None),
                        "type": getattr(raw_call, "type", None) or "function",
                        "function": {
                            "name": getattr(getattr(raw_call, "function", None), "name", None),
                            "arguments": getattr(getattr(raw_call, "function", None), "arguments", "") or "",
                        },
                    })

            # 思考模型思维链（与 content 相互独立，供上层展示/降级分析）
            if hasattr(message, "reasoning_content"):
                rc = getattr(message, "reasoning_content")
                if rc:
                    reasoning_content = str(rc)

    usage = None
    resp_usage = getattr(resp, "usage", None)
    if resp_usage is not None:
        usage = {
            "prompt_tokens": getattr(resp_usage, "prompt_tokens", 0),
            "completion_tokens": getattr(resp_usage, "completion_tokens", 0),
            "total_tokens": getattr(resp_usage, "total_tokens", 0),
        }

    raw: Optional[Dict[str, Any]] = None
    if hasattr(resp, "model_dump"):
        try:
            raw = resp.model_dump()
        except Exception:  # pragma: no cover
            raw = None

    # ⚠️ 必须带默认值：原先写作 ``getattr(resp, "model")``（**无默认值**），属性缺失
    # 即抛 AttributeError。虽然官方 SDK 的 ChatCompletion 里 ``model`` 是必填字段、
    # 生产上基本不可达，但一旦发生，异常会走到执行器的 ``except BaseException``——
    # 后果不是"没降级"，而是**把一个健康的候选记成失败**（误伤熔断计数），
    # 日志里还只留下一条与真实问题无关的报错。带上默认值后回归 `target.id`。
    actual_model = getattr(resp, "model", None) or target.id
    return AsyncOpenAICallResult(
        content=content,
        model_id=actual_model or target.id,
        usage=usage,
        raw=raw,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )


async def _streaming_chat_call(
    *,
    client: AsyncOpenAI,
    params: Dict[str, Any],
    target: ModelTarget,
) -> AsyncOpenAICallResult:
    """流式收流 + 旁路推送，**返回与 ``async_openai_chat_caller`` 相同的完整结果**。

    为什么必须返回完整结果：上游（``run_with_attempt_budget`` 的重试、
    ``execute_with_candidate_fallback`` 的候选降级与熔断回写、以及节点里的
    结构化解析与全部闸门）都依赖"一次调用一个完整结果"这个契约。
    只要契约不变，它们一行都不用改。

    ⚠️ 这里刻意不经过 ``_chat_completion_with_langfuse``：那个包装不感知流式。
    代价是流式调用在 langfuse 里不可见（spec 风险 3，已知并记录）。
    """
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None
    # tool_calls 在流里是按 index 分片下发的，必须按 index 聚合成完整调用
    tool_calls_acc: Dict[int, Dict[str, Any]] = {}
    received_any_chunk: bool = False

    try:
        stream = await client.chat.completions.create(**params)
        async for chunk in stream:
            received_any_chunk = True
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = {
                    "prompt_tokens": getattr(chunk_usage, "prompt_tokens", None),
                    "completion_tokens": getattr(chunk_usage, "completion_tokens", None),
                    "total_tokens": getattr(chunk_usage, "total_tokens", None),
                }
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            text = getattr(delta, "content", None)
            if text:
                content_parts.append(text)
                # 旁路推送：失败被 emit 内部吞掉，绝不影响主链路
                _emit_delta({"kind": "delta", "text": text})
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)
            for tc in (getattr(delta, "tool_calls", None) or []):
                idx = int(getattr(tc, "index", 0) or 0)
                slot = tool_calls_acc.setdefault(idx, {
                    "id": None,
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                })
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["function"]["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["function"]["arguments"] += fn.arguments
    except Exception as exc:  # noqa: BLE001
        # 方案 A（spec §10.1）：**首 chunk 之前**失败、且异常特征像"参数组合不被
        # 厂商接受"时，同一次尝试内退回非流式重发，且不推任何 delta。
        # 探测请求不计入"增加调用次数"（用户已认可）；厂商不支持流式时
        # 用户看到的就是"这段没有逐字、直接出结果"。
        if not received_any_chunk and _looks_like_params_rejected(exc):
            logger.warning(
                "流式参数不被接受，本次尝试退回非流式重发: provider=%s modelId=%s err=%s",
                target.candidate.provider,
                target.id,
                exc,
            )
            resp = await client.chat.completions.create(**_strip_stream(params))
            return _standardize_completion(resp, target)
        # 其它异常（超时/连接/5xx/首 chunk 后断流）原样抛出，
        # 交由现有 run_with_attempt_budget 重试与候选降级处理。
        raise

    content: str = "".join(content_parts)
    if not content and reasoning_parts:
        # 与非流式路径同口径：content 为空时回落 reasoning_content
        content = "".join(reasoning_parts)

    return AsyncOpenAICallResult(
        content=content,
        model_id=target.candidate.model or target.id,
        usage=usage,
        raw=None,
        tool_calls=[tool_calls_acc[i] for i in sorted(tool_calls_acc)] or None,
        reasoning_content="".join(reasoning_parts) or None,
    )


def _looks_like_params_rejected(exc: BaseException) -> bool:
    """异常是否像"参数组合不被厂商接受"（spec §10.1 第 2 条）。"""
    return getattr(exc, "status_code", None) in (400, 422)


def _strip_stream(params: Dict[str, Any]) -> Dict[str, Any]:
    """剥掉流式相关参数，供"探测回落"复用同一份调用参数。

    ⚠️ 必须是"除 stream 外完全相同"的 params —— 两条路径若参数有差异，
    回落后的结果与原来就不一致了。
    """
    out = dict(params)
    out.pop("stream", None)
    out.pop("stream_options", None)
    return out
