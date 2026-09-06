# -*- coding: utf-8 -*-
"""AsyncOpenAI 兼容客户端 + Chat Completions 异步 Caller 实现。

为重写版 ``AsyncModelRoutingExecutor`` 提供可插拔的 OpenAI 协议调用器，
同时将返回值结构化为 ``AsyncOpenAICallResult``，供外层兼容壳
（``infrastructure.llm.model_router.ModelRouter.chat``）转化为旧的
``LLMResponse``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from openai import APIError, AsyncOpenAI, RateLimitError

from app.llm_model_router.model_router_config import AIModelProperties
from app.llm_model_router.model_router_enums import ModelProvider, ModelTarget

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
      - timeout_ms（来自 tier）可映射为 httpx 级别的 timeout。
    """
    # 1) 优先使用 candidate 级别的 url / api_key，其次 provider 级别
    api_key: Optional[str] = None
    if provider is not None:
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
# Chat Caller：给 AsyncModelRoutingExecutor.execute_with_fallback(..., caller=)
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
    **extra_kwargs: Any,
) -> AsyncOpenAICallResult:
    """异步调用 AsyncOpenAI.chat.completions.create。

    说明：
      - ``messages`` 接受 Sequence[Dict[str, Any]]（或 list[dict]），兼容
        orchestrator 的 Sequence[Dict[str,str]] 以及 chat.py 的 list[dict] 两种入参。
      - ``thinking`` 字段目前主流 Provider 不直接在 SDK 参数中接收，此处将其放入
        ``extra_body``，供 DashScope/SiliconFlow 等非官方兼容协议的服务端消费。
        如果目标 Provider 是标准 OpenAI，则字段被忽略不影响。
      - 所有 Provider 异常（APIError/RateLimitError）都向上抛出，由 Executor 统一
        做熔断计数 + 降级。
    """
    if stream:  # 兼容上层偶发 stream=True，但本调用器是非流式的
        stream = False

    # 1) 组装参数
    params: Dict[str, Any] = {
        "model": target.candidate.model or target.id,
        "messages": [dict(m) for m in messages],  # 转 list[dict]
        "temperature": temperature,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if response_format is not None:
        params["response_format"] = response_format
    if tools is not None:
        params["tools"] = tools
    if tool_choice is not None:
        params["tool_choice"] = tool_choice

    # thinking 字段 → 放入 extra_body（非标准参数，避免 AsyncOpenAI 校验时拒绝）
    if thinking is not None:
        extra_body = dict(extra_kwargs.pop("extra_body", None) or {})
        # DashScope 兼容接口要求 thinking 为 JSON 对象
        if isinstance(thinking, bool):
            extra_body["enable_thinking"] =  thinking
        else:
            extra_body.setdefault("enable_thinking", thinking)
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
    choice = resp.choices[0] if getattr(resp, "choices", None) else None
    content: str = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None
    if choice is not None:
        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content") or ""
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

    actual_model = getattr(resp, "model") or target.id
    return AsyncOpenAICallResult(
        content=content,
        model_id=actual_model or target.id,
        usage=usage,
        raw=raw,
        tool_calls=tool_calls,
        reasoning_content=reasoning_content,
    )
