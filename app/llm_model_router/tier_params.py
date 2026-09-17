# -*- coding: utf-8 -*-
"""档位参数读取：把"某一档用哪个模型、单次预算多少、重试几次、要不要思考"
收敛成**一个入口**，供路由链路与知识图谱抽取链路共用。

为什么需要它：知识图谱抽取（LightRAG）过去只从档位候选池里"借"了模型名，其余
参数各自硬编码——超时用它自身的全局默认（实测 240s）、思考开关写死百炼方言、
凭据写死百炼。结果是**同一档模型在不同链路上行为不一致**，超时口径也无法从配置
推导。本模块让两条链路读同一份参数。

⚠️ 本模块刻意只做"读参数"，不把 LightRAG 接入统一路由：抽取类调用量大且自带响应
缓存与并发池，接入路由会污染健康计数（误伤对话链路）并丢掉缓存。取舍见本变更
design.md 的 D4。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.llm_model_router.model_router_config import (
    AIModelProperties,
    expand_secret,
    infer_api_style,
    normalize_api_style,
)

logger = logging.getLogger(__name__)

DEEP_TIER_NAME: str = "deep"
"""需要思考的档位名（与 `app.main` 构建 `deep_thinking_tier` 时的取值一致）。"""

_TIER_FIELDS: Dict[str, Tuple[str, str, str]] = {
    "fast": (
        "llm_tier_fast_parsed",
        "llm_tier_fast_timeout_ms",
        "llm_tier_fast_retries",
    ),
    "standard": (
        "llm_tier_standard_parsed",
        "llm_tier_standard_timeout_ms",
        "llm_tier_standard_retries",
    ),
    "deep": (
        "llm_tier_deep_parsed",
        "llm_tier_deep_timeout_ms",
        "llm_tier_deep_retries",
    ),
}


@dataclass(frozen=True)
class TierParams:
    """某一档的调用参数（全部读自配置，不含任何"就地写死"的兜底值）。"""

    tier: str
    model: str
    timeout_s: Optional[float]
    retries: int
    thinking: bool
    candidate: AIModelProperties.ModelCandidate


def _build_candidate(raw: Dict[str, Any], settings: Any) -> AIModelProperties.ModelCandidate:
    """把一条 ``LLM_MODELS`` 条目映射成候选（与 ``app.main._build_global_router`` 同口径）。

    沿用同一套凭据解析规则，尤其是跨厂商网关未配独立 key 时必须**不要**错用全局
    百炼 key（否则真调用必 401）。
    """
    mid: str = str(raw.get("model_id") or "").strip()
    explicit_key: str = expand_secret(str(raw.get("api_key") or "").strip())
    explicit_base: str = str(raw.get("base_url") or "").strip()
    base: Optional[str] = explicit_base or settings.openai_api_base
    if explicit_key:
        key: str = explicit_key
    elif not explicit_base or explicit_base == settings.openai_api_base:
        key = settings.openai_api_key
    else:
        key = ""
    api_style: str = normalize_api_style(
        raw.get("api_style") or raw.get("provider") or infer_api_style(base)
    )

    def _opt_bool(name: str) -> Optional[bool]:
        return None if raw.get(name) is None else bool(raw.get(name))

    return AIModelProperties.ModelCandidate(
        id=mid,
        provider=api_style,
        model=str(raw.get("model") or mid),
        url=base or None,
        api_key=key or None,
        api_style=api_style,
        priority=int(raw.get("priority", 0)),
        enabled=bool(raw.get("enabled", True)),
        supports_thinking=bool(raw.get("supports_thinking", True)),
        supports_json_schema=_opt_bool("supports_json_schema"),
        supports_json_object=_opt_bool("supports_json_object"),
        thinking_can_disable=bool(raw.get("thinking_can_disable", True)),
    )


def _single_model_candidate(settings: Any) -> AIModelProperties.ModelCandidate:
    """未配 ``LLM_MODELS`` 时的单模型兜底（与 ``app.main`` 同口径）。"""
    mid: str = settings.openai_llm_model
    base: Optional[str] = settings.openai_api_base or None
    style: str = infer_api_style(base or "")
    return AIModelProperties.ModelCandidate(
        id=mid,
        provider=style,
        model=mid,
        url=base,
        api_key=settings.openai_api_key,
        api_style=style,
        priority=0,
        enabled=True,
        supports_thinking=True,
        supports_json_schema=None,
        supports_json_object=None,
        thinking_can_disable=True,
    )


def read_tier_params(settings: Any, tier: str = "standard") -> TierParams:
    """读取指定档位的调用参数。

    Args:
        settings: 应用配置（``app.config.Settings``）。用鸭子类型读取，便于测试替身。
        tier: 档位名（``fast`` / ``standard`` / ``deep``）。

    Returns:
        TierParams：模型名、单次预算（秒）、重试次数、是否需要思考，以及完整候选
        （供厂商方言翻译复用，例如思考开关该按哪个厂商的协议注入）。

    Raises:
        ValueError: 档位名未知。
    """
    fields = _TIER_FIELDS.get(tier)
    if fields is None:
        raise ValueError(f"未知档位：{tier!r}（可选 {sorted(_TIER_FIELDS)}）")
    candidates_attr, timeout_attr, retries_attr = fields

    parsed_models: List[Dict[str, Any]] = list(
        getattr(settings, "llm_models_parsed", None) or []
    )
    pool: List[str] = list(getattr(settings, candidates_attr, None) or [])
    model: str = str(pool[0]) if pool else str(settings.openai_llm_model)

    candidate: Optional[AIModelProperties.ModelCandidate] = None
    for raw in parsed_models:
        if str(raw.get("model_id") or "").strip() == model:
            candidate = _build_candidate(raw, settings)
            break
    if candidate is None:
        if parsed_models:
            logger.warning(
                "档位 %s 的首个候选 %r 不在 LLM_MODELS 注册表中，退回单模型配置；"
                "请检查 %s 与 LLM_MODELS 是否一致",
                tier,
                model,
                candidates_attr,
            )
        candidate = _single_model_candidate(settings)

    timeout_ms: Optional[int] = getattr(settings, timeout_attr, None)
    return TierParams(
        tier=tier,
        model=model,
        timeout_s=(timeout_ms / 1000.0) if timeout_ms else None,
        retries=max(0, int(getattr(settings, retries_attr, 0) or 0)),
        thinking=(tier == DEEP_TIER_NAME),
        candidate=candidate,
    )
