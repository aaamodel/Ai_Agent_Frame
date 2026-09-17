# -*- coding: utf-8 -*-
"""异步版模型路由执行器：按候选顺序调用，失败自动降级下一个。

整个调用链是 async：
  - ``client_resolver`` 仍允许为同步 callable（查表返回客户端对象即可）；
  - 单次调用固定走 OpenAI 兼容协议的 ``async_openai_chat_caller``。历史上曾以
    ``caller`` 参数注入以支持"多协议/多 capability"，但实际只有一个 caller 实现、
    一种 capability（CHAT），该泛型属于投机设计，已内联为本模块直接调用；
  - 结果通过``permit.mark_success/mark_failure`` 反馈给 AsyncModelHealthStore。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, List, Optional, Tuple

from app.llm_model_router.async_model_health import AsyncModelHealthStore, CallPermit
from app.llm_model_router.async_openai_caller import (
    AsyncOpenAICallResult,
    async_openai_chat_caller,
)
from app.llm_model_router.model_router_config import AIModelProperties
from app.llm_model_router.model_router_enums import ModelCapability, ModelTarget

logger = logging.getLogger(__name__)

# 外层安全网的宽限（秒）：正常情况下 HTTP 层会先按请求级预算超时并抛出可辨识的
# 异常；这个宽限只用于兜住"连 HTTP 层超时都没能触发"的挂死情况。
_TIMEOUT_GRACE_S: float = 1.0


def is_retryable(exc: BaseException) -> bool:
    """该异常是否值得重试（瞬时故障）。

    非瞬时故障（参数错误 / 鉴权失败 / 模型不存在）重试只是重复烧钱，
    直接交给候选降级或上层处理。
    """
    if isinstance(exc, asyncio.TimeoutError):
        return True
    try:
        # 就近导入：只在错误分支用到这些类型
        from openai import APIConnectionError, APIStatusError, RateLimitError
    except Exception:  # pragma: no cover - openai 不可用时不做重试判断
        return False
    # APITimeoutError 继承自 APIConnectionError，已被这条覆盖
    if isinstance(exc, (APIConnectionError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError):
        return int(getattr(exc, "status_code", 0) or 0) >= 500
    return False


async def run_with_attempt_budget(
    one_attempt: Callable[[], Awaitable[Any]],
    *,
    timeout_s: Optional[float],
    retries: int,
    budget_label: str,
    subject: str,
) -> Tuple[Optional[Any], Optional[BaseException]]:
    """通用「单次尝试独立预算」执行器。

    ⚠️ 这是"逐次独立预算 + 只在瞬时故障重试"的**唯一实现**：模型路由的候选调用与
    知识图谱抽取的调用都走它，避免两处各写一套而口径漂移。

    为什么要分两层：只用一层 ``wait_for`` 包住整个调用时，"首次尝试"与"它的重试"
    会共用同一份预算，第二次尝试注定被掐掉（实测 ``Retrying request ... in 0.47s``
    紧接 ``tier-timeout (50000ms)``）。所以：**HTTP 层**按请求携带单次预算、
    **外层** ``wait_for`` 只作 ``+宽限`` 的安全网。

    Args:
        one_attempt: 无参可调用，每次调用产生**一次**尝试的 awaitable。
        timeout_s: 单次尝试预算（秒）；``None`` 表示不设预算。
        retries: 额外重试次数（不含首次）。
        budget_label: 预算的可读描述（写进日志）。
        subject: 调用主体标识（模型 id / 组件名，写进日志）。

    Returns:
        ``(结果, 最后一次异常)``；成功时异常为 ``None``。
    """
    attempts: int = 1 + max(0, int(retries))
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            call = one_attempt()
            if timeout_s is None:
                return await call, None
            return await asyncio.wait_for(call, timeout=timeout_s + _TIMEOUT_GRACE_S), None
        except BaseException as exc:  # noqa: BLE001 - 含 cancel，逐次判断是否值得重试
            last_error = exc
            elapsed_ms: float = (time.perf_counter() - started) * 1000.0
            logger.warning(
                "%s 第 %d/%d 次尝试失败（单次预算 %s，该次耗时 %.0fms）: %s",
                subject,
                attempt,
                attempts,
                budget_label,
                elapsed_ms,
                type(exc).__name__,
            )
            if attempt >= attempts or not is_retryable(exc):
                break
    return None, last_error


class AsyncModelRoutingExecutor:
    """异步调用带熔断保护的候选执行器。超时由 tier 配置统一驱动（集中到此处查表）。"""

    def __init__(
        self,
        health_store: AsyncModelHealthStore,
        properties: AIModelProperties,
    ) -> None:
        self._health_store = health_store
        self._properties = properties

    async def execute_with_candidate_fallback(
        self,
        capability: ModelCapability,
        targets: List[ModelTarget],
        client_resolver: Callable[[ModelTarget], Optional[Any]],
        **execution_kwargs,
    ) -> AsyncOpenAICallResult:
        """
        按 targets 顺序发起异步调用，第一个成功者即返回结果。

        ⚠️ 命名说明：本方法是本执行器**唯一**的执行入口（不存在"不带 fallback 的
        execute"）。这里的 fallback 指的是**候选级故障转移**（failover）——在
        targets 之间逐个转移，而非"主路径失败后启用的备用路径"。后者请见
        ReAct 的 FC→文本协议、Orchestrator 的 plan→react，那些才是真正的 fallback。

        Args:
            capability: 能力类型（仅用于日志/错误信息）
            targets: Selector 输出的候选目标列表（已按 tier/priority/健康 排序过滤）
            client_resolver: ``ModelTarget → Optional[客户端对象]``；返回 None 视为
                客户端缺失，跳过该候选并 warning。
            **execution_kwargs: 透传给 ``async_openai_chat_caller`` 的调用参数
                （messages/temperature/max_tokens/thinking/response_format/
                tools/tool_choice 等）。

        Returns:
            首个成功候选的 AsyncOpenAICallResult。

        Raises:
            RuntimeError: 所有候选全部失败（附带最后一个异常的 cause 链）。
        """
        label = capability.display_name
        if not targets:
            raise RuntimeError(f"No {label} model candidates available")

        last_exception: Optional[BaseException] = None
        for target in targets:
            # ------- 1) 解析客户端（同步查表即可，不 async）-------
            client = client_resolver(target)
            if client is None:
                logger.warning(
                    "%s provider client missing: provider=%s, modelId=%s",
                    label,
                    target.candidate.provider,
                    target.id,
                )
                continue

            # ------- 2) 健康熔断许可（异步，内部拿 lock）-------
            permit: Optional[CallPermit] = await self._health_store.allow_call(target.id)
            if permit is None:
                # Selector 理论上已过滤；但并发下可能刚好进入 OPEN，此处双重保险
                logger.info(
                    "%s model rejected by circuit breaker, skip: modelId=%s",
                    label,
                    target.id,
                )
                continue

            # ------- 3) 异步调用（按尝试逐次独立预算）+ 成功/失败 状态回写 -------
            result, attempt_error = await self._call_with_attempts(
                client, target, execution_kwargs
            )
            if result is None:
                last_exception = attempt_error
                try:
                    await self._health_store.mark_failure(target.id)
                except Exception as health_err:  # pragma: no cover - 防御性分支
                    logger.exception(
                        "%s mark_failure 内部异常: modelId=%s err=%s",
                        label,
                        target.id,
                        health_err,
                    )
                # 超时与普通失败分开记录：前者要能看出"预算不够"，后者要能看出真实错误
                if isinstance(attempt_error, asyncio.TimeoutError):
                    logger.warning(
                        "%s 已耗尽配置的尝试次数，超时收场 (%s), fallback to next. modelId=%s",
                        label,
                        self._timeout_label(target),
                        target.id,
                    )
                else:
                    logger.warning(
                        "%s model failed, fallback to next. modelId=%s, provider=%s",
                        label,
                        target.id,
                        target.candidate.provider,
                        exc_info=attempt_error,
                    )
                # permit 因失败已通过 mark_failure 重置，无需额外 release
                continue

            try:
                await self._health_store.mark_success(target.id)
            except Exception as health_err:  # pragma: no cover
                logger.exception(
                    "%s mark_success 内部异常: modelId=%s err=%s",
                    label,
                    target.id,
                    health_err,
                )
            return result

        # ------- 所有候选均失败 -------
        msg = f"All {label} model candidates failed"
        if last_exception is not None:
            raise RuntimeError(msg) from last_exception
        raise RuntimeError(msg)

    # ------------------------------------------------------------------
    # 辅助：按「单次尝试独立预算」执行（tier 配置集中查表，统一口径）
    # ------------------------------------------------------------------
    async def _call_with_attempts(
        self,
        client: Any,
        target: ModelTarget,
        execution_kwargs: dict,
    ) -> Tuple[Optional[AsyncOpenAICallResult], Optional[BaseException]]:
        """按「每次尝试独立预算」调用该候选，返回 ``(结果, 最后一次异常)``。

        ⚠️ 为什么不能只用一层 ``wait_for`` 包住整个候选调用：那样"首次尝试"与
        "它触发的重试"会**共用同一份预算**，第二次尝试注定被掐掉——实测日志形态为
        ``Retrying request ... in 0.47s`` 紧接 ``tier-timeout (50000ms(tier=fast))``，
        看起来像"模型算得慢"，实际是预算被前一次尝试吃掉了。

        实现分两层：
          - **HTTP 层**：``request_timeout`` = 该档位的单次预算，由底层 httpx 触发，
            异常类型可辨识（属于"提供方慢"而非"我们掐断"）；
          - **外层**：``wait_for(单次预算 + 宽限)`` 仅作安全网，兜住挂死。

        重试只针对**可重试故障**（超时 / 连接错误 / 限流 / 5xx）。参数错误、鉴权失败、
        模型不存在这类重试只是重复烧钱，直接交给候选降级。
        """
        timeout_s: Optional[float] = self._resolve_timeout(target)
        tier = self._tier_of(target)

        def _one_attempt() -> Any:
            return async_openai_chat_caller(
                client, target, request_timeout=timeout_s, **execution_kwargs
            )

        return await run_with_attempt_budget(
            _one_attempt,
            timeout_s=timeout_s,
            retries=int(getattr(tier, "retries", 0) or 0),
            budget_label=self._timeout_label(target),
            subject=target.id,
        )

    def _tier_of(self, target: ModelTarget) -> Optional[Any]:
        """由 target.tier_name 反查 tier 配置（超时与重试次数共用这一份查表）。"""
        tier_name = getattr(target, "tier_name", None)
        if not tier_name:
            return None
        return (self._properties.chat.tiers or {}).get(tier_name)

    def _resolve_timeout(self, target: ModelTarget) -> Optional[float]:
        """返回**单次尝试**的超时秒数；目标无 tier 或 tier 配置缺失时返回 None。"""
        tier = self._tier_of(target)
        if tier is None or not tier.timeout_ms:
            return None
        return tier.timeout_ms / 1000.0

    def _timeout_label(self, target: ModelTarget) -> str:
        tier = self._tier_of(target)
        if tier is None:
            return "<no-tier>"
        tier_name = getattr(target, "tier_name", None)
        return f"{tier.timeout_ms}ms(tier={tier_name})"
