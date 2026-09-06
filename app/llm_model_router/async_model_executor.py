# -*- coding: utf-8 -*-
"""异步版模型路由执行器：按候选顺序调用，失败自动降级下一个。

与同步版 ``ModelRoutingExecutor`` 思路一致，但整个调用链是 async：
  - ``client_resolver`` 仍允许为同步 callable（查表返回客户端对象即可）；
  - ``caller`` 必须是 async callable，签名为
    ``async caller(client, target: ModelTarget, **execution_kwargs) -> TResult``；
  - 结果通过``permit.mark_success/mark_failure`` 反馈给 AsyncModelHealthStore。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, List, Optional, TypeVar

from app.llm_model_router.async_model_health import AsyncModelHealthStore, CallPermit
from app.llm_model_router.model_router_config import AIModelProperties
from app.llm_model_router.model_router_enums import ModelCapability, ModelTarget

logger = logging.getLogger(__name__)

TClient = TypeVar("TClient")
TResult = TypeVar("TResult")


class AsyncModelRoutingExecutor:
    """异步调用带熔断保护的候选执行器。超时由 tier 配置统一驱动（集中到此处查表）。"""

    def __init__(
        self,
        health_store: AsyncModelHealthStore,
        properties: AIModelProperties,
    ) -> None:
        self._health_store = health_store
        self._properties = properties

    async def execute_with_fallback(
        self,
        capability: ModelCapability,
        targets: List[ModelTarget],
        client_resolver: Callable[[ModelTarget], Optional[TClient]],
        caller: Callable[..., Awaitable[TResult]],
        **execution_kwargs,
    ) -> TResult:
        """
        按 targets 顺序发起异步调用，第一个成功者即返回结果。

        Args:
            capability: 能力类型（仅用于日志/错误信息）
            targets: Selector 输出的候选目标列表（已按 tier/priority/健康 排序过滤）
            client_resolver: ``ModelTarget → Optional[客户端对象]``；返回 None 视为
                客户端缺失，跳过该候选并 warning。
            caller: ``async def caller(client, target, **execution_kwargs) -> TResult``
            **execution_kwargs: 透传给 caller 的实际调用参数（messages/temperature/
                max_tokens/thinking/response_format/tools/tool_choice 等）。

        Returns:
            首个成功的 caller 返回值。

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

            # ------- 3) 异步调用 + 成功/失败 状态回写 -------
            try:
                # 🔴 Tier 级超时：依据 target.tier_name 反查 tier 配置（FAST 15s /
                # STANDARD 30s / DEEP 60s）。超时统一在此按配置查表，无需 Selector 预解析。
                timeout_s: Optional[float] = self._resolve_timeout(target)
                if timeout_s is not None:
                    result = await asyncio.wait_for(
                        caller(client, target, **execution_kwargs),
                        timeout=timeout_s,
                    )
                else:
                    result = await caller(client, target, **execution_kwargs)
            except asyncio.TimeoutError as exc:
                last_exception = exc
                try:
                    await self._health_store.mark_failure(target.id)
                except Exception as health_err:  # pragma: no cover - 防御性分支
                    logger.exception(
                        "%s 超时→mark_failure 内部异常: modelId=%s err=%s",
                        label,
                        target.id,
                        health_err,
                    )
                logger.warning(
                    "%s model tier-timeout (%s), fallback to next. modelId=%s",
                    label,
                    self._timeout_label(target),
                    target.id,
                )
                continue
            except BaseException as exc:  # noqa: BLE001 - 需要捕获所有异常包含 cancel
                last_exception = exc
                try:
                    await self._health_store.mark_failure(target.id)
                except Exception as health_err:  # pragma: no cover - 防御性分支
                    logger.exception(
                        "%s mark_failure 内部异常: modelId=%s err=%s",
                        label,
                        target.id,
                        health_err,
                    )
                logger.warning(
                    "%s model failed, fallback to next. modelId=%s, provider=%s",
                    label,
                    target.id,
                    target.candidate.provider,
                    exc_info=exc,
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
    # 辅助：由 target.tier_name 反查 tier 配置中的超时（集中查表，统一口径）
    # ------------------------------------------------------------------
    def _resolve_timeout(self, target: ModelTarget) -> Optional[float]:
        """返回超时秒数；目标无 tier 或 tier 配置缺失时返回 None（不设超时）。"""
        tier_name = getattr(target, "tier_name", None)
        if not tier_name:
            return None
        tier = (self._properties.chat.tiers or {}).get(tier_name)
        if tier is None or not tier.timeout_ms:
            return None
        return tier.timeout_ms / 1000.0

    def _timeout_label(self, target: ModelTarget) -> str:
        tier_name = getattr(target, "tier_name", None)
        tier = (self._properties.chat.tiers or {}).get(tier_name) if tier_name else None
        if tier is None:
            return "<no-tier>"
        return f"{tier.timeout_ms}ms(tier={tier_name})"
