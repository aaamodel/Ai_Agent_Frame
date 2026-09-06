# -*- coding: utf-8 -*-
"""异步版模型选择器：与 AsyncModelHealthStore 协同，健康检查处正确 await。

同步版 ``ModelSelector`` 在 ``_build_model_target`` 中直接调用
``health_store.is_unavailable(model_id)``，当健康存储替换为 AsyncModelHealthStore
后该调用返回一个协程对象（永远为 True），导致**所有候选被判为不可用**。
本文件提供与同步版完全等价的算法、但全链路 async：

    is_unavailable() 同步判断  →  await async_is_unavailable()
    构建 targets 时逐个 await  →  保持选择逻辑不变，仅并发语义切换
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.llm_model_router.model_router_config import AIModelProperties, resolve_model_id
from app.llm_model_router.async_model_health import AsyncModelHealthStore
from app.llm_model_router.model_router_enums import Tier, ModelTarget

logger = logging.getLogger(__name__)


class AsyncModelSelector:
    """异步版候选目标构建器。方法签名与同步版一一对应（只是都加了 async / await）。"""

    def __init__(
        self,
        properties: AIModelProperties,
        health_store: AsyncModelHealthStore,
    ) -> None:
        self._properties = properties
        self._health_store = health_store

    # ------------------------------------------------------------------
    # 对外入口（与同步版 ModelSelector 同名，确保业务层可平滑切换）
    # ------------------------------------------------------------------
    async def select_chat_candidates(
        self,
        thinking: bool,
        override: Optional[Tier] = None,
        preferred_model_id: Optional[str] = None,
    ) -> List[ModelTarget]:
        group = self._properties.chat
        if group is None:
            return []
        tier_name = self._resolve_tier_name(group, thinking, override)
        return await self._build_tier_targets(
            group, tier_name, preferred_model_id, thinking
        )

    async def select_embedding_candidates(self) -> List[ModelTarget]:
        return await self._select_candidates(self._properties.embedding)

    async def select_rerank_candidates(self) -> List[ModelTarget]:
        return await self._select_candidates(self._properties.rerank)

    async def select_vlm_candidates(self) -> List[ModelTarget]:
        return await self._select_candidates(self._properties.vlm)

    # ----------------------------- private -----------------------------

    def _resolve_tier_name(
        self,
        group: AIModelProperties.ModelGroup,
        thinking: bool,
        override: Optional[Tier],
    ) -> str:
        # 统一的单条决策链（避免 thinking/override 双信号竞速）：
        #   显式 tier_override > thinking 默认提升到 deep > 全局 default_tier
        # 说明：thinking 不再"抢占"显式 tier（例如 react+thinking → FAST，只在该
        #       tier 内按 supports_thinking 过滤候选），此谓集中决策。
        if override is not None:
            return override.key
        if thinking and group.deep_thinking_tier:
            return group.deep_thinking_tier
        return group.default_tier or ""

    async def _build_tier_targets(
        self,
        group: AIModelProperties.ModelGroup,
        tier_name: str,
        preferred_model_id: Optional[str],
        require_thinking: bool,
    ) -> List[ModelTarget]:
        registry: Dict[str, AIModelProperties.ModelCandidate] = self._build_registry(
            group.candidates
        )

        ordered_ids: List[str] = []
        if preferred_model_id:
            preferred = registry.get(preferred_model_id)
            if preferred is None:
                logger.warning(
                    "Chat preferred model not registered: %s", preferred_model_id
                )
            elif require_thinking and not preferred.supports_thinking:
                logger.warning(
                    "Chat preferred model does not support thinking, ignored: %s",
                    preferred_model_id,
                )
            else:
                ordered_ids.append(preferred_model_id)

        tier = group.tiers.get(tier_name) if group.tiers else None
        if tier is None:
            logger.warning("Chat tier config missing: %s", tier_name)
        else:
            for mid in tier.candidates:
                if mid not in ordered_ids:
                    ordered_ids.append(mid)

        targets: List[ModelTarget] = []
        for mid in ordered_ids:
            candidate = registry.get(mid)
            if candidate is None:
                logger.warning(
                    "Chat tier candidate id not in registry: id=%s, tier=%s",
                    mid,
                    tier_name,
                )
                continue
            if candidate.enabled is False:
                continue
            if require_thinking and not candidate.supports_thinking:
                continue
            # 【关键】异步判断当前模型是否熔断 OPEN，避免同步版中 coroutine 被当真值的 Bug
            unavailable = await self._health_store.is_unavailable(mid)
            if unavailable:
                continue
            target = self._build_model_target(candidate, tier_name)
            if target is not None:
                targets.append(target)
        return targets

    async def _select_candidates(
        self, group: AIModelProperties.ModelGroup
    ) -> List[ModelTarget]:
        if group is None or not group.candidates:
            return []
        ordered_candidates = self._filter_and_sort_candidates(
            group.candidates, group.default_model
        )
        return await self._build_available_targets(ordered_candidates)

    def _filter_and_sort_candidates(
        self,
        candidates: List[AIModelProperties.ModelCandidate],
        first_choice_id: Optional[str],
    ) -> List[AIModelProperties.ModelCandidate]:
        filtered = [
            c for c in candidates if c is not None and c.enabled is not False
        ]

        def key_func(c):
            return (
                not (resolve_model_id(c) == first_choice_id),
                c.priority if c.priority is not None else float("inf"),
                c.id or "",
            )

        return sorted(filtered, key=key_func)

    async def _build_available_targets(
        self, candidates: List[AIModelProperties.ModelCandidate]
    ) -> List[ModelTarget]:
        targets: List[ModelTarget] = []
        for c in candidates:
            mid = resolve_model_id(c)
            if await self._health_store.is_unavailable(mid):
                continue
            target = self._build_model_target(c, None)
            if target is not None:
                targets.append(target)
        return targets

    def _build_model_target(
        self,
        candidate: AIModelProperties.ModelCandidate,
        tier_name: Optional[str],
    ) -> ModelTarget:
        """构造最小目标：只保留 id + candidate + tier_name。

        provider 客户端由 ModelRouter 预构建后经 client_resolver 提供，
        timeout 由 Executor 依据 tier_name 反查 tier 配置——此处不做任何预解析。
        """
        return ModelTarget(resolve_model_id(candidate), candidate, tier_name)

    def _build_registry(
        self, candidates: List[AIModelProperties.ModelCandidate]
    ) -> Dict[str, AIModelProperties.ModelCandidate]:
        registry: Dict[str, AIModelProperties.ModelCandidate] = {}
        if candidates:
            for c in candidates:
                if c is not None:
                    registry[resolve_model_id(c)] = c
        return registry
