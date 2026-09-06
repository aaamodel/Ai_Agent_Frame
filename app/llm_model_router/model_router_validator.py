# ==============================
# validator.py
# ==============================
"""
Startup validator for chat tier configuration.
"""

import logging
from typing import List, Set, Dict
from app.llm_model_router.model_router_config import AIModelProperties, resolve_model_id
from app.llm_model_router.model_router_enums import Tier

logger = logging.getLogger(__name__)


class ChatTierConfigValidator:
    """Validates chat tier configuration at startup (fail-fast for structural errors)."""

    def __init__(self, properties: AIModelProperties):
        self._properties = properties

    def validate(self) -> None:
        group = self._properties.chat
        if group is None:
            return

        errors: List[str] = []
        registry = self._build_registry(group, errors)
        registry_ids = set(registry.keys())
        tiers = group.tiers

        if not tiers:
            errors.append("ai.chat.tiers not configured")
        else:
            self._validate_tier_ref(group.default_tier, "default-tier", tiers, errors)
            self._validate_tier_ref(group.deep_thinking_tier, "deep-thinking-tier", tiers, errors)
            self._validate_tier_candidates(tiers, registry_ids, errors)
            self._validate_tier_enum_coverage(tiers, errors)
            self._validate_deep_thinking_candidates(group, tiers, registry, errors)

        if errors:
            raise ValueError("chat tier configuration validation failed:\n - " + "\n - ".join(errors))

        # soft warnings
        self._warn_deep_thinking_support(group, tiers, registry)
        logger.info("chat tier configuration validation passed: tiers=%s", list(tiers.keys()))

    # ---------- private ----------

    def _build_registry(self, group: AIModelProperties.ModelGroup, errors: List[str]) -> Dict[str, AIModelProperties.ModelCandidate]:
        registry = {}
        for c in group.candidates or []:
            if c is None:
                continue
            mid = resolve_model_id(c)
            if mid in registry:
                errors.append(f"chat candidates duplicate id: {mid}")
            else:
                registry[mid] = c
        return registry

    def _validate_tier_ref(self, tier_name: str, label: str,
                           tiers: Dict[str, AIModelProperties.TierConfig], errors: List[str]) -> None:
        if not tier_name:
            errors.append(f"{label} not configured")
        elif tier_name not in tiers:
            errors.append(f"{label} references non-existent tier: {tier_name}")

    def _validate_tier_candidates(self, tiers: Dict[str, AIModelProperties.TierConfig],
                                  registry_ids: Set[str], errors: List[str]) -> None:
        for tier_name, tier in tiers.items():
            timeout_ms = tier.timeout_ms if tier else None
            if timeout_ms is None:
                errors.append(f"tier {tier_name} missing timeout-ms (required for TTFT or call limit)")
            elif timeout_ms <= 0:
                errors.append(f"tier {tier_name} timeout-ms must be positive: {timeout_ms}")

            candidates = tier.candidates if tier else None
            if not candidates:
                errors.append(f"tier {tier_name} candidate list is empty")
                continue
            for mid in candidates:
                if mid not in registry_ids:
                    errors.append(f"tier {tier_name} references unregistered id: {mid}")

    def _validate_tier_enum_coverage(self, tiers: Dict[str, AIModelProperties.TierConfig],
                                     errors: List[str]) -> None:
        for tier_enum in Tier:
            if tier_enum.key not in tiers:
                errors.append(f"Tier enum {tier_enum.name} has no corresponding tier in ai.chat.tiers: {tier_enum.key}")

    def _validate_deep_thinking_candidates(self, group: AIModelProperties.ModelGroup,
                                           tiers: Dict[str, AIModelProperties.TierConfig],
                                           registry: Dict[str, AIModelProperties.ModelCandidate],
                                           errors: List[str]) -> None:
        deep_tier_name = group.deep_thinking_tier
        if not deep_tier_name:
            return
        deep = tiers.get(deep_tier_name)
        if deep is None or not deep.candidates:
            return
        has_thinking = any(
            registry.get(mid) is not None and
            registry[mid].enabled is not False and
            registry[mid].supports_thinking
            for mid in deep.candidates
        )
        if not has_thinking:
            errors.append(f"deep-thinking-tier {deep_tier_name} has no enabled candidate that supports thinking")

    def _warn_deep_thinking_support(self, group: AIModelProperties.ModelGroup,
                                    tiers: Dict[str, AIModelProperties.TierConfig],
                                    registry: Dict[str, AIModelProperties.ModelCandidate]) -> None:
        if not tiers or not group.deep_thinking_tier:
            return
        deep = tiers.get(group.deep_thinking_tier)
        if deep is None or not deep.candidates:
            return
        for mid in deep.candidates:
            cand = registry.get(mid)
            if cand is not None and not cand.supports_thinking:
                logger.warning("deep-thinking-tier candidate lacks supports-thinking, will be filtered: id=%s", mid)
