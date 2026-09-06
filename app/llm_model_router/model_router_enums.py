# ==============================
# enums.py
# ==============================
"""
Enum definitions for model providers, tiers, and capabilities.
"""

from enum import Enum


class ModelProvider(str, Enum):
    """Model provider identifiers."""
    OLLAMA = "ollama"
    BAI_LIAN = "bailian"
    SILICON_FLOW = "siliconflow"
    AI_HUB_MIX = "aihubmix"
    NOOP = "noop"

    def matches(self, provider: str) -> bool:
        return provider is not None and provider.lower() == self.value


class Tier(str, Enum):
    """Model tier keys used in chat group."""
    FAST = "fast"
    STANDARD = "standard"
    DEEP = "deep"

    @property
    def key(self) -> str:
        return self.value


class ModelCapability(str, Enum):
    """Model capability categories."""
    CHAT = "Chat"
    EMBEDDING = "Embedding"
    RERANK = "Rerank"

    @property
    def display_name(self) -> str:
        return self.value


# ==============================
# model_target.py
# ==============================
"""
Data class for model target.
"""

from dataclasses import dataclass
from typing import Optional
from app.llm_model_router.model_router_config import AIModelProperties


@dataclass
class ModelTarget:
    """Encapsulates a target model invocation context.

    只携带最小必要信息：模型 ID + 候选配置 + 所属 tier 名。
    provider / timeout 不再预解析塞入目标（详见 Selector/Executor 解耦），
    超时统一由 Executor 依据 ``tier_name`` 在 tier 配置中查询。
    """
    id: str
    candidate: AIModelProperties.ModelCandidate
    tier_name: Optional[str] = None
