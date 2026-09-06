"""
Configuration data classes for AI models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


def resolve_model_id(candidate: "AIModelProperties.ModelCandidate") -> str:
    """Resolve model ID: use explicit id or fallback to provider::model."""
    if candidate.id:
        return candidate.id
    return f"{candidate.provider or 'unknown'}::{candidate.model or 'unknown'}"


@dataclass
class AIModelProperties:
    """Root configuration for AI models."""

    @dataclass
    class ProviderConfig:
        url: Optional[str] = None
        api_key: Optional[str] = None
        endpoints: Dict[str, str] = field(default_factory=dict)

    @dataclass
    class ModelCandidate:
        id: Optional[str] = None
        provider: Optional[str] = None
        model: Optional[str] = None
        url: Optional[str] = None
        dimension: Optional[int] = None
        priority: int = 100
        enabled: bool = True
        supports_thinking: bool = False

    @dataclass
    class TierConfig:
        candidates: List[str] = field(default_factory=list)
        timeout_ms: Optional[int] = None

    @dataclass
    class ModelGroup:
        default_model: Optional[str] = None
        candidates: List["AIModelProperties.ModelCandidate"] = field(default_factory=list)
        default_tier: Optional[str] = None
        deep_thinking_tier: Optional[str] = None
        tiers: Dict[str, "AIModelProperties.TierConfig"] = field(default_factory=dict)

    @dataclass
    class Selection:
        failure_threshold: int = 2
        open_duration_ms: int = 30000

    @dataclass
    class Stream:
        message_chunk_size: int = 5

    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    chat: ModelGroup = field(default_factory=ModelGroup)
    embedding: ModelGroup = field(default_factory=ModelGroup)
    rerank: ModelGroup = field(default_factory=ModelGroup)
    vlm: ModelGroup = field(default_factory=ModelGroup)
    selection: Selection = field(default_factory=Selection)
    stream: Stream = field(default_factory=Stream)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIModelProperties":
        """Build configuration from nested dictionary (e.g., loaded YAML)."""
        # Simplistic conversion; can be extended with proper mapping
        def build_candidate(c: Dict) -> AIModelProperties.ModelCandidate:
            return AIModelProperties.ModelCandidate(
                id=c.get("id"),
                provider=c.get("provider"),
                model=c.get("model"),
                url=c.get("url"),
                dimension=c.get("dimension"),
                priority=c.get("priority", 100),
                enabled=c.get("enabled", True),
                supports_thinking=c.get("supports_thinking", False),
            )

        def build_tier(t: Dict) -> AIModelProperties.TierConfig:
            return AIModelProperties.TierConfig(
                candidates=t.get("candidates", []),
                timeout_ms=t.get("timeout_ms"),
            )

        def build_group(g: Dict) -> AIModelProperties.ModelGroup:
            return AIModelProperties.ModelGroup(
                default_model=g.get("default_model"),
                candidates=[build_candidate(c) for c in g.get("candidates", [])],
                default_tier=g.get("default_tier"),
                deep_thinking_tier=g.get("deep_thinking_tier"),
                tiers={k: build_tier(v) for k, v in g.get("tiers", {}).items()},
            )

        def build_provider(p: Dict) -> AIModelProperties.ProviderConfig:
            return AIModelProperties.ProviderConfig(
                url=p.get("url"),
                api_key=p.get("api_key"),
                endpoints=p.get("endpoints", {}),
            )

        return cls(
            providers={k: build_provider(v) for k, v in data.get("providers", {}).items()},
            chat=build_group(data.get("chat", {})),
            embedding=build_group(data.get("embedding", {})),
            rerank=build_group(data.get("rerank", {})),
            vlm=build_group(data.get("vlm", {})),
            selection=AIModelProperties.Selection(
                failure_threshold=data.get("selection", {}).get("failure_threshold", 2),
                open_duration_ms=data.get("selection", {}).get("open_duration_ms", 30000),
            ),
            stream=AIModelProperties.Stream(
                message_chunk_size=data.get("stream", {}).get("message_chunk_size", 5),
            ),
        )