"""
Configuration data classes for AI models.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# 厂商 API 方言（api_style）
# ---------------------------------------------------------------------------
# 不同 OpenAI 兼容厂商对「思考开关」与「结构化输出」的协议差异（2026-09 实测/官方文档）：
#
# api_style    | 思考开关                                        | json_schema | json_object
# -------------|-------------------------------------------------|-------------|------------
# openai       | 标准参数（不注入额外字段）                       | ✅          | ✅
# dashscope    | extra_body.enable_thinking = true/false（百炼） | ✅(strict)  | ✅
# deepseek     | extra_body.thinking = {"type":"enabled"/...}    | ❌(400)     | ✅
# zhipu        | extra_body.thinking = {"type":"enabled"/...}    | ✅(strict)*  | ✅
#
# 备注：
# - DeepSeek 官方 Chat Completions 2026-09-15 复测（deepseek-flash/v4-pro）：
#   json_schema 仍返回 400 "This response_format type is unavailable now"
#   （json_schema 仅其 Responses API 支持），自动降级 json_object；
#   如改用支持 json_schema 的第三方 DeepSeek 兼容网关，可在该模型条目显式配
#   "supports_json_schema": true 覆盖家族默认。
# - 智谱 GLM-4.7 及更高版本（open.bigmodel.cn / Z.AI api.z.ai）原生支持
#   strict json_schema（全字段必须进 required）；GLM-4.6 及更老版本不支持，
#   需在条目显式配 "supports_json_schema": false。
# - 智谱 GLM-5.3/5.3-flash 强制思考，传 thinking.type=disabled 会报错
#   → 用 thinking_can_disable=False 标注。
STYLE_OPENAI = "openai"
STYLE_DASHSCOPE = "dashscope"
STYLE_DEEPSEEK = "deepseek"
STYLE_ZHIPU = "zhipu"

API_STYLES = frozenset({STYLE_OPENAI, STYLE_DASHSCOPE, STYLE_DEEPSEEK, STYLE_ZHIPU})

# 各厂商结构化输出能力的家族默认值（条目级显式配置可覆盖）
_STYLE_JSON_DEFAULTS: Dict[str, Dict[str, bool]] = {
    STYLE_OPENAI: {"json_schema": True, "json_object": True},
    STYLE_DASHSCOPE: {"json_schema": True, "json_object": True},
    STYLE_DEEPSEEK: {"json_schema": False, "json_object": True},
    STYLE_ZHIPU: {"json_schema": True, "json_object": True},  # GLM-4.7+ 原生 strict 支持
}


def normalize_api_style(raw: Optional[str]) -> str:
    """归一化厂商方言名；无法识别时回退 openai。"""
    if not raw:
        return STYLE_OPENAI
    val = str(raw).strip().lower()
    # 常见别名归一
    alias = {
        "bailian": STYLE_DASHSCOPE,
        "aliyun": STYLE_DASHSCOPE,
        "qwen": STYLE_DASHSCOPE,
        "glm": STYLE_ZHIPU,
        "bigmodel": STYLE_ZHIPU,
        "zai": STYLE_ZHIPU,
    }
    val = alias.get(val, val)
    return val if val in API_STYLES else STYLE_OPENAI


# base_url 域名 → 厂商方言（条目未显式写 provider/api_style 时兜底推断，
# 保证旧版只配百炼 URL 的 .env 零改动自动识别为 dashscope 方言）
_BASE_URL_STYLE_HINTS = (
    ("dashscope.aliyuncs.com", STYLE_DASHSCOPE),
    ("aliyuncs.com", STYLE_DASHSCOPE),
    ("api.deepseek.com", STYLE_DEEPSEEK),
    ("deepseek", STYLE_DEEPSEEK),
    ("bigmodel.cn", STYLE_ZHIPU),
    ("z.ai", STYLE_ZHIPU),
)


def infer_api_style(base_url: Optional[str]) -> str:
    """未显式声明厂商时，按 base_url 域名推断 API 方言；无法识别回退 openai。"""
    u = (base_url or "").lower()
    for fragment, style in _BASE_URL_STYLE_HINTS:
        if fragment in u:
            return style
    return STYLE_OPENAI


def expand_secret(raw: Optional[str]) -> str:
    """展开 ``${ENV_VAR}`` 形式的密钥引用；普通字符串原样返回（空值返回空串）。"""
    s = str(raw or "").strip()
    if len(s) >= 3 and s.startswith("${") and s.endswith("}"):
        return os.environ.get(s[2:-1].strip(), "").strip()
    return s


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

        # ---- 多厂商接入（每个候选可独立网关/密钥/方言）----
        api_key: Optional[str] = None
        """候选级 API Key；优先级高于 provider 级 api_key。"""

        api_style: str = "openai"
        """厂商方言：openai / dashscope / deepseek / zhipu，决定思考开关等
        非标准参数如何翻译。"""

        supports_json_schema: Optional[bool] = None
        """是否支持 response_format=json_schema（严格结构化输出）。
        None=按 api_style 家族默认；不支持时 caller 自动降级 json_object。"""

        supports_json_object: Optional[bool] = None
        """是否支持 response_format=json_object。None=按家族默认。"""

        thinking_can_disable: bool = True
        """能否显式关闭思考。GLM-5.3/5.3-flash 等强制思考模型置 False，
        此时 thinking=False 不注入 disabled（避免 400）。"""

        def json_schema_supported(self) -> bool:
            if self.supports_json_schema is not None:
                return bool(self.supports_json_schema)
            return _STYLE_JSON_DEFAULTS.get(self.api_style, _STYLE_JSON_DEFAULTS[STYLE_OPENAI])["json_schema"]

        def json_object_supported(self) -> bool:
            if self.supports_json_object is not None:
                return bool(self.supports_json_object)
            return _STYLE_JSON_DEFAULTS.get(self.api_style, _STYLE_JSON_DEFAULTS[STYLE_OPENAI])["json_object"]

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
            style = normalize_api_style(c.get("api_style") or c.get("provider"))
            return AIModelProperties.ModelCandidate(
                id=c.get("id"),
                provider=c.get("provider"),
                model=c.get("model"),
                url=c.get("url"),
                dimension=c.get("dimension"),
                priority=c.get("priority", 100),
                enabled=c.get("enabled", True),
                supports_thinking=c.get("supports_thinking", False),
                api_key=c.get("api_key"),
                api_style=style,
                supports_json_schema=c.get("supports_json_schema"),
                supports_json_object=c.get("supports_json_object"),
                thinking_can_disable=c.get("thinking_can_disable", True),
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