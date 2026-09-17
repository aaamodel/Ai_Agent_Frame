# -*- coding: utf-8 -*-
"""知识图谱抽取链路的"档位参数同源"单测（第 2 组 / spec: `platform/llm-call-budget`）。

覆盖：
- 2.1 档位参数唯一入口（模型名 / 单次预算 / 重试次数 / 思考开关），含未注册模型回退；
- 2.4 思考开关按**厂商方言**翻译（可关闭 → 注入关闭；不可关闭 → 跳过且不报错）；
- 2.2 / 2.3 抽取调用确实携带档位预算、且不叠加第三方库自带的重试。

⚠️ 2.2/2.3 用**源码级守卫**而不是实例断言：导入 `light_rag` 会在模块级构造 LightRAG
实例并加载图谱工作区（有 IO 与副作用），而这里要守住的恰恰是"调用点怎么写"。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.llm_model_router.async_openai_caller import apply_thinking_dialect  # noqa: E402
from app.llm_model_router.model_router_config import (  # noqa: E402
    STYLE_DASHSCOPE,
    STYLE_ZHIPU,
    AIModelProperties,
)
from app.llm_model_router.tier_params import read_tier_params  # noqa: E402

_LIGHTRAG_SOURCE: str = (
    _REPO_ROOT / "app" / "infrastructure" / "knowledgebase" / "light_rag.py"
).read_text(encoding="utf-8")


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        openai_api_key="global-key",
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_llm_model="qwen3.8-flash",
        llm_models_parsed=[
            {"model_id": "qwen3.8-flash", "provider": "dashscope"},
            {
                "model_id": "glm-4.7",
                "provider": "zhipu",
                "base_url": "https://open.bigmodel.cn/api/paas/v4",
                "api_key": "${ZHIPU_TEST_KEY}",
                "thinking_can_disable": False,
            },
        ],
        llm_tier_fast_parsed=["glm-4.7"],
        llm_tier_fast_timeout_ms=50_000,
        llm_tier_fast_retries=1,
        llm_tier_standard_parsed=["qwen3.8-flash", "glm-4.7"],
        llm_tier_standard_timeout_ms=70_000,
        llm_tier_standard_retries=2,
        llm_tier_deep_parsed=["qwen3.8-flash"],
        llm_tier_deep_timeout_ms=90_000,
        llm_tier_deep_retries=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _candidate(*, api_style: str, can_disable: bool) -> AIModelProperties.ModelCandidate:
    return AIModelProperties.ModelCandidate(
        id=f"m-{api_style}",
        provider=api_style,
        model="m",
        api_key="k",
        url="https://example.invalid/v1",
        api_style=api_style,
        thinking_can_disable=can_disable,
    )


# ---------------------------------------------------------------------------
# 2.1 唯一入口
# ---------------------------------------------------------------------------
def test_reads_all_four_knobs_from_tier_config():
    settings = _settings()

    standard = read_tier_params(settings, "standard")
    assert standard.model == "qwen3.8-flash", "取候选池首个（LightRAG 只接受单模型）"
    assert standard.timeout_s == 70.0
    assert standard.retries == 2
    assert standard.thinking is False

    fast = read_tier_params(settings, "fast")
    assert fast.model == "glm-4.7" and fast.timeout_s == 50.0

    deep = read_tier_params(settings, "deep")
    assert deep.thinking is True, "deep 档才要求思考"
    assert deep.retries == 0


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        read_tier_params(_settings(), "nonexistent")


def test_candidate_carries_provider_dialect_and_resolved_key(monkeypatch):
    monkeypatch.setenv("ZHIPU_TEST_KEY", "resolved-by-env")
    candidate = read_tier_params(_settings(), "fast").candidate
    assert candidate.model == "glm-4.7"
    assert candidate.url == "https://open.bigmodel.cn/api/paas/v4"
    assert candidate.api_key == "resolved-by-env", "${ENV} 形式必须被展开"
    assert candidate.thinking_can_disable is False


def test_unregistered_model_falls_back_with_warning(caplog):
    settings = _settings(llm_tier_standard_parsed=["not-registered"])
    with caplog.at_level("WARNING"):
        params = read_tier_params(settings, "standard")
    assert params.model == "not-registered"
    assert params.candidate.id == settings.openai_llm_model
    assert any("不在 LLM_MODELS 注册表" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 2.4 思考开关按厂商方言翻译
# ---------------------------------------------------------------------------
def test_thinking_dialect_dashscope_injects_enable_thinking():
    extra: dict = {}
    apply_thinking_dialect(
        _candidate(api_style=STYLE_DASHSCOPE, can_disable=True), False, extra
    )
    assert extra == {"enable_thinking": False}


def test_thinking_dialect_zhipu_enabled():
    extra: dict = {}
    apply_thinking_dialect(_candidate(api_style=STYLE_ZHIPU, can_disable=True), True, extra)
    assert extra == {"thinking": {"type": "enabled"}}


def test_thinking_dialect_skips_when_model_cannot_disable():
    """不可关闭思考的模型：跳过注入且不报错（发 disabled 会 400）。"""
    extra: dict = {}
    apply_thinking_dialect(
        _candidate(api_style=STYLE_ZHIPU, can_disable=False), False, extra
    )
    assert extra == {}


# ---------------------------------------------------------------------------
# 2.2 / 2.3 调用点守卫
# ---------------------------------------------------------------------------
def test_extraction_call_carries_tier_budget_and_retries():
    assert "read_tier_params(light_rag_settings, _LIGHTRAG_TIER)" in _LIGHTRAG_SOURCE
    assert "timeout=int(_tier_params.timeout_s)" in _LIGHTRAG_SOURCE, (
        "单次预算必须按调用传递（HTTP 层），而不是各用各的默认"
    )
    assert "run_with_attempt_budget(" in _LIGHTRAG_SOURCE
    assert "retries=_tier_params.retries" in _LIGHTRAG_SOURCE
    assert "apply_thinking_dialect(_tier_params.candidate" in _LIGHTRAG_SOURCE


def test_extraction_call_bypasses_library_builtin_retry():
    """必须绕开 LightRAG 自带的 3 次重试，否则与档位重试叠加成 N×3。"""
    assert '__wrapped__' in _LIGHTRAG_SOURCE
    assert "_complete_raw(" in _LIGHTRAG_SOURCE
    assert "openai_complete_if_cache(" not in _LIGHTRAG_SOURCE.replace(
        "getattr(openai_complete_if_cache,", ""
    ).replace("_complete_raw = openai_complete_if_cache", "")


def test_extraction_keeps_concurrency_and_cache_untouched():
    """2.3：不改并发与缓存机制——**不接入统一路由实例**（只共用参数）。"""
    assert "openai_complete_if_cache" in _LIGHTRAG_SOURCE
    # 断言的是"不使用路由实例"，而不是子串（`app.llm_model_router` 是 import 路径）
    assert "ModelRouter" not in _LIGHTRAG_SOURCE
    assert "get_llm(" not in _LIGHTRAG_SOURCE
    assert "LightRAG(" in _LIGHTRAG_SOURCE
