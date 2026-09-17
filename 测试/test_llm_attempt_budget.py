# -*- coding: utf-8 -*-
"""模型调用的「单次尝试独立预算」单测。

覆盖：
- 1.1 档位重试次数的默认值与取值校验；
- 1.2 客户端关闭 SDK 隐式重试；
- 1.3 单次预算按**请求**传递，而不固化在客户端上；
- 1.4 每次尝试独立计时（首次超时不吞掉第二次的预算）；
- 1.5 逐次记录尝试，日志可区分"某次尝试超时"与"重试后仍失败"；
- 1.7 预算耗尽后返回**最后一次**失败原因。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.llm_model_router import async_model_executor as executor_module  # noqa: E402
from app.llm_model_router import async_openai_caller as caller_module  # noqa: E402
from app.llm_model_router.async_model_executor import (  # noqa: E402
    AsyncModelRoutingExecutor,
)
from app.llm_model_router.model_router_config import AIModelProperties  # noqa: E402


def _properties(*, timeout_ms: int = 1000, retries: int = 1) -> AIModelProperties:
    return AIModelProperties.from_dict(
        {
            "providers": {},
            "chat": {
                "candidates": [{"id": "m1", "provider": "dashscope", "model": "qwen-x"}],
                "tiers": {
                    "fast": {
                        "candidates": ["m1"],
                        "timeout_ms": timeout_ms,
                        "retries": retries,
                    }
                },
            },
        }
    )


def _target(tier: str = "fast") -> SimpleNamespace:
    """执行器与调用器只按鸭子类型取用这几个字段。"""
    return SimpleNamespace(
        id="m1",
        tier_name=tier,
        candidate=SimpleNamespace(provider="dashscope", model="qwen-x"),
    )


# ---------------------------------------------------------------------------
# 1.1 档位重试次数：默认值与取值校验
# ---------------------------------------------------------------------------
def test_tier_retries_explicit_value_is_used():
    assert _properties(retries=0).chat.tiers["fast"].retries == 0
    assert _properties(retries=2).chat.tiers["fast"].retries == 2


def test_tier_retries_defaults_to_one_when_absent():
    props = AIModelProperties.from_dict(
        {"providers": {}, "chat": {"candidates": [], "tiers": {"fast": {"timeout_ms": 1000}}}}
    )
    assert props.chat.tiers["fast"].retries == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    ((-3, 0), (99, 5), ("abc", 1)),
    ids=["负数收敛为0", "超上限收敛为5", "无法解析回退默认1"],
)
def test_tier_retries_is_clamped(raw, expected):
    props = AIModelProperties.from_dict(
        {
            "providers": {},
            "chat": {
                "candidates": [],
                "tiers": {"fast": {"timeout_ms": 1000, "retries": raw}},
            },
        }
    )
    assert props.chat.tiers["fast"].retries == expected


# ---------------------------------------------------------------------------
# 1.2 客户端关闭 SDK 隐式重试
# ---------------------------------------------------------------------------
def test_client_disables_sdk_retries():
    candidate = AIModelProperties.ModelCandidate(
        id="m1",
        provider="dashscope",
        model="qwen-x",
        api_key="test-key",
        url="https://example.invalid/v1",
        api_style="dashscope",
    )
    client = caller_module.async_build_openai_client(candidate, None)
    assert client is not None
    # 变更前这里是 SDK 默认的 2：隐式重试会把"多次尝试"藏进一次调用，
    # 与候选级超时共用预算，并吞掉首次失败的真实原因。
    assert client.max_retries == 0


# ---------------------------------------------------------------------------
# 1.3 单次预算按请求传递
# ---------------------------------------------------------------------------
def test_request_timeout_is_written_into_request_params(monkeypatch):
    captured: dict = {}

    async def fake_completion(client, params, *, model, provider):
        captured.update(params)
        return SimpleNamespace(choices=[], usage=None, model="qwen-x")

    monkeypatch.setattr(caller_module, "_chat_completion_with_langfuse", fake_completion)

    asyncio.run(
        caller_module.async_openai_chat_caller(
            None,
            _target(),
            messages=[{"role": "user", "content": "hi"}],
            request_timeout=12.5,
        )
    )
    assert captured["timeout"] == 12.5


def test_request_timeout_absent_leaves_client_default(monkeypatch):
    captured: dict = {}

    async def fake_completion(client, params, *, model, provider):
        captured.update(params)
        return SimpleNamespace(choices=[], usage=None, model="qwen-x")

    monkeypatch.setattr(caller_module, "_chat_completion_with_langfuse", fake_completion)

    asyncio.run(
        caller_module.async_openai_chat_caller(
            None, _target(), messages=[{"role": "user", "content": "hi"}]
        )
    )
    assert "timeout" not in captured


def test_each_tier_passes_its_own_budget(monkeypatch):
    """同一模型在不同档位 MUST 各自携带该档的单次预算。

    这是"不能把超时固化在客户端上"的直接理由：一个模型可能同时出现在多个档位
    （如 glm-4.7 在 FAST 与 STANDARD），而两档预算不同。
    """
    observed: list = []

    async def fake_caller(client, target, *, request_timeout=None, **kwargs):
        observed.append((target.tier_name, request_timeout))
        return "OK"

    monkeypatch.setattr(executor_module, "async_openai_chat_caller", fake_caller)
    props = AIModelProperties.from_dict(
        {
            "providers": {},
            "chat": {
                "candidates": [{"id": "m1", "provider": "dashscope", "model": "qwen-x"}],
                "tiers": {
                    "fast": {"candidates": ["m1"], "timeout_ms": 1000, "retries": 0},
                    "standard": {"candidates": ["m1"], "timeout_ms": 2500, "retries": 0},
                },
            },
        }
    )
    executor = AsyncModelRoutingExecutor(None, props)
    asyncio.run(executor._call_with_attempts(None, _target("fast"), {}))
    asyncio.run(executor._call_with_attempts(None, _target("standard"), {}))

    assert observed == [("fast", 1.0), ("standard", 2.5)]


def test_response_without_model_falls_back_to_target_id(monkeypatch):
    """响应不含 `model` 字段时不得抛 AttributeError。

    该行原先写作 ``getattr(resp, "model")``（**没有默认值**），属性缺失即抛
    AttributeError。官方 SDK 的 ChatCompletion 里 ``model`` 是必填字段、生产上
    基本不可达，但同类响应对象（测试替身 / 非标准网关）一旦触发，异常会走到执行器
    的 ``except BaseException`` —— 后果是把**一个健康的候选记成失败**（误伤熔断计数），
    且日志里只留下一条与真实问题无关的报错。
    """

    async def fake_completion(client, params, *, model, provider):
        return SimpleNamespace(choices=[], usage=None)  # 刻意不带 model

    monkeypatch.setattr(caller_module, "_chat_completion_with_langfuse", fake_completion)

    result = asyncio.run(
        caller_module.async_openai_chat_caller(
            None, _target(), messages=[{"role": "user", "content": "hi"}]
        )
    )
    assert result.model_id == "m1", "缺 model 时应回退为 target.id"


def test_message_without_content_field_does_not_raise(monkeypatch):
    """同上：``message`` 缺 ``content`` 属性时不得抛 AttributeError。

    ``or ""`` 只能兜住"值为 None"，兜不住"没有这个属性"——这里补的是后半句。
    """

    async def fake_completion(client, params, *, model, provider):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace())],  # 刻意不带 content
            usage=None,
            model="qwen-x",
        )

    monkeypatch.setattr(caller_module, "_chat_completion_with_langfuse", fake_completion)

    result = asyncio.run(
        caller_module.async_openai_chat_caller(
            None, _target(), messages=[{"role": "user", "content": "hi"}]
        )
    )
    assert result.content == ""


# ---------------------------------------------------------------------------
# 1.4 / 1.5 / 1.7 逐次尝试
# ---------------------------------------------------------------------------
def _run_attempts(monkeypatch, *, retries: int, timeout_ms: int, side_effects: list):
    """用假 caller 驱动 ``_call_with_attempts``。

    Returns:
        ``(结果, 最后一次异常, 每次尝试实际收到的 request_timeout 列表)``
    """
    observed: list = []

    async def fake_caller(client, target, *, request_timeout=None, **kwargs):
        observed.append(request_timeout)
        outcome = side_effects[len(observed) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(executor_module, "async_openai_chat_caller", fake_caller)
    executor = AsyncModelRoutingExecutor(
        None, _properties(timeout_ms=timeout_ms, retries=retries)
    )
    result, error = asyncio.run(executor._call_with_attempts(None, _target(), {}))
    return result, error, observed


def test_first_attempt_timeout_then_second_succeeds(monkeypatch):
    """首次尝试超时后，第二次 MUST 获得独立的完整预算并成功。"""
    result, error, timeouts = _run_attempts(
        monkeypatch,
        retries=1,
        timeout_ms=1000,
        side_effects=[asyncio.TimeoutError(), "OK"],
    )
    assert result == "OK"
    assert error is None
    assert len(timeouts) == 2, "首次超时后必须真的发生第二次尝试"
    assert timeouts == [1.0, 1.0], "每次尝试都必须拿到同一份独立预算"


def test_retries_zero_means_single_attempt(monkeypatch):
    result, error, timeouts = _run_attempts(
        monkeypatch,
        retries=0,
        timeout_ms=1000,
        side_effects=[asyncio.TimeoutError(), "OK"],
    )
    assert result is None
    assert isinstance(error, asyncio.TimeoutError)
    assert len(timeouts) == 1


def test_non_retryable_error_does_not_retry(monkeypatch):
    """参数错/鉴权失败这类重试只是重复烧钱，直接交给候选降级。"""
    boom = ValueError("bad request")
    result, error, timeouts = _run_attempts(
        monkeypatch, retries=2, timeout_ms=1000, side_effects=[boom, "OK"]
    )
    assert result is None
    assert error is boom
    assert len(timeouts) == 1


def test_exhausted_attempts_returns_last_error(monkeypatch):
    """1.7：预算耗尽后返回**最后一次**失败原因，而不是第一次。"""
    first = asyncio.TimeoutError("first")
    last = asyncio.TimeoutError("last")
    result, error, timeouts = _run_attempts(
        monkeypatch, retries=1, timeout_ms=1000, side_effects=[first, last]
    )
    assert result is None
    assert error is last
    assert len(timeouts) == 2


def test_attempt_logs_carry_index_budget_and_elapsed(monkeypatch, caplog):
    """1.5：日志能区分第几次尝试、单次预算、该次耗时。"""
    with caplog.at_level("WARNING"):
        _run_attempts(
            monkeypatch,
            retries=1,
            timeout_ms=1000,
            side_effects=[asyncio.TimeoutError(), asyncio.TimeoutError()],
        )
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "第 1/2 次尝试失败" in joined
    assert "第 2/2 次尝试失败" in joined
    assert "该次耗时" in joined
    assert "1000ms(tier=fast)" in joined
