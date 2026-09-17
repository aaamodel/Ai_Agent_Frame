# -*- coding: utf-8 -*-
"""真实 token 用量采集（用于「单轮 token 成本」指标）。

## 为什么需要单独采集

``AgentResponse`` / ``ChatResponse`` 都不返回 token 用量——项目把 usage 直接
推给了 Langfuse（``_enrich_langfuse_generation``），本地拿不到。而手册要求的
四个必测指标里就有「**单轮 token 成本**」，不能空着。

本模块的做法是**在真实调用点上旁路采集**，而不是估算：

    所有真实 LLM 调用都收敛到 ``app.llm_model_router.async_openai_caller``
    的模块级函数 ``_chat_completion_with_langfuse(client, params, ...)``，
    且该名字是**调用时**才在模块命名空间里查找的（``async_openai_chat_caller``
    内部直接引用它）。

因此在 ``async_openai_caller`` 模块上替换这个属性，就能拿到**每一次**真实
LLM 响应的 ``resp.usage``（prompt_tokens / completion_tokens / total_tokens），
用完即还原。

## 三条原则（避免"数据不准确"）

1. **绝不造假**：拿不到 usage 就是 ``0``，报告里显式写"不可用"，
   而不是用字符数估算一个看起来漂亮的数字。
2. **采集失败不影响被测链路**：任何异常只降级为"少记一次调用"，
   绝不向 LLM 调用方抛异常（否则会污染熔断/降级统计）。
3. **可追溯**：记录每次调用的模型名，便于按 ``monitoring/model_pricing.yaml``
   分模型计价（不同 tier 单价差 10 倍以上，混在一起算会失真）。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class UsageRecorder:
    """累计一次评测运行内的 LLM 调用与 token 用量。"""

    calls: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    by_model: Dict[str, Dict[str, int]] = field(default_factory=dict)
    """按模型聚合：``{model: {"calls": n, "input": a, "output": b, "total": c}}``。"""

    # ------------------------------------------------------------------
    def record(self, model: str, usage: Any) -> None:
        """记录一次调用的 usage（``usage`` 为 OpenAI 风格对象或 None）。"""
        self.calls += 1
        in_tok: int = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok: int = int(getattr(usage, "completion_tokens", 0) or 0)
        tot_tok: int = int(getattr(usage, "total_tokens", 0) or 0)
        if not tot_tok:
            tot_tok = in_tok + out_tok

        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.total_tokens += tot_tok

        bucket: Dict[str, int] = self.by_model.setdefault(
            str(model or "unknown"),
            {"calls": 0, "input": 0, "output": 0, "total": 0},
        )
        bucket["calls"] += 1
        bucket["input"] += in_tok
        bucket["output"] += out_tok
        bucket["total"] += tot_tok

    # ------------------------------------------------------------------
    def per_turn(self, turns: int) -> Optional[float]:
        """单轮平均 token；``turns <= 0`` 时返回 ``None``（表示不可计算）。

        返回 ``None`` 而不是 0：0 会被误读成"成本极低"，
        而 ``None`` 让 ``evaluate_gate`` 跳过该项，不会产生假信号。
        """
        if turns <= 0:
            return None
        return self.total_tokens / turns

    def as_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "by_model": self.by_model,
        }


@contextmanager
def record_llm_usage() -> Iterator[UsageRecorder]:
    """上下文管理器：在 with 块内旁路采集所有真实 LLM 调用的 usage。

    实现细节：替换 ``async_openai_caller._chat_completion_with_langfuse``。
    该函数是模块级全局名，且被同模块的 ``async_openai_chat_caller`` 以**运行时
    名称查找**方式调用，所以替换模块属性即可全局生效。

    异常安全：
        - 包装函数内部用 try/except 包住「读 usage + 记账」，异常只让本次调用
          **不被记账**，原始响应照常返回；
        - with 退出时无条件还原原函数（即使块内抛异常）。
    """
    from app.llm_model_router import async_openai_caller as caller_module

    recorder: UsageRecorder = UsageRecorder()
    original: Any = caller_module._chat_completion_with_langfuse

    async def _recording_chat_completion(
        client: Any, params: Dict[str, Any], *, model: str, provider: str
    ) -> Any:
        try:
            response: Any = await original(client, params, model=model, provider=provider)
        except BaseException:
            # 真实调用失败（含熔断用的 APIError）——只记账，不改变异常语义
            recorder.errors += 1
            raise
        try:
            recorder.record(str(params.get("model") or model), getattr(response, "usage", None))
        except Exception:  # noqa: BLE001 - 记账失败绝不影响已成功的真实调用
            pass
        return response

    caller_module._chat_completion_with_langfuse = _recording_chat_completion
    try:
        yield recorder
    finally:
        caller_module._chat_completion_with_langfuse = original


__all__ = ["UsageRecorder", "record_llm_usage"]
