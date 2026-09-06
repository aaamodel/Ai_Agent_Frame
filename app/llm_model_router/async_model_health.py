# -*- coding: utf-8 -*-
"""异步版模型健康存储：三态熔断器（CLOSED/OPEN/HALF_OPEN），基于 asyncio.Lock。

与同步版 ``ModelHealthStore`` 行为一一对应（相同的失败阈值/恢复窗口/Half-Open
单飞令牌语义），唯一区别是所有变更方法为 async，配合 FastAPI/Uvicorn 的事件循环
避免线程锁切换开销。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from app.llm_model_router.model_router_config import AIModelProperties


@dataclass
class CallPermit:
    """Permit granted for a model call."""
    model_id: str
    half_open_token: int  # 0 means not a half-open probe


class _State(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class _AsyncModelHealth:
    consecutive_failures: int = 0
    open_until_ms: int = 0
    half_open_inflight: bool = False
    half_open_token: int = 0
    state: _State = _State.CLOSED


class AsyncModelHealthStore:
    """异步版模型健康追踪 + 熔断器。参数语义与同步版一致（驱动配置 AIModelProperties）。"""

    def __init__(self, properties: AIModelProperties) -> None:
        self._properties = properties
        self._health: Dict[str, _AsyncModelHealth] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._probe_token_seq: int = 0

    # ------------------------------------------------------------------
    # 只读查询（Select 阶段使用，不需要锁但保持一致更安全）
    # ------------------------------------------------------------------
    async def is_unavailable(self, model_id: str) -> bool:
        """电路打开或 Half-Open 有在飞探测时返回 True（供 Selector 预选过滤）。"""
        if model_id is None:
            return False
        async with self._lock:
            health = self._health.get(model_id)
            if health is None:
                return False
            now_ms = int(time.monotonic() * 1000)
            if health.state == _State.OPEN and health.open_until_ms > now_ms:
                return True
            return health.state == _State.HALF_OPEN and health.half_open_inflight

    # ------------------------------------------------------------------
    # 许可授予
    # ------------------------------------------------------------------
    async def allow_call(self, model_id: str) -> Optional[CallPermit]:
        if model_id is None:
            return None
        now_ms = int(time.monotonic() * 1000)
        async with self._lock:
            health = self._health.get(model_id)
            if health is None:
                health = _AsyncModelHealth()
                self._health[model_id] = health

            if health.state == _State.OPEN:
                if health.open_until_ms > now_ms:
                    return None
                health.state = _State.HALF_OPEN
                health.half_open_inflight = True
                self._probe_token_seq += 1
                health.half_open_token = self._probe_token_seq
                return CallPermit(model_id, health.half_open_token)

            if health.state == _State.HALF_OPEN:
                if health.half_open_inflight:
                    return None
                health.half_open_inflight = True
                self._probe_token_seq += 1
                health.half_open_token = self._probe_token_seq
                return CallPermit(model_id, health.half_open_token)

            # CLOSED：直接放行，half_open_token=0 表示非探测
            return CallPermit(model_id, 0)

    # ------------------------------------------------------------------
    # 结果标记
    # ------------------------------------------------------------------
    async def mark_success(self, model_id: str) -> None:
        if model_id is None:
            return
        async with self._lock:
            health = self._health.get(model_id)
            if health is None:
                health = _AsyncModelHealth()
                self._health[model_id] = health
            health.state = _State.CLOSED
            health.consecutive_failures = 0
            health.open_until_ms = 0
            health.half_open_inflight = False

    async def mark_failure(self, model_id: str) -> None:
        if model_id is None:
            return
        now_ms = int(time.monotonic() * 1000)
        async with self._lock:
            health = self._health.get(model_id)
            if health is None:
                health = _AsyncModelHealth()
                self._health[model_id] = health

            if health.state == _State.HALF_OPEN:
                # 探测失败：立刻打开熔断 + 重置计数（与同步版行为一致）
                health.state = _State.OPEN
                health.open_until_ms = now_ms + self._properties.selection.open_duration_ms
                health.consecutive_failures = 0
                health.half_open_inflight = False
                return

            health.consecutive_failures += 1
            if health.consecutive_failures >= self._properties.selection.failure_threshold:
                health.state = _State.OPEN
                health.open_until_ms = now_ms + self._properties.selection.open_duration_ms
                health.consecutive_failures = 0

    # ------------------------------------------------------------------
    # 辅助：Half-Open 许可回滚（执行前 permit 成功但最终实际没发起调用的场景）
    # ------------------------------------------------------------------
    async def release_half_open_permit(self, permit: Optional[CallPermit]) -> None:
        if permit is None or permit.half_open_token <= 0:
            return
        async with self._lock:
            health = self._health.get(permit.model_id)
            if (
                health is not None
                and health.state == _State.HALF_OPEN
                and health.half_open_inflight
                and health.half_open_token == permit.half_open_token
            ):
                health.half_open_inflight = False
