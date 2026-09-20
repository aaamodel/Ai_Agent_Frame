# -*- coding: utf-8 -*-
"""LLM 输出的旁观通道。

设计要点（详见 specs/2026-09-20-llm-token-streaming-design.md §4.1）：

- 模型调用器**对外契约不变**：仍然"一次调用、一个完整结果"。
  它只是在被观测时顺带把增量推给通道。
- 因此上游的重试、熔断、候选降级、以及业务侧的解析路径全部无需改动。
- 用 ``ContextVar`` 而不是显式参数：``asyncio.to_thread`` 会**复制当前上下文**到
  工作线程，所以同步的 Pipeline 链路也能读到同一个通道，一套机制覆盖两种情况。

⚠️ ``emit()`` 内部吞掉一切异常。推送是旁路，绝不能因为它失败而影响主链路。

⚠️ 普通 ``threading.Thread`` **不会**复制上下文（只有 ``asyncio.to_thread`` 会），
   所以从裸线程里读不到通道——这不是缺陷，是机制边界。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional, Protocol

__all__ = [
    "StreamSink",
    "current_sink",
    "has_sink",
    "use_sink",
    "emit",
]


class StreamSink(Protocol):
    """输出通道。实现方**负责线程安全**（可能被工作线程调用）。"""

    def push(self, event: Dict[str, Any]) -> None:  # pragma: no cover - 协议
        ...


_current_sink: ContextVar[Optional[StreamSink]] = ContextVar(
    "agent_stream_sink", default=None
)


def current_sink() -> Optional[StreamSink]:
    """取当前上下文里的通道；没有则 None。"""
    return _current_sink.get()


def has_sink() -> bool:
    """当前是否有观测者。调用器用它决定走流式还是非流式。"""
    return _current_sink.get() is not None


@contextmanager
def use_sink(sink: Optional[StreamSink]) -> Iterator[None]:
    """在 with 块内装上下线通道；退出时恢复外层（支持嵌套）。

    传 None 等价于"本块内明确没有观测者"，用于临时屏蔽外层通道。
    """
    token = _current_sink.set(sink)
    try:
        yield
    finally:
        _current_sink.reset(token)


def emit(event: Dict[str, Any]) -> bool:
    """把一条事件推给当前通道。

    Returns:
        有通道且推送成功为 True；无通道或推送失败为 False。

    ⚠️ 任何异常都在此处被吞掉：旁路推送失败不该让用户拿不到答案。
    """
    sink = _current_sink.get()
    if sink is None:
        return False
    try:
        sink.push(event)
        return True
    except Exception:  # noqa: BLE001 - 旁路失败必须无声
        return False
