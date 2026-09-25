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
    "structural_visible",
]


class StreamSink(Protocol):
    """输出通道。实现方**负责线程安全**（可能被工作线程调用）。"""

    def push(self, event: Dict[str, Any]) -> None:  # pragma: no cover - 协议
        ...


_current_sink: ContextVar[Optional[StreamSink]] = ContextVar(
    "agent_stream_sink", default=None
)

# 结构化调用（带 response_format 的 JSON 调用）在当前通道上是否可见。
# 默认 True：改写阶段要从流式 JSON 里增量抽取 rewrite 字段，依赖结构化流。
# 答案阶段（chat.py 包裹整张图运行时）置 False：planner 计划 / distill 控制协议 /
# summarize 判定都是内部 JSON，不是给用户看的答案——它们流进 answer 通道会：
#   1. 把原始 JSON 当正文显示（前端排版事故，2026-09-23）；
#   2. 每次内部调用都发 attempt_start，把上一段【成功的】内部输出划成
#      "上段因模型切换已废弃"（GLM 已是最末候选，根本没发生降级）。
# 真正给用户的答案由图结束后的 _stream_final_answer 统一逐字推送。
_show_structural: ContextVar[bool] = ContextVar(
    "agent_stream_show_structural", default=True
)


def current_sink() -> Optional[StreamSink]:
    """取当前上下文里的通道；没有则 None。"""
    return _current_sink.get()


def has_sink() -> bool:
    """当前是否有观测者。调用器用它决定走流式还是非流式。"""
    return _current_sink.get() is not None


def structural_visible() -> bool:
    """当前通道是否放行结构化（response_format）调用的流式输出。"""
    return _show_structural.get()


@contextmanager
def use_sink(
    sink: Optional[StreamSink], *, show_structural: bool = True
) -> Iterator[None]:
    """在 with 块内装上下线通道；退出时恢复外层（支持嵌套）。

    传 None 等价于"本块内明确没有观测者"，用于临时屏蔽外层通道。

    ``show_structural=False`` 时，带 response_format 的结构化调用改走非流式，
    其 JSON 不进入通道（也不发 attempt_start）；自由文本 / FC 调用不受影响。
    """
    sink_token = _current_sink.set(sink)
    policy_token = _show_structural.set(show_structural)
    try:
        yield
    finally:
        _show_structural.reset(policy_token)
        _current_sink.reset(sink_token)


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
