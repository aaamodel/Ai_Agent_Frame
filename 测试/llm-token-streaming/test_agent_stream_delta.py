# -*- coding: utf-8 -*-
"""SSE 层接线单测：通道投递、跨线程、队列排空不丢尾。"""

import asyncio
import json

import pytest

from app.api.routes.chat import (
    _QueueSink,
    _bump_attempt,
    _drain_queue,
    _render_answer_delta,
    _render_rewrite_delta,
)
from app.core.agent.delta_extract import DisplayRouter


@pytest.mark.asyncio
async def test_queue_sink_delivers_from_worker_thread():
    """Pipeline 在 to_thread 的工作线程里 emit，主循环必须收得到。"""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sink = _QueueSink(loop=loop, queue=queue)

    await asyncio.to_thread(sink.push, {"kind": "delta", "text": "来自线程"})

    got = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert got == {"kind": "delta", "text": "来自线程"}


@pytest.mark.asyncio
async def test_drain_queue_emits_all_pending():
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(3):
        queue.put_nowait({"i": i})
    drained = [ev async for ev in _drain_queue(queue)]
    assert [e["i"] for e in drained] == [0, 1, 2]


@pytest.mark.asyncio
async def test_drain_until_task_does_not_lose_tail():
    """任务结束时队列里可能还剩事件，必须补排空。"""
    queue: asyncio.Queue = asyncio.Queue()
    drained = []

    async def producer():
        await asyncio.sleep(0.01)
        queue.put_nowait({"i": 0})
        queue.put_nowait({"i": 1})  # 任务结束前最后一刻才放进去

    task = asyncio.create_task(producer())
    async for ev in _drain_queue(queue, until=task):
        drained.append(ev)
    assert [e["i"] for e in drained] == [0, 1]


# ---------------------------------------------------------------------------
# Task 7：答案阶段渲染 + 换候选标注
# ---------------------------------------------------------------------------


def _payload_text(payloads) -> str:
    """把若干 SSE 负载里的 delta.text 拼起来。"""
    return "".join(
        json.loads(p[6:].decode("utf-8")).get("delta", {}).get("text", "")
        for p in payloads
    )


def _has_attempt_reset(payloads) -> bool:
    return any(
        json.loads(p[6:].decode("utf-8")).get("delta", {}).get("attempt_reset")
        for p in payloads
    )


def test_answer_router_suppresses_react_draft_end_to_end():
    """Review Focus #3：草稿标记无论怎么切片都不能漏到界面上。"""
    r = DisplayRouter("answer")
    draft = "Thought: 先查数据\nAction: sales_sql_query\nAction Input: {\"q\":\"x\"}\n"
    out = "".join(r.feed(draft[i:i + 6]) for i in range(0, len(draft), 6))
    assert out == ""


def test_attempt_reset_marks_and_router_restarts_clean():
    """spec §4.4：第 1 条 attempt_start 不插分隔；第 2 条插分隔，且路由器被复位。"""
    router = DisplayRouter("answer")

    # 开局第一条 attempt_start：不产生任何事件
    opening = list(_render_answer_delta(router, {"kind": "attempt_start"}, 0))
    assert opening == [], "第 1 条 attempt_start 不该产生任何事件"

    # 第一候选吐了半截内容后失败
    first = list(_render_answer_delta(router, {"kind": "delta", "text": "第一候选半截"}, 0))
    assert _payload_text(first) == "第一候选半截"
    assert router.visible_any() is True

    # 第二尝试开始（attempt_count 已为 1）：插分隔 + 复位分流器
    second = list(_render_answer_delta(router, {"kind": "attempt_start"}, 1))
    assert _has_attempt_reset(second) is True
    assert router.visible_any() is False, "分隔发出后分流器必须复位"
    assert router.mode == "undecided"

    # 第二尝试是全新输出，不能被旧 _raw 污染模式判定
    tail = list(_render_answer_delta(router, {"kind": "delta", "text": "第二候选"}, 1))
    assert _payload_text(tail) == "第二候选"


def test_attempt_reset_not_emitted_when_nothing_visible():
    """旧尝试没吐出任何可见内容（首 chunk 前失败）就不插分隔，但仍复位。"""
    router = DisplayRouter("answer")
    payloads = list(_render_answer_delta(router, {"kind": "attempt_start"}, 1))
    assert payloads == []
    assert router.visible_any() is False


def test_bump_attempt_counts_only_attempt_start():
    assert _bump_attempt(0, {"kind": "delta", "text": "x"}) == 0
    assert _bump_attempt(0, {"kind": "attempt_start"}) == 1


# ---------------------------------------------------------------------------
# 改写阶段：真实 schema 字段抽取 + 重试复位
# ---------------------------------------------------------------------------


def test_render_rewrite_extracts_real_schema_field():
    """端到端渲染：组合 schema JSON 分片进来，只吐出 rewrite 字段值。"""
    router = DisplayRouter("rewrite")
    text = '{"rewrite":"把负责人改成李娜","should_split":false}'
    payloads = []
    for i in range(0, len(text), 3):
        payloads.extend(_render_rewrite_delta(
            router, {"kind": "delta", "text": text[i:i + 3]}, 0
        ))
    assert _payload_text(payloads) == "把负责人改成李娜"


def test_render_rewrite_first_attempt_start_is_silent_but_resets():
    router = DisplayRouter("rewrite")
    payloads = list(_render_rewrite_delta(router, {"kind": "attempt_start"}, 0))
    assert payloads == []
    assert router.mode == "undecided"


def test_render_rewrite_second_attempt_with_partial_emits_reset_then_clean_text():
    router = DisplayRouter("rewrite")
    # 第一尝试吐了半截
    half = list(_render_rewrite_delta(
        router, {"kind": "delta", "text": '{"rewrite":"半截'}, 0
    ))
    assert _payload_text(half) == "半截"
    # 第二尝试开始：发 rewrite 版 attempt_reset
    reset = list(_render_rewrite_delta(router, {"kind": "attempt_start"}, 1))
    assert any(
        (lambda d: d.get("phase") == "rewrite" and d.get("attempt_reset"))(
            json.loads(p[6:].decode("utf-8"))["delta"]
        )
        for p in reset
    )
    # 复位后第二尝试的完整 JSON 只输出新值，不带半截、不带 JSON 结构
    tail_text = '{"rewrite":"完整改写","should_split":false}'
    tail = []
    for i in range(0, len(tail_text), 4):
        tail.extend(_render_rewrite_delta(
            router, {"kind": "delta", "text": tail_text[i:i + 4]}, 1
        ))
    assert _payload_text(tail) == "完整改写"
