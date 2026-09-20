# -*- coding: utf-8 -*-
"""SSE 层接线单测：通道投递、跨线程、队列排空不丢尾。"""

import asyncio
import json

import pytest

from app.api.routes.chat import _QueueSink, _bump_attempt, _drain_queue, _render_answer_delta
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


def test_attempt_reset_only_marks_from_second_attempt():
    """spec §4.4：第 1 条 attempt_start 只是"新一轮开始"，不插分隔。"""
    router = DisplayRouter("answer")

    # 第一次尝试：正常推送 + 一条 attempt_start（此时还没有过尝试）
    first = list(_render_answer_delta(router, {"kind": "delta", "text": "第一候选"}, 0))
    assert _payload_text(first) == "第一候选"

    opening = list(_render_answer_delta(router, {"kind": "attempt_start"}, 0))
    assert opening == [], "第 1 条 attempt_start 不该产生任何事件"

    # 已经完成过一次尝试（attempt_count == 1）后，再来 attempt_start 才插分隔
    second = list(_render_answer_delta(router, {"kind": "attempt_start"}, 1))
    assert _has_attempt_reset(second) is True


def test_bump_attempt_counts_only_attempt_start():
    assert _bump_attempt(0, {"kind": "delta", "text": "x"}) == 0
    assert _bump_attempt(0, {"kind": "attempt_start"}) == 1
