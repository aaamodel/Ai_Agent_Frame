# -*- coding: utf-8 -*-
"""SSE 层接线单测：通道投递、跨线程、队列排空不丢尾。"""

import asyncio

import pytest

from app.api.routes.chat import _QueueSink, _drain_queue


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
