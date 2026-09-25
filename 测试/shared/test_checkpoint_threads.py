# -*- coding: utf-8 -*-
"""线程枚举（待审批列表数据源）测试。

历史事故（2026-09-23）：langgraph-checkpoint-redis 0.5.2 的 ``alist(None)``
走 RediSearch，返回的 ``Document`` 在当前 redis-py 下不物化 return_fields
（没有 ``thread_id`` 属性），枚举整体抛 AttributeError → 待审批列表恒空。
修复：对 RedisSaver  duck-type 出客户端后走 SCAN，thread_id 直接从
checkpoint 主键解析（key 格式：``{prefix}:{thread_id}:{ns}:{checkpoint_id}``，
thread_id 本身允许含 ``:``——本项目 run_id = ``session_id:uuidhex``）。
"""

import asyncio
import fnmatch

import pytest

from app.core.agent.graph.checkpoint import alist_thread_ids


class _FakeAsyncKeys:
    def __init__(self, keys):
        self._keys = list(keys)

    def scan_iter(self, *, match=None, count=None):  # noqa: ARG002
        # 模拟服务端 MATCH：bytes 键按 utf-8 解码后再匹配（真实 SCAN 的
        # MATCH 在 Redis 服务端执行，不会把混合类型抛给客户端 fnmatch）
        keys = self._keys
        if match:
            def _matches(k):
                text = k if isinstance(k, str) else k.decode("utf-8", "ignore")
                return fnmatch.fnmatchcase(text, match)
            keys = [k for k in keys if _matches(k)]

        class _Iter:
            def __init__(self, items):
                self._items = list(items)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        return _Iter(keys)


class _FakeRedisSaver:
    """具备 AsyncRedisSaver 被枚举依赖的两个属性。"""

    def __init__(self, keys, prefix="agent_cp"):
        self._redis = _FakeAsyncKeys(keys)
        self._checkpoint_prefix = prefix


def _run(coro):
    return asyncio.run(coro)


def test_scan_parses_thread_ids_including_embedded_colons() -> None:
    saver = _FakeRedisSaver([
        # run_id 含 ':'（本项目 session_id:uuidhex 形态）
        "agent_cp:sess-aaa:47fa42106ac04ed1:__empty__:1f1b72a5-0001",
        "agent_cp:sess-aaa:47fa42106ac04ed1:__empty__:1f1b72a5-0000",  # 同线程旧点
        "agent_cp:sess-bbb:56359062367e41ec:__empty__:1f1b72a5-0002",
        # 写入暂存与最新指针键必须被排除
        "agent_cp_write:sess-bbb:56359062367e41ec:__empty__:cp-1:task:0",
        "agent_cp_latest:sess-bbb:56359062367e41ec:__empty__",
        # bytes 键也要能解析
        b"agent_cp:sess-ccc:8899:__empty__:1f1b72a5-0003",
        # 无关业务键
        "other:ignored",
    ])
    ids = _run(alist_thread_ids(saver))

    assert ids == [
        "sess-aaa:47fa42106ac04ed1",
        "sess-bbb:56359062367e41ec",
        "sess-ccc:8899",
    ]


def test_scan_respects_max_threads() -> None:
    keys = [
        f"agent_cp:sess-{i}:hex{i}:__empty__:cp-{i}" for i in range(10)
    ]
    saver = _FakeRedisSaver(keys)
    ids = _run(alist_thread_ids(saver, max_threads=3))
    assert len(ids) == 3
    assert len(set(ids)) == 3


def test_in_memory_saver_still_uses_storage_dict() -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    saver.storage["thread-x"] = {}
    saver.storage["thread-y"] = {}
    ids = _run(alist_thread_ids(saver))
    assert set(ids) == {"thread-x", "thread-y"}


@pytest.mark.asyncio
async def test_unknown_saver_alist_failure_degrades_to_empty() -> None:
    class _Broken:
        async def alist(self, config):  # noqa: ARG002
            raise AttributeError("'Document' object has no attribute 'thread_id'")
            yield  # pragma: no cover - 让其成为 async generator

    ids = await alist_thread_ids(_Broken())
    assert ids == []
