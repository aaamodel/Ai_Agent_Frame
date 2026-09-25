# -*- coding: utf-8 -*-
"""Checkpointer 工厂与暂停/审批快照辅助。

策略（计划第 6/7 条铁律）：

- 优先 ``AsyncRedisSaver``（``langgraph-checkpoint-redis``），函数内 import，
  缺包 / 连不上 Redis 时自动降级 ``InMemorySaver``：审批与断点续跑能力关闭，
  但主链路可跑、模块可导入。
- 是否暂停**只以** ``StateSnapshot.next`` 非空 + ``tasks[].interrupts[]``
  非空为准；stream chunk 里的 ``__interrupt__`` 仅用于推送事件，不作为判据。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from langgraph.checkpoint.memory import InMemorySaver

BACKEND_REDIS = "redis"
BACKEND_MEMORY = "memory"

# 与 langgraph.checkpoint.redis.util 的空 id 哨兵保持一致（函数内 import 可选包，
# 这里直接内联常量，避免对 redis 扩展包产生硬导入依赖）。
_EMPTY_ID_SENTINEL = "00000000-0000-0000-0000-000000000000"


def configurable_for(run_id: str) -> Dict[str, Any]:
    """构造 RunnableConfig["configurable"] 的基础部分（thread_id = run_id）。"""
    return {"thread_id": run_id}


async def build_checkpointer(
    *,
    backend: str = BACKEND_REDIS,
    redis_url: Optional[str] = None,
    redis_client: Optional[Any] = None,
    ttl_seconds: int = 86_400,
    checkpoint_prefix: str = "agent_cp",
) -> Tuple[Any, str]:
    """构造异步 checkpointer。

    Args:
        backend: ``"redis"``（默认，失败降级）或 ``"memory"``（显式内存模式）。
        redis_url: Redis 连接串（redis://[:pwd@]host:port/db）。
        redis_client: 可选的已建 ``redis.asyncio`` 客户端（lifespan 复用）。
        ttl_seconds: checkpoint TTL；<=0 表示不设置 TTL。
        checkpoint_prefix: Redis key 前缀，多服务共库时做隔离。

    Returns:
        ``(saver, actual_backend)``，actual_backend 为实际生效的后端名，
        调用方可据此决定是否开放审批/续跑能力。
    """
    if backend == BACKEND_MEMORY:
        logger.info("Agent checkpointer 使用显式内存模式（InMemorySaver）。")
        return InMemorySaver(), BACKEND_MEMORY

    try:
        # 函数内 import：未安装 redis 扩展包时主链路仍可降级运行
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        ttl: Optional[Dict[str, Any]] = None
        if ttl_seconds and ttl_seconds > 0:
            # RedisSaver TTL 约定：default_ttl（秒，0=永久）+ refresh_on_read
            ttl = {"default_ttl": int(ttl_seconds), "refresh_on_read": True}

        write_prefix: str = f"{checkpoint_prefix}_write"
        saver = AsyncRedisSaver(
            redis_url=redis_url,
            redis_client=redis_client,
            ttl=ttl,
            checkpoint_prefix=checkpoint_prefix,
            checkpoint_write_prefix=write_prefix,
        )
        # 建索引 / 初始化（首次连接失败会在这里暴露）
        await saver.asetup()
        logger.info(
            "Agent checkpointer 已连接 Redis（prefix={}, ttl={}s）。",
            checkpoint_prefix, ttl_seconds,
        )
        return saver, BACKEND_REDIS
    except Exception as redis_error:  # noqa: BLE001 — 降级是设计行为，需兜住一切连接/导入异常
        logger.warning(
            "AsyncRedisSaver 初始化失败，降级 InMemorySaver（审批/断点续跑将不可用）: {}",
            redis_error,
        )
        return InMemorySaver(), BACKEND_MEMORY


# ---------------------------------------------------------------------------
# 快照 / interrupt 辅助（HITL 唯一判定入口）
# ---------------------------------------------------------------------------
async def aget_snapshot(graph: Any, run_id: str) -> Any:
    """读取某 run 的最新 ``StateSnapshot``。

    必须传**编译后的图**（``graph.aget_state``）——checkpointer 原生
    ``aget_tuple`` 返回的是 ``CheckpointTuple``（只有 checkpoint/metadata/
    pending_writes），不含 ``next``/``tasks``/``values``，不能作为 HITL 判据。
    找不到线程时 ``aget_state`` 返回空 StateSnapshot（不抛异常）。
    """
    config: Dict[str, Any] = {"configurable": configurable_for(run_id)}
    if hasattr(graph, "aget_state"):
        return await graph.aget_state(config)
    # 兼容：极少数场景直接拿到 saver（无 next/tasks，仅能取 channel_values）
    return await graph.aget_tuple(config)


def snapshot_next_nodes(snapshot: Any) -> Tuple[str, ...]:
    """快照中等待恢复的节点名元组（空元组表示已到 END）。"""
    if snapshot is None:
        return tuple()
    next_nodes = getattr(snapshot, "next", None)
    if next_nodes is not None:
        return tuple(next_nodes or ())
    # CheckpointTuple 兜底：待执行节点在 metadata.next
    metadata = getattr(snapshot, "metadata", None) or {}
    return tuple(metadata.get("next") or ())


def snapshot_interrupts(snapshot: Any) -> List[Dict[str, Any]]:
    """提取快照中全部未处理 interrupt 的可 JSON 化载荷列表。"""
    if snapshot is None:
        return []
    payloads: List[Dict[str, Any]] = []
    for task in getattr(snapshot, "tasks", None) or []:
        for interrupt in getattr(task, "interrupts", None) or []:
            value = getattr(interrupt, "value", None)
            if value is None:
                continue
            payloads.append(value if isinstance(value, dict) else {"raw": value})
    return payloads


def is_paused_for_approval(snapshot: Any) -> bool:
    """是否正停在 interrupt 点等待人工审批。

    判据铁律：``next`` 非空 **且** 存在未处理 interrupt，两者同时成立。
    """
    return bool(snapshot_next_nodes(snapshot)) and bool(snapshot_interrupts(snapshot))


def snapshot_values(snapshot: Any) -> Dict[str, Any]:
    """从快照取 state values 的浅拷贝（最终状态一律以此为准，不用 chunk 累积）。"""
    if snapshot is None:
        return {}
    values = getattr(snapshot, "values", None)
    if values is not None:
        return dict(values)
    # CheckpointTuple 兜底：values 在 checkpoint.channel_values
    checkpoint = getattr(snapshot, "checkpoint", None) or {}
    return dict(checkpoint.get("channel_values") or {})


def snapshot_exists(snapshot: Any) -> bool:
    """该 run_id 是否存在已持久化的检查点。

    ``aget_state`` 对未知线程不抛异常，而是返回回显 config 的**空**快照
    （values/next/tasks 全空），因此不能用 ``config is not None`` 判定。
    正常 run 的初始 state 在首个节点后必然落盘，values 必非空；
    暂停 run 还会有非空 next。
    """
    if snapshot is None:
        return False
    return bool(snapshot_values(snapshot)) or bool(snapshot_next_nodes(snapshot))


# ---------------------------------------------------------------------------
# 线程枚举（待审批列表）
# ---------------------------------------------------------------------------
async def _alist_thread_ids_via_scan(
    redis_client: Any, prefix: str, max_threads: int
) -> List[str]:
    """用 SCAN 从 checkpoint 主键解析全部 thread_id。

    RedisSaver 主键格式（见 langgraph-checkpoint-redis base.py）::

        {prefix}:{thread_id}:{checkpoint_ns}:{checkpoint_id}

    本项目 thread_id=run_id 本身含 ``:``（``session_id:uuidhex``），因此用
    ``rsplit(":", 2)`` 只剥掉末尾的 checkpoint_id / checkpoint_ns 两段，
    中间整段归 thread_id。``{prefix}_write:`` / ``{prefix}_latest:`` 等派生
    键的首段分隔符是下划线，不会匹配 ``"{prefix}:"`` 前缀，天然排除。
    """
    marker = f"{prefix}:"
    thread_ids: List[str] = []
    seen: set = set()
    async for raw_key in redis_client.scan_iter(match=f"{marker}*", count=500):
        key = raw_key if isinstance(raw_key, str) else bytes(raw_key).decode(
            "utf-8", "ignore"
        )
        if not key.startswith(marker):
            continue
        parts = key.rsplit(":", 2)
        if len(parts) < 3:
            continue
        thread_encoded = parts[0][len(marker):]
        if not thread_encoded or thread_encoded in seen:
            continue
        seen.add(thread_encoded)
        thread_ids.append(
            "" if thread_encoded == _EMPTY_ID_SENTINEL else thread_encoded
        )
        if len(thread_ids) >= max_threads:
            break
    return thread_ids


async def alist_thread_ids(checkpointer: Any, *, max_threads: int = 500) -> List[str]:
    """枚举 checkpointer 中全部 run（thread）id，最新检查点所属线程在前、去重。

    - InMemorySaver：直接读其 ``storage``（thread_id -> {checkpoint_id: tuple}）；
    - AsyncRedisSaver：SCAN checkpoint 主键解析 thread_id（库自带 ``alist`` 与
      当前 redis-py 不兼容，见 ``_alist_thread_ids_via_scan``）；
    - 其它未知后端：退回标准 ``alist(None)``。

    后端不支持全量列举时返回空列表并告警——调用方据此诚实返回，不抛 500。
    """
    # InMemorySaver 的内存字典是公开属性，读键即可，免去逐条 aget_tuple
    storage = getattr(checkpointer, "storage", None)
    if isinstance(storage, dict):
        return list(storage.keys())[:max_threads]

    # AsyncRedisSaver：不走它的 alist()——langgraph-checkpoint-redis 0.5.2 与
    # 当前 redis-py 存在兼容问题：FT.SEARCH 返回的 Document 不物化 return_fields，
    # 库内 `doc["thread_id"]` 直接 AttributeError（2026-09-23 实测，待审批列表
    # 因此恒空）。thread_id 本就编码在 checkpoint 主键里，SCAN 解析即可。
    redis_client = getattr(checkpointer, "_redis", None)
    checkpoint_prefix = getattr(checkpointer, "_checkpoint_prefix", None)
    if redis_client is not None and checkpoint_prefix:
        try:
            return await _alist_thread_ids_via_scan(
                redis_client, str(checkpoint_prefix), max_threads
            )
        except Exception as scan_error:  # noqa: BLE001 — 扫描失败再降级通用 alist
            logger.warning(
                "SCAN 枚举 checkpointer 线程失败，回退通用 alist: {}", scan_error
            )

    thread_ids: List[str] = []
    seen: set = set()
    try:
        async for checkpoint_tuple in checkpointer.alist(None):
            configurable = (
                (getattr(checkpoint_tuple, "config", None) or {}).get("configurable") or {}
            )
            thread_id = configurable.get("thread_id")
            if thread_id and thread_id not in seen:
                seen.add(thread_id)
                thread_ids.append(str(thread_id))
                if len(thread_ids) >= max_threads:
                    break
    except Exception as enumerate_error:  # noqa: BLE001 — 列举是辅助能力，失败降级空列表
        logger.warning("枚举 checkpointer 线程失败（待审批列表将为空）: {}", enumerate_error)
        return []
    return thread_ids


def checkpointer_backend_name(checkpointer: Any) -> str:
    """按实例类型推断后端名（列表接口要告诉前端内存模式下重启即失效）。"""
    if isinstance(checkpointer, InMemorySaver):
        return BACKEND_MEMORY
    return BACKEND_REDIS
