# -*- coding: utf-8 -*-
"""向量集合注册表的数据访问（Postgres ``vector_collections`` + 文档集合枚举）。"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Set

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.models import Document, VectorCollection
from app.query_intent.kb_collection_registry import (
    ENGINE_GRAPH,
    ENGINE_RAG,
    KbCollectionDescriptor,
    KbCollectionRegistry,
)


async def list_vector_collections(session: AsyncSession) -> list[VectorCollection]:
    """返回全部集合描述行（按名称排序）。"""
    result = await session.execute(
        select(VectorCollection).order_by(VectorCollection.name.asc())
    )
    return list(result.scalars().all())


async def upsert_vector_collection(
    session: AsyncSession,
    name: str,
    description: Optional[str],
    engine: str = ENGINE_RAG,
) -> VectorCollection:
    """登记/更新集合描述。

    ``description`` 是集合**唯一**的路由语义字段——它同时回答"这个集合是什么"与
    "什么时候该选它"，因此不再有第二份"检索时机"文本。

    ``description`` 为 None 时表示「本次未提供」，保留库内既有值不覆盖；
    显式空串表示清空。``engine``（rag/graph）在建行时确定，不允许同名集合
    在两种引擎间漂移——撞名直接抛 ValueError（由路由层转 409）。
    """
    engine = engine if engine in (ENGINE_RAG, ENGINE_GRAPH) else ENGINE_RAG
    row = await session.get(VectorCollection, name)
    if row is None:
        row = VectorCollection(
            name=name,
            engine=engine,
            description=description,
        )
        session.add(row)
    else:
        if (row.engine or ENGINE_RAG) != engine:
            raise ValueError(
                f"集合 {name!r} 已注册为引擎 {row.engine!r}，"
                f"不能按 {engine!r} 引擎重复登记；请换一个集合名。"
            )
        if description is not None:
            row.description = description or None
    return row


async def list_document_collection_names(session: AsyncSession) -> list[str]:
    """从 documents.meta->>'collection' 枚举当前实际有文档的逻辑集合名。"""
    result = await session.execute(
        select(Document.meta["collection"].as_string())
        .distinct()
        .where(Document.meta["collection"].as_string().is_not(None))
    )
    return [str(name) for (name,) in result.all() if name]


async def refresh_kb_registry(
    session: AsyncSession,
    extra_active_names: Optional[Iterable[str]] = None,
    graph_active_names: Optional[Iterable[str]] = None,
) -> list[str]:
    """用 DB 最新状态刷新进程内动态集合注册表快照（RAG + 图谱两类集合）。

    只有**当前仍有文档**（或调用方明确补充，如 Milvus 中存在向量、LightRAG
    workspace 中有已处理文档）的集合才会成为意图路由候选，避免路由到空集合。

    Args:
        extra_active_names: RAG 引擎侧补充的 active 集合名（如 Milvus 实测枚举）。
        graph_active_names: 图谱引擎侧 active 的 workspace 名。

    Returns:
        刷新后注册的集合名列表。
    """
    try:
        descriptors = await list_vector_collections(session)
        desc_by_name: dict[str, VectorCollection] = {row.name: row for row in descriptors}
        rag_active: Set[str] = set(await list_document_collection_names(session))
        for name in extra_active_names or []:
            name = str(name or "").strip()
            if name:
                rag_active.add(name)
        graph_active: Set[str] = {
            str(name or "").strip()
            for name in (graph_active_names or [])
            if str(name or "").strip()
        }

        def _make_row(name: str, fallback_engine: str) -> KbCollectionDescriptor:
            row = desc_by_name.get(name)
            return KbCollectionDescriptor(
                name=name,
                description=str(row.description) if row and row.description else "",
                engine=str(row.engine) if row and row.engine else fallback_engine,
            )

        rows: list[KbCollectionDescriptor] = [
            _make_row(name, ENGINE_RAG) for name in sorted(rag_active)
        ]
        # 全局同名只保留一个（注册表主键即 name）；图谱侧跳过 RAG 已占用名
        rows.extend(
            _make_row(name, ENGINE_GRAPH)
            for name in sorted(graph_active)
            if name not in rag_active
        )
        KbCollectionRegistry.replace(rows)
        return [row.name for row in rows]
    except Exception as exc:  # noqa: BLE001 - 注册表刷新失败不应阻断上传主流程
        logger.warning("刷新动态 KB 集合注册表失败（不影响本次写库）: {}", exc)
        return KbCollectionRegistry.names()
