# -*- coding: utf-8 -*-
"""RAG 服务门面（LlamaIndex 重构）。

对外保持与旧版一致的接口约定（文档同名类 RAGService）：

    RAGService(embed_model, milvus_uri, collection_name, overwrite, ...)
        ├─ MilvusIndexManager      —— 存储层（LlamaIndex MilvusVectorStore）
        ├─ BM25IndexBuilder        —— 关键词层（rank_bm25 纯 Python）
        └─ HybridRetriever         —— 混合检索（向量 + BM25 + RRF）

    - async ingest_texts(texts, metadatas) -> list[str]：写入向量库并刷新 BM25。
    - async seed_bm25()：从 Milvus 全量重建 BM25 内存索引（启动预热）。
    - async retrieve_contexts(query, top_k, collection_names=None)
        -> list[RetrievalResult]：统一检索出口（Agent 工具依赖的唯一接口）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

from loguru import logger

from app.core.rag.bm25_builder import BM25IndexBuilder
from app.core.rag.retriever import HybridRetriever
from app.infrastructure.vectordb.milvus_store import MilvusIndexManager
from app.models.agent_schemas import RetrievalResult


class RAGService:
    """LlamaIndex 驱动的知识库混合检索服务门面。

    Attributes:
        index_manager: MilvusIndexManager（存储层）。
        bm25_builder: BM25IndexBuilder（关键词层）。
        _retriever: HybridRetriever（向量 + BM25 + RRF）。
        top_k_default: 未显式指定时的默认召回数。
    """

    def __init__(
        self,
        embed_model: BaseEmbedding,
        milvus_uri: str,
        collection_name: str,
        overwrite: bool = False,
        dim: int = 1024,
        top_k_default: int = 10,
        token: str = "",
    ) -> None:
        self.index_manager = MilvusIndexManager(
            uri=milvus_uri,
            embed_model=embed_model,
            collection_name=collection_name,
            dim=dim,
            overwrite=overwrite,
            token=token,
        )
        self.bm25_builder: BM25IndexBuilder = BM25IndexBuilder()
        self.top_k_default = max(1, int(top_k_default))
        self._retriever: HybridRetriever = HybridRetriever(
            vector_index_holder=self.index_manager,
            bm25_builder=self.bm25_builder,
            similarity_top_k=self.top_k_default,
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def build_nodes(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[TextNode]:
        """按 texts/metadatas 构建 TextNode 列表（不插入）。"""
        if len(texts) != len(metadatas):
            raise ValueError("texts 与 metadatas 长度必须一致")
        nodes: List[TextNode] = []
        for text, meta in zip(texts, metadatas):
            if not text or not str(text).strip():
                continue
            meta = dict(meta or {})
            meta.setdefault("source", "vector")
            nodes.append(TextNode(text=str(text), metadata=meta))
        return nodes

    async def ingest_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """将文本切片写入 Milvus（自动 embedding），并同步刷新 BM25 索引。"""
        nodes = self.build_nodes(texts, metadatas)
        if not nodes:
            return []
        ids = await self._loop().run_in_executor(None, self.index_manager.insert_nodes, nodes)
        bm25_items = [{"id": node.node_id, "text": node.text} for node in nodes]
        self.bm25_builder.update(bm25_items)
        self.bm25_builder.rebuild_retriever(self.top_k_default)
        return ids

    # ------------------------------------------------------------------
    # 启动预热：从既有集合重建 BM25
    # ------------------------------------------------------------------
    async def seed_bm25(self) -> None:
        """从 Milvus 全量节点重建 BM25 内存索引。失败不阻塞（向量检索仍可用）。"""
        try:
            nodes = await self._loop().run_in_executor(None, self.index_manager.get_all)
            if not nodes:
                logger.info("RAG BM25 预热：集合为空，跳过。")
                return
            self.bm25_builder.update(
                [{"id": n.node_id, "text": n.text} for n in nodes]
            )
            self.bm25_builder.rebuild_retriever(self.top_k_default)
            logger.info("RAG BM25 预热完成：共 {} 个节点。", len(nodes))
        except Exception as seed_error:
            logger.warning("RAG BM25 预热失败（向量检索不受影响）: {}", seed_error)

    # ------------------------------------------------------------------
    # 检索（工具唯一依赖接口）
    # ------------------------------------------------------------------
    async def retrieve_contexts(
        self,
        query: str,
        top_k: Optional[int] = None,
        collection_names: Optional[List[str]] = None,
    ) -> List[RetrievalResult]:
        """混合检索 query，返回按相关度降序的 RetrievalResult 列表。

        Args:
            query: 检索问题。
            top_k: 想召回的最大切片数（默认 self.top_k_default）。
            collection_names: 可选集合白名单（意图路由硬约束）。提供时仅召回
                metadata["collection"] 命中白名单的节点，其余全部剔除。
        """
        target_top_k: int = max(1, int(top_k if top_k is not None else self.top_k_default))
        query = (query or "").strip()
        if not query:
            return []

        try:
            hits: List[NodeWithScore] = await asyncio.to_thread(
                self._retriever.retrieve, query
            )
        except Exception as retr_error:
            logger.exception("RAG 混合检索失败: {}", retr_error)
            return []

        allow_collections: Optional[set[str]] = None
        if collection_names:
            allow_collections = {str(c).strip() for c in collection_names if str(c).strip()}
            if not allow_collections:
                allow_collections = None

        out: List[RetrievalResult] = []
        for hit in hits:
            node = hit.node
            meta: Dict[str, Any] = dict(getattr(node, "metadata", None) or {})
            if allow_collections is not None:
                node_collection = str(meta.get("collection", "") or "").strip()
                if node_collection not in allow_collections:
                    continue
            score = float(hit.score) if hit.score is not None else 0.0
            out.append(
                RetrievalResult(
                    id=node.node_id,
                    content=node.text,
                    score=score,
                    metadata=meta,
                    source="hybrid",
                )
            )
            if len(out) >= target_top_k:
                break
        return out

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _loop() -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop