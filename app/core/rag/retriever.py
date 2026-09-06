# -*- coding: utf-8 -*-
"""混合检索器：向量 Top-K + BM25 关键词，通过 RRF（倒数排名融合）合并。

检索层（LlamaIndex 重构）：封装 VectorIndexRetriever 与 BM25Retriever，用
Reciprocal Rank Fusion 融合两路结果，对外 /_retrieve 返回 NodeWithScore 列表。

说明：不使用 QueryFusionRetriever 的 LLM 多查询生成，避免为召回额外引入一次
LLM 调用（符合「第一时间回复」的目标）；RRF 融合在两路检索结果上手工完成。
"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore

from app.core.rag.bm25_builder import BM25IndexBuilder, _to_query_str

# RRF 常驻常数（smooth constant），典型 K=60
_RRF_K = 60


class HybridRetriever(BaseRetriever):
    """向量 + BM25 混合检索器（RRF 融合）。

    Attributes:
        _vector_index_holder: 提供 index 的调用方（懒取 vector retriever）。
        _bm25_builder: BM25IndexBuilder，corpus 变更后重建检索器。
        _similarity_top_k: 融合后的最终召回数。
        _vector_retriever: 懒构建的向量检索器。
    """

    def __init__(
        self,
        vector_index_holder: object,
        bm25_builder: BM25IndexBuilder,
        similarity_top_k: int = 10,
    ) -> None:
        self._vector_index_holder = vector_index_holder
        self._bm25_builder = bm25_builder
        self._similarity_top_k = max(1, int(similarity_top_k))
        self._vector_retriever: Optional[BaseRetriever] = None
        super().__init__()

    def _ensure_vector_retriever(self) -> BaseRetriever:
        if self._vector_retriever is None:
            index = self._vector_index_holder.index
            self._vector_retriever = index.as_retriever(
                similarity_top_k=self._similarity_top_k
            )
        return self._vector_retriever

    def _fuse(
        self, vector_hits: List[NodeWithScore], bm25_hits: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """RRF 融合两路命中，按融合分降序取 top_k。"""
        rank_score: dict[str, float] = {}
        nodes: dict[str, object] = {}

        for hits in (vector_hits, bm25_hits):
            for rank, hit in enumerate(hits, start=1):
                nid = hit.node.node_id
                rank_score[nid] = rank_score.get(nid, 0.0) + 1.0 / (_RRF_K + rank)
                if nid not in nodes:
                    nodes[nid] = hit.node

        fused = sorted(
            rank_score.items(), key=lambda kv: kv[1], reverse=True
        )[: self._similarity_top_k]
        return [NodeWithScore(node=nodes[nid], score=score) for nid, score in fused]

    def _retrieve(self, query: object) -> List[NodeWithScore]:
        qstr = _to_query_str(query)
        if not qstr:
            return []
        try:
            vector_hits = self._ensure_vector_retriever().retrieve(qstr)
        except Exception as exc:  # 向量侧故障不阻断 BM25
            vector_hits = []
        bm25_retriever = self._bm25_builder.get_retriever(self._similarity_top_k)
        try:
            bm25_hits = bm25_retriever.retrieve(qstr)
        except Exception:
            bm25_hits = []
        return self._fuse(vector_hits, bm25_hits)

    async def _aretrieve(self, query: object) -> List[NodeWithScore]:
        return self._retrieve(query)