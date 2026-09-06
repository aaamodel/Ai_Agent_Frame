# -*- coding: utf-8 -*-
"""RAG 子系统（LlamaIndex 重构）：存储 → 混合检索。

分层：
    - retriever.HybridRetriever  —— 向量 + BM25 + RRF 混合检索
    - bm25_builder.BM25IndexBuilder —— 关键词检索（rank_bm25 纯 Python）
    - rag_service.RAGService     —— 对外门面（写入 / 预热 / 检索统一出口）
"""

from app.core.rag.bm25_builder import BM25IndexBuilder, BM25Retriever
from app.core.rag.rag_service import RAGService
from app.core.rag.retriever import HybridRetriever

__all__ = ["RAGService", "HybridRetriever", "BM25IndexBuilder", "BM25Retriever"]