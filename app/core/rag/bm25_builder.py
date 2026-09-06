# -*- coding: utf-8 -*-
"""BM25 内存索引构建器（纯 Python，基于 rank_bm25）。

规避 llama-index-retrievers-bm25 对 pystemmer 的 C 编译依赖（Python 3.14 无
MSVC SDK 头文件会编译失败），改用纯 Python 的 rank_bm25.BM25Okapi 实现关键词
检索，并封装为一个 LlamaIndex BaseRetriever，供 QueryFusionRetriever 做 RRF 融合。
"""

from __future__ import annotations

import re
import threading
from typing import Dict, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from rank_bm25 import BM25Okapi


def _to_query_str(query: object) -> str:
    """把 LlamaIndex ``_retrieve`` 入参（str 或 QueryBundle）归一化为字符串。"""
    qstr = getattr(query, "query_str", None)
    if isinstance(qstr, str):
        return qstr
    return str(query).strip()

from loguru import logger


def _tokenize(text: str) -> List[str]:
    """简体中文 + 英文单词分词（字母数字与 CJK 逐字切分，忽略空白与标点）。"""
    text = text.lower()
    tokens: List[str] = []
    # 匹配连续的字母数字串 OR 单个汉字
    for match in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text):
        tokens.append(match)
    return tokens


class BM25IndexBuilder:
    """维护一份 id → 文本 的映射，并按需构建 BM25Okapi 检索器。"""

    def __init__(self) -> None:
        self._id_to_text: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._retriever: Optional[BM25Retriever] = None
        self._revision = 0

    def update(self, items: List[Dict[str, str]]) -> None:
        """增量登记节点文本（items: [{"id": ..., "text": ...}]）。"""
        with self._lock:
            for item in items:
                text = item.get("text")
                if text:
                    self._id_to_text[item.get("id") or ""] = text
            self._revision += 1

    def rebuild_retriever(self, similarity_top_k: int = 10) -> "BM25Retriever":
        """据当前全量文本重建 BM25 检索器（corpus 变化后调用）。"""
        with self._lock:
            nodes = [
                TextNode(id_=nid, text=text)
                for nid, text in self._id_to_text.items()
                if text
            ]
            bm25 = BM25Okapi([_tokenize(n.text) for n in nodes]) if nodes else None
            self._retriever = BM25Retriever(
                nodes=nodes, bm25=bm25, similarity_top_k=similarity_top_k
            )
            return self._retriever

    def get_retriever(self, similarity_top_k: int = 10) -> "BM25Retriever":
        """返回当前 BM25 检索器；尚无文本时返回空检索器（不抛错）。"""
        if self._retriever is None:
            return self.rebuild_retriever(similarity_top_k)
        return self._retriever

    @property
    def size(self) -> int:
        return len(self._id_to_text)


class BM25Retriever(BaseRetriever):
    """用 rank_bm25.BM25Okapi 实现的关键词检索器（LlamaIndex BaseRetriever 协议）。"""

    def __init__(
        self,
        nodes: List[TextNode],
        bm25: Optional[BM25Okapi],
        similarity_top_k: int = 10,
    ) -> None:
        self._nodes = nodes
        self._bm25 = bm25
        self._top_k = max(1, int(similarity_top_k))
        super().__init__()

    def _retrieve(self, query: object) -> List[NodeWithScore]:
        qstr = _to_query_str(query)
        if self._bm25 is None or not self._nodes or not qstr:
            return []
        tokens = _tokenize(qstr)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(self._nodes)),
            key=lambda i: scores[i],
            reverse=True,
        )[: self._top_k]
        out: List[NodeWithScore] = []
        for idx in ranked:
            node = self._nodes[idx]
            score = float(scores[idx])
            if score > 0.0:
                out.append(NodeWithScore(node=node, score=score))
        return out