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
    """维护一份 id → TextNode 的映射，并按需构建 BM25Okapi 检索器。

    ⚠️ 存的是**整个 TextNode**（而不是只有文本），这一点很关键：BM25 的召回结果
    会和向量召回一起进 RRF 融合后直接返回。如果这里丢掉了 metadata，命中节点上的
    ``document_id`` / ``filename`` / ``collection`` 全是空值，后果有两个，且都表现为
    "什么都没召回"，日志上完全看不出来：

    1. 评测侧文档级判分两条通道同时落空 → ``Recall@5`` 恒为 0；
    2. ``retrieve_contexts`` 的集合白名单过滤按 ``metadata['collection']`` 匹配，
       空标签会被整段剔除 → 关键词通道的命中全部被过滤掉。
    """

    def __init__(self) -> None:
        self._id_to_node: Dict[str, TextNode] = {}
        self._lock = threading.Lock()
        self._retriever: Optional[BM25Retriever] = None
        self._revision = 0

    def update(self, nodes: List[TextNode]) -> None:
        """增量登记节点（保留 metadata，供召回后判分/过滤使用）。"""
        with self._lock:
            for node in nodes:
                if node.text:
                    self._id_to_node[str(node.node_id)] = node
            self._revision += 1

    def remove_ids(self, ids: List[str]) -> None:
        """按 id 删除已登记节点（向量库删除文档后同步调用）。"""
        with self._lock:
            for nid in ids:
                self._id_to_node.pop(str(nid), None)
            self._revision += 1

    def replace_all(self, nodes: List[TextNode]) -> None:
        """用给定全量节点**替换**内存语料（删除向量后的全量重建路径）。

        与启动时的 ``update`` 合并语义不同：这里必须先清空，否则被删文档
        会残留在 BM25 倒排里，关键词通道仍能召回已删除内容。
        """
        with self._lock:
            self._id_to_node = {
                str(node.node_id): node for node in nodes if node.text
            }
            self._revision += 1

    def rebuild_retriever(self, similarity_top_k: int = 10) -> "BM25Retriever":
        """据当前全量节点重建 BM25 检索器（corpus 变化后调用）。"""
        with self._lock:
            nodes = list(self._id_to_node.values())
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
        return len(self._id_to_node)


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