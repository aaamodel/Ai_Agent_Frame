# ============================================================================
# intent_vector_retriever.py - 意图树向量检索器（调整二）
# ============================================================================
"""意图树向量化检索器：用 Embedding Top-K 召回替代「全量意图树塞 Prompt」。

背景（调整二）：
  原实现 DefaultIntentClassifier._build_prompt 每次调用都把整棵意图树
  （15~20 个叶子节点）的全量描述（ID/路径/描述/示例/Tools）序列化进
  System Prompt（约 2000~3000 tokens），拖高网络吞吐与 LLM 首 Token
  延迟（TTFT）。

本实现：
  1) 首次使用时对全部叶子节点做一次性 embedding（懒预热 + 单例复用，
     节点规模固定后零重复成本）；
  2) 查询时仅对「用户问题」做 1 次 embedding，内存余弦相似度暴力检索
     Top-K 候选节点（15~20 节点规模下 <1ms，效果等同 Milvus 且零网络
     开销；未来意图节点规模增长到数百以上时，可将本类内部索引平滑切换
     为 Milvus collection，对外接口不变）；
  3) SYSTEM 类交互节点（问候/关于机器人）始终附加进候选集，防止语义
     检索漏召回交互导向意图；
  4) embedding 失败时返回 None，调用方回退全量意图清单（可用性兜底）。

接口约定：
  - retrieve(question) -> Optional[list[IntentNode]]：
    None = 向量索引不可用（调用方回退全量）；list = Top-K 候选节点。
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Optional, Protocol

from app.query_intent.intent_classify_resolver.intent_model import IntentNode
from app.query_intent.rag_constant import INTENT_VECTOR_TOP_K

logger = logging.getLogger(__name__)


class _EmbedProtocol(Protocol):
    """最小 embedding 协议（与 LTMEmbedProtocol 对齐，避免跨层依赖）。"""

    def embed_query(self, text: str) -> list[float]:
        ...


class IntentTreeVectorRetriever:
    """意图树叶子节点向量检索器（内存余弦索引 + 懒预热单例复用）。

    Attributes:
        embed_model: embedding 实现（生产为 QwenEmbeddingImpl，冒烟可注入
            假实现）；必须实现 embed_query(text) -> list[float]。
        tree_provider: 意图树数据提供者（通常是 DefaultIntentClassifier），
            需提供 load_intent_tree_data() -> 对象含 leaf_nodes 属性。
        top_k: 每次召回的候选节点数上限（不含强制附加的 SYSTEM 节点）。
    """

    def __init__(
        self,
        embed_model: Any,
        tree_provider: Any,
        top_k: int = INTENT_VECTOR_TOP_K,
    ) -> None:
        self._embed_model = embed_model
        self._tree_provider = tree_provider
        self._top_k = max(1, int(top_k))
        self._index_ready = False
        self._index_degraded = False
        self._lock = threading.Lock()
        # (node, normalized_vector) 列表；SYSTEM 节点单独记录
        self._vector_index: list[tuple[IntentNode, list[float]]] = []
        self._system_nodes: list[IntentNode] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """作废内存向量索引（动态 KB 集合增删后调用）。

        下次 ``retrieve`` 会重新从 tree_provider 拉取全量叶子（含新集合节点）
        并重新 embedding 预热；同时解除一次性 degraded 标记，给索引重建机会。
        """
        with self._lock:
            self._vector_index = []
            self._system_nodes = []
            self._index_ready = False
            self._index_degraded = False
        invalidate = getattr(self._tree_provider, "invalidate_tree_cache", None)
        if callable(invalidate):
            invalidate()
        logger.info("意图树向量索引已 reset，等待下次请求重新预热。")

    def retrieve(self, question: str) -> Optional[list[IntentNode]]:
        """对用户问题召回 Top-K 候选意图节点。

        Returns:
            Optional[list[IntentNode]]：None 表示向量索引不可用（调用方应
            回退全量意图清单）；否则返回按相似度降序的候选节点列表
            （已附加 SYSTEM 交互节点并去重）。
        """
        if not question or not str(question).strip():
            return None

        if not self._ensure_index():
            return None

        try:
            query_vector = self._normalize(self._embed_model.embed_query(question))
        except Exception as embed_error:
            logger.warning(
                "意图检索 query embedding 失败，回退全量意图清单：%s",
                embed_error,
            )
            return None

        scored: list[tuple[float, IntentNode]] = []
        for node, node_vector in self._vector_index:
            similarity = sum(a * b for a, b in zip(query_vector, node_vector))
            scored.append((similarity, node))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        candidates: list[IntentNode] = [node for _, node in scored[: self._top_k]]

        # SYSTEM 交互节点始终附加（防语义检索漏召回问候/关于机器人类意图）
        for system_node in self._system_nodes:
            if system_node not in candidates:
                candidates.append(system_node)

        logger.info(
            "意图向量检索完成：question=%r，召回候选 %d 个（Top-K=%d）：%s",
            str(question)[:80],
            len(candidates),
            self._top_k,
            [node.id for node in candidates],
        )
        return candidates

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_index(self) -> bool:
        """懒预热：首次调用时对全部叶子节点做一次性 embedding。

        Returns:
            bool：True = 索引可用；False = 预热失败（degraded，本次及后续
            直接回退全量清单，避免每个请求重复尝试 15 次 API 调用）。
        """
        if self._index_ready:
            return True
        if self._index_degraded:
            return False

        with self._lock:
            if self._index_ready:
                return True
            if self._index_degraded:
                return False

            try:
                tree_data = self._tree_provider.load_intent_tree_data()
                leaf_nodes: list[IntentNode] = list(
                    getattr(tree_data, "leaf_nodes", None) or []
                )
                if not leaf_nodes:
                    logger.warning(
                        "意图树叶子节点为空，向量检索器降级（回退全量清单路径）"
                    )
                    self._index_degraded = True
                    return False

                vectors: list[tuple[IntentNode, list[float]]] = []
                system_nodes: list[IntentNode] = []
                for node in leaf_nodes:
                    if node.is_system():
                        system_nodes.append(node)
                        continue
                    text = self._build_node_text(node)
                    vector = self._normalize(
                        self._embed_model.embed_query(text)
                    )
                    vectors.append((node, vector))

                self._vector_index = vectors
                self._system_nodes = system_nodes
                self._index_ready = True
                logger.info(
                    "意图树向量索引预热完成：叶子节点 %d 个（SYSTEM 附加 %d 个）",
                    len(vectors),
                    len(system_nodes),
                )
                return True
            except Exception as index_error:
                self._index_degraded = True
                logger.warning(
                    "意图树向量索引预热失败，永久降级为全量清单回退：%s",
                    index_error,
                )
                return False

    @staticmethod
    def _build_node_text(node: IntentNode) -> str:
        """拼接节点检索文本：full_path + description + examples。"""
        parts: list[str] = [
            str(getattr(node, "full_path", "") or ""),
            str(getattr(node, "name", "") or ""),
            str(getattr(node, "description", "") or ""),
        ]
        examples = getattr(node, "examples", None) or []
        for example in examples[:5]:
            if isinstance(example, str) and example.strip():
                parts.append(example.strip())
        return " | ".join(part for part in parts if part)

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        """L2 归一化（零向量原样返回，余弦退化为点积但不报错）。"""
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            return list(vector)
        return [value / norm for value in vector]
