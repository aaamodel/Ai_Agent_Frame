# -*- coding: utf-8 -*-
"""向量知识库集合的进程内注册表（意图识别动态集合路由）。

为什么需要它：
    硬编码意图树（IntentTreeFactory）里的 KB 节点是静态的，但用户可以随时通过
    ``/documents/upload`` 上传一个**新逻辑集合**并填写「功能描述 / 检索时机」。
    意图识别必须能把问题匹配到这些新集合，再经 ``slots.top_kb_node.collection_names``
    → prepare_node 硬约束 → ``rag_knowledge_search(collection_names=...)`` 透传到
    Agent 编排层。

设计：
    - 集合描述的权威存储在 Postgres ``vector_collections`` 表；本模块只是进程内
      只读快照（上传/删除接口在事务提交后调 ``replace`` 刷新，启动时从 DB 加载）。
    - DefaultIntentClassifier 每次加载意图树后，把快照里的集合**合并为 knowledge
      域下的动态 KB 叶子节点**，复用既有「向量召回 + LLM 打分」管线，不新造匹配机制。
    - 只包含**当前仍有文档**的集合（由刷新方计算 active 集合后传入），避免把请求
      路由到空集合。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from loguru import logger

from app.query_intent.intent_classify_resolver.intent_model import IntentNode
from app.query_intent.intent_data_base import IntentKind, IntentLevel

_KNOWLEDGE_ROOT_ID = "knowledge"
_DYNAMIC_ID_PREFIX = "knowledge-dynamic-"
_RAG_TOOL = "rag_knowledge_search"
_GRAPH_TOOL = "knowledge_graph_search"

ENGINE_RAG = "rag"
ENGINE_GRAPH = "graph"

_HINT_DYNAMIC_KB: str = (
    "调用 rag_knowledge_search，query 传改写后的完整问题，"
    "并把 collection_names 限定为该集合；答案来自该集合对应的私有文档，"
    "检索不到时如实告知，不要编造，也不要跨集合编造答案。"
)

_HINT_DYNAMIC_GRAPH: str = (
    "调用 knowledge_graph_search，query 传改写后的完整问题，"
    "并把 collection 参数设置为该知识图谱集合名；答案来自该集合抽取的"
    "实体、关系与跨文档图谱上下文，检索不到时如实告知，不要编造。"
)


@dataclass(frozen=True)
class KbCollectionDescriptor:
    """一个可被意图路由命中的逻辑集合。

    engine:
        - ``rag``：Milvus 文档向量库，命中后走 rag_knowledge_search；
        - ``graph``：LightRAG 知识图谱 workspace，命中后走 knowledge_graph_search。
    """

    name: str
    description: str = ""
    retrieval_hint: str = ""
    engine: str = ENGINE_RAG

    def intent_text(self) -> str:
        """供 embedding / LLM 匹配的检索文本（功能描述 + 检索时机）。"""
        label = "知识图谱集合" if self.engine == ENGINE_GRAPH else "知识库集合"
        parts: List[str] = [f"{label}「{self.name}」"]
        if self.description.strip():
            parts.append(self.description.strip())
        if self.retrieval_hint.strip():
            parts.append(f"适用检索时机：{self.retrieval_hint.strip()}")
        return "。".join(parts)


class KbCollectionRegistry:
    """进程内只读快照 + 动态意图节点构建（线程安全单例式 API）。"""

    _rows: tuple[KbCollectionDescriptor, ...] = ()
    _lock = threading.Lock()

    # ------------------------------------------------------------------
    # 快照维护
    # ------------------------------------------------------------------
    @classmethod
    def replace(cls, rows: Iterable[KbCollectionDescriptor]) -> None:
        """用最新描述符全量替换快照（按集合名去重、去空白、保序）。"""
        deduped: dict[str, KbCollectionDescriptor] = {}
        for row in rows or []:
            name = str(getattr(row, "name", "") or "").strip()
            if not name:
                continue
            engine = str(getattr(row, "engine", "") or ENGINE_RAG).strip()
            if engine not in (ENGINE_RAG, ENGINE_GRAPH):
                engine = ENGINE_RAG
            deduped[name] = KbCollectionDescriptor(
                name=name,
                description=str(getattr(row, "description", "") or ""),
                retrieval_hint=str(getattr(row, "retrieval_hint", "") or ""),
                engine=engine,
            )
        with cls._lock:
            cls._rows = tuple(deduped.values())
        logger.info("KbCollectionRegistry 已刷新：动态集合 {} 个 {}", len(deduped), list(deduped))

    @classmethod
    def rows(cls) -> tuple[KbCollectionDescriptor, ...]:
        with cls._lock:
            return cls._rows

    @classmethod
    def names(cls) -> list[str]:
        return [row.name for row in cls.rows()]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._rows = ()

    # ------------------------------------------------------------------
    # 意图树合并
    # ------------------------------------------------------------------
    @classmethod
    def build_dynamic_nodes(cls) -> List[IntentNode]:
        """把快照构造成 knowledge 域下的动态 KB 叶子节点。

        engine=graph 的集合挂 knowledge_graph_search（LightRAG 图谱检索），
        其余挂 rag_knowledge_search（Milvus 向量检索）。
        """
        nodes: List[IntentNode] = []
        for descriptor in cls.rows():
            is_graph = descriptor.engine == ENGINE_GRAPH
            node = IntentNode(
                id=f"{_DYNAMIC_ID_PREFIX}{descriptor.engine}-{descriptor.name}",
                name=descriptor.name,
                description=descriptor.intent_text(),
                level=IntentLevel.CATEGORY,
                parent_id=_KNOWLEDGE_ROOT_ID,
                kind=IntentKind.KB,
                examples=[],
                collection_name=descriptor.name,
                collection_names=[descriptor.name],
                agent_tool_names=[_GRAPH_TOOL if is_graph else _RAG_TOOL],
                tool_usage_hint=_HINT_DYNAMIC_GRAPH if is_graph else _HINT_DYNAMIC_KB,
            )
            node.full_path = f"企业知识问答 > {descriptor.name}"
            nodes.append(node)
        return nodes

    @classmethod
    def merge_into_tree(cls, roots: List[IntentNode]) -> List[IntentNode]:
        """把动态集合节点挂到 knowledge 根下（原地合并，幂等）。

        - 找不到 knowledge 根或快照为空时原样返回；
        - 按节点 id 去重（静态树已存在同名 id 时不覆盖）。
        """
        rows = cls.rows()
        if not rows or not roots:
            return roots

        knowledge: Optional[IntentNode] = next(
            (node for node in roots if getattr(node, "id", None) == _KNOWLEDGE_ROOT_ID),
            None,
        )
        if knowledge is None:
            return roots
        if knowledge.children is None:
            knowledge.children = []

        existing_ids = {n.id for n in knowledge.children}
        added = 0
        for node in cls.build_dynamic_nodes():
            if node.id in existing_ids:
                continue
            knowledge.children.append(node)
            existing_ids.add(node.id)
            added += 1
        if added:
            logger.info("意图树已合并 {} 个动态 KB 集合节点。", added)
        return roots

    @classmethod
    def is_dynamic_node_id(cls, node_id: Any) -> bool:
        return str(node_id or "").startswith(_DYNAMIC_ID_PREFIX)
