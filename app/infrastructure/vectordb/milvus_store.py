# -*- coding: utf-8 -*-
"""LlamaIndex MilvusVectorStore 封装：统一管理知识库集合与索引。

存储层（LlamaIndex 重构）：
- 内部持有 MilvusVectorStore + VectorStoreIndex，对外只暴露极简接口：
  insert_nodes / get_all / index 访问。
- overwrite 仅在显式开启时才会删除重建集合（用于旧集合一次性迁移重建），
  迁移完成后应关闭避免每次启动清空数据。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.base import MILVUS_ID_FIELD

from loguru import logger


def _milvus_quote(value: str) -> str:
    """转义 Milvus 表达式字符串字面量（反斜杠与双引号）。"""
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


class MilvusIndexManager:
    """管理 RAG 知识库的 Milvus 集合与 LlamaIndex 索引（懒加载）。

    Attributes:
        collection_name: Milvus 集合名。
        embed_model: LlamaIndex BaseEmbedding（如 DashScopeEmbedding）。
        overwrite: 启动是否重建集合（一次性迁移时置 True，完成后应关闭）。
        _index: 懒加载的 VectorStoreIndex；None 表示尚未初始化。
    """

    def __init__(
        self,
        uri: str,
        embed_model: BaseEmbedding,
        collection_name: str,
        dim: int = 1024,
        overwrite: bool = False,
        token: str = "",
        batch_size: int = 100,
    ) -> None:
        if uri is None or not str(uri).strip():
            raise ValueError("MilvusIndexManager uri 不能为空")
        self.collection_name = collection_name
        self.embed_model = embed_model

        Settings.embed_model = embed_model  # 全局默认嵌入，分词器/索引均可复用
        self.vector_store = MilvusVectorStore(
            uri=uri,
            token=token or "",
            collection_name=collection_name,
            dim=dim,
            overwrite=bool(overwrite),
            similarity_metric="IP",          # 文本嵌入常配 IP 余弦（向量已归一化）
            batch_size=batch_size,
        )
        self._index: Optional[VectorStoreIndex] = None
        self._index_ready = False

    @property
    def index(self) -> VectorStoreIndex:
        """懒加载的 VectorStoreIndex（线程内首次访问创建）。"""
        if not self._index_ready:
            self._index = VectorStoreIndex.from_vector_store(
                self.vector_store, embed_model=self.embed_model
            )
            self._index_ready = True
        return self._index  # type: ignore[return-value]

    def insert_nodes(self, nodes: List[TextNode]) -> List[str]:
        """将预构建的 TextNode 列表写入集合（自动 embedding 后落 Milvus）。"""
        if not nodes:
            return []
        inserted = self.index.insert_nodes(nodes)
        logger.info(
            "LlamaIndex 已写入集合 [{}] {} 个节点。",
            self.collection_name,
            len(nodes),
        )
        return inserted or [node.node_id for node in nodes]

    async def ainsert_nodes(self, nodes: List[TextNode]) -> List[str]:
        """异步写入节点（转线程池执行同步 insert_nodes）。"""
        return await asyncio_to_thread(self.insert_nodes, nodes)

    def get_all(self) -> List[TextNode]:
        """拉取集合内全部节点（用于启动时重建 BM25 内存索引）。

        注意：llama_index 的 MilvusVectorStore 没有 get_all，且 from_vector_store
        每次进程都是空索引（index_struct.nodes_dict 为空），读不到历史持久化节点。
        因此这里直接用 MilvusClient 分批枚举集合主键（MILVUS_ID_FIELD，即 node_id），
        再分批经 vector_store.get_nodes 重建带文本的 TextNode。
        """
        try:
            client = self.vector_store.client
            pk_field: str = str(MILVUS_ID_FIELD)  # "id"
            node_ids: List[str] = []
            offset: int = 0
            batch_size: int = 1000
            while True:
                rows = client.query(
                    collection_name=self.collection_name,
                    output_fields=[pk_field],
                    limit=batch_size,
                    offset=offset,
                )
                if not rows:
                    break
                node_ids.extend(str(row.get(pk_field)) for row in rows if row.get(pk_field) is not None)
                if len(rows) < batch_size:
                    break
                offset += len(rows)

            if not node_ids:
                return []

            nodes: List[TextNode] = []
            for start in range(0, len(node_ids), batch_size):
                chunk: List[str] = node_ids[start : start + batch_size]
                fetched: List[Optional[TextNode]] = self.vector_store.get_nodes(chunk) or []
                nodes.extend(node for node in fetched if node is not None)
            logger.info("MilvusIndexManager.get_all 拉取集合 [{}] {} 个节点。", self.collection_name, len(nodes))
            return nodes
        except Exception as exc:
            logger.warning("MilvusIndexManager.get_all 失败（跳过 BM25 预热）: {}", exc)
            return []

    async def aget_all(self) -> List[TextNode]:
        """异步拉取全部节点。"""
        return await asyncio_to_thread(self.get_all)

    # ------------------------------------------------------------------
    # 集合 / 文件管理（管理面 API）
    # ------------------------------------------------------------------
    def _query_all_rows(
        self, output_fields: List[str], expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """按 filter 分页枚举集合实体（动态字段 metadata 可直接 output/过滤）。"""
        client = self.vector_store.client
        rows: List[Dict[str, Any]] = []
        offset: int = 0
        batch_size: int = 1000
        query_kwargs: Dict[str, Any] = {}
        if expr:
            query_kwargs["filter"] = expr
        while True:
            batch = client.query(
                collection_name=self.collection_name,
                output_fields=output_fields,
                limit=batch_size,
                offset=offset,
                **query_kwargs,
            )
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < batch_size:
                break
            offset += len(batch)
        return rows

    def list_logical_files(self) -> Dict[str, Dict[str, int]]:
        """枚举物理集合内的「逻辑集合 → 文件 → 切片数」。

        逻辑集合是节点 metadata 里的 ``collection`` 标签（所有文档物理上写在
        同一个 Milvus collection）。filename/collection 都在 Milvus 动态字段里。
        """
        grouped: Dict[str, Dict[str, int]] = {}
        rows = self._query_all_rows(["collection", "filename"])
        for row in rows:
            tag = str(row.get("collection") or "").strip()
            filename = str(row.get("filename") or "").strip()
            if not tag:
                tag = "_untagged"
            files = grouped.setdefault(tag, {})
            if filename:
                files[filename] = files.get(filename, 0) + 1
        return grouped

    def delete_by_filename(self, collection_tag: str, filename: str) -> List[str]:
        """删除指定逻辑集合下某个文件的全部向量，返回被删节点 id 列表。"""
        expr = (
            f'collection == "{_milvus_quote(collection_tag)}"'
            f' and filename == "{_milvus_quote(filename)}"'
        )
        client = self.vector_store.client
        rows = self._query_all_rows([MILVUS_ID_FIELD], expr)
        ids = [
            str(row[MILVUS_ID_FIELD])
            for row in rows
            if row.get(MILVUS_ID_FIELD) is not None
        ]
        if not ids:
            return []
        for start in range(0, len(ids), 1000):
            client.delete(
                collection_name=self.collection_name,
                ids=ids[start : start + 1000],
            )
        client.flush(self.collection_name)
        logger.info(
            "Milvus 已删除集合标签 [{}] 文件 [{}] 的 {} 个向量节点。",
            collection_tag,
            filename,
            len(ids),
        )
        return ids


def asyncio_to_thread(func, *args: Any, **kwargs: Any) -> Any:
    """把同步阻塞调用放到线程池，避免阻塞事件循环。"""
    import asyncio

    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, lambda: func(*args, **kwargs))