# -*- coding: utf-8 -*-
"""长期记忆：向量库存储与按会话召回。"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Protocol, runtime_checkable

from app.models.agent_schemas import MemoryItem
from openai import OpenAI
from loguru import logger

from pymilvus import MilvusClient, DataType

import http
import dashscope
from dashscope import TextEmbedding

@runtime_checkable
class LTMEmbedProtocol(Protocol):
    """嵌入模型接口。"""

    def embed_query(self, text: str) -> list[float]:
        ...


@runtime_checkable
class LTMCollectionProtocol(Protocol):
    """Milvus Collection 最小接口。"""

    def insert(self, data: Any, **kwargs: Any) -> Any:
        ...

    def search(
        self,
        data: list[list[float]],
        anns_field: str,
        param: dict[str, Any],
        limit: int,
        expr: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        ...

    def delete(self, expr: str, **kwargs: Any) -> Any:
        ...

    def flush(self, **kwargs: Any) -> Any:
        ...


# -*- coding: utf-8 -*-
"""
基础设施层：长期记忆协议的具体实现（Milvus 仓储与 OpenAI Embedding）。
"""

# -*- coding: utf-8 -*-




class QwenEmbeddingImpl:
    """
    嵌入模型实现类：对接阿里云百炼平台（DashScope）的通义千问向量模型。
    实现 LTMEmbedProtocol 契约。
    """

    def __init__(self, api_key: str, model: str = "text-embedding-v3"):
        """
        :param api_key: 阿里云百炼平台的 API Key (DASHSCOPE_API_KEY)
        :param model: 向量模型名称，推荐使用最新通用模型 'text-embedding-v3'
                      也可根据业务选择 'text-embedding-v1' 或 'text-embedding-v2'
        """
        self.api_key = api_key
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        """将文本转化为通义千问稠密向量"""
        try:
            # 调用百炼平台的文本向量服务
            response = TextEmbedding.call(
                model=self.model,
                input=text,
                api_key=self.api_key
            )

            # 百炼平台标准状态码校验
            if response.status_code == http.HTTPStatus.OK:
                # text-embedding-v3 返回的 embeddings 是一个包含 dict 的 list
                # 结构为: [{'embedding': [0.1, 0.2, ...], 'text_index': 0}]
                return response.output['embeddings'][0]['embedding']
            else:
                logger.error(
                    f"Qwen Embedding 请求失败: 状态码={response.status_code}, "
                    f"错误码={response.code}, 错误信息={response.message}"
                )
                raise RuntimeError(f"DashScope Error: {response.message}")

        except Exception as e:
            logger.error(f"Qwen Embedding 转化过程中发生异常: {e}")
            raise


class OpenAIEmbeddingImpl:
    """
    嵌入模型实现类：对接 OpenAI 兼容的 Embedding API。
    实现 LTMEmbedProtocol 契约。
    """

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-3-small"):
        """
        :param api_key: 大模型 API 密钥
        :param base_url: 大模型 API 基础路径 (例如 https://api.deepseek.com/v1)
        :param model: 向量模型名称
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        """将文本转化为稠密向量"""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding 转化失败: {e}")
            raise


# -*- coding: utf-8 -*-
"""
基础设施层：长期记忆协议的具体实现（采用最新 MilvusClient 现代标准）。
"""




class MilvusCollectionWrapper:
    """
    Milvus 集合包装类：处理连接、自动建表、创建索引（基于新版 MilvusClient 接口）。
    实现 LTMCollectionProtocol 契约。
    """

    def __init__(
            self,
            collection_name: str,
            dim: int,
            host: str = "127.0.0.1",
            port: str = "19530"
    ):
        self.collection_name = collection_name
        self.dim = dim

        # 使用新版推荐的 MilvusClient，一行代码搞定连接
        endpoint = f"http://{host}:{port}"
        self.client = MilvusClient(uri=endpoint)

        # 自动化建表和索引
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """检查并创建符合长期记忆期望的简易 Schema 集合"""
        if self.client.has_collection(self.collection_name):
            return

        logger.info(f"Milvus 中未找到集合 {self.collection_name}，开始使用 MilvusClient 自动创建...")

        # 使用 MilvusClient 的 create_schema / add_field 体系
        schema = self.client.create_schema(auto_id=False, description="Agent 长期记忆存储库")

        # ✨ 修复点：将下面所有的 dtype 替换为 datatype
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="session_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="meta", datatype=DataType.VARCHAR, max_length=65535)

        # 配置索引参数
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="L2",
            params={"M": 8, "efConstruction": 64}
        )

        # 创建集合并建立索引
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"Milvus 集合 {self.collection_name} 初始化成功！")

    # ---------------------------------------------------------------------------
    # 下面 4 个方法将 MilvusClient 的原生方法完美适配为原脚手架要求的 Protocol 格式
    # ---------------------------------------------------------------------------
    def insert(self, data: Any, **kwargs: Any) -> Any:
        # LongTermMemory 传过来的是 [{'pk': ..., 'embedding': ...}] 格式，MilvusClient 原生支持
        return self.client.insert(collection_name=self.collection_name, data=data)

    def search(
            self,
            data: list[list[float]],
            anns_field: str,
            param: dict[str, Any],
            limit: int,
            expr: str | None = None,
            output_fields: list[str] | None = None,
            **kwargs: Any,
    ) -> Any:

        raw_res = self.client.search(
            collection_name=self.collection_name,
            data=data,
            anns_field=anns_field,
            search_params=param,
            limit=limit,
            filter=expr,
            output_fields=output_fields
        )



        return  raw_res

    def delete(self, expr: str, **kwargs: Any) -> Any:
        return self.client.delete(collection_name=self.collection_name, filter=expr)

    def flush(self, **kwargs: Any) -> Any:
        # 新版 MilvusClient 默认自动 flush，这里直接跳过即可
        pass


class LongTermMemory:
    """长期记忆：基于向量数据库的持久化记忆。"""

    vector_field: str = "embedding"
    content_field: str = "content"
    session_field: str = "session_id"
    pk_field: str = "pk"
    meta_field: str = "meta"

    metric_param: dict[str, Any] = {"metric_type": "L2", "params": {"nprobe": 16}}

    def __init__(self, milvus_collection: Any, embedding_model: Any) -> None:
        """
        :param milvus_collection: Milvus Collection，需包含向量、文本、会话 ID 等字段
        :param embedding_model: 含 ``embed_query`` 的嵌入模型
        """
        self._coll = milvus_collection
        self._embed = embedding_model

    def _ensure_embed(self) -> None:
        if not isinstance(self._embed, LTMEmbedProtocol):
            raise TypeError("embedding_model 需实现 embed_query")

    def _ensure_coll(self) -> None:
        if not isinstance(self._coll, LTMCollectionProtocol):
            raise TypeError("milvus_collection 需支持 insert/search/delete")

    async def store(self, session_id: str, content: str, metadata: dict[str, Any]) -> str:
        """写入一条长期记忆，返回 memory_id。"""
        self._ensure_embed()
        self._ensure_coll()

        memory_id = str(uuid.uuid4())
        meta = dict(metadata)
        meta["memory_id"] = memory_id

        def _sync() -> None:
            vec = self._embed.embed_query(content)
            row = {
                self.pk_field: memory_id,
                self.vector_field: vec,
                self.content_field: content,
                self.session_field: session_id,
                self.meta_field: json.dumps(meta, ensure_ascii=False),
            }
            # pymilvus 2.4+ 支持实体字典列表，字段名需与 Collection Schema 一致
            self._coll.insert([row])
            try:
                self._coll.flush()
            except Exception as fe:
                logger.warning("flush 失败（可忽略）: {}", fe)

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("长期记忆写入失败: {}", e)
            raise RuntimeError(f"store 失败: {e}") from e

        return memory_id

    async def recall(self, query: str, session_id: str, top_k: int = 5) -> list[MemoryItem]:
        """按语义在指定会话内召回记忆。"""
        self._ensure_embed()
        self._ensure_coll()

        def _sync() -> list[MemoryItem]:
            vec = self._embed.embed_query(query)
            # 转义单引号，避免 expr 注入
            sid = session_id.replace("'", "\\'")
            expr = f'{self.session_field} == "{sid}"'
            out = self._coll.search(
                data=[vec],
                anns_field=self.vector_field,
                param=self.metric_param,
                limit=top_k,
                expr=expr,
                output_fields=[self.pk_field, self.content_field, self.meta_field],
            )
            items: list[MemoryItem] = []
            hits = out[0] if out else []


            # 此时 hit 就是标准的 Python dict
            for hit in hits:
                entity = hit.get("entity") or {}

                rid = str(entity.get(self.pk_field) or hit.get("id") or "")
                text = str(entity.get(self.content_field) or "")

                meta_raw = entity.get(self.meta_field) or "{}"
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
                except json.JSONDecodeError:
                    meta = {}

                score = float(hit.get("distance", 0.0) or 0.0)

                items.append(
                    MemoryItem(id=rid, content=text, score=score, metadata=meta),
                )
            return items

        try:
            return await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("长期记忆召回失败: {}", e)
            raise RuntimeError(f"recall 失败: {e}") from e

    async def forget(self, memory_id: str) -> None:
        """按主键删除一条记忆。"""
        self._ensure_coll()

        def _sync() -> None:
            mid = memory_id.replace("'", "\\'")
            expr = f'{self.pk_field} == "{mid}"'
            self._coll.delete(expr)
            try:
                self._coll.flush()
            except Exception as fe:
                logger.warning("flush 失败（可忽略）: {}", fe)

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("长期记忆删除失败: {}", e)
            raise RuntimeError(f"forget 失败: {e}") from e
