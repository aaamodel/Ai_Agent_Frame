# -*- coding: utf-8 -*-
"""DashScope Embedding → LlamaIndex BaseEmbedding 适配器。

将阿里云百炼 text-embedding-v3（OpenAI 兼容协议）接入 LlamaIndex，作为
RAG 重写后唯一的嵌入层。同时提供同步/异步的单条与批量 embedding。
"""

from __future__ import annotations

from typing import List

from llama_index.core.embeddings import BaseEmbedding
from openai import AsyncOpenAI, OpenAI


class DashScopeEmbedding(BaseEmbedding):
    """适配阿里云 DashScope text-embedding-v3（OpenAI 兼容协议）。

    Attributes:
        model_name: 向量模型名称（如 text-embedding-v3）。
        _client / _async_client: OpenAI 兼容同步/异步客户端。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        """构造 DashScope 嵌入适配器。

        Args:
            model: 向量模型名，如 "text-embedding-v3"。
            api_key: DashScope 兼容模式 API Key。
            base_url: OpenAI 兼容 Base URL；None 时不需要指定（由客户端默认）。
        """
        super().__init__(model_name=model)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _get_text_embedding(self, text: str) -> List[float]:
        """同步单条 embedding。"""
        resp = self._client.embeddings.create(model=self.model_name, input=text)
        return resp.data[0].embedding


    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """同步批量 embedding。"""
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self.model_name, input=texts)
        ordered = sorted(resp.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def _get_query_embedding(self, query: str) -> List[float]:
        """同步单条查询 embedding。

        ⚠️ 这里必须是**单条**语义（``str -> List[float]``），签名不能写成批量
        ``(List[str]) -> List[List[float]]``：LlamaIndex 的 ``BaseEmbedding`` 按单条
        调用（``get_query_embedding`` / ``get_agg_embedding_from_queries`` 都是逐条），
        写成批量会返回二维数组，向量检索通道在构造 ``EmbeddingEndEvent`` 时直接抛
        pydantic ``ValidationError``；该异常又被 ``HybridRetriever`` 的兜底 try 吞掉，
        最终表现为「向量通道 0 召回」，日志上完全看不出异常。
        """
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """异步单条 embedding。"""
        resp = await self._async_client.embeddings.create(
            model=self.model_name, input=text
        )
        return resp.data[0].embedding

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """异步查询 embedding。"""
        return await self._aget_text_embedding(query)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """异步批量 embedding。"""
        if not texts:
            return []
        resp = await self._async_client.embeddings.create(model=self.model_name, input=texts)
        ordered = sorted(resp.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]