# -*- coding: utf-8 -*-
"""文件路径：app/core/tools/builtin/rag_search.py"""

from typing import Any, List, Dict
from app.core.tools.base import BaseTool, ToolParameter
from app.core.rag.rag_service import RAGService
from app.models.agent_schemas import RetrievalResult


class RagSearchTool(BaseTool):
    """知识库混合检索工具。

    将现有的多路 RAG 检索能力封装为标准 Agent 工具，允许大模型根据需要动态召回本地企业私有文档。
    """

    def __init__(self, rag_service: RAGService) -> None:
        """初始化检索工具并定义输入 Schema 约束。

        Args:
            rag_service: 已初始化的本地 RAG 核心协同服务单例。
        """
        super().__init__()
        self.name = "rag_knowledge_search"
        self.description = (
            "从私有知识库检索原文片段，用于获取具体事实、数字、条款、"
            "操作步骤等可直接引用的文档证据。"
        )
        """
        self.description = (
            "【证据检索工具】从私有知识库文档中检索与问题最相关的原文片段，"
            "用于获取【具体事实、数字、定义、条款、标准、操作步骤、文档原文】等可直接作为回答依据的证据。"
            "优先用于回答‘是什么’、‘多少’、‘具体规定是什么’、‘原文怎么说’、"
            "‘某个具体问题的答案是什么’等问题。"
            "该工具返回的是文档文本证据，不负责建立或推导多个实体之间的复杂关系。"
            "如果问题主要需要跨文档发现实体、比较实体、追踪关系链或理解多个实体之间的关联，"
            "应优先使用 knowledge_graph_search。"
        )
        """
        self._rag_service = rag_service

        # 显式声明工具参数规范
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="检索问题或关键词",
                required=True
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="期望召回的高相关性文本切片最大数量，默认值为 5。",
                required=False
            ),
            ToolParameter(
                name="collection_names",
                type="array",
                description="限定的知识库集合名列表，如 [\'hr_docs\']；未指定时可省略，由系统在默认全库检索。",
                required=False)
        ]
        
    async def execute(self, **kwargs: Any) -> str:
        """异步执行知识库检索并将结构化结果序列化为可供模型阅读的文本块。

        Args:
            **kwargs: 大模型解析出的具名参数字典，预期包含 'query'、可选的
                'top_k' 与可选的 'collection_names'（意图路由指定集合白名单）。

        Returns:
            str: 格式化后的 Markdown 文本块，包含召回的切片内容及来源标记。
        """
        search_query: str = kwargs.get("query", "")
        if not search_query.strip():
            return "错误：检索检索词（query）不能为空。"

        # 安全处理可能传入的各种类型的 top_k 参数
        raw_top_k: Any = kwargs.get("top_k", 5)
        try:
            target_top_k: int = int(raw_top_k)
        except (ValueError, TypeError):
            target_top_k = 5

        # 【改进点 2 · KB 命中引导】容错解析 collection_names：接受数组 / 单字符串 / None
        target_collection_names: List[str] = []
        raw_collection_names: Any = kwargs.get("collection_names")
        if isinstance(raw_collection_names, str):
            if raw_collection_names.strip():
                target_collection_names = [raw_collection_names.strip()]
        elif isinstance(raw_collection_names, (list, tuple)):
            target_collection_names = [
                str(item).strip() for item in raw_collection_names if str(item).strip()
            ]

        try:
            # 内部无缝调用原有核心服务的纯检索接口（指定集合时走定向检索）
            retrieved_chunks: List[RetrievalResult] = await self._rag_service.retrieve_contexts(
                query=search_query,
                top_k=target_top_k,
                collection_names=target_collection_names or None,
            )

            if not retrieved_chunks:
                scope_hint: str = (
                    f"（限定集合: {', '.join(target_collection_names)}）"
                    if target_collection_names else ""
                )
                return f"针对查询项 '{search_query}'，知识库{scope_hint}未匹配到任何高相关性的文档片段。"

            # 结构化串联，生成标准的 Observation 汇报文本
            formatted_observations: List[str] = [f"--- 知识库检索结果 (查询: {search_query}) ---"]
            for index, chunk in enumerate(retrieved_chunks, start=1):
                # 兼容不同字段设计，提取文本核心
                chunk_content: str = getattr(chunk, "content", getattr(chunk, "text", str(chunk)))
                chunk_source: str = getattr(chunk, "source", getattr(chunk, "metadata", {}).get("source", "未知来源"))
                formatted_observations.append(
                    f"[{index}] 来源文献: {chunk_source}\n内容片段: {chunk_content}\n"
                )

            return "\n".join(formatted_observations)

        except Exception as execution_error:
            # 故障平滑包装，确保 Agent 状态机不因底层连接或数据库故障而崩溃
            return f"知识库检索执行期间发生异常错误: {str(execution_error)}"