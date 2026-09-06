# -*- coding: utf-8 -*-
"""app/core/tools/builtin/graph_search.py
内置本地知识图谱检索工具（基于 LightRAG 引擎）。"""

from __future__ import annotations

from typing import Any, Optional
from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter
# 注意：这里禁止模块级 `from lightrag import QueryParam` —— lightrag 会传递拉起
# langchain_text_splitters / sentence_transformers / torch / transformers 等重型
# 依赖（实测拖慢 app.main 导入 ~76s），导致 uvicorn reload 双进程启动超过 3 分钟。
# QueryParam 统一在 execute() 内懒加载。


class KnowledgeGraphSearchTool(BaseTool):
    """基于 LightRAG 知识图谱与向量双索引执行本地私有资产检索的工具。"""

    def __init__(self, rag_instance: Optional[Any] = None) -> None:
        """初始化知识图谱检索工具。

        Args:
            rag_instance: 可选的 LightRAG 引擎实例。若不传，则默认从基础设施层导入。
        """
        super().__init__()
        self.name = "knowledge_graph_search"
        self.description = (
            "【全局图谱与多文档关系检索工具】用于处理涉及【全局总结、跨文档比对、复杂业务实体网络或链路推导】的宏观提问。"
            "当用户提问涉及‘总结一下...’、‘它们之间有什么关系...’、‘有哪些共同点...’，"
            "或者需要横跨多个业务线/产品线进行概念关联和逻辑推理时，必须优先使用此工具。"
        )
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="具体需要向私有知识库检索的问题，建议传入完整的业务长句或具体问题。",
                required=True,
            )
        ]

        # 优先使用构造函数注入的实例，其次尝试从指定的基础设施层动态加载全局单例


        try:
            from app.infrastructure.knowledgebase.light_rag import rag_instance
            self._rag_engine = rag_instance
        except ImportError as exc:
            logger.error("从基础设施层导入全局 rag_instance 失败，请检查路径。错误原因: {}", exc)
            self._rag_engine = None

    async def execute(self, **kwargs: Any) -> str:
        """执行本地图谱与向量的混合检索。

        Args:
            **kwargs: 必须包含 'query' 键值对。

        Returns:
            str: 检索出的关联实体、关系链和全局上下文合并后的深度背景文本。
        """
        search_query = str(kwargs.get("query", "")).strip()
        if not search_query:
            raise ValueError("关键检索参数 'query' 不能为空")

        if self._rag_engine is None:
            logger.error("LightRAG 引擎实例未就绪，放弃本次私有知识库检索。")
            return "【系统提示】本地私有知识库服务当前不可用，请完全依赖你自身拥有的通用知识和记忆进行回答。"

        try:
            logger.info("开始本地知识图谱检索，正在查询: '{}'", search_query)

            # 懒加载 QueryParam：避免模块级导入 lightrag 拖慢整个应用启动
            from lightrag import QueryParam

            # 显式初始化底层的命名空间与存储连接，确保服务高可用
            await self._rag_engine.initialize_storages()

            # 使用混合检索模式（hybrid）同时兼顾实体拓扑关系与语义稠密向量检索
            graph_search_response = await self._rag_engine.aquery(
                search_query,
                param=QueryParam(mode="hybrid")
            )

            logger.info(
                "本地知识图谱检索成功，查询语句='{}'",
                search_query
            )
            return graph_search_response

        except Exception as exc:
            # 捕获异常，并使用标准的异常追踪日志打印，确保生产环境可追溯
            logger.exception("本地知识图谱在执行检索时发生未预期异常，原因: {}", exc)

            # 优雅降级：向大模型返回系统提示，防止整个 Agent 编排流水线崩溃
            return (
                f"【系统提示】在检索私有知识库时发生异常。请优先使用你（大模型）自身"
                f"掌握的通用领域知识尽可能准确地回答用户的问题：「{search_query}」。"
            )