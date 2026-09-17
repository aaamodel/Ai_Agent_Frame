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
            "从私有知识图谱检索实体、实体关系和跨文档上下文，"
            "用于回答\"A 和 B 有什么关系\"\"A 依赖哪些东西\"等关联性问题。"
        )
        """
        self.description = (
            "【关系与上下文检索工具】从私有知识图谱及其关联文档中检索【实体、实体关系、关系链和跨文档上下文】，"
            "用于回答需要发现或理解多个业务实体之间关联的问题。"
            "优先用于‘A 和 B 有什么关系’、‘A 依赖哪些东西’、‘哪些产品具有共同特征’、"
            "‘这些业务之间如何关联’、‘跨多个文档综合分析某个实体网络’等问题。"
            "该工具的重点不是返回某一条文档原文，而是发现【实体之间的连接、上下游关系、共同关联和跨文档上下文】。"
            "如果问题只需要查找一个具体事实、数字、条款或原文证据，应优先使用 rag_knowledge_search。"
        )
        """
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="检索问题，建议传完整业务长句",
                required=True,
            ),
            ToolParameter(
                name="collection",
                type="string",
                description="知识图谱集合名；不传则用默认集合",
                required=False,
            ),
        ]
        # description=(
        # "知识图谱集合（LightRAG workspace）名。当意图识别的「意图路由硬约束」"
        #"明确指定集合时必须传入同名集合，严禁查其他图谱集合；"
        #"无明确指定时留空，使用默认知识图谱集合。")

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
            **kwargs: 必须包含 'query' 键值对；可选 'collection' 指定图谱集合。

        Returns:
            str: 检索出的关联实体、关系链和全局上下文合并后的深度背景文本。
        """
        search_query = str(kwargs.get("query", "")).strip()
        if not search_query:
            raise ValueError("关键检索参数 'query' 不能为空")

        collection = str(kwargs.get("collection", "") or "").strip()

        # 意图硬约束指定了图谱集合时，按 workspace 取对应 LightRAG 实例
        rag_engine = self._rag_engine
        if collection:
            try:
                # 懒加载：避免模块级导入 lightrag 拖慢应用启动
                from app.infrastructure.knowledgebase.light_rag import get_lightrag

                rag_engine = get_lightrag(collection)
                logger.info("知识图谱检索定向到集合（workspace）: {}", collection)
            except Exception as exc:  # noqa: BLE001 - 集合解析失败时降级默认实例
                logger.warning(
                    "图谱集合 {} 的 LightRAG 实例获取失败，降级默认集合：{}",
                    collection,
                    exc,
                )
                rag_engine = self._rag_engine

        if rag_engine is None:
            logger.error("LightRAG 引擎实例未就绪，放弃本次私有知识库检索。")
            return "【系统提示】本地私有知识库服务当前不可用，请完全依赖你自身拥有的通用知识和记忆进行回答。"

        try:
            logger.info("开始本地知识图谱检索，正在查询: '{}'", search_query)

            # 懒加载 QueryParam：避免模块级导入 lightrag 拖慢整个应用启动
            from lightrag import QueryParam

            # 显式初始化底层的命名空间与存储连接，确保服务高可用
            await rag_engine.initialize_storages()

            # 使用混合检索模式（hybrid）同时兼顾实体拓扑关系与语义稠密向量检索
            graph_search_response = await rag_engine.aquery(
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