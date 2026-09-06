# -*- coding: utf-8 -*-
"""工具工厂：在这里实例化所有具体工具，并统一注入到注册中心。"""

from typing import Optional
from sqlalchemy.ext.asyncio import async_sessionmaker  # 假设你使用的工厂类型

from app.core.tools.builtin.feishu import FeishuBitableTool
from app.core.tools.builtin.graph_search import KnowledgeGraphSearchTool
from app.core.tools.builtin.localexcel import LocalExcelTool
from app.core.tools.builtin.rag_search import RagSearchTool
from app.core.tools.builtin.todo_list import AgentTodoPlannerTool
from app.core.tools.registry import ToolRegistry
from app.core.tools.builtin.database import DatabaseQueryTool, DescribeTableTool, ListTablesTool
from app.core.tools.builtin.search import WebSearchTool
from app.core.tools.builtin.doubao_search import DoubaoWebSearchTool

from app.core.backends.filesystem import FilesystemBackend
from app.core.tools.builtin.filesystem_tools import FileReadTool, FileListTool, FileGrepTool
# 🌟【新引入】引入 RAG 核心服务类型声明，用于类型体操约束
from app.core.rag.rag_service import RAGService


def bootstrap_tools(
        db_session_factory=None,
        fs_backend: Optional[FilesystemBackend] = None,
        rag_service: Optional[RAGService] = None,  # 🌟【改动点 1】开放 RAG 服务注入通道
) -> ToolRegistry:
    """一键初始化所有内置工具（含高效文件检索系统与知识检索系统），并返回装载完毕的注册中心。"""

    registry = ToolRegistry()
    # 2. 注入原生封装的工具（第一种路线）
    registry.register(FeishuBitableTool())
    # 2. 实例化基础通用计算与检索工具
    """
    db_tool = DatabaseQueryTool(session_factory=db_session_factory)
    db_describe_table_tool = DescribeTableTool()
    db_list_table_tool = ListTablesTool()
    """
    # 联网搜索双梯队：
    #   第一梯队  web_search          ← DoubaoWebSearchTool（豆包搜索 API 直连）

    #     使用时机由工具 description/SYSTEM_PROMPT 提示词约束）
    # 同时保留结构性兜底：豆包工具内部失败/空结果时自动降级调用 Tavily
    tavily_search_tool = WebSearchTool()
    web_search_tool = DoubaoWebSearchTool(fallback=tavily_search_tool)
    # 🌟【改动点 2】将上层注入的 rag_service 喂给 RagSearchTool 实例
    # 这样大模型在 Thought 循环里调用 rag_search 时，才能真正通过连接池查到 Milvus 数据
    rag_search_tool = RagSearchTool(rag_service=rag_service)

    graph_search_tool = KnowledgeGraphSearchTool()
    todo_list_tool = AgentTodoPlannerTool()

    # 3. 容错防御：文件后端
    if fs_backend is None:
        fs_backend = FilesystemBackend(virtual_mode=True)

    # 4. 实例化文件系统工具
    file_read_tool = FileReadTool(fs_backend=fs_backend)
    file_list_tool = FileListTool(fs_backend=fs_backend)
    file_grep_tool = FileGrepTool(fs_backend=fs_backend)

    # 5. 将所有打工人（工具实例）统一登记到注册中心内
    """
    registry.register(db_tool)
    registry.register(db_describe_table_tool)
    registry.register(db_list_table_tool)
    """
    # 🌟【改动点 3】Excel 工具注入规范数据根（fs_backend.cwd），
    # 让相对路径稳定解析到项目根，并在模型臆造路径时返回真实文件清单接地。
    registry.register(LocalExcelTool(base_dir=str(fs_backend.cwd)))

    registry.register(web_search_tool)
    registry.register(tavily_search_tool)  # 第二梯队备选（提示词限制使用时机）

    registry.register(rag_search_tool)  # 正确持有了依赖的工具登记
    registry.register(graph_search_tool)
    registry.register(todo_list_tool)

    registry.register(file_read_tool)
    registry.register(file_list_tool)
    registry.register(file_grep_tool)

    return registry