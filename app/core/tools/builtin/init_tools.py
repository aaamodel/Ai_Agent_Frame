# -*- coding: utf-8 -*-
"""工具工厂：在这里实例化所有具体工具，并统一注入到注册中心。"""

from typing import Optional
from sqlalchemy.ext.asyncio import async_sessionmaker  # 假设你使用的工厂类型

from app.core.tools.builtin.feishu import FeishuBitableTool
from app.core.tools.builtin.graph_search import KnowledgeGraphSearchTool

from app.core.tools.builtin.sql_vanna import SalesSqlQueryTool, SalesSqlWriteTool
from app.core.tools.builtin.sales_report import SalesReportExportTool
from app.core.tools.builtin.rag_search import RagSearchTool
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
        model_router=None,
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
    # 联网搜索：**只注册一个对外工具** web_search ← DoubaoWebSearchTool（豆包 API 直连）。
    # Tavily（WebSearchTool）作为其**内部降级通道**注入，不注册、不进意图白名单，
    # 大模型既看不到也调不到它；"何时降级"由 DoubaoWebSearchTool 的代码分支决定
    # （空结果或调用异常 → 自动降级，见 doubao_search._try_fallback）。
    tavily_search_tool = WebSearchTool()
    web_search_tool = DoubaoWebSearchTool(fallback=tavily_search_tool)
    # 🌟【改动点 2】将上层注入的 rag_service 喂给 RagSearchTool 实例
    # 这样大模型在 Thought 循环里调用 rag_search 时，才能真正通过连接池查到 Milvus 数据
    rag_search_tool = RagSearchTool(rag_service=rag_service)

    graph_search_tool = KnowledgeGraphSearchTool()

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
    # ── 销售业务库（SQLite）自然语言取数 ──────────────────────────────
    # Excel 取数三件套已退役（见变更 migrate-sales-excel-to-sqlite-vanna）：
    # Excel 不携带 schema，模型只能猜列名与类型，于是要补偿出 9 个参数、difflib 模糊匹配、
    # "预览几行"推断结构——换到 SQLite 后 schema 由建表语句定死，这些补偿全部不需要。
    # ⚠️ 查询工具只读（另有只读护栏拒绝非 SELECT）；写工具登记进危险名单走人工审批。
    registry.register(SalesSqlQueryTool())
    registry.register(SalesSqlWriteTool())

    # 销售分析报表导出（写操作，人工审批）
    registry.register(SalesReportExportTool(base_dir=str(fs_backend.cwd)))

    registry.register(web_search_tool)
    # ⚠️ 刻意不注册 tavily_search_tool：它只是 web_search 的内部降级通道，
    # 一旦注册就会重新出现在工具清单里，把"代码层降级"退化成"模型自己选工具"。

    registry.register(rag_search_tool)  # 正确持有了依赖的工具登记
    registry.register(graph_search_tool)

    registry.register(file_read_tool)
    registry.register(file_list_tool)
    registry.register(file_grep_tool)

    return registry