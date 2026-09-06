# -*- coding: utf-8 -*-
"""内置只读数据库查询工具（SQLAlchemy 异步会话）。"""

from __future__ import annotations

from typing import Any
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tools.base import BaseTool, ToolParameter

class ListTablesTool(BaseTool):
    """列出当前数据库 public 模式下所有的表名。"""

    def __init__(self, session_factory: Any | None = None) -> None:
        super().__init__()
        self.name = "list_tables"
        self.description = "获取当前数据库 'public' 模式下的所有表名列表。通常在生成 SQL 前调用，以避免猜测表名。"
        self.parameters = []  # 无需参数
        self._session_factory = session_factory

    def _get_fallback_session_factory(self) -> Any:
        """动态获取全局会话工厂，逻辑与 DatabaseQueryTool 保持一致。"""
        try:
            from app.infrastructure.database.session import async_session_factory
            if async_session_factory is not None:
                logger.info("⚡ list_tables 工具成功连接到全局 async_session_factory！")
                return async_session_factory
            raise RuntimeError(
                "检测到全局变量 async_session_factory 存在，但其值为 None。\n"
                "请检查应用启动文件（如 main.py），确保在 Agent 运行前已成功执行 `configure_session(init_engine())`。"
            )
        except ImportError as e:
            logger.error(f"无法从基础设施中导入会话工厂: {e}")
            raise RuntimeError(f"基础设施路径不匹配，无法导入 async_session_factory: {e}")

    async def execute(self, **kwargs: Any) -> Any:
        try:
            factory = self._session_factory or self._get_fallback_session_factory()
            async with factory() as sess:
                result = await sess.execute(
                    text("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;")
                )
                tables = [row[0] for row in result.all()]

            if not tables:
                return "数据库 'public' 模式下没有任何表。"

            # 返回易读的列表字符串，便于大模型直接理解
            formatted = "当前数据库包含以下表名：\n" + "\n".join(f"- {t}" for t in tables)
            return formatted

        except Exception as e:
            logger.error(f"list_tables 执行出错: {e}")
            return f"工具 [list_tables] 执行期间发生异常: {str(e)}"


class DescribeTableTool(BaseTool):
    """获取指定表的列信息（列名、类型、是否可为空等）。"""

    def __init__(self, session_factory: Any | None = None) -> None:
        super().__init__()
        self.name = "describe_table"
        self.description = "获取指定表的详细列信息（列名、数据类型、是否可为空）。在编写涉及该表的 SQL 前调用，可避免字段名猜测错误。"
        self.parameters = [
            ToolParameter(
                name="table_name",
                type="string",
                description="需要查看结构的表名（区分大小写，通常为小写）",
                required=True,
            )
        ]
        self._session_factory = session_factory

    def _get_fallback_session_factory(self) -> Any:
        """动态获取全局会话工厂，逻辑与 DatabaseQueryTool 保持一致。"""
        try:
            from app.infrastructure.database.session import async_session_factory
            if async_session_factory is not None:
                logger.info("⚡ describe_table 工具成功连接到全局 async_session_factory！")
                return async_session_factory
            raise RuntimeError(
                "检测到全局变量 async_session_factory 存在，但其值为 None。\n"
                "请检查应用启动文件（如 main.py），确保在 Agent 运行前已成功执行 `configure_session(init_engine())`。"
            )
        except ImportError as e:
            logger.error(f"无法从基础设施中导入会话工厂: {e}")
            raise RuntimeError(f"基础设施路径不匹配，无法导入 async_session_factory: {e}")

    async def execute(self, **kwargs: Any) -> Any:
        table_name = str(kwargs.get("table_name", "")).strip()
        if not table_name:
            return "错误：必须提供 table_name 参数。"

        try:
            factory = self._session_factory or self._get_fallback_session_factory()
            async with factory() as sess:
                # 查询列信息：列名、数据类型、字符最大长度、是否可为空、默认值
                result = await sess.execute(
                    text(
                        """
                        SELECT 
                            column_name,
                            data_type,
                            character_maximum_length,
                            is_nullable,
                            column_default
                        FROM information_schema.columns
                        WHERE table_schema = 'public' AND table_name = :table_name
                        ORDER BY ordinal_position;
                        """
                    ),
                    {"table_name": table_name},
                )
                columns = result.mappings().all()

            if not columns:
                return f"表 '{table_name}' 在 'public' 模式下不存在，或者没有任何列。"

            # 格式化为清晰的多行字符串
            lines = [f"表 {table_name} 的列信息："]
            for col in columns:
                name = col["column_name"]
                dtype = col["data_type"]
                max_len = col.get("character_maximum_length")
                nullable = "可空" if col["is_nullable"] == "YES" else "非空"
                default = col.get("column_default") or "无默认值"
                size_info = f"({max_len})" if max_len else ""
                lines.append(f"  - {name}: {dtype}{size_info} | {nullable} | 默认: {default}")
            return "\n".join(lines)

        except Exception as e:
            logger.error(f"describe_table 执行出错: {e}")
            return f"工具 [describe_table] 执行期间发生异常: {str(e)}"
class DatabaseQueryTool(BaseTool):
    """执行受控的只读 SQL（默认仅允许 SELECT）。"""

    def __init__(self, session_factory: Any | None = None) -> None:
        """
        :param session_factory: 可选，返回 AsyncSession 的异步上下文工厂；
            若为 None，execute 时会尝试从全局基础设施中自动动态获取。
        """
        super().__init__()
        self.name = "database_query"
        self.description = "在只读模式下执行 SQL 查询并返回行列表（禁止写操作）。"
        self.parameters = [
            ToolParameter(
                name="sql",
                type="string",
                description="只读 SQL，必须以 SELECT 开头",
                required=True,
            )
        ]
        self._session_factory = session_factory

    def _validate_sql(self, sql: str) -> str:
        s = sql.strip().rstrip(";")
        lower = s.lower()
        if not lower.startswith("select"):
            raise ValueError("仅允许 SELECT 查询")
        forbidden = ("insert", "update", "delete", "drop", "alter", "truncate", "create")
        for bad in forbidden:
            if bad in lower:
                raise ValueError(f"查询包含禁止关键字: {bad}")
        return s

    def _get_fallback_session_factory(self) -> Any:
        """
        🔍 自动指路：直接对接 app/infrastructure/database/session.py 中的全局工厂
        """
        try:
            # 🎯 精准导入你刚刚提供文件中的全局变量
            from app.infrastructure.database.session import async_session_factory

            if async_session_factory is not None:
                logger.info("⚡ database_query 工具成功连接到全局 async_session_factory 基础设施！")
                return async_session_factory

            # 如果是 None，说明你的 FastAPI 在启动时，还没有来得及执行 configure_session(engine)
            raise RuntimeError(
                "检测到全局变量 async_session_factory 存在，但其值为 None。\n"
                "请检查应用启动文件（如 main.py），确保在 Agent 运行前已成功执行 `configure_session(init_engine())`。"
            )
        except ImportError as e:
            logger.error(f"无法从基础设施中导入会话工厂，请检查路径。错误信息: {e}")
            raise RuntimeError(f"基础设施路径不匹配，无法导入 async_session_factory: {e}")

    async def execute(self, **kwargs: Any) -> Any:
        sql = str(kwargs.get("sql", "")).strip()
        if not sql:
            raise ValueError("参数 sql 不能为空")

        try:
            safe_sql = self._validate_sql(sql)
            session: AsyncSession | None = kwargs.get("session")
            factory = self._session_factory
            if session is None and factory is None:
                factory = self._get_fallback_session_factory()

            async def _run(sess: AsyncSession) -> list[dict[str, Any]]:
                result = await sess.execute(text(safe_sql))
                rows = result.mappings().all()
                return [dict(r) for r in rows]

            if session is not None:
                return await _run(session)

            async with factory() as sess:  # type: ignore[misc]
                return await _run(sess)

        except Exception as e:
            error_str = str(e)
            logger.error(f"数据库工具执行出错: {error_str}")

            # 🎯 针对表不存在的异常进行高级定制化召回
            if "UndefinedTableError" in error_str or "relation" in error_str:
                try:
                    # 顺手查一下当前库里到底有什么表，抓现行给大模型看
                    current_factory = self._session_factory or self._get_fallback_session_factory()
                    async with current_factory() as sess:
                        # 适用于 PostgreSQL 的查表 SQL
                        schema_res = await sess.execute(text(
                            "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
                        ))
                        tables = [row[0] for row in schema_res.all()]

                    return (
                        f"数据库执行失败：你指定的表在数据库中不存在。\n"
                        f"【系统级提示】：当前数据库的 'public' 模式下，实际存在的有效表名列表为：{tables}。\n"
                        f"请比对列表。如果这里面没有你想要的表，说明系统中完全没有这部分数据，请换用其他非数据库工具，禁止盲目重试！"
                    )
                except Exception as meta_err:
                    pass

            # 其他原生错误正常返回
            return f"工具 [database_query] 执行期间发生异常: {error_str}"