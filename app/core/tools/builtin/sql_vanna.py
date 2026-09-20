# -*- coding: utf-8 -*-
"""销售业务库的自然语言取数工具（Vanna 核心，注册为 BaseTool 子类）。

为什么替换掉 Excel 取数工具：
    Excel 不携带 schema，模型面对单元格只能猜"这一列是产品线还是公司全称"，
    于是要补偿出 9 个参数、`difflib` 模糊匹配、"预览几行"推断结构。换到 SQLite 后
    `CREATE TABLE` 把 schema 定死，`LIKE` / `GROUP BY` / `JOIN` 全是标准能力。

设计要点（详见变更的 design.md）：
  - **两个工具**而不是参照实现的"生成 SQL + 执行 SQL"双工具：本项目每次工具调用都要
    模型决策一轮，拆两步会让"取一次数"变成两轮决策，还把 SQL 塞进对话历史。合一后
    SQL 仍随结果返回，可审计性不降（D3）。
  - **只支持 SQLite**：参照实现的 PostgreSQL 分支与跨库路由**不引入**（D5）。
  - **模型不走 ModelRouter**：按需求硬编码 DashScope OpenAI 兼容端点（D6）。
    代价是这一路不受熔断降级 / 调用预算 / Langfuse 追踪覆盖，属于已知盲区。

依赖版本：项目锁定 Vanna **0.7.3**，因此沿用参照实现的原始导入路径
``vanna.chromadb.chromadb_vector`` / ``vanna.openai.openai_chat``。

⚠️ 版本差异备忘（踩过一次）：曾误装 Vanna 2.0.2，其包结构已重构，
``vanna.chromadb.*`` 不存在（2.x 把它挪到 ``vanna.legacy.*``）。当前回到 0.7.3，
**不要**改用 legacy 路径，也不要再升级到 2.x——``connect_sqlite`` 与
``connect_to_sqlite`` 的命名在两个大版本间也不同（0.7.3 只有后者）。
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter
from dotenv import load_dotenv

load_dotenv()
# ── 硬编码的模型通道配置（不接 ModelRouter，见模块文档）──────────────────────
# 模型名：优先专用变量，其次沿用项目既有的 OPENAI_LLM_MODEL，最后兜底。
# ⚠️ 选型的现实约束：DashScope 账号开了 "use free tier only"，
#    `qwen-plus` 会直接 403（Free quota exhausted），`qwen3.8-flash` 在额度内。
_SQL_LLM_MODEL: str = (
    os.getenv("SALES_SQL_MODEL") or os.getenv("OPENAI_LLM_MODEL") or "qwen3.8-flash"
)
_DEFAULT_DB_PATH: str = os.path.join("data", "sales.db")
_DEFAULT_CHROMA_PATH: str = os.path.join("data", "chroma_sales_sql")

#: 只读通道允许的语句类型
_READONLY_PREFIXES = ("select", "with")
#: 只读通道显式禁止的语句（先于前缀判断拦截，避免 `with ... insert` 之类绕过）
_FORBIDDEN_SQL_TOKENS = (
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "pragma", "vacuum", "reindex", "begin", "commit",
)


def _load_sqlite_conn() -> sqlite3.Connection:
    """业务库连接（单文件 SQLite）。"""
    path = os.getenv("SALES_DB_PATH", _DEFAULT_DB_PATH)
    return sqlite3.connect(path, check_same_thread=False)


class SalesVanna:
    """Vanna 实例 + 一个 SQLite 连接。

    惰性初始化：Vanna 会拉起 ChromaDB 并做一次 DDL 训练，成本不低，
    不该在进程启动时无条件付出（且工具可能整场请求都没被用到）。
    """

    def __init__(self, db_path: Optional[str] = None, chroma_path: Optional[str] = None) -> None:
        from openai import OpenAI  # noqa: PLC0415 - 惰性：避免未配置 Key 就导入失败
        from vanna.chromadb.chromadb_vector import ChromaDB_VectorStore  # noqa: PLC0415
        from vanna.openai.openai_chat import OpenAI_Chat  # noqa: PLC0415

        api_key = os.getenv("OPENAI_API_KEY") or ""
        base_url = os.getenv("OPENAI_API_BASE") or ""
        if not api_key or not base_url:
            raise RuntimeError(
                "销售 SQL 工具未配置模型通道：需要 OPENAI_API_KEY 与 OPENAI_API_BASE"
                "（当前硬编码走 DashScope OpenAI 兼容端点，不接 ModelRouter）。"
            )

        config: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "model": _SQL_LLM_MODEL,
            "path": chroma_path or _DEFAULT_CHROMA_PATH,
            "collection_name": "sales_sql_schemas",
        }

        class _Vanna(ChromaDB_VectorStore, OpenAI_Chat):
            """与参照实现 vanna_sql.py 的 MyVanna 同构（去掉了 pg/多引擎路由）。"""

            def __init__(self, cfg: Dict[str, Any]) -> None:
                client = cfg.get("client") or OpenAI(
                    api_key=cfg.get("api_key"), base_url=cfg.get("base_url")
                )
                ChromaDB_VectorStore.__init__(self, config=cfg)
                OpenAI_Chat.__init__(self, client=client, config=cfg)

        self._vn = _Vanna(config)
        self._conn = sqlite3.connect(db_path or os.getenv("SALES_DB_PATH", _DEFAULT_DB_PATH),
                                     check_same_thread=False)
        self._vn.connect_to_sqlite(db_path or os.getenv("SALES_DB_PATH", _DEFAULT_DB_PATH))

    # ------------------------------------------------------------------
    def sync_ddl(self) -> List[str]:
        """把当前库的表结构增量同步进向量库；返回发生变更的表名。

        新增 → 训练；变更 → 删旧训练新；删除 → 移除训练数据。
        schema 知识 MUST 与实际库同源，硬编码表名清单会随结构变化而失效。
        """
        rows = self._conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        current: Dict[str, str] = {name: (sql or "").strip() for name, sql in rows}

        trained: Dict[str, Any] = {}
        frame = self._vn.get_training_data()
        if frame is not None and not getattr(frame, "empty", True):
            for _, row in frame[frame["training_data_type"] == "ddl"].iterrows():
                ddl = str(row.get("content") or row.get("ddl") or "")
                name = _extract_table_name(ddl)
                if name:
                    trained[name] = {"id": row.get("id"), "ddl": ddl.strip()}

        changed: List[str] = []
        for name, ddl in current.items():
            if name not in trained:
                self._vn.train(ddl=ddl)
                changed.append(name)
            elif trained[name]["ddl"] != ddl:
                self._vn.remove_training_data(id=trained[name]["id"])
                self._vn.train(ddl=ddl)
                changed.append(name)
        for name, info in trained.items():
            if name not in current:
                self._vn.remove_training_data(id=info["id"])
                changed.append(name)

        if changed:
            logger.info("销售 SQL 工具的 DDL 训练数据已同步: {}", changed)
        return changed

    def generate_sql(self, question: str) -> str:
        return self._vn.generate_sql(question=question) or ""

    def run_readonly(self, sql: str):
        """执行只读 SQL，返回 DataFrame。"""
        return self._vn.run_sql(sql=sql)

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn


def _extract_table_name(ddl: str) -> str:
    """从 DDL 中取表名。"""
    clean = ddl.replace('"', "").replace("`", "").replace("'", "")
    match = re.search(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w\.]+)", clean, re.IGNORECASE)
    return match.group(1) if match else ""


def _strip_sql_noise(sql: str) -> str:
    """剥掉代码围栏与注释，露出真正的语句。

    ⚠️ 必须做：Vanna 的提示词约定"中间查询"要以注释 `-- intermediate_sql` 开头，
    模型也常返回 ```sql 围栏。若不先剥离，下面的前缀判断会把**合法 SELECT 误判为
    非只读**并拒绝——实测已出现一次。
    """
    text = (sql or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"--[^\n]*", " ", text)
    return text.strip().rstrip(";").strip()


def _assert_readonly(sql: str) -> None:
    """只读护栏：非 SELECT/WITH 一律拒绝。"""
    text = _strip_sql_noise(sql)
    if not text:
        raise ValueError("SQL 为空（去掉注释后没有任何语句）")
    lowered = text.lower()
    for token in _FORBIDDEN_SQL_TOKENS:
        if re.search(r"\b" + token + r"\b", lowered):
            raise ValueError(
                f"查询通道只读，禁止 {token.upper()} 语句；修改数据请改用写入工具。"
            )
    if not lowered.startswith(_READONLY_PREFIXES):
        raise ValueError("查询通道只读，只接受 SELECT / WITH 开头的语句。")


_vanna_singleton: Optional[SalesVanna] = None


def get_vanna() -> SalesVanna:
    """获取（必要时初始化）共享的 Vanna 实例。"""
    global _vanna_singleton
    if _vanna_singleton is None:
        _vanna_singleton = SalesVanna()
        _vanna_singleton.sync_ddl()
    return _vanna_singleton


def reset_vanna() -> None:
    """丢弃实例（测试夹具用）。"""
    global _vanna_singleton
    _vanna_singleton = None


class SalesSqlQueryTool(BaseTool):
    """自然语言 → 业务库查询结果（只读）。"""

    name: str = "sales_sql_query"
    description: str = (
        "用一句中文直接向销售业务库要答案（底层在 SQLite 上执行 SQL）。"
        "适用于线索、市场活动、竞品、竞品动态、输赢单、月度业绩、产品报价等业务数据的"
        "查询与统计：筛选、分组聚合、排序、TopN、占比、同环比、多表关联。"
        "不适用于规则与口径类问题（字段含义、阶段流转规则、折扣权限、组合策略）——"
        "那些请用 rag_knowledge_search 检索知识库。"
        "本工具只读，不能修改数据。"
    )

    def __init__(self) -> None:
        super().__init__()
        self.parameters = [
            ToolParameter(
                name="question",
                type="string",
                description="一句中文业务问题，如：三季度哪条产品线赢单金额最高？",
                required=True,
            ),
        ]

    async def execute(self, **kwargs: Any) -> str:
        question: str = str(kwargs.get("question") or "").strip()
        if not question:
            return "错误：未提供查询问题"

        try:
            vanna = get_vanna()
        except Exception as exc:  # noqa: BLE001 - 初始化失败必须可见，不能静默不可用
            return f"错误：销售 SQL 工具初始化失败（{type(exc).__name__}）：{exc}"

        try:
            sql = vanna.generate_sql(question)
            if not sql:
                return "错误：未能生成有效的 SQL。请换一种问法（明确表名/条件/时间范围）。"
            _assert_readonly(sql)
            frame = vanna.run_readonly(sql)
        except ValueError as exc:
            return f"错误：{exc}"
        except Exception as exc:  # noqa: BLE001
            return f"错误：SQL 生成或执行失败（{type(exc).__name__}）：{exc}"

        if frame is None or getattr(frame, "empty", True):
            return f"## 查询：{question}\n\n生成 SQL：\n```sql\n{sql}\n```\n\n查询成功，但没有匹配的数据。"

        try:
            table = frame.to_markdown(index=False)
        except Exception:  # noqa: BLE001 - to_markdown 依赖 tabulate，缺失时退回字符串
            table = frame.to_string()
        return (
            f"## 查询：{question}\n\n生成 SQL：\n```sql\n{sql}\n```\n\n结果：\n{table}"
        )


class SalesSqlWriteTool(BaseTool):
    """受约束的业务库写入：条件列 + 目标列 → 参数化 UPDATE。"""

    name: str = "sales_sql_write"
    description: str = (
        "修改销售业务库中的某一条记录（生成 UPDATE 并在库内执行）。"
        "要求条件唯一命中一行：命中 0 行或命中多行都会被拒绝，不会误改数据。"
        "需要人工审批。不适用于批量写入或新增记录。"
    )

    def __init__(self, db_path: Optional[str] = None) -> None:
        super().__init__()
        self._db_path = db_path
        self.parameters = [
            ToolParameter(name="table", type="string",
                          description="目标业务表名（中文表名，如 线索）", required=True),
            ToolParameter(name="filter_column", type="string",
                          description="条件列名（必须是该表的真实列名）", required=True),
            ToolParameter(name="filter_value", type="string",
                          description="条件值：该列等于此值的行才被修改", required=True),
            ToolParameter(name="target_column", type="string",
                          description="要修改的列名（必须是该表的真实列名）", required=True),
            ToolParameter(name="new_value", type="string",
                          description="写入的新值", required=True),
        ]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path or os.getenv("SALES_DB_PATH", _DEFAULT_DB_PATH))

    async def execute(self, **kwargs: Any) -> str:
        table = str(kwargs.get("table") or "").strip()
        filter_column = str(kwargs.get("filter_column") or "").strip()
        filter_value = kwargs.get("filter_value")
        target_column = str(kwargs.get("target_column") or "").strip()
        new_value = kwargs.get("new_value")

        if not all([table, filter_column, target_column]) or filter_value is None:
            return "错误：必须提供 table / filter_column / filter_value / target_column / new_value"

        conn = self._connect()
        try:
            known_tables = [row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()]
            if table not in known_tables:
                return f"错误：表 {table!r} 不存在。可选业务表：{'、'.join(known_tables)}"

            columns = [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]
            for column in (filter_column, target_column):
                if column not in columns:
                    return f"错误：表 {table!r} 中没有列 {column!r}。真实列：{'、'.join(columns)}"

            # 值一律参数化；列名已过白名单，可安全拼接
            hit = conn.execute(
                f'SELECT rowid, "{target_column}" FROM "{table}" WHERE "{filter_column}" = ?',
                (filter_value,),
            ).fetchall()

            if not hit:
                return (
                    f"错误：条件 {filter_column}={filter_value!r} 在表 [{table}] 中命中 0 行，"
                    "未做任何修改。请核对取值后重试。"
                )
            if len(hit) > 1:
                return (
                    f"错误：条件 {filter_column}={filter_value!r} 命中了 {len(hit)} 行，"
                    "写操作要求唯一命中，已拒绝修改（防止误改多条记录）。请收紧条件后重试。"
                )

            old_value = hit[0][1]
            cursor = conn.execute(
                f'UPDATE "{table}" SET "{target_column}" = ? WHERE "{filter_column}" = ?',
                (new_value, filter_value),
            )
            conn.commit()
            if cursor.rowcount != 1:
                conn.rollback()
                return "错误：更新影响行数不为 1，已回滚。"
            return (
                f"成功：表 [{table}] 中 {filter_column}={filter_value!r} 的唯一一行，"
                f"列 {target_column!r} 已由 {old_value!r} 更新为 {new_value!r}。"
            )
        except sqlite3.Error as exc:
            conn.rollback()
            return f"错误：写入失败（{type(exc).__name__}）：{exc}"
        finally:
            conn.close()
