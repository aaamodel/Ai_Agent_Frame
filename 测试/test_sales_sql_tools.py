# -*- coding: utf-8 -*-
"""销售 SQL 工具的专项测试（只读护栏 + 受约束写入 + 初始化失败可见）。

覆盖 `agent/sql-query-tool` 的只读与写入护栏部分。
需要真实模型通道的用例（端到端问答）不放在这里，由 `test_sales_sql_e2e.py` 承担。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.sales_db.schema import create_all  # noqa: E402
from app.core.tools.builtin.sql_vanna import (  # noqa: E402
    SalesSqlQueryTool,
    SalesSqlWriteTool,
    _assert_readonly,
    get_vanna,
    reset_vanna,
)


@pytest.fixture()
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    """临时业务库：只插 3 行线索，够覆盖命中/未命中/多命中。"""
    target = tmp_path / "probe.db"
    conn = sqlite3.connect(str(target))
    create_all(conn)
    conn.executemany(
        "INSERT INTO 线索 (线索编号, 公司全称, 行业, 区域, 负责人) VALUES (?,?,?,?,?)",
        [("LD-001", "甲公司", "制造业", "华北", "王强"),
         ("LD-002", "乙公司", "制造业", "华东", "李娜"),
         ("LD-003", "丙公司", "金融服务", "华南", "张伟")],
    )
    conn.commit()
    monkeypatch.setenv("SALES_DB_PATH", str(target))
    yield conn
    conn.close()


# ── 只读护栏 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("sql", [
    "UPDATE 线索 SET 负责人='x'",
    "DELETE FROM 线索 WHERE 线索编号='LD-001'",
    "INSERT INTO 线索 (线索编号) VALUES ('x')",
    "DROP TABLE 线索",
    "ALTER TABLE 线索 ADD COLUMN x TEXT",
    "ATTACH DATABASE 'other.db' AS o",
    "PRAGMA table_info(线索)",
    "VACUUM",
])
def test_readonly_guard_rejects_write_statements(sql: str) -> None:
    with pytest.raises(ValueError, match="只读"):
        _assert_readonly(sql)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM 线索",
    "WITH t AS (SELECT * FROM 线索) SELECT * FROM t",
    "-- intermediate_sql\nSELECT COUNT(*) FROM 线索",
    "```sql\nSELECT * FROM 线索\n```",
    "/* 说明 */ SELECT * FROM 线索;",
])
def test_readonly_guard_allows_read_statements(sql: str) -> None:
    """前导注释/围栏必须先剥离，否则会把合法 SELECT 误杀（实测踩过一次）。"""
    _assert_readonly(sql)


def test_readonly_guard_rejects_comment_only_sql() -> None:
    with pytest.raises(ValueError, match="为空"):
        _assert_readonly("-- 只有注释没有语句")


# ── 受约束写入 ──────────────────────────────────────────────────────────
async def _write(**kwargs) -> str:
    return await SalesSqlWriteTool().execute(**kwargs)


@pytest.mark.asyncio
async def test_write_unique_hit_succeeds(db: sqlite3.Connection) -> None:
    result = await _write(table="线索", filter_column="线索编号", filter_value="LD-001",
                          target_column="负责人", new_value="赵磊")
    assert "成功" in result
    assert db.execute("SELECT 负责人 FROM 线索 WHERE 线索编号='LD-001'").fetchone()[0] == "赵磊"


@pytest.mark.asyncio
async def test_write_zero_hit_is_rejected(db: sqlite3.Connection) -> None:
    result = await _write(table="线索", filter_column="线索编号", filter_value="NOT-EXIST",
                          target_column="负责人", new_value="赵磊")
    assert "命中 0 行" in result and "未做任何修改" in result
    assert db.execute("SELECT COUNT(*) FROM 线索 WHERE 负责人='赵磊'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_write_multi_hit_is_rejected_with_count(db: sqlite3.Connection) -> None:
    result = await _write(table="线索", filter_column="行业", filter_value="制造业",
                          target_column="负责人", new_value="赵磊")
    assert "命中了 2 行" in result and "拒绝" in result
    assert db.execute("SELECT COUNT(*) FROM 线索 WHERE 负责人='赵磊'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_write_rejects_unknown_table(db: sqlite3.Connection) -> None:
    result = await _write(table="不存在的表", filter_column="线索编号", filter_value="LD-001",
                          target_column="负责人", new_value="x")
    assert "不存在" in result


@pytest.mark.asyncio
async def test_write_rejects_unknown_column_and_lists_real_ones(db: sqlite3.Connection) -> None:
    result = await _write(table="线索", filter_column="不存在的列", filter_value="LD-001",
                          target_column="负责人", new_value="x")
    assert "没有列" in result and "线索编号" in result


@pytest.mark.asyncio
async def test_write_parameterizes_values_so_injection_cannot_escape(db: sqlite3.Connection) -> None:
    """列名走白名单、值走 ? 占位：注入串既当不了列名，也改不动别的行。"""
    result = await _write(table="线索", filter_column="负责人' OR '1'='1", filter_value="x",
                          target_column="备注", new_value="y")
    assert "没有列" in result
    # 全表未被误改
    assert db.execute("SELECT COUNT(*) FROM 线索 WHERE 备注='y'").fetchone()[0] == 0


@pytest.mark.asyncio
async def test_write_rejects_incomplete_arguments(db: sqlite3.Connection) -> None:
    result = await _write(table="线索", filter_column="线索编号", target_column="负责人",
                          new_value="x")
    assert "必须提供" in result


# ── 初始化失败必须可见（不能静默不可用）────────────────────────────────
@pytest.mark.asyncio
async def test_query_reports_init_failure_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_BASE", "")
    reset_vanna()
    result = await SalesSqlQueryTool().execute(question="有多少条线索？")
    assert "初始化失败" in result


def test_get_vanna_raises_without_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_BASE", "")
    reset_vanna()
    with pytest.raises(RuntimeError, match="模型通道"):
        get_vanna()
    reset_vanna()
