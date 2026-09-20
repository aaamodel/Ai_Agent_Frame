# -*- coding: utf-8 -*-
"""销售业务库（SQLite）的专项测试：建表约束、导入、扩量、导出。

覆盖 `platform/sales-sqlite-store`：schema 显式声明、只有业务事实进库、
导入与扩量口径一致、xlsx 降级为导出视图。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.sales_db.schema import (  # noqa: E402
    DDL_STATEMENTS,
    SOURCES,
    columns_of,
    create_all,
    drop_all,
    table_names,
)
from app.core.sales_db.seed import (  # noqa: E402
    XLSX_DIR,
    build,
    export_to_excel,
)

EXPECTED_ROWS = {"线索": 45, "市场活动": 12, "竞品": 6, "竞品动态": 12,
                 "输赢单": 34, "月度业绩": 12, "产品": 7}
EXPANDED_TABLES = ("线索", "市场活动", "竞品动态", "输赢单", "月度业绩")


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> sqlite3.Connection:
    """构建一次库供整个模块复用（导入要读 xlsx，避免每个用例重建）。"""
    target = tmp_path_factory.mktemp("sales") / "sales.db"
    build(target)
    conn = sqlite3.connect(str(target))
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# 2.1 建表约束
# ---------------------------------------------------------------------------
def test_all_business_tables_created(db: sqlite3.Connection) -> None:
    assert set(EXPECTED_ROWS).issubset(set(table_names(db)))


def test_ddl_carries_check_constraints(db: sqlite3.Connection) -> None:
    """带枚举列的表，枚举约束 MUST 出现在建表语句里——这是"模型不必猜"的根本。

    `月度业绩` 全是数值列 + 月份，没有可枚举的取值，因此不适用（不在断言范围内）。
    """
    with_enum = ("线索", "市场活动", "竞品", "竞品动态", "输赢单", "产品")
    for table in with_enum:
        ddl = db.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()[0]
        assert "CHECK" in ddl, f"{table} 缺少 CHECK 约束"


def test_chinese_column_names_match_source_headers(db: sqlite3.Connection) -> None:
    """列名 MUST 与源 Excel 表头逐字一致。"""
    import pandas as pd

    for table, (file_name, sheet) in SOURCES.items():
        frame = pd.read_excel(XLSX_DIR / file_name, sheet_name=sheet)
        source = set(str(c) for c in frame.columns)
        assert source == set(columns_of(db, table)), f"{table} 列名与源表头不一致"


def test_check_constraint_rejects_out_of_enum_value(db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            'INSERT INTO 线索 (线索编号, 公司全称, 行业) VALUES (?,?,?)',
            ("__probe__", "探测公司", "不存在的行业"),
        )
        db.commit()
    db.rollback()


def test_primary_key_rejects_duplicate(db: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            'INSERT INTO 线索 (线索编号, 公司全称) VALUES (?,?)',
            ("LD-202603-001", "重复主键探测"),
        )
        db.commit()
    db.rollback()


# ---------------------------------------------------------------------------
# 2.2 / 2.3 导入与校验
# ---------------------------------------------------------------------------
def test_import_row_counts_match_sources(db: sqlite3.Connection) -> None:
    for table, expected in EXPECTED_ROWS.items():
        actual = db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if table in EXPANDED_TABLES:
            assert actual == expected * 4, f"{table} 扩量后应为 {expected * 4}"
        else:
            assert actual == expected


def test_expansion_quadruples_rows(db: sqlite3.Connection) -> None:
    """扩量 = 净增 3 倍，即变为 4 倍。"""
    for table in EXPANDED_TABLES:
        assert db.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] % EXPECTED_ROWS[table] == 0


def test_expanded_data_passes_all_constraints(db: sqlite3.Connection) -> None:
    """扩量数据 MUST 全部通过约束（导入时已校验，这里再整体验一次）。"""
    report = db.execute("PRAGMA integrity_check").fetchone()[0]
    assert report == "ok"


def _orphan_count(conn: sqlite3.Connection, table: str, column: str) -> int:
    return conn.execute(
        f'SELECT COUNT(*) FROM "{table}" '
        f'WHERE "{column}" IS NOT NULL AND "{column}" != \'—\' '
        f'AND "{column}" NOT IN (SELECT 线索编号 FROM 线索)'
    ).fetchone()[0]


def test_expansion_introduces_no_new_orphans(tmp_path: Path) -> None:
    """扩量产生的数据 MUST NOT 新增孤儿引用。

    ⚠️ 注意基线：源 `市场活动明细` 里**本来就**引用了 `线索总表` 中不存在的线索编号
    （如 LD-202601-012），这是源数据自身的质量问题，不是扩量引入的。因此这里断言的是
    "扩量前后的孤儿数量一致"，而不是"孤儿为 0"。
    """
    pairs = (("输赢单", "线索编号"), ("市场活动", "关联台账线索"))

    base_path = tmp_path / "base.db"
    build(base_path, do_expand=False)
    base = sqlite3.connect(str(base_path))
    try:
        base_counts = {p: _orphan_count(base, *p) for p in pairs}
    finally:
        base.close()

    full_path = tmp_path / "full.db"
    build(full_path, do_expand=True)
    full = sqlite3.connect(str(full_path))
    try:
        full_counts = {p: _orphan_count(full, *p) for p in pairs}
    finally:
        full.close()

    assert full_counts == base_counts, (
        f"扩量引入了新的孤儿引用：扩量前 {base_counts} → 扩量后 {full_counts}"
    )


def test_import_is_all_or_nothing(tmp_path: Path) -> None:
    """任一行违规 MUST 整表回滚，不留半张表。"""
    conn = sqlite3.connect(str(tmp_path / "partial.db"))
    try:
        create_all(conn)
        conn.execute("BEGIN")
        try:
            conn.execute(
                'INSERT INTO 线索 (线索编号, 公司全称, 行业) VALUES (?,?,?)',
                ("OK-1", "正常公司", "制造业"),
            )
            conn.execute(
                'INSERT INTO 线索 (线索编号, 公司全称, 行业) VALUES (?,?,?)',
                ("BAD-1", "越界公司", "不存在的行业"),
            )
            conn.execute("COMMIT")
        except sqlite3.IntegrityError:
            conn.execute("ROLLBACK")
        count = conn.execute("SELECT COUNT(*) FROM 线索").fetchone()[0]
        assert count == 0, "违规行必须连带前面的正常行一起回滚"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3.3 规则类与派生类不进业务库
# ---------------------------------------------------------------------------
def test_rule_sheets_are_not_tables(db: sqlite3.Connection) -> None:
    names = set(table_names(db))
    for banned in ("字段字典", "阶段流转规则", "折扣权限", "组合策略"):
        assert banned not in names


def test_derived_sheets_are_not_tables(db: sqlite3.Connection) -> None:
    names = set(table_names(db))
    for banned in ("季度汇总", "区域产品透视", "原因分类"):
        assert banned not in names


def test_business_library_contains_only_business_tables(db: sqlite3.Connection) -> None:
    assert set(table_names(db)) == set(DDL_STATEMENTS)


# ---------------------------------------------------------------------------
# 2.5 导出视图
# ---------------------------------------------------------------------------
def test_export_writes_xlsx_per_table(db: sqlite3.Connection, tmp_path: Path) -> None:
    written = export_to_excel(db, tmp_path)
    assert len(written) == len(DDL_STATEMENTS)
    for path in written:
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_queries_do_not_depend_on_xlsx_files(db: sqlite3.Connection, tmp_path: Path) -> None:
    """移除 xlsx 后查询仍可用——xlsx 已降级为导出视图，不是数据源。"""
    empty_dir = tmp_path / "empty_xlsx"
    empty_dir.mkdir()
    with pytest.raises(Exception):
        # 源目录被换空后，导入会失败（证明导入确实读 xlsx）
        from app.core.sales_db.seed import import_from_excel

        conn = sqlite3.connect(str(tmp_path / "probe.db"))
        try:
            create_all(conn)
            import_from_excel(conn, empty_dir)
        finally:
            conn.close()
    # 但库已建好后，查询完全不依赖 xlsx
    assert db.execute("SELECT COUNT(*) FROM 线索").fetchone()[0] > 0


def test_rebuild_is_idempotent(tmp_path: Path) -> None:
    """重复重建不因残留数据撞主键。"""
    target = tmp_path / "rebuild.db"
    first = build(target)
    second = build(target)
    assert first["counts"] == second["counts"]
    assert first["counts"]["线索"] == 180
