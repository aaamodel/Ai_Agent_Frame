# -*- coding: utf-8 -*-
"""销售业务库的构建：从 Excel 导入 → 扩量 → 导出为 xlsx 视图。

用法：

    python -m app.core.sales_db.seed --rebuild      # 建库 + 导入 + 扩量
    python -m app.core.sales_db.seed --export       # 导出 xlsx（导出视图）

设计要点见 design.md D2 / D8：
  - 必须**先建表再 append**，靠 SQLite 的 CHECK/PK 做"导入即校验"；
  - 任一行违规 → 整表回滚，绝不留半张表；
  - 扩量只使用既有枚举取值，且关联字段取自扩量后的主表，不产生孤儿引用。
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app.core.sales_db.schema import DDL_STATEMENTS, SOURCES, create_all, drop_all

#: 项目根（本文件位于 app/core/sales_db/seed.py → parents[3]）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_DB: Path = PROJECT_ROOT / "data" / "sales.db"
XLSX_DIR: Path = PROJECT_ROOT / "raw_data" / "sales_intel"

#: 固定随机种子，保证扩量结果可复现（评测基线不能被随机数打乱）
SEED: int = 20260919


# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def import_from_excel(conn: sqlite3.Connection, xlsx_dir: Path = XLSX_DIR) -> Dict[str, int]:
    """从源 Excel 导入全部业务表；返回 {表名: 导入行数}。

    ⚠️ 整表事务：任一行违反主键 / CHECK / NOT NULL 都会整体回滚，
    不会出现"导入了一半"的表。
    """
    imported: Dict[str, int] = {}
    for table, (file_name, sheet) in SOURCES.items():
        frame = pd.read_excel(xlsx_dir / file_name, sheet_name=sheet)
        columns = [c for c in _table_columns(conn, table) if c in frame.columns]
        subset = frame[columns].where(pd.notna(frame[columns]), None)

        rows: List[tuple] = [tuple(record) for record in subset.itertuples(index=False, name=None)]
        placeholders = ", ".join("?" for _ in columns)
        quoted = ", ".join(f'"{c}"' for c in columns)
        try:
            conn.execute("BEGIN")
            conn.executemany(
                f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})', rows
            )
            conn.execute("COMMIT")
        except sqlite3.Error:
            conn.execute("ROLLBACK")
            raise
        imported[table] = len(rows)
    return imported


# ---------------------------------------------------------------------------
# 扩量（净增 3 倍 = 变为 4 倍）
# ---------------------------------------------------------------------------
_NAME_HEAD = ("华澜", "鼎信", "悦邻", "中恒", "云杉", "谷雨", "迈极", "悦动",
              "津门", "南洋", "星桥", "砺石", "青柚", "沐辰", "恒益", "晟远")
_NAME_TAIL = ("智造科技", "金融服务集团", "商业连锁", "建设集团", "零售科技",
              "餐饮管理", "生物医药", "智能装备", "物流服务", "教育科技")


def _existing(conn: sqlite3.Connection, table: str, column: str) -> List[Any]:
    rows = conn.execute(f'SELECT DISTINCT "{column}" FROM "{table}"').fetchall()
    return [row[0] for row in rows if row[0] is not None]


def _gen_线索(conn: sqlite3.Connection, count: int, rng: random.Random) -> List[tuple]:
    leads = _existing(conn, "线索", "线索编号")
    owners = _existing(conn, "线索", "负责人")
    industries = ["制造业", "金融服务", "零售连锁", "汽车零部件",
                  "医疗器械", "物流运输", "教育服务", "能源化工"]
    regions = ["华北", "华东", "华南", "西南"]
    channels = ["市场活动", "官网留资", "转介绍", "电销", "内容营销"]
    statuses = ["线索", "MQL", "SQL", "商机", "方案验证", "商务谈判", "赢单", "输单", "搁置"]
    products = ["智能客服平台", "工单系统", "企业知识库"]
    actions = ["补全公司与联系人信息", "需求初步沟通", "输出需求确认清单",
               "安排方案演示", "推进商务报价", "确认合同条款"]
    notes = ["决策链清晰，重点跟进", "客户预算待确认", "已约方案演示", "待补充联系人信息"]

    rows: List[tuple] = []
    for index in range(count):
        month = 3 + (index % 10)
        seq = 100 + index
        rows.append((
            f"LD-2026{month:02d}-{seq:03d}",
            f"{rng.choice(_NAME_HEAD)}{rng.choice(_NAME_TAIL)}",
            rng.choice(industries),
            rng.choice([120, 300, 800, 1200, 2000, 3500]),
            rng.choice(regions),
            rng.choice(channels),
            rng.choice(statuses),
            rng.choice(products),
            rng.choice([5, 8, 15, 25, 45, 60, 90]),
            rng.choice(owners or ["王强"]),
            f"2026-{month:02d}-{rng.randint(1, 28):02d}",
            f"2026-09-{rng.randint(1, 6):02d}",
            rng.choice(actions),
            rng.randint(1, 12),
            rng.choice(notes),
        ))
    return rows


def _gen_市场活动(conn: sqlite3.Connection, count: int, rng: random.Random) -> List[tuple]:
    lead_ids = _existing(conn, "线索", "线索编号")
    types = ["线上研讨会", "私域活动", "行业展会", "内容营销"]
    regions = ["华北", "华东", "华南", "西南", "全国"]
    products = ["智能客服平台", "企业知识库", "工单系统"]
    owners = ["王强", "李娜", "张伟", "陈静"]
    names = ["客户成功私享会", "行业趋势研讨会", "产品体验日", "标杆客户发布会",
             "解决方案直播", "区域沙龙", "生态伙伴大会"]
    rows: List[tuple] = []
    for index in range(count):
        month = 1 + (index % 12)
        seq = 20 + index
        cost = round(rng.uniform(1.5, 9.0), 1)
        leads = rng.randint(60, 210)
        rows.append((
            f"MA-2026{month:02d}-{seq:02d}",
            f"{rng.choice(['智能客服', '工单自动化', '知识库落地', 'AI 质检'])}{rng.choice(names)}",
            rng.choice(types),
            f"2026-{month:02d}-{rng.randint(1, 28):02d}",
            rng.choice(regions),
            rng.choice(products),
            cost,
            leads,
            max(1, int(leads * rng.uniform(0.2, 0.35))),
            rng.randint(3, 14),
            rng.randint(1, 5),
            rng.randint(20, 140),
            rng.choice(lead_ids) if lead_ids else None,
            rng.choice(owners),
        ))
    return rows


def _unique_months(existing: List[str], count: int) -> List[str]:
    """生成 ``count`` 个不与既有月份冲突的 ``YYYY-MM``（先向后、再向前延伸）。"""
    taken = set(existing)
    candidates: List[str] = []
    year, month = 2026, 9
    for _ in range(240):
        candidates.append(f"{year}-{month:02d}")
        month += 1
        if month > 12:
            month, year = 1, year + 1
    year, month = 2025, 8
    for _ in range(240):
        candidates.append(f"{year}-{month:02d}")
        month -= 1
        if month < 1:
            month, year = 12, year - 1
    picked: List[str] = []
    for candidate in candidates:
        if candidate not in taken and candidate not in picked:
            picked.append(candidate)
        if len(picked) >= count:
            break
    return picked


def _gen_竞品动态(conn: sqlite3.Connection, count: int, rng: random.Random) -> List[tuple]:
    rivals = _existing(conn, "竞品", "竞品名称")
    events = ["发布新版本", "调整价格体系", "推出私有化部署", "生态合作签约",
              "区域促销活动", "上线行业模板", "升级 SLA 能力"]
    impacts = ["低", "中", "高：需关注赢单率变化", "中：影响单产品市场", "低：ICP 重叠小"]
    actions = ["更新竞品对比话术", "组合策略前置", "无需动作，持续观察",
               "促销期避正面比价", "补充行业案例与资质材料"]
    sources = ["官网产品发布页", "官网公告", "自媒体", "销售一线反馈", "行业媒体"]
    rows: List[tuple] = []
    seen: set = set()
    # 复合主键 (日期, 竞品)：撞车就重摇，直到凑够 count 条唯一记录
    guard = 0
    while len(rows) < count and guard < count * 50:
        guard += 1
        month = 3 + (len(rows) % 7)
        day = rng.randint(1, 28)
        rival = rng.choice(rivals or ["智齿"])
        key = (f"2026-{month:02d}-{day:02d}", rival)
        if key in seen:
            continue
        seen.add(key)
        rows.append((
            key[0],
            rival,
            f"{rival}{rng.choice(events)}",
            rng.choice(impacts),
            rng.choice(actions),
            rng.choice(sources),
            rng.choice(["A", "B", "C"]),
        ))
    return rows


def _gen_输赢单(conn: sqlite3.Connection, count: int, rng: random.Random) -> List[tuple]:
    lead_ids = _existing(conn, "线索", "线索编号")
    regions = ["华北", "华东", "华南", "西南"]
    products = ["智能客服平台", "企业知识库", "工单系统"]
    versions = ["标准版", "团队版", "专业版", "企业版", "旗舰版"]
    rivals = ["—", "语雀企业版", "智齿", "网易七鱼", "易维帮助台"]
    owners = ["王强", "陈静", "李娜", "张伟"]
    reasons = ["—", "价格竞争", "竞品方案", "预算冻结", "决策链变动", "需求放弃"]
    rows: List[tuple] = []
    for index in range(count):
        month = 6 + (index % 3)
        seq = 50 + index
        result = rng.choice(["赢单", "输单"])
        rows.append((
            f"WL-2026{month:02d}-{seq:02d}",
            f"2026-{month:02d}",
            rng.choice(lead_ids) if lead_ids else "—",
            f"{rng.choice(_NAME_HEAD)}{rng.choice(_NAME_TAIL)}",
            rng.choice(regions),
            rng.choice(products),
            rng.choice(versions),
            rng.choice([5, 8, 12, 15, 17, 18, 45, 90]),
            result,
            rng.choice(reasons) if result == "输单" else "—",
            "标准版首单，客户计划次年扩购" if result == "赢单" else "客户压缩预算，需求降级",
            "—" if result == "赢单" else rng.choice([r for r in rivals if r != "—"]),
            rng.choice(owners),
            rng.choice(["在台账", "已归档"]),
        ))
    return rows


def _gen_月度业绩(conn: sqlite3.Connection, count: int, rng: random.Random) -> List[tuple]:
    rows: List[tuple] = []
    # 月份是主键：必须先拿到 count 个不冲突的月份，再逐月生成
    for label in _unique_months(_existing(conn, "月度业绩", "月份"), count):
        new_leads = rng.randint(95, 165)
        mql = max(1, int(new_leads * rng.uniform(0.25, 0.35)))
        sql = max(1, int(mql * rng.uniform(0.35, 0.45)))
        deals = max(1, int(sql * rng.uniform(0.4, 0.55)))
        won = max(1, int(deals * rng.uniform(0.25, 0.45)))
        lost = max(0, deals - won)
        amount = won * rng.randint(20, 35)
        rows.append((
            label,
            new_leads, mql, sql, deals, won, lost, amount,
            f"{won / max(1, won + lost) * 100:.1f}%",
            rng.randint(5, 7),
            round(amount / max(1, won), 1),
        ))
    return rows


def _gen_线索_like_reference() -> None:
    """参照类表（竞品 / 产品）的扩量由调用方单独处理，见 expand()。"""
    return None


_GENERATORS = {
    "线索": _gen_线索,
    "市场活动": _gen_市场活动,
    "竞品动态": _gen_竞品动态,
    "输赢单": _gen_输赢单,
    "月度业绩": _gen_月度业绩,
}


def expand(
    conn: sqlite3.Connection,
    factor: int = 4,
    rng: random.Random | None = None,
) -> Dict[str, int]:
    """把业务表行数扩到 ``factor`` 倍（净增 factor-1 倍）；返回 {表名: 新增行数}。

    ⚠️ 顺序有依赖：先扩 `线索`，再扩引用它的表，保证关联字段无孤儿。
    竞品 / 产品 属**参照数据**（一个竞品一行、一个 SKU 一行），合成新条目会污染
    业务语义，因此保持原样——这是有意偏离"全部 4 倍"的例外，已在报告中说明。
    """
    rng = rng or random.Random(SEED)
    added: Dict[str, int] = {}
    for table in ("线索", "市场活动", "竞品动态", "输赢单", "月度业绩"):
        current: int = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        need: int = current * (factor - 1)
        if need <= 0:
            continue
        rows = _GENERATORS[table](conn, need, rng)
        columns = _table_columns(conn, table)
        placeholders = ", ".join("?" for _ in columns)
        quoted = ", ".join(f'"{c}"' for c in columns)
        try:
            conn.execute("BEGIN")
            conn.executemany(
                f'INSERT INTO "{table}" ({quoted}) VALUES ({placeholders})', rows
            )
            conn.execute("COMMIT")
        except sqlite3.Error:
            conn.execute("ROLLBACK")
            raise
        added[table] = len(rows)
    return added


# ---------------------------------------------------------------------------
# 导出视图
# ---------------------------------------------------------------------------
def export_to_excel(conn: sqlite3.Connection, out_dir: Path = XLSX_DIR) -> List[str]:
    """把库中业务表导出为 xlsx（Excel 降级为导出视图后的产物）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for table in DDL_STATEMENTS:
        frame = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
        target = out_dir / f"{table}.xlsx"
        frame.to_excel(target, index=False)
        written.append(str(target))
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build(db_path: Path = DEFAULT_DB, *, do_expand: bool = True) -> Dict[str, Any]:
    """全量重建：建表 → 导入 → 扩量。返回各表最终行数。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        # 全量重建：先 DROP，保证重复执行不会因残留数据撞主键
        drop_all(conn)
        create_all(conn)
        imported = import_from_excel(conn)
        added = expand(conn) if do_expand else {}
        counts = {
            table: conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in DDL_STATEMENTS
        }
        return {"imported": imported, "added": added, "counts": counts}
    finally:
        conn.close()


def _main() -> None:
    parser = argparse.ArgumentParser(description="销售业务库构建/导出")
    parser.add_argument("--rebuild", action="store_true", help="建库 + 导入 + 扩量")
    parser.add_argument("--export", action="store_true", help="导出 xlsx")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    if args.rebuild:
        result = build(Path(args.db))
        print("导入:", result["imported"])
        print("新增:", result["added"])
        print("最终行数:", result["counts"])
    if args.export:
        conn = sqlite3.connect(args.db)
        try:
            for path in export_to_excel(conn):
                print("写出:", path)
        finally:
            conn.close()


if __name__ == "__main__":
    _main()
