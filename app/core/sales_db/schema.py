# -*- coding: utf-8 -*-
"""销售业务库（SQLite）的表结构。

为什么把 schema 显式写在这里：

    这是本变更的**全部意义所在**。Excel 不携带 schema，模型面对单元格只能猜
    "这一列是产品线还是公司全称"，于是要补偿出 9 个参数、`difflib` 模糊匹配、
    "预览几行"推断结构。写死 `CREATE TABLE` 之后，`PRIMARY KEY` / `CHECK` /
    `NOT NULL` 就是模型与代码都能直接读取的**确定约束**。

    因此建表 MUST 手写，`df.to_sql(if_exists="replace")` 按 dtype 推断的建表
    不会生成主键与 CHECK，等于把本变更的目标又丢回去。

枚举取值全部取自 `raw_data/sales_intel` 中数据的真实取值，不是拍脑袋列的。
"""

from __future__ import annotations

import sqlite3
from typing import Dict, List

#: 业务表 → 建表语句。中文列名与源 Excel 表头逐字一致，不允许改写成英文/拼音。
DDL_STATEMENTS: Dict[str, str] = {
    "线索": """
CREATE TABLE IF NOT EXISTS 线索 (
    线索编号        TEXT PRIMARY KEY,
    公司全称        TEXT NOT NULL,
    行业            TEXT CHECK(行业 IN ('制造业','金融服务','零售连锁','汽车零部件',
                                        '医疗器械','物流运输','教育服务','能源化工')),
    员工规模        INTEGER,
    区域            TEXT CHECK(区域 IN ('华北','华东','华南','西南')),
    来源渠道        TEXT CHECK(来源渠道 IN ('市场活动','官网留资','转介绍','电销','内容营销')),
    状态            TEXT CHECK(状态 IN ('线索','MQL','SQL','商机','方案验证',
                                        '商务谈判','赢单','输单','搁置')),
    意向产品        TEXT CHECK(意向产品 IN ('智能客服平台','工单系统','企业知识库')),
    "预估金额(万元)" INTEGER,
    负责人          TEXT CHECK(负责人 IN ('王强','李娜','刘洋','孙倩','张伟','赵磊','陈静')),
    创建日期        TEXT,
    最近跟进日期    TEXT,
    下一步动作      TEXT,
    MEDDIC评分      INTEGER,
    备注            TEXT
)
""",
    "市场活动": """
CREATE TABLE IF NOT EXISTS 市场活动 (
    活动编号        TEXT PRIMARY KEY,
    活动名称        TEXT NOT NULL,
    类型            TEXT CHECK(类型 IN ('线上研讨会','私域活动','行业展会','内容营销')),
    举办日期        TEXT,
    区域            TEXT CHECK(区域 IN ('华北','华东','华南','西南','全国')),
    目标产品线      TEXT CHECK(目标产品线 IN ('智能客服平台','企业知识库','工单系统')),
    "投入成本(万元)" REAL,
    获取线索数      INTEGER,
    MQL数           INTEGER,
    转化商机数      INTEGER,
    转化赢单数      INTEGER,
    "赢单金额(万元)" INTEGER,
    关联台账线索    TEXT,
    负责人          TEXT CHECK(负责人 IN ('王强','李娜','张伟','陈静'))
)
""",
    "竞品": """
CREATE TABLE IF NOT EXISTS 竞品 (
    竞品名称        TEXT PRIMARY KEY,
    所属公司        TEXT,
    对标我方产品线  TEXT,
    "定价区间(万元/年)" TEXT,
    核心优势        TEXT,
    主要劣势        TEXT,
    威胁等级        TEXT CHECK(威胁等级 IN ('高','中','低')),
    我方应对策略    TEXT,
    情报可信度基准  TEXT CHECK(情报可信度基准 IN ('A','B'))
)
""",
    "竞品动态": """
CREATE TABLE IF NOT EXISTS 竞品动态 (
    日期            TEXT,
    竞品            TEXT,
    事件            TEXT NOT NULL,
    对我方影响      TEXT,
    建议应对动作    TEXT,
    信息来源        TEXT,
    可信度          TEXT CHECK(可信度 IN ('A','B','C')),
    PRIMARY KEY (日期, 竞品)
)
""",
    "输赢单": """
CREATE TABLE IF NOT EXISTS 输赢单 (
    记录编号        TEXT PRIMARY KEY,
    关闭月份        TEXT,
    线索编号        TEXT,
    公司全称        TEXT NOT NULL,
    区域            TEXT CHECK(区域 IN ('华北','华南','华东','西南')),
    产品线          TEXT CHECK(产品线 IN ('智能客服平台','企业知识库','工单系统')),
    版本            TEXT CHECK(版本 IN ('标准版','团队版','专业版','企业版','旗舰版')),
    "金额(万元)"    INTEGER,
    结果            TEXT CHECK(结果 IN ('赢单','输单')),
    原因分类        TEXT,
    原因说明        TEXT,
    竞争对手        TEXT,
    负责人          TEXT,
    台账关联        TEXT
)
""",
    "月度业绩": """
CREATE TABLE IF NOT EXISTS 月度业绩 (
    月份            TEXT PRIMARY KEY,
    新增线索        INTEGER,
    MQL             INTEGER,
    SQL             INTEGER,
    商机数          INTEGER,
    赢单数          INTEGER,
    输单数          INTEGER,
    "赢单金额(万元)" INTEGER,
    赢单率          TEXT,
    销售人数        INTEGER,
    "平均客单价(万元)" REAL
)
""",
    "产品": """
CREATE TABLE IF NOT EXISTS 产品 (
    产品线          TEXT CHECK(产品线 IN ('智能客服平台','工单系统','企业知识库')),
    版本            TEXT CHECK(版本 IN ('标准版','团队版','专业版','企业版','旗舰版')),
    目标客户        TEXT,
    "标牌价(万元/年)" INTEGER,
    "年成本(万元)"   REAL,
    毛利率          TEXT,
    主要竞品        TEXT,
    一句话卖点      TEXT,
    PRIMARY KEY (产品线, 版本)
)
""",
}

#: 业务表 → (源 xlsx 文件, sheet 名)。派生表与规则表**不在**这里（见 design D2/D7）。
SOURCES: Dict[str, tuple] = {
    "线索": ("客户线索台账.xlsx", "线索总表"),
    "市场活动": ("市场活动效果.xlsx", "市场活动明细"),
    "竞品": ("竞品追踪台账.xlsx", "竞品档案"),
    "竞品动态": ("竞品追踪台账.xlsx", "动态时间线"),
    "输赢单": ("输赢单分析表.xlsx", "输赢单明细"),
    "月度业绩": ("销售业绩月度表.xlsx", "月度汇总"),
    "产品": ("产品与报价表.xlsx", "产品目录"),
}


def drop_all(conn: sqlite3.Connection) -> None:
    """删除全部业务表（供"全量重建"使用，使重建可重复执行）。"""
    for table in DDL_STATEMENTS:
        conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.commit()


def create_all(conn: sqlite3.Connection) -> None:
    """建全部业务表（幂等）。"""
    for ddl in DDL_STATEMENTS.values():
        conn.execute(ddl)
    conn.commit()


def table_names(conn: sqlite3.Connection) -> List[str]:
    """当前库中的用户表名（排除 SQLite 内部表）。"""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return [row[0] for row in rows]


def columns_of(conn: sqlite3.Connection, table: str) -> List[str]:
    """表的真实列清单（写操作的白名单依据）。"""
    try:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.Error:
        return []
    return [row[1] for row in rows]
