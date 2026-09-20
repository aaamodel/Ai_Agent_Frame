# -*- coding: utf-8 -*-
"""技能文档结构化事实提取的单测。

针对的病根（2026-09 实测）：技能文档整篇交给模型做摘要，摘要把「数据资产地图」
里逐字写明的路径全丢了，重规划只能盲猜目录，连续两次扫 /data 均为空。
提取器必须**原样保留**这些路径，且不产生任何模型调用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.skill.asset_map import (  # noqa: E402
    extract_asset_facts,
    merge_facts,
    render_facts_for_prompt,
)

# 取自真实技能文档的结构（含中文全角括号与反引号包裹的路径）
SKILL_MD = """# 公司销售情报与销售分析助手

## 数据资产地图（先看这里）

| 资产 | 位置 | 用途(调用工具）                                             |
|------|------|------------------------------------------------------|
| 客户线索台账.xlsx | `raw_data/sales_intel/客户线索台账.xlsx` | 线索查询（sales_sql_query）/ 单格更新（sales_sql_write，人工审批） |
| 产品与报价表.xlsx | `raw_data/sales_intel/产品与报价表.xlsx` | 产品的报价与毛利查询（sales_sql_query）                         |
| 销售业绩月度表.xlsx | `raw_data/sales_intel/销售业绩月度表.xlsx` | 业绩分析（sales_sql_query）                               |

注意：数据基准目录是项目根（本表路径均为相对项目根的路径）。
"""


# ---------------------------------------------------------------------------
# 3.2 确定性提取
# ---------------------------------------------------------------------------
def test_extracts_all_assets_verbatim():
    facts = extract_asset_facts(SKILL_MD)
    assert len(facts) == 3

    by_name = {fact.name: fact for fact in facts}
    assert by_name["客户线索台账.xlsx"].location == "raw_data/sales_intel/客户线索台账.xlsx"
    assert by_name["产品与报价表.xlsx"].location == "raw_data/sales_intel/产品与报价表.xlsx"
    assert by_name["销售业绩月度表.xlsx"].location == "raw_data/sales_intel/销售业绩月度表.xlsx"


def test_extracts_the_consuming_tool():
    facts = {fact.name: fact for fact in extract_asset_facts(SKILL_MD)}
    assert facts["客户线索台账.xlsx"].tool == "sales_sql_query"
    assert facts["销售业绩月度表.xlsx"].tool == "sales_sql_query"


def test_locations_are_free_of_backticks_and_spaces():
    """位置列常写成 `path`，反引号必须被清掉——否则工具调用会拿到非法路径。"""
    for fact in extract_asset_facts(SKILL_MD):
        assert "`" not in fact.location
        assert " " not in fact.location


# ---------------------------------------------------------------------------
# 3.3 解析不出时回退（不中断）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "只是一段普通说明，没有任何表格。",
        "| 列A | 列B |\n|---|---|\n| 值1 | 值2 |",  # 没有"位置"列
        "| 资产 | 位置 |\n| 客户表.xlsx | 没有斜杠也没有点 |",  # 位置不像路径
        "| 资产 | 位置 |\n|---|---|",  # 只有表头
        "| 坏表 |\n|---|\n",  # 结构不完整
    ],
)
def test_returns_empty_when_structure_absent(text):
    assert extract_asset_facts(text) == []


def test_merge_dedups_and_preserves_existing():
    existing = [{"name": "旧表.xlsx", "location": "raw_data/old.xlsx", "tool": ""}]
    new = extract_asset_facts(SKILL_MD)
    merged = merge_facts(existing, new)

    names = {item["name"] for item in merged}
    assert "旧表.xlsx" in names
    assert len(merged) == 4

    # 重复并入同一批不应增长
    assert len(merge_facts(merged, new)) == 4


def test_merge_tolerates_bad_existing():
    assert merge_facts(None, extract_asset_facts(SKILL_MD)) != []
    assert merge_facts("not-a-list", []) == []


# ---------------------------------------------------------------------------
# 3.4 渲染给后续环节
# ---------------------------------------------------------------------------
def test_render_includes_locations():
    text = render_facts_for_prompt(
        [{"name": "客户线索台账.xlsx", "location": "raw_data/sales_intel/客户线索台账.xlsx", "tool": ""}]
    )
    assert "raw_data/sales_intel/客户线索台账.xlsx" in text


def test_render_empty_when_no_facts():
    assert render_facts_for_prompt([]) == ""
    assert render_facts_for_prompt(None) == ""
    assert render_facts_for_prompt([{"name": "缺位置", "location": ""}]) == ""


def test_render_is_length_capped():
    many = [
        {"name": f"表{i}.xlsx", "location": f"raw_data/sales_intel/表{i}.xlsx", "tool": ""}
        for i in range(100)
    ]
    assert len(render_facts_for_prompt(many, max_chars=200)) <= 200
