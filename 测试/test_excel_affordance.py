# -*- coding: utf-8 -*-
"""Excel 概览「可聚合能力」披露单测（spec: `agent/excel-query`）。

覆盖：
- 3.1 可聚合判定的三种列型（纯数值 / 纯文本 / **文本形式存放的数字**），
      并断言"判定为可聚合"的列真实调用聚合能跑通（披露口径与聚合口径同源）；
- 3.2 四种组合（有/无分组列 × 有/无数值列）都有明确标注，不留空；
- 3.3 每张表至多一组可直接照抄的示例，且按示例调用聚合成功；
- 3.4 披露只是建议：用合法但未被建议的列名照常执行，不被拒绝。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.tools.builtin.localexcel import LocalExcelReadTool  # noqa: E402

_AMOUNT = "2026Q3 赢单金额(万元,截至08)"

FRAME = pd.DataFrame(
    {
        "区域": ["华东", "华南", "华东", "华南"],
        "产品线": ["A", "A", "B", "B"],
        _AMOUNT: ["53", "40", "12", "7"],  # 文本形式存放的数字（Excel 常见）
        "备注": ["好", "一般", "差", "好"],  # 纯文本
        "赢单数": [2, 1, 3, 4],  # 真·数值 dtype
    }
)
NUMERIC_ONLY = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
TEXT_ONLY = pd.DataFrame({"区域": ["华东", "华南", "华东"], "备注": ["好", "差", "好"]})
NEITHER = pd.DataFrame({"常量": ["x", "x", "x"]})  # 基数=1 不能分组；全文本不能聚合


def _tool() -> LocalExcelReadTool:
    """只测纯渲染/判定方法，用 __new__ 跳过 BaseTool 的初始化。"""
    return object.__new__(LocalExcelReadTool)


def _line(block: str, prefix: str) -> str:
    return next(line for line in block.splitlines() if line.startswith(prefix))


# ---------------------------------------------------------------------------
# 3.1 判定与聚合同源
# ---------------------------------------------------------------------------
def test_aggregatable_judgement_covers_three_column_shapes():
    tool = _tool()
    assert tool._is_aggregatable_column(FRAME, _AMOUNT) is True, "文本形式的数字也应可聚合"
    assert tool._is_aggregatable_column(FRAME, "赢单数") is True
    assert tool._is_aggregatable_column(FRAME, "区域") is False
    assert tool._is_aggregatable_column(FRAME, "备注") is False


def test_disclosed_aggregatable_columns_really_aggregate():
    """同源断言：披露说可聚合的列，真实调用必须跑出结果而非报错。"""
    tool = _tool()
    for column in (_AMOUNT, "赢单数"):
        assert tool._is_aggregatable_column(FRAME, column)
        out = tool._render_aggregate(FRAME, "产品线", column, "sum")
        assert "错误" not in out, f"披露为可聚合的 {column} 真实聚合却失败：{out}"


def test_text_numeric_column_aggregates_with_correct_values():
    tool = _tool()
    out = tool._render_aggregate(FRAME, "产品线", _AMOUNT, "sum")
    assert "93" in out and "19" in out, f"按产品线求和应为 A=93 / B=19：{out}"


# ---------------------------------------------------------------------------
# 3.2 四种组合都要有明确标注
# ---------------------------------------------------------------------------
def test_affordance_lines_are_never_blank():
    tool = _tool()
    for frame in (FRAME, NUMERIC_ONLY, TEXT_ONLY, NEITHER):
        block = tool._render_aggregation_affordances("S", frame)
        assert _line(block, "可用作分组列:").split(":", 1)[1].strip()
        assert _line(block, "可聚合数值列:").split(":", 1)[1].strip()


def test_explicit_none_when_column_kinds_absent():
    tool = _tool()
    neither = tool._render_aggregation_affordances("S", NEITHER)
    assert "可用作分组列: 无" in neither
    assert "可聚合数值列: 无" in neither

    no_group = tool._render_aggregation_affordances("S", NUMERIC_ONLY)
    assert "可用作分组列: 无" in no_group
    assert "可聚合数值列: " in no_group and "可聚合数值列: 无" not in no_group

    no_numeric = tool._render_aggregation_affordances("S", TEXT_ONLY)
    assert "可聚合数值列: 无" in no_numeric
    assert "可用作分组列: 无" not in no_numeric


def test_example_is_omitted_when_not_aggregatable():
    tool = _tool()
    assert "可直接调用" not in tool._render_aggregation_affordances("S", TEXT_ONLY)
    assert "可直接调用" not in tool._render_aggregation_affordances("S", NEITHER)


# ---------------------------------------------------------------------------
# 3.3 示例可直接照抄且数量受控
# ---------------------------------------------------------------------------
def test_example_params_are_single_and_copy_ready():
    tool = _tool()
    block = tool._render_aggregation_affordances("区域产品透视", FRAME)
    examples = [line for line in block.splitlines() if line.startswith("可直接调用:")]
    assert len(examples) == 1, "每张表至多一组示例，穷举会让概览体积失控"

    example = examples[0]
    assert 'sheet_name="区域产品透视"' in example
    assert 'group_by="区域"' in example
    assert f'agg_column="{_AMOUNT}"' in example
    assert 'agg_func="sum"' in example

    # 按示例照抄调用，必须成功且数值正确
    out = tool._render_aggregate(FRAME, "区域", _AMOUNT, "sum")
    assert "错误" not in out
    assert "65" in out and "47" in out, f"按区域求和应为 华东=65 / 华南=47：{out}"


def test_overview_renders_affordances_for_every_sheet():
    tool = _tool()
    overview = tool._render_overview({"区域产品透视": FRAME, "常量表": NEITHER}, head_rows=5)
    assert overview.count("可用作分组列:") == 2
    assert overview.count("可聚合数值列:") == 2
    assert "可直接调用" in overview


# ---------------------------------------------------------------------------
# 3.4 披露只是建议，不改变既有执行语义
# ---------------------------------------------------------------------------
def test_disclosure_is_advisory_only():
    """用合法但未被建议为分组列的列名，照常执行。"""
    tool = _tool()
    block = tool._render_aggregation_affordances("S", FRAME)
    # "赢单数" 是数值 dtype → 不会出现在分组列建议里
    assert "赢单数" not in _line(block, "可用作分组列:")
    # 但拿它当分组键真实调用聚合，照常执行（只是不推荐）
    out = tool._render_aggregate(FRAME, "赢单数", _AMOUNT, "sum")
    assert "错误" not in out
