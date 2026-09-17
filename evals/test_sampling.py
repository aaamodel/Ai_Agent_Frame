# -*- coding: utf-8 -*-
"""分层抽样单测。

覆盖的是「**容易出现偏差的场景**」而不是 Happy Path：
边界样本排在文件末尾会不会被漏掉、应拒答样本会不会被漏掉、
同样的 limit 是否永远得到同一批用例（可被 CI 依赖的前提）。

运行：``pytest evals/test_sampling.py -q``
"""

from __future__ import annotations

import pytest

from evals.sampling import (
    intent_stratum,
    rag_stratum,
    select_cases,
    select_stratified,
    tool_stratum,
)


# =====================================================================
# 分层键
# =====================================================================
def test_intent_stratum_encodes_boundary_prefix():
    assert intent_stratum({"boundary": True, "expected_intent": "knowledge-hr"}) == (
        "boundary|knowledge-hr"
    )
    assert intent_stratum({"boundary": False, "expected_intent": "knowledge-hr"}) == (
        "regular|knowledge-hr"
    )


def test_rag_stratum_isolates_unanswerable():
    assert rag_stratum({"unanswerable": True}) == "unanswerable"
    assert rag_stratum({"expected_doc_name": "报价.md"}) == "doc=报价.md"


def test_tool_stratum_uses_expected_tool():
    assert tool_stratum({"expected_tool": "web_search"}) == "tool=web_search"


# =====================================================================
# 边界行为
# =====================================================================
def test_limit_none_or_over_size_returns_all():
    cases = [{"id": f"I{i}", "expected_intent": f"n{i}", "boundary": False} for i in range(5)]
    assert len(select_cases("intent", cases, None)) == 5
    assert len(select_cases("intent", cases, 0)) == 5
    assert len(select_cases("intent", cases, 99)) == 5


def test_select_cases_unknown_kind_raises():
    with pytest.raises(ValueError):
        select_cases("unknown", [{"id": "X"}], limit=2)


# =====================================================================
# 核心：不会因为"排在文件末尾"而被漏掉
# =====================================================================
def test_boundary_samples_never_dropped():
    """回归：B 类样本在文件末尾，简单切片会把它们整体漏掉。"""
    cases = [
        {"id": "I1", "expected_intent": "a", "boundary": False},
        {"id": "I2", "expected_intent": "a", "boundary": False},
        {"id": "I3", "expected_intent": "b", "boundary": False},
        {"id": "I4", "expected_intent": "b", "boundary": False},
        {"id": "B1", "expected_intent": "c", "boundary": True},
        {"id": "B2", "expected_intent": "c", "boundary": True},
    ]
    for limit in (1, 2, 3, 4, 5):
        picked = select_cases("intent", cases, limit)
        assert len(picked) == min(limit, len(cases))
        assert any(c["boundary"] for c in picked), (
            f"limit={limit} 的抽样里一条边界样本都没有——这正是要修的偏差"
        )


def test_unanswerable_samples_never_dropped():
    cases = [
        {"id": "R1", "expected_doc_name": "报价.md"},
        {"id": "R2", "expected_doc_name": "报价.md"},
        {"id": "R3", "expected_doc_name": "方法论.md"},
        {"id": "R4", "unanswerable": True},
    ]
    for limit in (1, 2, 3):
        picked = select_cases("rag", cases, limit)
        assert any(c.get("unanswerable") for c in picked), (
            f"limit={limit} 漏掉了全部应拒答样本"
        )


def test_tool_sample_spreads_across_tools():
    """同一批 expected_tool 成簇排列时，抽样要跨工具轮转而不是扎堆。"""
    cases = [
        {"id": "T1", "expected_tool": "rag_knowledge_search"},
        {"id": "T2", "expected_tool": "rag_knowledge_search"},
        {"id": "T3", "expected_tool": "web_search"},
        {"id": "T4", "expected_tool": "file_read_tool"},
    ]
    picked = select_cases("tool", cases, 3)
    tools = {c["expected_tool"] for c in picked}
    assert len(tools) == 3, "应轮转到 3 个不同工具，而不是全抽同一个"


# =====================================================================
# 确定性
# =====================================================================
def test_selection_is_deterministic_and_order_preserved():
    cases = [
        {"id": "I1", "expected_intent": "a", "boundary": False},
        {"id": "B1", "expected_intent": "z", "boundary": True},
        {"id": "I2", "expected_intent": "b", "boundary": False},
        {"id": "B2", "expected_intent": "y", "boundary": True},
        {"id": "I3", "expected_intent": "c", "boundary": False},
    ]
    first = [c["id"] for c in select_cases("intent", cases, 3)]
    second = [c["id"] for c in select_cases("intent", cases, 3)]
    assert first == second, "同样的 limit 必须得到同一批用例（否则无法被 CI 依赖）"
    # 结果按文件原始顺序返回
    indices = [next(i for i, c in enumerate(cases) if c["id"] == cid) for cid in first]
    assert indices == sorted(indices)


def test_select_stratified_respects_custom_key():
    """通用 API：自定义分层键 + 保护位。"""
    cases = [{"group": g, "id": i} for i, g in enumerate(["a", "a", "b", "b", "c"])]
    picked = select_stratified(
        cases, 3, key=lambda c: c["group"], protected=(lambda k: k == "c",)
    )
    groups = [c["group"] for c in picked]
    # 保护位保证 c 一定入选，其余在 a/b 间轮转
    assert "c" in groups
    assert len(set(groups)) == 3
