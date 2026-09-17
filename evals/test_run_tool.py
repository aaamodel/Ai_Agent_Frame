# -*- coding: utf-8 -*-
"""run_tool 记账对账的纯函数单测（不依赖 Milvus/Redis/模型）。

重点覆盖 C2：plan_execute 路径 steps 不落工具调用，但 invoke 旁路有记录时，
必须回填判分序列，而不是把"账本缺口"误判成"没调工具"。
"""

from __future__ import annotations

from evals.metrics import key_arg_matches, tool_call_success
from evals.runners.run_tool import reconcile_tool_calls


def test_reconcile_steps_authoritative():
    """steps 有记录时以 steps 为准（invoke 旁路不重复计入）。"""
    calls = [{"tool": "rag_knowledge_search", "args": {"query": "SLA"}}]
    invocations = [
        {"tool": "rag_knowledge_search", "args": {"query": "SLA"}},
    ]
    judging, called, incomplete, source = reconcile_tool_calls(
        calls, [], invocations
    )
    assert source == "steps"
    assert called == ["rag_knowledge_search"]
    assert incomplete is False
    assert judging[0]["args"] == {"query": "SLA"}


def test_reconcile_bypass_backfill_when_steps_missing():
    """C2 场景（T02）：steps 全空，旁路记录里有 acceptable 工具 → 回填并判 PASS。"""
    invocations = [
        {"tool": "file_read_tool", "args": {"file_path": "a.xlsx"}},
        {"tool": "rag_knowledge_search", "args": {"query": "两个表怎么对账"}},
    ]
    judging, called, incomplete, source = reconcile_tool_calls(
        [], [], invocations
    )
    assert source == "invoke_bypass"
    assert incomplete is True  # 缺口本身仍然暴露
    assert called == ["file_read_tool", "rag_knowledge_search"]
    # 回填序列带真实入参
    assert judging[1]["args"] == {"query": "两个表怎么对账"}
    assert judging[1]["source"] == "invoke_bypass"

    # 成功率：acceptable_tools 命中 rag_knowledge_search → True（修复前误判 False）
    assert tool_call_success(
        called,
        expected_tool="knowledge_graph_search",
        acceptable_tools=[
            "local_excel_read_tool",
            "file_read_tool",
            "rag_knowledge_search",
        ],
    )
    # 关键参数命中率也能判（取第一个命中工具的入参）
    assert key_arg_matches(judging[1]["args"], {"query": "对账"})["query"] is True


def test_reconcile_approval_takes_precedence_over_bypass():
    """只有审批挂起工具名时不触发旁路回填（缺口有合理解释：HITL）。"""
    judging, called, incomplete, source = reconcile_tool_calls(
        [],
        ["sales_report_export_tool"],
        [{"tool": "sales_report_export_tool", "args": {}}],
    )
    assert source == "approval"
    assert called == ["sales_report_export_tool"]
    assert incomplete is False
    assert all("source" not in c for c in judging)


def test_reconcile_nothing_called():
    """三路全空（T01 真实幻觉场景）：如实记为无工具调用，不能假造。"""
    judging, called, incomplete, source = reconcile_tool_calls([], [], [])
    assert source == "none"
    assert called == []
    assert judging == []
    assert incomplete is False
    assert not tool_call_success(called, expected_tool="rag_knowledge_search")


def test_reconcile_dedup_preserves_order():
    """同一工具被多次调用只保留一个名字，顺序按首次出现。"""
    invocations = [
        {"tool": "rag_knowledge_search", "args": {"query": "a"}},
        {"tool": "file_read_tool", "args": {}},
        {"tool": "rag_knowledge_search", "args": {"query": "b"}},
    ]
    _, called, _, source = reconcile_tool_calls([], [], invocations)
    assert source == "invoke_bypass"
    assert called == ["rag_knowledge_search", "file_read_tool"]
