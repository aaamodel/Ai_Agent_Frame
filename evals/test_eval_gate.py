# -*- coding: utf-8 -*-
"""质量门与黄金集校验（CI 的"红线"就在这里）。

这个文件在 CI 里跑两件事：

1. **数据校验**（永远可跑，不依赖 Milvus/Redis/模型 API）
   —— 黄金集的字段完整性、ID 唯一性、取值是否落在**真实枚举**内。
   价值：别人改了一个意图叶子名或工具名，这里会直接红，逼他同步更新黄金集，
   避免出现"标注指向一个已经不存在的意图"这种静默腐化。

2. **质量门判定** —— 读 ``evals/_results/latest.json``，跌破 ``thresholds.yaml``
   就失败。文件不存在时 **skip**（过渡期合法状态），而不是 fail。

额外还测了 ``report.py`` 的纯渲染逻辑（用合成结果），这样"报告生成器"本身
也是被测试覆盖的，不依赖任何中间件。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from evals.metrics import evaluate_gate
from evals.report import (
    _collect_misses,
    flatten_metrics,
    pick_failure_case,
    render_report,
)

# =====================================================================
# 冻结的真实枚举（来源见注释），用于校验黄金集没有标注到不存在的目标
# =====================================================================
# app/query_intent/intent_classify_resolver/intent_tree.py::IntentTreeFactory.build_intent_tree
KNOWN_INTENT_LEAF_IDS: frozenset[str] = frozenset(
    {
        "knowledge-hr",
        "knowledge-it",
        "knowledge-finance",
        "knowledge-biz-system",
        "knowledge-entity-relation",
        "knowledge-product",
        "web-live-info",
        "data-sales-report",
        "data-excel-ops",
        "data-bitable-ops",
        "files-locate",
        "files-content",
        "sys-welcome",
        "sys-about-bot",
    }
)

# app/query_intent/rag_constant.py::REGISTERED_ENABLED_TOOL_NAMES
KNOWN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "rag_knowledge_search",
        "knowledge_graph_search",
        "web_search",
        "local_excel_read_tool",
        "local_excel_query_tool",
        "local_excel_write_tool",
        "sales_report_export_tool",
        "feishu_bitable_tool",
        "file_list_tool",
        "file_grep_tool",
        "file_read_tool",
    }
)

# 联网搜索的 Tavily 已从注册表移除（变成 web_search 的内部降级通道），
# 因此不再存在"acceptable_tools 可引用的非注册工具"——acceptable 必须是真的注册工具。


# =====================================================================
# 一、黄金集数据校验
# =====================================================================
@pytest.fixture(scope="module")
def all_cases(intent_cases, rag_cases, tool_cases) -> Dict[str, List[Dict[str, Any]]]:
    return {"intent": intent_cases, "rag": rag_cases, "tool": tool_cases}


def test_golden_sets_have_expected_size(all_cases) -> None:
    """手册要 50 条（15+20+15）；实际允许略多，但不允许少。"""
    assert len(all_cases["intent"]) >= 15, "意图黄金集不足 15 条"
    assert len(all_cases["rag"]) >= 20, "RAG 黄金集不足 20 条"
    assert len(all_cases["tool"]) >= 15, "工具黄金集不足 15 条"


def test_case_ids_are_unique(all_cases) -> None:
    for name, cases in all_cases.items():
        ids: List[str] = [str(c.get("id")) for c in cases]
        assert len(ids) == len(set(ids)), f"{name} 黄金集存在重复 id：{ids}"


def test_intent_cases_schema(intent_cases) -> None:
    for case in intent_cases:
        assert case.get("id"), f"缺 id：{case}"
        assert str(case.get("query", "")).strip(), f"{case['id']} 缺 query"
        expected: str = str(case.get("expected_intent", ""))
        assert expected, f"{case['id']} 缺 expected_intent"
        assert expected in KNOWN_INTENT_LEAF_IDS, (
            f"{case['id']} 的 expected_intent={expected} 不在真实意图叶子枚举内，"
            f"请核对 intent_tree.py"
        )
        assert "boundary" in case, f"{case['id']} 缺 boundary 标记"


def test_intent_has_five_boundary_samples(intent_cases) -> None:
    """手册明确要求 5 条易混淆边界样本。"""
    boundary: List[Dict[str, Any]] = [c for c in intent_cases if c.get("boundary")]
    assert len(boundary) >= 5, f"边界样本只有 {len(boundary)} 条，手册要求 5 条"


def test_rag_cases_schema(rag_cases) -> None:
    for case in rag_cases:
        assert str(case.get("query", "")).strip(), f"{case['id']} 缺 query"
        assert str(case.get("expected_doc_name", "")).strip(), (
            f"{case['id']} 缺 expected_doc_name（RAG 召回判定必须有文档标识）"
        )
        # expected_doc_id 允许为 None（尚未回填），但键必须存在，避免"忘了标注"
        assert "expected_doc_id" in case, f"{case['id']} 缺 expected_doc_id 键"
        doc_id: Any = case.get("expected_doc_id")
        if doc_id is not None:
            assert str(doc_id).startswith("doc-"), (
                f"{case['id']} 的 expected_doc_id={doc_id} 不符合 doc-<md5[:12]> 约定"
            )


def test_tool_cases_schema(tool_cases) -> None:
    for case in tool_cases:
        assert str(case.get("query", "")).strip(), f"{case['id']} 缺 query"
        expected: str = str(case.get("expected_tool", ""))
        assert expected, f"{case['id']} 缺 expected_tool"
        assert expected in KNOWN_TOOL_NAMES, (
            f"{case['id']} 的 expected_tool={expected} 不在注册工具枚举内"
        )
        for alt in case.get("acceptable_tools") or []:
            assert str(alt) in KNOWN_TOOL_NAMES, (
                f"{case['id']} 的 acceptable_tools 含未知工具 {alt}"
            )
        assert isinstance(case.get("key_args") or {}, dict), (
            f"{case['id']} 的 key_args 必须是对象"
        )


def test_tool_cases_expected_tool_reachable_from_intent_tree(tool_cases) -> None:
    """每个期望工具都必须至少被一个意图叶子声明过。

    否则说明黄金集在要求模型调用一个"意图树压根不会放行"的工具——
    这种用例必然失败，属于标注错误而非模型问题。
    """
    # 意图树里被声明过的工具集合（按 intent_tree.py 的 agent_tool_names 汇总）
    declared: frozenset[str] = frozenset(
        {
            "rag_knowledge_search",
            "knowledge_graph_search",
            "web_search",
            "local_excel_read_tool",
            "local_excel_write_tool",
            "sales_report_export_tool",
            "feishu_bitable_tool",
            "file_list_tool",
            "file_grep_tool",
            "file_read_tool",
        }
    )
    for case in tool_cases:
        assert str(case["expected_tool"]) in declared, (
            f"{case['id']} 期望的 {case['expected_tool']} 未被任何意图叶子声明过"
        )


def test_thresholds_schema(thresholds) -> None:
    assert "version" in thresholds, "thresholds.yaml 必须带 version（便于追溯红线版本）"
    for section in ("intent", "rag", "tool", "cost"):
        assert section in thresholds, f"thresholds.yaml 缺 section: {section}"
        assert isinstance(thresholds[section], dict), f"{section} 必须是映射"


# =====================================================================
# 二、质量门判定
# =====================================================================
def test_quality_gate_passes_when_no_violation(thresholds) -> None:
    perfect: Dict[str, float] = {
        "intent_accuracy": 1.0,
        "boundary_accuracy": 1.0,
        "recall@5": 1.0,
        "hit@5": 1.0,
        "mrr": 1.0,
        "tool_success_rate": 1.0,
        "key_arg_recall": 1.0,
        "avg_tokens_per_turn": 1.0,
        "p95_latency_ms": 1.0,
    }
    assert evaluate_gate(perfect, thresholds) == []


def test_quality_gate_detects_recall_regression(thresholds) -> None:
    """核心能力：**能量化证明一个改动变差了**。

    模拟 chunk size 512 -> 1024 之后 Recall@5 从 0.86 掉到 0.58，
    质量门必须明确报出违规（这就是面试里"我证明了一个改动是错的"的机制）。
    """
    before: Dict[str, float] = {"recall@5": 0.86}
    after: Dict[str, float] = {"recall@5": 0.58}

    assert evaluate_gate(before, thresholds) == [], "改善前的基线不该违规"
    violations: List[str] = evaluate_gate(after, thresholds)
    assert len(violations) == 1
    assert "recall_at_5_min" in violations[0]
    assert "recall@5" in violations[0]


def test_quality_gate_skips_uncollected_metrics(thresholds) -> None:
    """指标没采到（如 Milvus 没起）必须跳过，不能判失败。"""
    assert evaluate_gate({}, thresholds) == []
    assert evaluate_gate({"mrr": 0.9}, thresholds) == []


def test_quality_gate_handles_max_direction(thresholds) -> None:
    """cost 段是"越小越好"，方向不能搞反。"""
    limit: float = float(thresholds["cost"]["avg_tokens_per_turn_max"])
    assert evaluate_gate({"avg_tokens_per_turn": limit - 1}, thresholds) == []
    over: List[str] = evaluate_gate({"avg_tokens_per_turn": limit + 1}, thresholds)
    assert len(over) == 1 and ">" in over[0]


def test_eval_gate_pytest_gate(eval_results, thresholds) -> None:
    """读真实评测结果跑一次质量门。

    ``eval_results`` fixture 在结果文件缺失时 skip，所以这个用例在
    "还没跑过真实链路"的 CI 上是跳过而不是失败。
    """
    flat: Dict[str, float] = {}
    for section, metrics in (eval_results.get("sections") or {}).items():
        if isinstance(metrics, dict):
            flat.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    flat.update(
        {k: float(v) for k, v in (eval_results.get("metrics") or {}).items()
         if isinstance(v, (int, float))}
    )
    if not flat:
        pytest.skip("latest.json 中没有任何数值指标（可能只跑了部分评测）")

    violations: List[str] = evaluate_gate(flat, thresholds)
    assert violations == [], "质量门未通过：\n" + "\n".join(violations)


# =====================================================================
# 三、报告生成器（纯函数，无需中间件）
# =====================================================================
def _fake_runs(*, rag_recall_hit: bool = True, tool_ok: bool = True) -> Dict[str, Any]:
    """构造合成结果，用于在不连任何真实服务的前提下测报告渲染。"""
    return {
        "intent": {
            "ok": True,
            "result": {
                "metrics": {
                    "total": 2,
                    "intent_accuracy": 0.5,
                    "boundary_total": 1,
                    "boundary_accuracy": 0.0,
                    "mean_latency_ms": 1200.0,
                    "p95_latency_ms": 1500.0,
                    "llm_usage": {"calls": 4, "input_tokens": 800, "output_tokens": 200,
                                  "total_tokens": 1000},
                },
                "records": [
                    {"id": "I01", "query": "q1", "gold": "knowledge-hr",
                     "pred": "knowledge-hr", "correct": True, "boundary": False,
                     "latency_ms": 900.0},
                    {"id": "B01", "query": "q2", "gold": "data-excel-ops",
                     "pred": "data-sales-report", "correct": False, "boundary": True,
                     "latency_ms": 1500.0},
                ],
            },
        },
        "rag": {
            "ok": True,
            "result": {
                "metrics": {"total": 2, "recall@5": 0.5, "hit@5": 0.5, "mrr": 0.25,
                            "mean_latency_ms": 300.0, "p95_latency_ms": 400.0},
                "records": [
                    {"id": "R01", "query": "报价", "expected_doc_name": "公司产品知识库.md",
                     "expected_doc_names": ["公司产品知识库"],
                     "retrieved_ids": ["doc-aaa"], "expected_ids": ["doc-bbb"],
                     "retrieved_doc_names": ["公司产品知识库"], "latency_ms": 300.0,
                     "top1_doc_name": "公司产品知识库", "top1_score": 0.8,
                     "top1_content_head": "产品报价表…"},
                    {"id": "R02", "query": "输单原因", "expected_doc_name": "销售方法论.md",
                     "expected_doc_names": ["销售方法论"],
                     "retrieved_ids": ["doc-ccc"], "expected_ids": [],
                     "retrieved_doc_names": ["竞品情报.md"], "latency_ms": 300.0,
                     "top1_doc_name": "竞品情报", "top1_score": 0.6,
                     "top1_content_head": "竞品动态…"} if not rag_recall_hit else {
                        "id": "R02", "query": "输单原因", "expected_doc_name": "销售方法论.md",
                        "expected_doc_names": ["销售方法论"],
                        "retrieved_ids": ["doc-ddd"], "expected_ids": [],
                        "retrieved_doc_names": ["销售方法论"], "latency_ms": 300.0,
                        "top1_doc_name": "销售方法论", "top1_score": 0.9,
                        "top1_content_head": "7 类输单原因…"},
                ],
            },
        },
        "tool": {
            "ok": True,
            "result": {
                "metrics": {"total": 2, "tool_success_rate": 0.5 if not tool_ok else 1.0,
                            "key_arg_recall": 0.5 if not tool_ok else 1.0,
                            "key_arg_cases": 2, "error_rate": 0.0,
                            "mean_latency_ms": 9000.0, "p95_latency_ms": 12000.0,
                            "avg_tokens_per_turn": 3500.0,
                            "llm_usage": {"calls": 12, "input_tokens": 5000,
                                          "output_tokens": 2000, "total_tokens": 7000}},
                "records": [
                    {"id": "T01", "query": "差旅报销", "expected_tool": "rag_knowledge_search",
                     "acceptable_tools": [], "called_tools": ["rag_knowledge_search"],
                     "inspect_first_n": 3, "arg_matches": {"query": True},
                     "latency_ms": 6000.0, "mode_used": "react",
                     "allowed_tools": ["rag_knowledge_search"], "tool_not_whitelisted": []},
                    {"id": "T03", "query": "智齿动态", "expected_tool": "web_search",
                     "acceptable_tools": [],
                     "called_tools": [] if not tool_ok else ["web_search"],
                     "inspect_first_n": 3, "arg_matches": {"query": False},
                     "latency_ms": 12000.0, "mode_used": "react",
                     "allowed_tools": ["web_search"], "tool_not_whitelisted": []},
                ],
            },
        },
    }


def test_flatten_metrics_prefers_end_to_end_p95(thresholds) -> None:
    """p95 必须取端到端口径（工具组），而不是意图组的 1500ms。"""
    flat: Dict[str, float] = flatten_metrics(_fake_runs())
    assert flat["p95_latency_ms"] == pytest.approx(12000.0)
    assert flat["recall@5"] == pytest.approx(0.5)
    assert flat["intent_accuracy"] == pytest.approx(0.5)
    assert flat["avg_tokens_per_turn"] == pytest.approx(3500.0)


def test_render_report_flags_violations(thresholds) -> None:
    """合成一个 Recall@5 = 0.5（低于 0.70）的结果，报告必须报违规。"""
    markdown: str = render_report(_fake_runs(), thresholds, generated_at="2026-01-01 00:00:00")
    assert "# Ai_Agent_Frame 评测报告" in markdown
    assert "质量门未通过" in markdown
    assert "recall_at_5_min" in markdown
    assert "阈值版本" in markdown


def test_render_report_survives_all_runners_failed(thresholds) -> None:
    """三组全挂（中间件未起）时仍必须产出报告，并明确写"未采集"。"""
    runs: Dict[str, Any] = {
        "intent": {"ok": False, "error": "ConnectionError: redis down"},
        "rag": {"ok": False, "error": "MilvusException: unreachable"},
        "tool": {"ok": False, "error": "ModuleNotFoundError: langfuse"},
    }
    markdown: str = render_report(runs, thresholds)
    assert markdown.count("未采集") >= 3
    assert "不可用" in markdown


def test_pick_failure_case_prefers_rag_miss(thresholds) -> None:
    """RAG 完全没召回时，失败案例应优先挑 RAG（优先级 1，高于意图边界）。"""
    failure = pick_failure_case(_fake_runs(rag_recall_hit=False))
    assert failure is not None
    assert failure["stage"] == "RAG 检索"
    assert failure["id"] == "R02"


def test_pick_failure_case_falls_back_to_intent(thresholds) -> None:
    """RAG 全命中时，退而挑意图边界样本分错。"""
    failure = pick_failure_case(_fake_runs(rag_recall_hit=True))
    assert failure is not None
    assert "意图分类" in failure["stage"]
    assert failure["id"] == "B01"


def test_pick_failure_case_returns_none_when_all_pass(thresholds) -> None:
    runs = _fake_runs(rag_recall_hit=True, tool_ok=True)
    runs["intent"]["result"]["records"] = [
        {"id": "I01", "query": "q1", "gold": "knowledge-hr", "pred": "knowledge-hr",
         "correct": True, "boundary": False, "latency_ms": 900.0}
    ]
    assert pick_failure_case(runs) is None


def test_collect_misses_uses_tool_acceptable_tools() -> None:
    """acceptable_tools 命中必须算成功（否则合理路径被误判为失败）。"""
    records: List[Dict[str, Any]] = [
        {"id": "T17", "query": "q", "expected_tool": "knowledge_graph_search",
         "acceptable_tools": ["rag_knowledge_search"], "called_tools": ["rag_knowledge_search"],
         "inspect_first_n": 3}
    ]
    assert _collect_misses("tool", records) == []


def test_collect_misses_respects_inspect_first_n() -> None:
    """期望工具出现在第 4 步，但 inspect_first_n=3 时必须判失败。"""
    records: List[Dict[str, Any]] = [
        {"id": "T01", "query": "q", "expected_tool": "rag_knowledge_search",
         "acceptable_tools": [],
         "called_tools": ["file_list_tool", "file_read_tool", "web_search",
                          "rag_knowledge_search"],
         "inspect_first_n": 3}
    ]
    misses: List[Dict[str, Any]] = _collect_misses("tool", records)
    assert len(misses) == 1 and misses[0]["id"] == "T01"
