# -*- coding: utf-8 -*-
"""指标函数自身的单测（**重要**：指标错了，整套 eval 都是废纸）。

覆盖点（每一个容易出错的边界）：
    - Recall@5 的 0/1 退化、k 越界、空期望、重复检索去重；
    - MRR 的名次倒数、未命中、空期望；
    - accuracy 的长度不一致必须报错（不静默对齐）；
    - percentile 与 numpy 的 linear 口径一致（手算基准值）；
    - token_cost 输入/输出分开计价；
    - key_arg_matches 的别名、数值归一化、包含判定；
    - tool_call_success 的 acceptable_tools 与 inspect_first_n；
    - evaluate_gate 的 min/max、缺失键跳过语义。

运行：``pytest evals/test_metrics.py -q``
"""

from __future__ import annotations

import pytest

from evals.metrics import (
    MIN_SAMPLES_BOUNDARY,
    MIN_SAMPLES_PERCENTILE,
    accuracy,
    aggregate_intent_results,
    aggregate_rag_results,
    aggregate_tool_results,
    confusion_counts,
    evaluate_gate,
    format_metric,
    hit_at_k,
    key_arg_matches,
    key_arg_recall,
    mean,
    mrr,
    normalize_value,
    per_class_metrics,
    percentile,
    recall_at_k,
    sample_gate,
    token_cost,
    tool_call_success,
    tool_call_success_rate,
)


# =====================================================================
# Recall@k
# =====================================================================
def test_recall_at_k_basic():
    """手册示例：命中 / 未命中 / 超出 k 不算命中。"""
    assert recall_at_k(["a", "b", "c"], {"b"}, k=5) == 1.0
    assert recall_at_k(["a", "b", "c"], {"z"}, k=5) == 0.0
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0  # 第 3 位超出 k=2


def test_recall_at_k_partial_and_multi_expected():
    """多期望标注：按命中比例给分（这是 Recall 与 Hit-rate 的本质区别）。"""
    assert recall_at_k(["a", "b", "c", "d"], {"a", "c", "z"}, k=5) == pytest.approx(2 / 3)
    assert recall_at_k(["a", "b", "c", "d"], {"a", "b"}, k=2) == 1.0


def test_recall_at_k_duplicate_retrieval_not_double_counted():
    """检索结果重复命中同一文档，不应把召回刷到 > 1。"""
    assert recall_at_k(["a", "a", "a"], {"a", "z"}, k=5) == pytest.approx(0.5)


def test_recall_at_k_edge_cases():
    assert recall_at_k([], {"a"}, k=5) == 0.0            # 检索为空
    assert recall_at_k(["a"], set(), k=5) == 0.0         # 期望为空 -> 不判定
    assert recall_at_k(["a"], {"a"}, k=0) == 0.0         # k=0 不检索
    assert recall_at_k(["a"], {"a"}, k=-1) == 0.0
    # str/int 混用时统一转字符串比对
    assert recall_at_k([1, 2, 3], {"2"}, k=3) == 1.0


def test_hit_at_k():
    assert hit_at_k(["a", "b"], {"b", "z"}, k=5) == 1.0
    assert hit_at_k(["a", "b"], {"z"}, k=5) == 0.0
    assert hit_at_k(["a", "b"], {"b"}, k=1) == 0.0
    assert hit_at_k([], {"a"}) == 0.0


# =====================================================================
# MRR
# =====================================================================
def test_mrr():
    """手册示例 + 多种名次。"""
    assert mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert mrr(["a", "b", "c"], {"a"}) == 1.0
    assert mrr(["a", "b", "c"], {"b"}) == pytest.approx(0.5)
    assert mrr(["a", "b", "c"], {"z"}) == 0.0
    assert mrr([], {"a"}) == 0.0
    assert mrr(["a"], set()) == 0.0


# =====================================================================
# 分类指标
# =====================================================================
def test_accuracy_basic():
    assert accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert accuracy(["a", "b"], ["a", "x"]) == 0.5
    assert accuracy([], []) == 0.0


def test_accuracy_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])


def test_confusion_counts_and_per_class():
    golds = ["a", "a", "b", "b"]
    preds = ["a", "b", "b", "b"]
    matrix = confusion_counts(golds, preds)
    assert matrix["a"]["a"] == 1
    assert matrix["a"]["b"] == 1
    assert matrix["b"]["b"] == 2

    metrics_a = per_class_metrics(golds, preds, "a")
    assert metrics_a["precision"] == pytest.approx(1.0)
    assert metrics_a["recall"] == pytest.approx(0.5)
    assert metrics_a["f1"] == pytest.approx(2 / 3)
    assert metrics_a["support"] == 2.0


def test_per_class_metrics_no_positive_returns_zero():
    """类别从未出现 -> 全部 0.0，不抛 ZeroDivisionError。"""
    assert per_class_metrics(["a"], ["a"], "zzz")["f1"] == 0.0


# =====================================================================
# 统计
# =====================================================================
def test_mean():
    assert mean([1, 2, 3]) == 2.0
    assert mean([]) == 0.0


def test_percentile_matches_numpy_linear_convention():
    """手算基准：numpy.percentile([1..10], 95) == 9.55（linear 插值）。"""
    data = list(range(1, 11))
    assert percentile(data, 95) == pytest.approx(9.55)
    assert percentile(data, 50) == pytest.approx(5.5)
    assert percentile(data, 0) == pytest.approx(1.0)
    assert percentile(data, 100) == pytest.approx(10.0)
    # 单元素 / 空
    assert percentile([7], 95) == 7.0
    assert percentile([], 95) == 0.0
    # p 越界被截断
    assert percentile(data, 200) == pytest.approx(10.0)
    assert percentile(data, -10) == pytest.approx(1.0)


def test_percentile_single_value_repeated():
    assert percentile([2.0, 2.0, 2.0], 99) == pytest.approx(2.0)


def test_token_cost():
    # 1000 输入 @¥0.002/1k + 500 输出 @¥0.006/1k
    assert token_cost(1000, 500, 0.002, 0.006) == pytest.approx(0.002 + 0.003)
    # 负数被截断为 0
    assert token_cost(-10, 0, 1.0, 1.0) == 0.0
    assert token_cost(0, 0, 1.0, 1.0) == 0.0


def test_normalize_value():
    """normalize_value 的归一化规则。"""
    assert normalize_value(None) == ""
    assert normalize_value(True) == "true"
    assert normalize_value(5) == "5"
    assert normalize_value(5.0) == "5"
    assert normalize_value(5.50) == "5.5"
    assert normalize_value("  ABC ") == "abc"
    # 浮点误差不应导致 "45.000000000000006" 这种脏值进入比对
    assert normalize_value(0.1 + 0.2) == "0.3"


# =====================================================================
# 关键参数命中
# =====================================================================
def test_key_arg_matches_exact_and_alias():
    expected = {
        "file_path": {"value": "raw_data/sales_intel/客户线索台账.xlsx",
                      "aliases": ["path", "excel_path"]},
        "sheet_name": "线索总表",
    }
    predicted_hit = {
        "path": "raw_data/sales_intel/客户线索台账.xlsx",
        "sheet_name": "线索总表",
    }
    assert key_arg_matches(predicted_hit, expected) == {
        "file_path": True, "sheet_name": True,
    }

    predicted_miss = {"file_path": "其他表.xlsx", "sheet_name": "字段字典"}
    assert key_arg_matches(predicted_miss, expected) == {
        "file_path": False, "sheet_name": False,
    }


def test_key_arg_matches_substring_and_normalization():
    """长查询词包含短标注词 -> 视为命中；数值 45 与 45.0 等价。"""
    expected = {"query": "专业版报价", "count": 5}
    predicted = {"query": "智能客服平台专业版报价是多少", "count": 5.0}
    assert key_arg_matches(predicted, expected) == {"query": True, "count": True}


def test_key_arg_matches_empty_expected_returns_empty():
    assert key_arg_matches({"a": 1}, {}) == {}
    assert key_arg_matches({}, {"a": 1}) == {"a": False}
    assert key_arg_recall({"a": 1}, {}) is None
    assert key_arg_recall({"a": 1, "b": 2}, {"a": 1, "b": 9}) == pytest.approx(0.5)


# =====================================================================
# 工具调用成功判定
# =====================================================================
def test_tool_call_success():
    assert tool_call_success(["rag_knowledge_search"], "rag_knowledge_search") is True
    assert tool_call_success(["web_search"], "rag_knowledge_search") is False
    # acceptable_tools 兜底
    assert tool_call_success(
        ["file_list_tool"], "local_excel_read_tool",
        acceptable_tools=["file_list_tool"],
    ) is True
    # inspect_first_n 只看前 N 次：第 3 次才调对，只看前 2 次 -> 失败
    assert tool_call_success(
        ["web_search", "web_search", "rag_knowledge_search"],
        "rag_knowledge_search", inspect_first_n=2,
    ) is False
    assert tool_call_success([], "web_search") is False


def test_tool_call_success_rate():
    cases = [
        {"expected_tool": "web_search", "called_tools": ["web_search"]},
        {"expected_tool": "web_search", "called_tools": ["rag_knowledge_search"]},
        {"expected_tool": "file_read_tool", "called_tools": ["file_list_tool"],
         "acceptable_tools": ["file_list_tool"]},
    ]
    assert tool_call_success_rate(cases) == pytest.approx(2 / 3)
    assert tool_call_success_rate([]) == 0.0


# =====================================================================
# 结果聚合
# =====================================================================
def test_aggregate_intent_results():
    records = [
        {"gold": "knowledge-hr", "pred": "knowledge-hr", "latency_ms": 100,
         "input_tokens": 800, "output_tokens": 50},
        {"gold": "sys-welcome", "pred": "sys-about-bot", "latency_ms": 300,
         "input_tokens": 200, "output_tokens": 20, "boundary": True},
    ]
    result = aggregate_intent_results(records)
    assert result["total"] == 2
    assert result["intent_accuracy"] == pytest.approx(0.5)
    assert result["boundary_total"] == 1
    assert result["boundary_scored"] == 1
    assert result["boundary_accuracy"] == 0.0
    assert result["mean_latency_ms"] == pytest.approx(200.0)
    assert result["input_tokens"] == 1000


def test_aggregate_intent_results_no_boundary_is_na():
    """C1：--limit 切片漏掉全部边界样本时，boundary_accuracy=None（未采集），不是 0。"""
    records = [
        {"gold": "knowledge-hr", "pred": "knowledge-hr", "latency_ms": 100},
        {"gold": "sys-welcome", "pred": "sys-welcome", "latency_ms": 200},
    ]
    result = aggregate_intent_results(records)
    assert result["boundary_total"] == 0
    assert result["boundary_scored"] == 0
    assert result["boundary_accuracy"] is None
    # 有意图样本时意图准确率照常计算
    assert result["intent_accuracy"] == 1.0


def test_aggregate_intent_results_empty_is_na():
    """C1：一条意图样本都没采集到时，intent_accuracy 也是 None，而不是 0 分。"""
    result = aggregate_intent_results([])
    assert result["total"] == 0
    assert result["intent_scored"] == 0
    assert result["intent_accuracy"] is None
    assert result["boundary_accuracy"] is None


def test_aggregate_rag_results_prefers_doc_id_then_name():
    records = [
        # doc_id 已回填且命中
        {"retrieved_ids": ["n1", "n2"], "expected_ids": ["n2"],
         "retrieved_doc_names": [], "expected_doc_names": []},
        # doc_id 未回填（空），靠文件名兜底命中
        {"retrieved_ids": [], "expected_ids": [],
         "retrieved_doc_names": ["公司产品知识库.md"],
         "expected_doc_names": ["公司产品知识库.md"]},
    ]
    result = aggregate_rag_results(records, k=5)
    assert result["recall@5"] == 1.0
    assert result["hit@5"] == 1.0
    assert result["mrr"] == pytest.approx((0.5 + 1.0) / 2)


def test_aggregate_tool_results():
    records = [
        {"expected_tool": "rag_knowledge_search", "called_tools": ["rag_knowledge_search"],
         "arg_matches": {"query": True}, "latency_ms": 1000},
        {"expected_tool": "web_search", "called_tools": ["web_search"],
         "arg_matches": {"query": False}, "latency_ms": 3000, "error": "timeout"},
    ]
    result = aggregate_tool_results(records)
    assert result["tool_success_rate"] == 1.0
    assert result["key_arg_recall"] == pytest.approx(0.5)
    assert result["error_rate"] == pytest.approx(0.5)
    assert result["p95_latency_ms"] == pytest.approx(2900.0)


# =====================================================================
# 最小样本量守卫（"样本不足"而不是"分数低"）
# =====================================================================
def test_sample_gate():
    assert sample_gate(3, 20) == {"n": 3, "required": 20}
    assert sample_gate(20, 20) is None      # 恰好达标
    assert sample_gate(21, 20) is None


def test_p95_flagged_insufficient_when_n_below_min():
    """n=3 时 P95 的插值位置 ≈ 1.9（几乎等于最大值），不可进质量门。

    数值仍然保留给人看（"不编数字"原则：不用估算替代，也不隐藏）。
    """
    records = [{"latency_ms": 100}, {"latency_ms": 200}, {"latency_ms": 9000}]
    result = aggregate_tool_results(
        [{"expected_tool": "web_search", "called_tools": [], "latency_ms": r["latency_ms"]}
         for r in records]
    )
    assert result["p95_latency_ms"] == pytest.approx(percentile([100, 200, 9000], 95))
    assert result["sample_insufficient"]["p95_latency_ms"] == {
        "n": 3, "required": MIN_SAMPLES_PERCENTILE,
    }


def test_p95_not_flagged_when_n_enough():
    records = [
        {"expected_tool": "web_search", "called_tools": ["web_search"], "latency_ms": i}
        for i in range(1, MIN_SAMPLES_PERCENTILE + 1)
    ]
    result = aggregate_tool_results(records)
    assert result["sample_insufficient"] == {}


def test_boundary_insufficient_between_one_and_five():
    """0 条 -> None（未采集）；1~4 条 -> 有值但标记不足；>=5 条 -> 正常判定。"""
    def _recs(n: int, correct: bool):
        return [
            {"gold": "knowledge-hr", "pred": "knowledge-hr" if correct else "web-live-info",
             "latency_ms": 100, "boundary": True}
            for _ in range(n)
        ]

    assert aggregate_intent_results(_recs(0, True))["boundary_accuracy"] is None

    partial = aggregate_intent_results(_recs(3, True))
    assert partial["boundary_accuracy"] == 1.0
    assert partial["sample_insufficient"]["boundary_accuracy"] == {
        "n": 3, "required": MIN_SAMPLES_BOUNDARY,
    }

    full = aggregate_intent_results(_recs(MIN_SAMPLES_BOUNDARY, True))
    assert full["boundary_accuracy"] == 1.0
    assert full["sample_insufficient"].get("boundary_accuracy") is None


def test_format_metric_handles_none_and_numbers():
    assert format_metric(None) == "不可用"
    assert format_metric(0.5) == "0.500"
    assert format_metric(13583.6667, ".0f") == "13584"
    # 非数值兜底，避免 print 阶段把整个 runner 打挂
    assert format_metric("abc") == "abc"


# =====================================================================
# 质量门
# =====================================================================
def test_evaluate_gate_pass_and_fail():
    thresholds = {
        "intent": {"accuracy_min": 0.85},
        "rag": {"recall_at_5_min": 0.75, "mrr_min": 0.60},
        "tool": {"call_success_min": 0.80},
        "cost": {"avg_tokens_per_turn_max": 4000, "p95_latency_ms_max": 8000},
    }

    ok_results = {
        "intent_accuracy": 0.93, "recall@5": 0.85, "mrr": 0.72,
        "tool_success_rate": 0.87, "avg_tokens_per_turn": 3200, "p95_latency_ms": 6100,
    }
    assert evaluate_gate(ok_results, thresholds) == []

    bad_results = dict(ok_results)
    bad_results["intent_accuracy"] = 0.80
    bad_results["p95_latency_ms"] = 9000
    violations = evaluate_gate(bad_results, thresholds)
    assert len(violations) == 2
    assert any("intent.accuracy_min" in v for v in violations)
    assert any("cost.p95_latency_ms_max" in v for v in violations)


def test_evaluate_gate_missing_keys_skipped():
    """指标没跑出来（键缺失）时跳过，不误报红。"""
    thresholds = {"intent": {"accuracy_min": 0.85}, "rag": {"recall_at_5_min": 0.75}}
    assert evaluate_gate({"intent_accuracy": 0.9}, thresholds) == []
    # 阈值未配置该键 -> 也跳过
    assert evaluate_gate({"recall@5": 0.1}, {"rag": {}}) == []


def test_evaluate_gate_none_values_skipped():
    """C1：指标值为 None（零分母/未采集）时跳过，不能 float(None) 报错也不能判违规。"""
    thresholds = {"intent": {"accuracy_min": 0.85, "boundary_accuracy_min": 0.6}}
    results = {"intent_accuracy": None, "boundary_accuracy": None}
    assert evaluate_gate(results, thresholds) == []
