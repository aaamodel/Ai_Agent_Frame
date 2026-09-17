# -*- coding: utf-8 -*-
"""answer_quality 判分器的单测。

测试重点是**那些检索指标抓不到、只有判生成质量才能发现的错误**：
- 召回对了但答错数字（R01 真实场景）
- 顺序敏感题答对了要素但顺序错
- 混淆邻近条目（命中干扰项）
"""

from __future__ import annotations

import pytest

from evals.answer_quality import (
    aggregate_answer_results,
    check_order,
    detect_abstain,
    detect_distractors,
    evaluate_answer,
    extract_numbers,
    fact_hit,
    fact_hit_detail,
    fact_hit_rate,
    normalize_answer,
)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
def test_normalize_strips_fullwidth_and_space():
    # 全角数字 + 空格 + 「万元」都应被统一
    assert normalize_answer("４５万元") == "45万"
    assert normalize_answer(" 专业版\n45万\t") == "专业版45万"


def test_normalize_treats_punctuation_as_separator():
    """分隔符类标点会被消除，让不同写法对齐。

    这是 R12/R19 能通过自检的关键：标注写「A=官方发布」「中恒/工单客服联动」，
    要点写「A 官方发布」「中恒 工单客服联动」，归一化后一致。
    """
    assert normalize_answer("A=官方发布可直接引用") == normalize_answer("A 官方发布可直接引用")
    assert normalize_answer("中恒/工单客服联动") == normalize_answer("中恒 工单客服联动")
    assert normalize_answer("45 万 / 年") == "45万年"


def test_normalize_keeps_semantic_symbols():
    """承载语义的符号必须保留，否则「≤200 席」「10~12 分」会失去区分度。"""
    assert "%" in normalize_answer("达成率 73.1%")
    assert "~" in normalize_answer("10~12 分")
    assert "≤" in normalize_answer("≤200 席")


def test_extract_numbers_does_not_glue_across_punctuation():
    """数字提取不能跨标点粘连 —— 否则 6/7/8 月会变成 678，要点永远匹配不上。"""
    nums = extract_numbers("6/7/8 月 23.1%/18.2%/20.0%")
    assert "6" in nums and "7" in nums and "8" in nums
    assert "678" not in nums


def test_normalize_handles_none():
    assert normalize_answer(None) == ""
    assert normalize_answer(123) == "123"


def test_extract_numbers_dedup_keeps_order():
    assert extract_numbers("目标 130 万，实际 95 万，95 万") == ["130", "95"]


# ---------------------------------------------------------------------------
# 要点命中
# ---------------------------------------------------------------------------
def test_fact_hit_exact():
    assert fact_hit("专业版价格是 45 万/年", "专业版 45 万/年") is True


def test_fact_hit_numeric_fallback_for_writing_variants():
    """「45万一年」与「45 万/年」写法不同但数字一致 —— 这是数字兜底存在的理由。"""
    assert fact_hit("专业版 45万一年", "专业版 45 万/年") is True
    mode = fact_hit_detail("专业版 45万一年", "专业版 45 万/年")[1]
    assert mode == "numeric"


def test_fact_hit_supports_aliases():
    """中文同义表述多，要点应支持别名：任一命中即算命中。

    没有这个能力时，「含质检 / 与质检 / 支持质检」只有第一种写法能命中，
    会把正确答案判成未命中，导致生成质量被系统性低估。
    """
    fact = ["含质检", "与质检", "支持质检"]
    assert fact_hit("并支持质检", fact) is True
    assert fact_hit("完全没有提到", fact) is False
    assert fact_hit_detail("与质检", fact) == (True, "exact")


def test_fact_hit_rejects_wrong_number():
    """答成另一个版本的价��，要点必须不命中。"""
    assert fact_hit("旗舰版 90 万/年", "专业版 45 万/年") is False


def test_fact_hit_detail_modes():
    assert fact_hit_detail("含 RAG 知识库", "含 RAG 知识库") == (True, "exact")
    assert fact_hit_detail("完全无关的回答", "含 RAG 知识库") == (False, "miss")


def test_fact_hit_rate_none_when_no_facts():
    """没有标注时返回 None，不能记 0 分 —— 那是两种完全不同的情况。"""
    assert fact_hit_rate("任意答案", []) is None


def test_fact_hit_rate_basic():
    facts = ["价格 45 万", "含质检"]
    assert fact_hit_rate("价格 45 万，含质检", facts) == 1.0
    assert fact_hit_rate("价格 45 万", facts) == 0.5


# ---------------------------------------------------------------------------
# 顺序 / 干扰项
# ---------------------------------------------------------------------------
def test_check_order_ok_and_violation():
    facts = ["线索", "MQL", "SQL", "商机"]
    assert check_order("线索→MQL→SQL→商机", facts) is True
    assert check_order("商机→SQL→MQL→线索", facts) is False


def test_check_order_ignores_missing_facts():
    """缺项不参与次序比较，缺项由 min_facts 判罚，职责不重叠。"""
    facts = ["线索", "MQL", "绝不存在的要点"]
    assert check_order("线索→MQL", facts) is True


def test_detect_distractors_flags_wrong_version():
    """R01 真实风险：召回到产品知识库，但答成标准版/旗舰版的价。"""
    hit = detect_distractors("标准版 15 万，旗舰版 90 万起", ["15 万", "90 万"])
    assert hit == ["15 万", "90 万"]


def test_detect_distractors_empty_when_clean():
    assert detect_distractors("专业版 45 万", ["15 万", "90 万"]) == []


# ---------------------------------------------------------------------------
# 综合判定
# ---------------------------------------------------------------------------
def test_evaluate_answer_passes_on_good_answer():
    """完整答案应全要点命中。要点用别名形式，覆盖「含质检/与质检」这类写法差异。"""
    result = evaluate_answer(
        "专业版 45 万/年，支持 ≤200 席，包含 RAG 知识库与质检。",
        ["专业版 45 万/年", ["≤200 席", "200 席"], ["RAG 知识库", "知识库"], ["质检"]],
        {"min_facts": 2, "number_strict": True, "distractors": ["15 万", "90 万"]},
    )
    assert result["passed"] is True
    assert result["hit_count"] == 4
    assert result["distractors_hit"] == []


def test_evaluate_answer_fails_on_wrong_number_even_if_retrieval_ok():
    """核心用例：召回正确、但数字答错 —— Recall@5 全绿，本判分必须红。"""
    result = evaluate_answer(
        "标准版 15 万一年，旗舰版 90 万起。",
        ["专业版 45 万/年", "≤200 席"],
        {"min_facts": 2, "number_strict": True, "distractors": ["15 万", "90 万"]},
    )
    assert result["passed"] is False
    assert result["hit_count"] == 0
    assert result["distractors_hit"] == ["15 万", "90 万"]
    assert "干扰项" in result["reason"]


def test_evaluate_answer_fails_when_below_min_facts():
    result = evaluate_answer("只答了一个要点：含质检", ["价格 45 万", "含质检"], {"min_facts": 2})
    assert result["passed"] is False
    assert result["hit_count"] == 1
    assert "min_facts" in result["reason"]
    assert result["missing_facts"] == ["价格 45 万"]


def test_evaluate_answer_order_sensitive():
    judge = {"min_facts": 3, "order_sensitive": True}
    facts = ["线索", "MQL", "SQL", "商机"]
    assert evaluate_answer("线索→MQL→SQL→商机", facts, judge)["passed"] is True
    bad = evaluate_answer("商机→SQL→MQL→线索", facts, judge)
    assert bad["passed"] is False
    assert "顺序" in bad["reason"]


def test_evaluate_answer_no_facts_is_not_a_pass():
    """无标注不能算通过，否则质量门形同虚设。"""
    result = evaluate_answer("随便答点什么", [], {"min_facts": 1})
    assert result["passed"] is False
    assert "无法判分" in result["reason"]


def test_evaluate_answer_tolerates_missing_judge():
    result = evaluate_answer("价格 45 万", ["价格 45 万"])
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def test_aggregate_answer_results_basic():
    records = [
        {"case_id": "R01", "passed": True, "hit_rate": 1.0, "distractors_hit": []},
        {"case_id": "R02", "passed": False, "hit_rate": 0.5, "distractors_hit": ["15 万"]},
    ]
    agg = aggregate_answer_results(records)
    assert agg["sample_count"] == 2
    assert agg["pass_rate"] == 0.5
    assert agg["avg_fact_hit_rate"] == pytest.approx(0.75)
    assert agg["distractor_hit_rate"] == 0.5
    assert agg["failed_cases"] == ["R02"]


# ---------------------------------------------------------------------------
# 不可答样本 / 拒答（幻觉治理）
# ---------------------------------------------------------------------------
def test_detect_abstain():
    assert detect_abstain("抱歉，知识库中没有找到相关信息") is True
    assert detect_abstain("这个问题我无法回答") is True
    assert detect_abstain("专业版 45 万一年") is False


def test_evaluate_answer_must_abstain_passes_when_abstaining():
    """不可答样本：说「不知道」才是正确答案。"""
    result = evaluate_answer("抱歉，知识库中没有相关信息，无法回答。", [], {"must_abstain": True})
    assert result["must_abstain"] is True
    assert result["abstained"] is True
    assert result["passed"] is True


def test_evaluate_answer_must_abstain_fails_on_hallucination():
    """不可答样本：编一个具体数字 = 幻觉，必须判失败。

    这是不可答样本存在的全部意义 —— 24 条「一定能答上来」的题测不出这个。
    """
    result = evaluate_answer("目标 130 万，实际 95 万。", [], {"must_abstain": True})
    assert result["passed"] is False
    assert "幻觉" in result["reason"]


def test_aggregate_separates_abstain_from_answerable():
    records = [
        {"case_id": "R01", "passed": True, "hit_rate": 1.0, "must_abstain": False},
        {"case_id": "R25", "passed": True, "must_abstain": True, "abstained": True},
        {"case_id": "R26", "passed": False, "must_abstain": True, "abstained": False},
    ]
    agg = aggregate_answer_results(records)
    assert agg["answerable_count"] == 1
    # 可答样本的命中率不应被不可答样本污染（后者 hit_rate 为 None）
    assert agg["avg_fact_hit_rate"] == 1.0
    assert agg["abstain_count"] == 2
    assert agg["abstain_success_rate"] == 0.5


def test_aggregate_answer_results_skips_ungraded():
    """没有 passed 字段的记录应被跳过，而不是当 0 分。"""
    agg = aggregate_answer_results([{"case_id": "R99"}])
    assert agg["sample_count"] == 0
    assert agg["pass_rate"] is None


def test_aggregate_answer_results_empty():
    agg = aggregate_answer_results([])
    assert agg["sample_count"] == 0
    assert "无可用判分结果" in agg["note"]
