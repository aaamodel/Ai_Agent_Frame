# -*- coding: utf-8 -*-
"""**证据缺口检测专项测试**（任务 7.2）。

这个算法是本变更的关键之一，因此单独成套、可直接看出输入输出。

它比对的是**两个数据集**：

  数据集 A（问题侧核心词）  ← 本轮目标 / 子问题 / 意图树命中路径 / 已提取事实，
                             均缺失时回退到改写后的问题；经分词 + 停用词过滤
  数据集 B（本步返回内容）  ← 本步 `observation`（**只比本步**，不掺历史结论）

判据：A 中的词在 B 中的**字面出现率**，低于阈值即置位。

为什么需要它：本 trace 的典型失败不是"没取到数据"，而是"取到了数据但数据不对"
——Excel 返回了销售流水，而用户问的是行业优先级。仅以"取数失败"为触发条件
覆盖不到这种情形。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.agent import step_correction  # noqa: E402
from app.core.agent.step_correction import (  # noqa: E402
    EVIDENCE_COVERAGE_THRESHOLD,
    build_question_keywords,
    detect_evidence_gap,
    extract_keywords,
)

# ── 数据集 A：取自本次 trace 的真实问题与目标 ────────────────────────────────
TRACE_QUESTION = "我们优先做哪些行业？哪些行业算次优先？"
TRACE_GOAL = "交付行业优先级排序结论"

#: 从问题/目标抽出的核心词（"行业""优先级"是问题真正的落点）
KEYWORDS_A = {"行业", "优先级"}

# ── 数据集 B：两类真实返回 ──────────────────────────────────────────────────
OBS_SALES_RECORDS = (
    "2026年8月销售流水明细：客户A 成交额 12 万元，客户B 成交额 8.4 万元；"
    "本月退货率 3.1%，环比下降 2pct。"
)
OBS_KNOWLEDGE_BASE_MISS = "未匹配到任何高相关性的文档片段。"
OBS_IRRELEVANT_NEWS = "今日体育新闻：某足球比赛 2:1 结束。"
OBS_ON_TOPIC = "行业优先级排序结论：金融行业优先级最高，其次为先进制造与医疗健康行业。"


def _state(**overrides) -> dict:
    base: dict = {
        "user_input": TRACE_QUESTION,
        "intent": {"slots": {"agent_goal": TRACE_GOAL}},
        "extracted_facts": [],
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════
# 7.2 ① B 中不含 A 的核心词 → 置位（销售记录 vs 行业优先级）
# ═══════════════════════════════════════════════════════════════════════════
def test_sales_records_against_industry_priority_triggers_gap():
    """本 trace 的核心场景：拿到了数据，但数据与问题对不上。"""
    gap, coverage = detect_evidence_gap(KEYWORDS_A, OBS_SALES_RECORDS)
    assert gap is True
    assert coverage == 0.0


def test_irrelevant_news_triggers_gap():
    gap, _ = detect_evidence_gap(KEYWORDS_A, OBS_IRRELEVANT_NEWS)
    assert gap is True


def test_updated_but_empty_newsstyle_return_triggers_gap():
    """"知识库无匹配"也属于证据缺口：这一步没有带来问题所需的信息。"""
    gap, _ = detect_evidence_gap(KEYWORDS_A, OBS_KNOWLEDGE_BASE_MISS)
    assert gap is True


# ═══════════════════════════════════════════════════════════════════════════
# 7.2 ② B 含 A 的核心词 → 不置位
# ═══════════════════════════════════════════════════════════════════════════
def test_on_topic_return_does_not_trigger_gap():
    gap, coverage = detect_evidence_gap(KEYWORDS_A, OBS_ON_TOPIC)
    assert gap is False
    assert coverage == 1.0


def test_partial_coverage_above_threshold_does_not_trigger():
    keywords = {"行业", "优先级", "排序", "结论"}
    observation = "行业优先级如下：金融 > 先进制造（排序依据为增速）"
    gap, coverage = detect_evidence_gap(keywords, observation)
    assert coverage == 0.75
    assert gap is False


def test_coverage_exactly_at_threshold_does_not_trigger():
    """阈值是"低于才置位"——等于阈值不置位。"""
    keywords = {"甲", "乙", "丙"}
    observation = "丙"  # 1/3 ≈ 0.333 < 0.3? 否 → 0.333 > 0.3 → 不置位
    gap, coverage = detect_evidence_gap(keywords, observation)
    assert coverage == pytest.approx(1 / 3)
    assert gap is (coverage < EVIDENCE_COVERAGE_THRESHOLD)


# ═══════════════════════════════════════════════════════════════════════════
# 7.2 ③ A 为空 → 不置位（宁可漏触发，不得误伤）
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("keywords", [set(), frozenset(), [], None, ["", None]])
def test_empty_dataset_a_never_triggers(keywords):
    gap, coverage = detect_evidence_gap(keywords, OBS_SALES_RECORDS)
    assert gap is False
    assert coverage == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 7.2 ④ B 为空 → 不置位（纯推理步骤无返回内容，覆盖率无意义）
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("observation", ["", "   ", None, "\n\t"])
def test_empty_dataset_b_never_triggers(observation):
    gap, coverage = detect_evidence_gap(KEYWORDS_A, observation)
    assert gap is False
    assert coverage == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 7.2 ⑤ 覆盖率不掺入历史结论
# ═══════════════════════════════════════════════════════════════════════════
def test_history_must_not_be_mixed_in():
    """这是最容易被写错的一条。

    历史结论往往与问题高度相关（覆盖率很高）。若把它掺进数据集 B，
    "本步返回跑题"就会被历史结论的覆盖率掩盖掉，检测彻底失效。
    """
    history = "历史结论：行业优先级排序已完成，行业维度覆盖金融/制造/医疗；优先级依据为增速。"
    this_step = OBS_IRRELEVANT_NEWS

    # 只比本步 → 正确置位
    assert detect_evidence_gap(KEYWORDS_A, this_step)[0] is True

    # 反证：把历史掺进来就会被掩盖（这正是"不得掺入"的理由）
    assert detect_evidence_gap(KEYWORDS_A, history + this_step)[0] is False


def test_signature_admits_only_one_return_text():
    """从结构上杜绝"把历史结论一起传进来"。

    签名里只有一份「返回内容」入参——想掺历史就必须由调用方自己拼字符串，
    而那正是 `test_history_must_not_be_mixed_in` 反证过的错误做法。
    """
    params = list(inspect.signature(detect_evidence_gap).parameters)
    assert params[:2] == ["question_keywords", "observation"]


# ═══════════════════════════════════════════════════════════════════════════
# §5.1 数据集 A 的抽取：优先级与回退
# ═══════════════════════════════════════════════════════════════════════════
def test_keywords_come_from_goal_first():
    keywords = build_question_keywords(_state())
    assert "行业" in keywords
    assert "优先级" in keywords


def test_keywords_include_sub_questions():
    state = _state(intent={"slots": {
        "agent_goal": TRACE_GOAL,
        "per_sub_questions": ["各组线索转化率对比"],
    }})
    keywords = build_question_keywords(state)
    assert "线索" in keywords or "转化率" in keywords


def test_keywords_include_intent_tree_path():
    state = _state(intent={"slots": {
        "top_kb_node": {"full_path": "企业知识问答 > 销售数据统计"},
    }})
    keywords = build_question_keywords(state)
    assert "销售" in keywords


def test_keywords_include_extracted_fact_names():
    state = _state(intent={"slots": {}},
                   extracted_facts=[{"name": "客户线索台账.xlsx", "location": "raw_data/x.xlsx"}])
    keywords = build_question_keywords(state)
    assert any("客户线索" in item or "台账" in item for item in keywords)


def test_falls_back_to_question_when_all_sources_missing():
    state = {"user_input": TRACE_QUESTION, "intent": {"slots": {}}, "extracted_facts": []}
    keywords = build_question_keywords(state)
    assert "行业" in keywords


def test_returns_empty_when_nothing_available():
    assert build_question_keywords({}) == set()


def test_stopwords_and_short_tokens_are_filtered():
    keywords = extract_keywords(TRACE_QUESTION)
    for noise in ("我们", "哪些", "的", "了", "和", "是"):
        assert noise not in keywords, f"停用词 {noise} 不应进入核心词集合"


def test_keywords_are_pure_local_no_meaningful_noise():
    """分词结果里不应出现纯标点或单字噪声。"""
    keywords = extract_keywords("我们优先做哪些行业？哪些行业算次优先？")
    assert all(len(item) >= 2 for item in keywords)


# ═══════════════════════════════════════════════════════════════════════════
# §5.5 零额外调用：既不发模型请求，也不做向量化
# ═══════════════════════════════════════════════════════════════════════════
def test_no_network_or_model_call(monkeypatch):
    import httpx

    def _boom(*args, **kwargs):
        raise AssertionError("证据缺口检测不得发起任何网络 / 模型调用")

    monkeypatch.setattr(httpx, "post", _boom)
    monkeypatch.setattr(httpx, "get", _boom)

    keywords = build_question_keywords(_state())
    gap, _ = detect_evidence_gap(keywords, OBS_SALES_RECORDS)
    assert isinstance(gap, bool)


def test_source_has_no_model_or_vector_dependency():
    source = inspect.getsource(step_correction)
    for banned in ("model_router", "openai", "langchain", "embedding", "sentence_transformers"):
        assert banned not in source, f"不得依赖 {banned}"


def test_threshold_is_configurable():
    keywords = {"行业", "优先级", "排序", "结论"}
    observation = "行业"  # 覆盖率 0.25
    assert detect_evidence_gap(keywords, observation, threshold=0.3)[0] is True
    assert detect_evidence_gap(keywords, observation, threshold=0.2)[0] is False
