# -*- coding: utf-8 -*-
"""多轮改写"历史污染"确定性兜底的单测。

复现 2026-09-24 线上事故：
  Q1 = "MEDDIC 打分最多能打多少分？打到几分算 A 类？"
  Q2 = "线索第一次筛选是哪个部门负责的？以及看下线索多久更新一次"（自足新问题）
  glm-4.7 把 Q1 整句拼进 Q2 的 rewrite，并多拆出 MEDDIC 子问题，导致误触发
  plan_execute 又去搜了一遍 MEDDIC。

兜底策略：
  1. 主改写大段覆盖历史旧问题、而当前问题不含该内容 → 回退归一化原问题；
  2. 子问题逐条剔除污染项，存活不足 2 条取消拆分；
  3. 当前问题含指代/省略信号时不干预（历史介入合法）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.query_intent.rewrite.combined_rewrite_intent_service import (  # noqa: E402
    AgentCombinedRewriteIntentService,
)
from app.query_intent.rewrite.multi_question_rewrite_service import (  # noqa: E402
    AgentMultiQuestionRewriteService,
    is_history_contaminated,
)

Q1 = "MEDDIC 打分最多能打多少分？打到几分算 A 类？"
Q2 = "线索第一次筛选是哪个部门负责的？以及看下线索多久更新一次"


def _parse(raw_json: str, fallback_question: str, prior_user_questions=None):
    # 解析层不读实例状态（纯入参 → DTO），object.__new__ 跳过 dataclass 初始化。
    service = object.__new__(AgentMultiQuestionRewriteService)
    return service._parse_agent_rewrite(
        raw_response_text=raw_json,
        fallback_question=fallback_question,
        available_tool_ids=[],
        pre_rule_plan_hint=None,
        prior_user_questions=prior_user_questions,
    )


def _payload(**overrides) -> str:
    base = {
        "rewrite": Q2,
        "agent_goal": "给出线索首筛部门与更新频率结论",
        "should_split": False,
        "sub_questions": [],
        "complexity_analysis": {
            "estimated_steps": 2,
            "estimated_tool_calls": 2,
            "has_multi_step_dependency": False,
            "has_external_data_dependency": True,
            "need_creative_output": False,
            "reasoning_notes": "知识库检索",
        },
        "suggested_tools": [],
        "suggested_skills": [],
        "explicit_plan_hint": None,
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# 判定函数本身
# ═══════════════════════════════════════════════════════════════════════════
def test_detects_full_old_question_concatenation():
    assert is_history_contaminated(Q1 + Q2, Q2, [Q1]) is True


def test_clean_self_sufficient_question_not_flagged():
    assert is_history_contaminated(Q2, Q2, [Q1]) is False


def test_coreference_question_is_never_flagged():
    # 当前问题含指代"它"：结合历史把"它"还原成 MEDDIC 是合法改写
    current = "它最多能打多少分？"
    rewrite = "MEDDIC 打分最多能打多少分？"
    assert is_history_contaminated(rewrite, current, [Q1]) is False


def test_short_prior_question_ignored():
    assert is_history_contaminated("你好呀，今天天气如何", "今天天气如何", ["你好呀"]) is False


# ═══════════════════════════════════════════════════════════════════════════
# 解析层：事故复现
# ═══════════════════════════════════════════════════════════════════════════
def test_parser_strips_merged_old_question_from_rewrite():
    raw = _payload(
        rewrite=Q1 + Q2,
        should_split=True,
        sub_questions=[
            "MEDDIC 打分最多能打多少分？打到几分算 A 类？",
            "线索第一次筛选是哪个部门负责的？",
            "线索多久更新一次？",
        ],
    )
    result = _parse(raw, fallback_question=Q2, prior_user_questions=[Q1])

    assert result.rewritten_question == Q2, "主改写必须回退为当前问题"
    assert result.should_split is True
    assert result.sub_questions == [
        "线索第一次筛选是哪个部门负责的？",
        "线索多久更新一次？",
    ], "MEDDIC 子问题是历史污染，必须剔除"
    # 存活子问题对应模型原始 question_index 2、3
    assert result.sub_question_source_indexes == [2, 3]


def test_parser_collapses_split_when_only_one_clean_sub_remains():
    raw = _payload(
        rewrite=Q1 + "线索多久更新一次？",
        should_split=True,
        sub_questions=[
            "MEDDIC 打分最多能打多少分？打到几分算 A 类？",
            "线索多久更新一次？",
        ],
    )
    result = _parse(raw, fallback_question="线索多久更新一次？", prior_user_questions=[Q1])

    assert result.should_split is False
    assert result.sub_questions == ["线索多久更新一次？"]
    assert result.sub_question_source_indexes is None


def test_parser_without_history_is_unchanged():
    raw = _payload(
        rewrite=Q1 + Q2,
        should_split=True,
        sub_questions=[Q1, "线索第一次筛选是哪个部门负责的？"],
    )
    # 没有历史可比时不做任何剔除（保持旧行为）
    result = _parse(raw, fallback_question=Q2, prior_user_questions=None)
    assert result.rewritten_question == Q1 + Q2
    assert len(result.sub_questions) == 2


def test_parser_keeps_legitimate_coreference_rewrite():
    current = "它最多能打多少分？"
    raw = _payload(
        rewrite="MEDDIC 打分最多能打多少分？",
        should_split=False,
        sub_questions=[],
    )
    result = _parse(raw, fallback_question=current, prior_user_questions=[Q1])
    assert result.rewritten_question == "MEDDIC 打分最多能打多少分？"


# ═══════════════════════════════════════════════════════════════════════════
# 组合链路：意图打分 question_index 按剔除后的源序号对位
# ═══════════════════════════════════════════════════════════════════════════
def test_combined_scores_follow_sub_question_source_indexes():
    class _Node:
        def __init__(self, node_id: str):
            self.id = node_id

    node_dept = _Node("kb_lead_dept")
    node_freq = _Node("kb_lead_freq")
    payload = json.loads(
        _payload(
            rewrite=Q2,
            should_split=True,
            sub_questions=[
                "MEDDIC 打分最多能打多少分？打到几分算 A 类？",
                "线索第一次筛选是哪个部门负责的？",
                "线索多久更新一次？",
            ],
        )
    )
    payload["intent_classifications"] = [
        {"question_index": 0, "results": [{"id": "kb_main", "score": 0.9}]},
        # 模型为被污染的 MEDDIC 子问题打的分（index=1）必须被丢弃
        {"question_index": 1, "results": [{"id": "kb_meddic", "score": 0.95}]},
        {"question_index": 2, "results": [{"id": "kb_lead_dept", "score": 0.88}]},
        {"question_index": 3, "results": [{"id": "kb_lead_freq", "score": 0.7}]},
    ]
    raw = json.dumps(payload, ensure_ascii=False)
    service = object.__new__(AgentCombinedRewriteIntentService)
    scores = service._extract_precomputed_scores(
        raw_response_text=raw,
        primary_question=Q2,
        sub_questions=[
            "线索第一次筛选是哪个部门负责的？",
            "线索多久更新一次？",
        ],
        id_to_node={"kb_lead_dept": node_dept, "kb_lead_freq": node_freq},
        source_indexes=[2, 3],
    )
    dept_scores = scores["线索第一次筛选是哪个部门负责的？"]
    freq_scores = scores["线索多久更新一次？"]
    assert dept_scores[0].node is node_dept
    assert freq_scores[0].node is node_freq
    # 主问题批次的 kb_main 不在 id_to_node → 整批丢弃；污染子问题的 index=1 无映射
    assert Q2 not in scores
