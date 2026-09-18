# -*- coding: utf-8 -*-
"""显式步骤提示（explicit_plan_hint）判定的单测。

覆盖两件事：
1. **词级判定**：不得因单个字符的子串命中就判定为"用户给出了步骤"。
   历史事故：正则 `(先|步骤|第\\d+步|\\d+[.、)\\s])` 让「我们优**先**做哪些行业」
   命中（匹配到的只有「先」），于是被当成"用户明确说出的步骤"。
2. **产物是步骤语义片段，不是整句问题**：该值会被 ModeDecider 原样注入 planner
   的冷启动参考段，若它等于用户问题，那段参考就纯粹是问题复述。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.query_intent.rewrite.multi_question_rewrite_service import (  # noqa: E402
    AgentMultiQuestionRewriteService,
)


@pytest.fixture()
def service():
    """只测纯判定逻辑，不需要构造带 LLM 依赖的服务实例。

    `_extract_plan_hint_if_present` 只用到类级的正则常量，因此用 `__new__`
    取一个未初始化实例即可，避免为这个单测桩整套依赖。
    """
    return AgentMultiQuestionRewriteService.__new__(AgentMultiQuestionRewriteService)


# ---------------------------------------------------------------------------
# 1.1 词级判定：这些**不该**命中
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question",
    [
        "我们优先做哪些行业？哪些行业算次优先？",   # 本次 trace 的真实问句
        "优先考虑华东区",
        "首先分析一下",
        "这个方案率先落地",
        "帮我做个复盘",
        "A和B有什么关系",
        "2026年销售额",      # 年份不得被当成编号步骤
        "占比3.15是否正常",  # 小数不得被当成编号步骤
        "帮我看看先进制造行业",  # 「先进」里的「先」
    ],
)
def test_word_internal_markers_do_not_match(service, question):
    assert service._extract_plan_hint_if_present(question) is None


# ---------------------------------------------------------------------------
# 1.1 词级判定：这些**应当**命中
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question",
    [
        "先取数再对账最后出结论",
        "步骤1",
        "第2步",
        "先清洗数据，然后聚合，最后出报表",
    ],
)
def test_real_step_semantics_matches(service, question):
    assert service._extract_plan_hint_if_present(question) is not None


# ---------------------------------------------------------------------------
# 1.2 产物必须是步骤语义片段，不得是整句问题
# ---------------------------------------------------------------------------
def test_hint_is_step_segment_not_whole_question(service):
    """整句里只有一部分是步骤表述时，只保留带步骤语义的分句。"""
    question = "帮我做个复盘，步骤1先取数再分析最后出结论"
    hint = service._extract_plan_hint_if_present(question)

    assert hint is not None
    assert hint != question, "步骤提示不得等于整句问题"
    assert "帮我做个复盘" not in hint, "与步骤无关的分句必须被排除"
    assert "步骤1" in hint


def test_hint_excludes_non_step_parts_of_the_question(service):
    """核心不变量：hint 必须由"带步骤标记的分句"组成，不得夹带问题原文。

    注意：当**整句本来就是步骤表述**时（如「先取数再对账最后出结论」），
    hint 等于整句是**正确**的——此时"步骤语义部分"就是整句。真正要防的是
    历史事故那种情形：一句普通的商务问句被整体当成步骤提示回填。
    """
    # 混有非步骤部分 → 非步骤部分必须被剔除
    mixed = "帮我做个复盘，步骤1先取数再分析最后出结论"
    assert "帮我做个复盘" not in service._extract_plan_hint_if_present(mixed)

    # 纯步骤表述 → 允许等于整句（整句即步骤语义）
    pure = "先取数再对账最后出结论"
    assert service._extract_plan_hint_if_present(pure) == pure

    # 普通问句 → 必须整体不产出
    assert service._extract_plan_hint_if_present("我们优先做哪些行业？哪些行业算次优先？") is None


def test_single_marker_alone_is_not_a_step(service):
    """单个「先」不构成顺序——不构成"用户给出了步骤清单"。"""
    assert service._extract_plan_hint_if_present("先帮我看看数据") is None
    assert service._extract_plan_hint_if_present("最后再确认一次") is not None


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question", ["", "   ", None])
def test_empty_input_returns_none(service, question):
    assert service._extract_plan_hint_if_present(question) is None


def test_hint_is_length_capped(service):
    long_step = "先" + "取数" * 200 + "再" + "对账" * 200
    hint = service._extract_plan_hint_if_present(long_step)
    assert hint is not None
    assert len(hint) <= 120
