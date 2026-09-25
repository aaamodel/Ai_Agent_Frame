# -*- coding: utf-8 -*-
"""规划器输入块的单测：哪些块该进、哪些不该进。

对应四条需求：
1. 与用户问题逐字相同的冷启动参考段 → 整段不注入；
2. 意图锚点必须达到置信度门槛，缺失/非法按不足处理；
3. 规划器侧只注入技能清单，不注入"如何读取技能"的操作指引；
4. 技能段标题必须紧邻清单内容，无技能时不得留下空标题。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.agent.graph.nodes.plan_node import _build_planner_skills_block  # noqa: E402
from app.core.agent.planner import PLAN_SYSTEM_PROMPT, REPLAN_SYSTEM_PROMPT  # noqa: E402

QUESTION = "我们优先做哪些行业？哪些行业算次优先？"


# 显式给出 agent_goal：否则目标块会回退成 user_input，把问题原文带进来，
# 干扰"hint 段是否被注入"的断言。
GOAL = "交付行业优先级排序结论"


def _state(**overrides) -> dict:
    base: dict = {
        "user_input": QUESTION,
        "intent": {"intent": "general", "confidence": 1.0, "slots": {"agent_goal": GOAL}},
        "skills_index": "",
        "skills_prompt": "",
    }
    base.update(overrides)
    return base


def _with_hint(slots_hint: str) -> dict:
    return _state(
        intent={
            "intent": "general",
            "confidence": 1.0,
            "slots": {"agent_goal": GOAL, "initial_plan_hint": slots_hint},
        }
    )


# ---------------------------------------------------------------------------
# 2.1 冷启动参考段
# ---------------------------------------------------------------------------
def test_hint_equal_to_question_is_not_injected():
    block = _build_planner_skills_block(_with_hint(QUESTION), 0.5)
    assert "宏观计划参考" not in block
    assert QUESTION not in block


def test_hint_equal_to_question_ignoring_whitespace():
    """换行/缩进差异不影响"逐字相同"判定。"""
    block = _build_planner_skills_block(_with_hint("我们优先做哪些行业？\n哪些行业算次优先？"), 0.5)
    assert "宏观计划参考" not in block


def test_hint_with_real_step_semantics_is_injected():
    block = _build_planner_skills_block(_with_hint("先取数再对账最后出结论"), 0.5)
    assert "宏观计划参考" in block
    assert "先取数再对账最后出结论" in block


def test_empty_hint_not_injected():
    assert "宏观计划参考" not in _build_planner_skills_block(_with_hint(""), 0.5)
    assert "宏观计划参考" not in _build_planner_skills_block(_with_hint("   "), 0.5)
    assert "宏观计划参考" not in _build_planner_skills_block(_state(), 0.5)


# ---------------------------------------------------------------------------
# 2.2 意图锚点置信度门槛
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("confidence", [0.5, 0.75, 1.0])
def test_intent_injected_when_confidence_meets_threshold(confidence):
    state = _state(intent={"intent": "销售分析", "confidence": confidence, "slots": {}})
    block = _build_planner_skills_block(state, 0.5)
    assert "当前识别用户意图" in block
    assert "销售分析" in block


@pytest.mark.parametrize("confidence", [0.0, 0.1, 0.49])
def test_intent_not_injected_when_confidence_below_threshold(confidence):
    state = _state(intent={"intent": "销售分析", "confidence": confidence, "slots": {}})
    assert "当前识别用户意图" not in _build_planner_skills_block(state, 0.5)


@pytest.mark.parametrize("confidence", [None, "", "abc", [], {}])
def test_intent_not_injected_when_confidence_missing_or_invalid(confidence):
    """缺失或无法解析一律按"不足"处理，不得默认视为达标。"""
    state = _state(intent={"intent": "销售分析", "confidence": confidence, "slots": {}})
    assert "当前识别用户意图" not in _build_planner_skills_block(state, 0.5)


def test_intent_key_absent_is_treated_as_insufficient():
    state = _state(intent={"intent": "销售分析", "slots": {}})
    assert "当前识别用户意图" not in _build_planner_skills_block(state, 0.5)


def test_general_intent_never_injected_even_with_high_confidence():
    state = _state(intent={"intent": "general", "confidence": 1.0, "slots": {}})
    assert "当前识别用户意图" not in _build_planner_skills_block(state, 0.5)


def test_injected_intent_carries_its_confidence():
    state = _state(intent={"intent": "销售分析", "confidence": 0.75, "slots": {}})
    assert "0.75" in _build_planner_skills_block(state, 0.5)


# ---------------------------------------------------------------------------
# 2.3 规划器侧只注入技能清单
# ---------------------------------------------------------------------------
def test_only_index_is_used_not_the_full_executor_prompt():
    """即使 state 里有完整的执行侧规约，规划器也只用极简清单。"""
    full_prompt = (
        "## Skills System\n**How to Use Skills (Progressive Disclosure):**\n"
        "1. Identify Relevance\n2. Read Full Instructions\n"
    )
    state = _state(
        skills_index="- sales-intel：销售情报与销售分析（Source File: `/skills/x/SKILL.md`）",
        skills_prompt=full_prompt,
    )
    block = _build_planner_skills_block(state, 0.5)
    assert "sales-intel" in block
    assert "How to Use Skills" not in block
    assert "Read Full Instructions" not in block


# ---------------------------------------------------------------------------
# 2.4 技能段标题紧邻清单内容
# ---------------------------------------------------------------------------
def test_header_is_immediately_followed_by_the_list():
    state = _state(skills_index="- alpha：做 A\n- beta：做 B")
    block = _build_planner_skills_block(state, 0.5)
    lines = [line.strip() for line in block.splitlines()]
    header_at = lines.index("## 可用技能")
    assert lines[header_at + 1].startswith("- "), "标题下方必须紧邻清单内容"


def test_no_header_when_no_skills():
    assert "## 可用技能" not in _build_planner_skills_block(_state(), 0.5)


# ---------------------------------------------------------------------------
# 3. 规划器系统提示词必须内嵌 JSON 输出契约
#    （GLM-4.7 实测会静默忽略 response_format=json_schema：服务端返回 200
#      但自由发挥键名 plan/sub_tasks，解析器 subtasks=[] → 两次 parse 失败
#      → fallback 单任务。提示词必须自带结构契约作为第二道约束。）
# ---------------------------------------------------------------------------
_CONTRACT_FIELDS = ("subtasks", "action_type", "tool_name", "tool_args_hint", "covers_sub_questions")


@pytest.mark.parametrize("prompt", [PLAN_SYSTEM_PROMPT, REPLAN_SYSTEM_PROMPT])
def test_planner_prompt_declares_subtasks_contract(prompt):
    assert '"subtasks"' in prompt
    for field in _CONTRACT_FIELDS:
        assert f'"{field}"' in prompt, f"契约缺少字段 {field}"


@pytest.mark.parametrize("prompt", [PLAN_SYSTEM_PROMPT, REPLAN_SYSTEM_PROMPT])
def test_planner_prompt_forbids_alias_top_level_keys(prompt):
    """必须明确禁止模型实测自造的 plan / tasks / sub_tasks 顶层键。"""
    assert "sub_tasks" in prompt
    assert "plan" in prompt
    assert "tasks" in prompt
