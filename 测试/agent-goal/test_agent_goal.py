# -*- coding: utf-8 -*-
"""agent_goal（本轮目标）归一化 + 解析层回填的单测。

覆盖三件事：
1. **正常值**：LLM 产出的目标原样保留；
2. **空值回退**：空串 / null / None / 未产出 → 回退为「改写后的问题」；
3. **超长截断**：超过上限硬截断，且**回退值本身**同样受上限约束。

第 3 条里的"回退值也要截断"容易被漏掉：一条 400 字的 rewrite 若原样变成目标，
目标就失去了"一句话锚点"的意义，每次注入都在烧上下文。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.query_intent.intent_dto import (  # noqa: E402
    AGENT_GOAL_MAX_CHARS,
    is_agent_goal_missing,
    normalize_agent_goal,
)
from app.query_intent.rewrite.multi_question_rewrite_service import (  # noqa: E402
    AgentMultiQuestionRewriteService,
)

REWRITTEN = "改写后的问题"


def _parse(raw_json: str):
    """直接调用解析层。

    该方法不读任何实例状态（纯入参 → DTO），故用 ``object.__new__`` 跳过
    dataclass 初始化，避免测试依赖 llm_service / 配置项等与本题无关的构造参数。
    """
    service = object.__new__(AgentMultiQuestionRewriteService)
    return service._parse_agent_rewrite(
        raw_response_text=raw_json,
        fallback_question=REWRITTEN,
        available_tool_ids=[],
        pre_rule_plan_hint=None,
    )


def _payload(**overrides) -> str:
    base = {
        "rewrite": REWRITTEN,
        "agent_goal": "输出 8 月销售复盘，含赢单率与输单原因结论",
        "should_split": False,
        "sub_questions": [],
        "complexity_analysis": {
            "estimated_steps": 3,
            "estimated_tool_calls": 2,
            "has_multi_step_dependency": True,
            "has_external_data_dependency": True,
            "need_creative_output": False,
            "reasoning_notes": "需要先取数再汇总",
        },
        "suggested_tools": [],
        "suggested_skills": [],
        "explicit_plan_hint": None,
    }
    base.update(overrides)
    return json.dumps(base, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 分支 1：正常值
# ---------------------------------------------------------------------------
def test_normal_goal_is_kept_verbatim():
    goal = "输出 8 月销售复盘，含赢单率与输单原因结论"
    assert normalize_agent_goal(goal, REWRITTEN) == goal
    assert _parse(_payload()).agent_goal == goal


def test_goal_is_stripped():
    assert normalize_agent_goal("  交付分析结论  ", REWRITTEN) == "交付分析结论"


# ---------------------------------------------------------------------------
# 分支 2：空值回退（四种"空"的写法都要兜住）
# ---------------------------------------------------------------------------
def test_empty_goal_falls_back_to_rewritten_question():
    assert normalize_agent_goal("", REWRITTEN) == REWRITTEN
    assert normalize_agent_goal("   ", REWRITTEN) == REWRITTEN
    assert normalize_agent_goal(None, REWRITTEN) == REWRITTEN
    assert normalize_agent_goal("null", REWRITTEN) == REWRITTEN
    assert normalize_agent_goal("None", REWRITTEN) == REWRITTEN
    # 解析层同样回退到「改写后的问题」（而不是原始问题）
    assert _parse(_payload(agent_goal="")).agent_goal == REWRITTEN


def test_fallback_to_empty_when_both_empty():
    """两边都空时返回空串——调用方按"无目标"处理，不得因此抛异常。"""
    assert normalize_agent_goal("", "") == ""
    assert normalize_agent_goal(None, None) == ""


# ---------------------------------------------------------------------------
# 分支 3：超长截断
# ---------------------------------------------------------------------------
def test_overlong_goal_is_truncated():
    long_goal = "交付" + "很长的目标描述" * 20
    assert len(long_goal) > AGENT_GOAL_MAX_CHARS
    assert normalize_agent_goal(long_goal, REWRITTEN) == long_goal[:AGENT_GOAL_MAX_CHARS]
    assert len(_parse(_payload(agent_goal=long_goal)).agent_goal) == AGENT_GOAL_MAX_CHARS


def test_overlong_fallback_is_also_truncated():
    """回退值本身超长时同样要截断（spec：回退为改写后问题，「必要时截断」）。"""
    long_question = "查一下" + "跨多个系统的长问题描述" * 20
    assert len(long_question) > AGENT_GOAL_MAX_CHARS
    assert normalize_agent_goal("", long_question) == long_question[:AGENT_GOAL_MAX_CHARS]


# ---------------------------------------------------------------------------
# 与 schema 的联动：字段**不在 required 里**，漏输出不牵连其他字段（design D8）
# ---------------------------------------------------------------------------
def test_is_agent_goal_missing_covers_null_like():
    assert is_agent_goal_missing("") is True
    assert is_agent_goal_missing("   ") is True
    assert is_agent_goal_missing(None) is True
    assert is_agent_goal_missing("null") is True
    assert is_agent_goal_missing("None") is True
    assert is_agent_goal_missing("交付复盘报告") is False


def test_agent_goal_is_required_in_schema():
    """D8 契约层：字段必须留在 `required` 里——模型被明确要求 100% 输出。"""
    from app.query_intent.llm_schemas import (  # noqa: PLC0415 - 就近导入便于阅读
        AgentRewriteIntentCombinedSchema,
        AgentRewriteSchema,
        pydantic_to_openai_response_format,
    )

    for schema_cls in (AgentRewriteSchema, AgentRewriteIntentCombinedSchema):
        schema = pydantic_to_openai_response_format(schema_cls)["json_schema"]["schema"]
        assert "agent_goal" in schema["properties"]
        assert "agent_goal" in schema.get("required", [])


def test_missing_goal_field_keeps_other_fields_intact():
    """D8 校验层：漏输出该字段 → 只置空 + 回退，**其余字段照常生效**。

    契约层是必填，但"模型偶尔没输出好"属极端事件：我们接受它（本轮目标没起作用），
    只是**不能**让一个字段把整次改写带走——否则 strict 未严格执行时（qwen 系已知会这样）
    连本来正确的 rewrite / 复杂度一起丢，整轮退化为规则兜底。
    """
    payload = json.loads(_payload())
    payload.pop("agent_goal")
    result = _parse(json.dumps(payload, ensure_ascii=False))

    assert result is not None, "漏输出 agent_goal 不应导致整次改写解析失败"
    assert result.agent_goal == REWRITTEN          # 置空后按 D5 回退
    assert result.rewritten_question == REWRITTEN  # ↓ 以下均原样保留
    assert result.should_split is False
    assert result.complexity_analysis.estimated_steps == 3
    assert result.complexity_analysis.estimated_tool_calls == 2


def test_wrong_type_goal_keeps_other_fields_intact():
    """类型不对（模型给了数字）同样只容错该字段，不牵连其他字段。"""
    result = _parse(_payload(agent_goal=12345))

    assert result is not None, "agent_goal 类型不合法不应导致整次改写解析失败"
    assert result.agent_goal == REWRITTEN
    assert result.complexity_analysis.estimated_steps == 3


def test_other_field_errors_are_still_strict():
    """容错边界刻意收窄：别的必填字段（rewrite）出错时**照旧判定失败**。

    否则"只对 agent_goal 容错"就变成了"什么都接受"，畸形输出会被静默吞掉。
    """
    payload = json.loads(_payload())
    payload.pop("rewrite")
    assert _parse(json.dumps(payload, ensure_ascii=False)) is None


def test_tolerance_applies_to_combined_schema():
    """同一容错器必须对**组合 schema** 也生效（主链路解析点的关键前提）。

    主链路若只丢目标、却也把意图打分丢掉，Stage2 就会因"没有预计算打分"
    退回旧的两段链路，白白多一次 LLM 调用——所以这里专门断言打分字段完好。
    """
    from app.query_intent.llm_schemas import (  # noqa: PLC0415 - 就近导入便于阅读
        AgentRewriteIntentCombinedSchema,
        validate_tolerating_agent_goal,
    )

    payload = json.loads(_payload())
    payload.pop("agent_goal")
    payload["intent_classifications"] = [{"question_index": 0, "results": []}]
    text = json.dumps(payload, ensure_ascii=False)

    with pytest.raises(Exception) as parse_error:
        AgentRewriteIntentCombinedSchema.model_validate_json(text)

    repaired = validate_tolerating_agent_goal(
        AgentRewriteIntentCombinedSchema, text, parse_error.value
    )
    assert repaired is not None, "组合 schema 也应容错 agent_goal"
    assert repaired.agent_goal == ""
    assert repaired.rewrite == REWRITTEN                       # 其余字段完好
    assert repaired.intent_classifications[0].question_index == 0  # 意图打分没被牵连
