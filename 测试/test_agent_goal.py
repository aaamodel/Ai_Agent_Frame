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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.query_intent.intent_dto import (  # noqa: E402
    AGENT_GOAL_MAX_CHARS,
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
# 与 schema 的联动：字段为 required，缺失即整体解析失败（已录入 design 风险表）
# ---------------------------------------------------------------------------
def test_missing_goal_field_fails_whole_parse():
    """漏输出该字段 → 解析返回 None → 上游退回规则兜底（不中断、不抛异常）。

    这是**刻意保留**的行为：字段若改为可选默认值，模型漏输出时会静默退化成
    "改写后问题的副本"，恰好是规范明令禁止的形态。代价是这一轮丢 LLM 改写，
    已在 design.md 风险表记录。
    """
    payload = json.loads(_payload())
    payload.pop("agent_goal")
    assert _parse(json.dumps(payload, ensure_ascii=False)) is None
