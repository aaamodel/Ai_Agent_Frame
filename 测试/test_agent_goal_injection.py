# -*- coding: utf-8 -*-
"""agent_goal 读取与全链路注入的单测。

覆盖四块：
- 2.2 统一读取函数的三种情况（槽位有值 / 槽位缺失 / 值为空串）；
- 3.1 / 3.2 / 3.3 三个注入点分别注入目标；
- 3.4 三处注入内容**完全一致**（防"一处用目标、一处用改写后的问题"的漂移）；
- 4.1 目标缺失或为空时链路不中断，注入的是回退值。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.agent.graph.nodes._common import (  # noqa: E402
    AGENT_GOAL_SLOT_KEY,
    agent_goal_from_state,
    agent_goal_trace_payload,
    build_extra_system,
    render_plan_ledger,
)
from app.core.agent.graph.nodes.execute_node import (  # noqa: E402
    REACT_SYSTEM_PROMPT,
    _bootstrap_fc_messages,
    _react_text_turn,
)
from app.core.agent.graph.nodes.plan_node import _build_planner_skills_block  # noqa: E402
from app.query_intent.intent_dto import normalize_agent_goal  # noqa: E402

QUESTION = "帮我做一份 2026 年 8 月的销售复盘，重点看赢单率和输单原因"
GOAL = "输出 8 月销售复盘，含赢单率与输单原因结论"

_MISSING = object()

PLAN = [
    {"id": "task_1", "description": "读月度汇总表 8 月行", "tool_name": "local_excel_read_tool"},
    {"id": "task_2", "description": "统计输单原因", "tool_name": "local_excel_read_tool"},
    {"id": "task_3", "description": "汇总成复盘结论", "tool_name": None},
]


def _state(goal=GOAL, question: str = QUESTION) -> dict:
    """构造最小可用状态：``goal is _MISSING`` 表示槽位里**没有该键**。"""
    slots = {} if goal is _MISSING else {AGENT_GOAL_SLOT_KEY: goal}
    return {"intent": {"intent": "sales", "slots": slots}, "user_input": question}


# ---------------------------------------------------------------------------
# 2.2 读取：三种情况
# ---------------------------------------------------------------------------
def test_slot_present_returns_slot_value():
    assert agent_goal_from_state(_state(GOAL)) == GOAL


def test_slot_missing_falls_back_to_user_input():
    assert agent_goal_from_state(_state(_MISSING)) == QUESTION


def test_slot_empty_falls_back_to_user_input():
    assert agent_goal_from_state(_state("")) == QUESTION
    assert agent_goal_from_state(_state("   ")) == QUESTION


def test_fallback_value_matches_normalize_layer():
    """「槽位缺失」与「模型输出空值」必须得到同一个结果。

    这是 D5 的核心断言：两处兜底若各写一份，就会出现同一轮里目标不一致。
    """
    assert agent_goal_from_state(_state(_MISSING)) == normalize_agent_goal(None, QUESTION)
    assert agent_goal_from_state(_state("")) == normalize_agent_goal("", QUESTION)


def test_degenerate_states_do_not_raise():
    assert agent_goal_from_state({}) == ""
    assert agent_goal_from_state({"intent": None}) == ""
    assert agent_goal_from_state(
        {"intent": {"slots": None}, "user_input": QUESTION}
    ) == QUESTION


# ---------------------------------------------------------------------------
# 3.1 ReAct 系统附加段（FC 与文本协议共用同一个注入点）
# ---------------------------------------------------------------------------
def test_react_extra_system_injects_goal():
    assert GOAL in build_extra_system(_state())


def test_fc_protocol_system_prompt_contains_goal():
    """真实走一遍 FC 首轮消息组装。"""
    messages = _bootstrap_fc_messages(_state())
    system_content = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert GOAL in system_content


def test_text_protocol_system_prompt_contains_goal():
    """文本协议的系统段按其真实拼装方式组装（生产常量 + 生产函数）。"""
    extra_system = build_extra_system(_state())
    text_system = REACT_SYSTEM_PROMPT + ("\n\n" + extra_system if extra_system else "")
    assert GOAL in text_system


def test_both_protocols_consume_the_shared_injection_point():
    """结构守卫：两条协议的系统段都必须取自 ``build_extra_system``。

    行为测试只能覆盖 FC 首轮（文本协议是 async 且依赖 deps）。而"某条协议改用
    别的拼装方式"这种缺口，只有真实请求才会暴露——所以这里直接守住调用点。
    """
    import inspect

    for func, label in (
        (_bootstrap_fc_messages, "FC"),
        (_react_text_turn, "文本"),
    ):
        assert "build_extra_system(state)" in inspect.getsource(func), (
            f"{label} 协议的系统段未取自 build_extra_system"
        )


# ---------------------------------------------------------------------------
# 3.2 规划提示词
# ---------------------------------------------------------------------------
def test_plan_prompt_injects_goal():
    assert GOAL in _build_planner_skills_block(_state())


# ---------------------------------------------------------------------------
# 3.3 台账：目标在表格上方首行，且不在表格行内重复
# ---------------------------------------------------------------------------
def test_ledger_puts_goal_on_first_line():
    ledger = render_plan_ledger(PLAN, [], cursor=0, agent_goal=GOAL)
    assert ledger.splitlines()[0] == f"[本轮目标] {GOAL}"


def test_ledger_does_not_repeat_goal_in_rows():
    """目标不是子任务属性：表格里 MUST NOT 每个子任务重复一遍。"""
    ledger = render_plan_ledger(PLAN, [], cursor=0, agent_goal=GOAL)
    assert ledger.count(GOAL) == 1


def test_ledger_without_goal_has_no_goal_line():
    ledger = render_plan_ledger(PLAN, [], cursor=0)
    assert "[本轮目标]" not in ledger
    # 没有目标时表格本身照常渲染
    assert "task_1" in ledger


# ---------------------------------------------------------------------------
# 3.4 三处注入同一份内容
# ---------------------------------------------------------------------------
def test_three_injection_points_share_the_same_goal():
    state = _state()
    react_text = build_extra_system(state)
    plan_text = _build_planner_skills_block(state)
    ledger_text = render_plan_ledger(
        PLAN, [], cursor=0, agent_goal=agent_goal_from_state(state)
    )
    for text, label in ((react_text, "ReAct 系统段"), (plan_text, "规划提示词"), (ledger_text, "台账")):
        assert GOAL in text, f"{label} 未注入目标"
        # 漂移检测：任一注入点若用的是"改写后的问题"，这里就会命中
        assert QUESTION not in text, f"{label} 注入的是改写后的问题而非目标（漂移）"


# ---------------------------------------------------------------------------
# 4.1 目标缺失 / 为空时链路继续，且注入回退值
# ---------------------------------------------------------------------------
def test_missing_goal_still_injects_fallback_value():
    state = _state(_MISSING)
    assert QUESTION in build_extra_system(state)
    assert QUESTION in _build_planner_skills_block(state)


def test_missing_goal_does_not_break_ledger():
    ledger = render_plan_ledger(
        PLAN, [], cursor=0, agent_goal=agent_goal_from_state(_state(_MISSING))
    )
    assert ledger.splitlines()[0] == f"[本轮目标] {QUESTION}"
    assert "task_1" in ledger


def test_both_empty_skips_goal_line_without_raising():
    """两侧都空属退化情形：不注入占位噪音，也不抛异常。"""
    empty_state = {"intent": {"slots": {}}, "user_input": ""}
    assert agent_goal_from_state(empty_state) == ""
    assert "本轮目标" not in build_extra_system(empty_state)
    assert "[本轮目标]" not in render_plan_ledger(PLAN, [], cursor=0, agent_goal="")


# ---------------------------------------------------------------------------
# 4.2 留痕：trace 中能看到本轮目标值
# ---------------------------------------------------------------------------
def test_goal_trace_payload_records_slot_source():
    payload = agent_goal_trace_payload(_state())
    assert payload["agent_goal"] == GOAL
    assert payload["source"] == "slots"
    assert payload["chars"] == len(GOAL)


def test_goal_trace_payload_marks_fallback():
    """槽位没供上时 source=fallback——这是抽查目标质量时的关键信号。"""
    payload = agent_goal_trace_payload(_state(_MISSING))
    assert payload["agent_goal"] == QUESTION
    assert payload["source"] == "fallback"


def test_prepare_node_emits_goal_trace_event():
    """结构守卫：入口节点必须真的把这条事件发出去。

    prepare 是 `START → prepare` 的唯一后继，两种模式都必经，因此它是一轮只记
    一次的天然位置。事件发送依赖 deps.tracer，纯函数测不到，故守住调用点。
    """
    import inspect

    from app.core.agent.graph.nodes.prepare_node import prepare_node as prepare_node_func

    source = inspect.getsource(prepare_node_func)
    assert '"agent.goal"' in source
    assert "agent_goal_trace_payload(state)" in source
