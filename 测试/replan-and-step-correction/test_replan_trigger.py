# -*- coding: utf-8 -*-
"""重规划触发收窄与次数上限的单测。

对应需求：
- 触发条件收窄为"方向性错误"（全部结论跑题），缺口性质必须结构化给出；
- 所有候选工具都已证明未取得有效数据时禁止重规划；
- 同一轮至多重规划一次。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.agent.graph.builder import (  # noqa: E402
    ROUTE_PERSIST,
    ROUTE_REPLAN,
    route_after_summarize,
)
from app.core.agent.graph.nodes.summarize_node import (  # noqa: E402
    INSUFFICIENCY_KIND_NO_DATA,
    INSUFFICIENCY_KIND_OFF_TOPIC,
    normalize_gap_kind,
)
from app.core.agent.graph.state import make_initial_state  # noqa: E402


def _state(**overrides) -> dict:
    base: dict = {
        "insufficiency_signal": "证据不足",
        "insufficiency_kind": INSUFFICIENCY_KIND_OFF_TOPIC,
        "replan_attempts": 0,
        "max_replan": 1,
        "budget": {},
        "active_tool_names": ["rag_knowledge_search", "web_search"],
        "subtask_results": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 5.1 缺口性质归一化
# ---------------------------------------------------------------------------
def test_off_topic_is_recognized():
    assert normalize_gap_kind("off_topic") == INSUFFICIENCY_KIND_OFF_TOPIC
    assert normalize_gap_kind(" OFF_TOPIC ") == INSUFFICIENCY_KIND_OFF_TOPIC


@pytest.mark.parametrize("raw", [None, "", "  ", "no_data", "garbage", 123, []])
def test_anything_else_is_treated_as_no_data(raw):
    """缺失/非法一律按非方向性错误处理——宁可少重规划。"""
    assert normalize_gap_kind(raw) == INSUFFICIENCY_KIND_NO_DATA


# ---------------------------------------------------------------------------
# 5.2 只有方向性错误才允许重规划
# ---------------------------------------------------------------------------
def test_off_topic_routes_to_replan():
    assert route_after_summarize(_state()) == ROUTE_REPLAN


@pytest.mark.parametrize("kind", [INSUFFICIENCY_KIND_NO_DATA, None, "", "unknown"])
def test_non_off_topic_routes_to_persist(kind):
    assert route_after_summarize(_state(insufficiency_kind=kind)) == ROUTE_PERSIST


def test_no_signal_routes_to_persist():
    assert route_after_summarize(_state(insufficiency_signal=None)) == ROUTE_PERSIST


# ---------------------------------------------------------------------------
# 5.3 候选工具全部失效 → 禁止重规划
# ---------------------------------------------------------------------------
def test_all_candidates_exhausted_blocks_replan():
    state = _state(
        subtask_results=[
            {"tool_name": "rag_knowledge_search", "status": "empty_data", "llm_output": ""},
            {"tool_name": "web_search", "status": "ok", "llm_output": "", "observation": ""},
        ],
    )
    assert route_after_summarize(state) == ROUTE_PERSIST


def test_replan_allowed_when_some_candidate_still_viable():
    state = _state(
        subtask_results=[
            {"tool_name": "rag_knowledge_search", "status": "empty_data", "llm_output": ""},
        ],
    )
    assert route_after_summarize(state) == ROUTE_REPLAN


# ---------------------------------------------------------------------------
# 5.4 次数硬上限
# ---------------------------------------------------------------------------
def test_second_replan_is_blocked():
    assert route_after_summarize(_state(replan_attempts=1)) == ROUTE_PERSIST


def test_default_max_replan_is_one():
    assert make_initial_state(
        run_id="r", session_id="s", trace_id="t", user_input="q",
        intent=None, should_plan=True, mode_source="test",
    )["max_replan"] == 1
