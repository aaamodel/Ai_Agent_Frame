# -*- coding: utf-8 -*-
"""Planner 业务规则 ⑧：子问题覆盖校验。

背景：只靠提示词要求"每个子问题一个子任务"是**软约束**，模型仍会把多个子问题
合并成一次检索；一旦该次检索没能同时覆盖，就会有子问题无答案，而链路当时没有
任何机制能发现。所以覆盖校验必须落在解析层。

校验分两档：
    首次解析（strict_coverage=True）缺失 → 抛错 → 调用方重试一次；
    重试后仍缺失（strict_coverage=False）→ 告警放行（fail-open）。
第二档是刻意的：无限重试会把链路卡死，静默丢弃又会漏答，只能把问题暴露出来。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.agent.planner import _parse_subtasks  # noqa: E402


def _payload(items):
    return {"subtasks": items}


def _subtask(task_id: str, covers):
    return {
        "id": task_id,
        "title": f"任务{task_id}",
        "description": f"解决子问题 {covers}",
        "action_type": "reasoning",
        "covers_sub_questions": covers,
    }


def test_all_sub_questions_covered():
    """每个子问题都有独立子任务 → 正常返回，且覆盖声明被保留。"""
    tasks = _parse_subtasks(
        _payload([_subtask("t1", [1]), _subtask("t2", [2])]), None, 2, True
    )
    assert len(tasks) == 2
    assert [t.covers_sub_questions for t in tasks] == [[1], [2]]


def test_missing_coverage_raises_when_strict():
    """首次解析发现子问题无人覆盖 → 抛错，让调用方发起重新规划。"""
    with pytest.raises(ValueError, match="没有任何子任务覆盖"):
        _parse_subtasks(_payload([_subtask("t1", [1])]), None, 2, True)


def test_missing_coverage_fail_open_when_not_strict():
    """重试后仍缺失 → 不抛错、保留计划（fail-open），避免卡死链路。"""
    tasks = _parse_subtasks(_payload([_subtask("t1", [1])]), None, 2, False)
    assert len(tasks) == 1


def test_no_split_skips_validation():
    """未拆分子问题（count=0）→ 完全不做覆盖校验。"""
    tasks = _parse_subtasks(_payload([_subtask("t1", None)]), None, 0, True)
    assert len(tasks) == 1


def test_missing_declaration_is_detected():
    """模型干脆不声明 covers_sub_questions → 所有子问题视为未覆盖。"""
    with pytest.raises(ValueError, match="没有任何子任务覆盖"):
        _parse_subtasks(
            _payload([{"id": "t1", "title": "x", "description": "y",
                       "action_type": "reasoning"}]), None, 2, True
        )


def test_non_integer_entries_are_ignored():
    """声明里混入非法值（字符串/None）不应让解析崩溃，只丢弃非法项。"""
    tasks = _parse_subtasks(
        _payload([_subtask("t1", [1, "bad", None, 2])]), None, 2, True
    )
    assert tasks[0].covers_sub_questions == [1, 2]
