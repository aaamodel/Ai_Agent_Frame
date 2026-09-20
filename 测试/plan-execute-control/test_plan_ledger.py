# -*- coding: utf-8 -*-
"""plan 台账渲染 + 游标推进的单测。

覆盖两件事：
1. 台账必须由 plan + subtask_results **推导**（不依赖任何额外状态副本）；
2. 「执行」与「是否解决」两列语义分离——执行失败强制"否"，
   因为那些步骤没有走结论提炼，模型并未看到数据，让它自评就是编。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.agent.graph.nodes._common import (  # noqa: E402
    PLAN_STATUS_DONE,
    PLAN_STATUS_FAILED,
    PLAN_STATUS_PENDING,
    PLAN_STATUS_RUNNING,
    PLAN_STATUS_SKIPPED,
    SOLVED_NO,
    SOLVED_PARTIAL,
    SOLVED_UNKNOWN,
    SOLVED_YES,
    next_pending_cursor,
    render_plan_ledger,
)

PLAN = [
    {"id": "task_1", "description": "读字段字典确认列名", "tool_name": "file_read_tool"},
    {"id": "task_2", "description": "查私有化上线周期", "tool_name": "rag_knowledge_search"},
    {"id": "task_3", "description": "查实施费标准", "tool_name": "rag_knowledge_search"},
    {"id": "task_4", "description": "整合最终回复", "tool_name": None},
    {"id": "task_5", "description": "导出报表", "tool_name": "sales_report_export_tool"},
]


def test_ledger_shows_all_five_statuses():
    """一次渲染里同时出现：已完成 / 失败 / 执行中 / 已跳过 / 待执行。"""
    results = [
        {"subtask_id": "task_1", "status": "ok", "solved": "yes"},
        {"subtask_id": "task_2", "status": "empty_data"},
    ]
    ledger = render_plan_ledger(PLAN, results, cursor=2, skipped_task_ids=["task_4"])
    for status in (
        PLAN_STATUS_DONE, PLAN_STATUS_FAILED, PLAN_STATUS_RUNNING,
        PLAN_STATUS_SKIPPED, PLAN_STATUS_PENDING,
    ):
        assert status in ledger, f"台账缺少状态 {status}"


def _row_of(ledger: str, task_id: str) -> str:
    """取台账中某个子任务所在行。

    ⚠️ 必须按**行**断言而不是子串断言：表头里就有"是否解决"，
    用 `"是" in ledger` 这类判断恒为真，测不出东西。
    """
    rows = [line for line in ledger.splitlines() if task_id in line]
    assert len(rows) == 1
    return rows[0]


def test_failed_step_forces_solved_no():
    """执行失败 → 「是否解决」强制为否，不依赖模型自评。"""
    results = [{"subtask_id": "task_1", "status": "error", "solved": "yes"}]
    row = _row_of(render_plan_ledger(PLAN[:1], results, cursor=0), "task_1")
    assert PLAN_STATUS_FAILED in row
    # 即便记录里模型自评 solved=yes，失败步骤也必须显示"否"
    assert row.rstrip().endswith(SOLVED_NO)


def test_model_self_assessment_is_rendered():
    """正常完成的步骤按模型自评显示 是 / 部分。"""
    yes = render_plan_ledger(PLAN[:1], [{"subtask_id": "task_1", "status": "ok",
                                         "solved": "yes"}], cursor=0)
    partial = render_plan_ledger(PLAN[:1], [{"subtask_id": "task_1", "status": "ok",
                                             "solved": "partial"}], cursor=0)
    assert _row_of(yes, "task_1").rstrip().endswith(SOLVED_YES)
    assert _row_of(partial, "task_1").rstrip().endswith(SOLVED_PARTIAL)


def test_pending_and_running_show_unknown_solved():
    """未执行/执行中的步骤没有"是否解决"的概念，显示占位符。"""
    ledger = render_plan_ledger(PLAN, [], cursor=2)
    assert _row_of(ledger, "task_3").rstrip().endswith(SOLVED_UNKNOWN)
    assert _row_of(ledger, "task_5").rstrip().endswith(SOLVED_UNKNOWN)


def test_next_pending_cursor_skips_ids():
    """游标推进必须跳过被标记的子任务。"""
    assert next_pending_cursor(PLAN, 3, ["task_4"]) == 4
    assert next_pending_cursor(PLAN, 2, ["task_3", "task_4"]) == 4


def test_next_pending_cursor_none_when_all_remaining_skipped():
    """剩余全部被跳过（提前收尾的实现方式）→ None，路由据此进汇总。"""
    assert next_pending_cursor(PLAN, 2, ["task_3", "task_4", "task_5"]) is None


def test_next_pending_cursor_without_skips():
    """无跳过时退化为普通顺序推进。"""
    assert next_pending_cursor(PLAN, 0, None) == 0
    assert next_pending_cursor(PLAN, 5, []) is None


def test_next_pending_cursor_ignores_unknown_ids():
    """跳过列表里的未知 id 不应影响推进（容错）。"""
    assert next_pending_cursor(PLAN, 0, ["task_99"]) == 0
