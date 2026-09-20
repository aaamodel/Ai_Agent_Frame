# -*- coding: utf-8 -*-
"""控制协议单测：解析 + 调度决策折算。

核心契约：
- 控制指令不可解析时 MUST 降级为"继续执行"，且不得丢弃结论（由调用方保证）；
- 只允许「跳过」与「提前收尾」，不提供新增/修改子任务；
- 模型编造的 task id 必须被忽略（并留痕），不能污染跳过列表。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.agent.graph.nodes.execute_node import (  # noqa: E402
    _apply_subtask_control,
    _parse_subtask_outcome,
)

PLAN = [
    {"id": "task_1", "tool_name": "rag_knowledge_search"},
    {"id": "task_2", "tool_name": "rag_knowledge_search"},
    {"id": "task_3", "tool_name": None},
    {"id": "task_4", "tool_name": "sales_report_export_tool"},
]


def test_parse_valid_json():
    raw = json.dumps({
        "conclusion": "8 月赢单金额 53 万",
        "solved": "yes",
        "next_action": "continue",
        "skip_task_ids": None,
        "reason": "",
    }, ensure_ascii=False)
    parsed = _parse_subtask_outcome(raw)
    assert parsed is not None
    assert parsed["conclusion"] == "8 月赢单金额 53 万"
    assert parsed["solved"] == "yes"


def test_parse_plain_text_returns_none():
    """纯文本（降级档）→ None，调用方据此保留原文并继续。"""
    assert _parse_subtask_outcome("这是一段普通结论文本") is None
    assert _parse_subtask_outcome("") is None


def test_parse_missing_required_field_returns_none():
    """缺 conclusion → 不可解析（降级，不抛异常）。"""
    assert _parse_subtask_outcome(json.dumps({"solved": "yes"})) is None


def test_parse_invalid_solved_value_returns_none():
    """solved 不是三态之一 → 按未解析处理。"""
    raw = json.dumps({"conclusion": "x", "solved": "maybe"})
    assert _parse_subtask_outcome(raw) is None


def test_skip_ids_merged_and_unknown_ignored():
    """只接受计划里真实存在的 id；编造的 id 被忽略并留痕。"""
    rec: dict = {}
    update = _apply_subtask_control(
        rec,
        {"next_action": "continue", "skip_task_ids": ["task_3", "task_99"], "reason": "答案已拿到"},
        PLAN, cursor=1, control_enabled=True, skipped_ids=["task_2"],
    )
    # task_2（原有） + task_3（合法）合并；task_99 被丢弃
    assert update["skipped_task_ids"] == ["task_2", "task_3"]
    assert rec["control_ignored_ids"] == ["task_99"]
    assert rec["control_reason"] == "答案已拿到"
    assert "early_finish" not in update


def test_finish_skips_all_remaining():
    """finish 归一为"跳过剩余全部"，并置 early_finish 留痕。"""
    rec: dict = {}
    update = _apply_subtask_control(
        rec,
        {"next_action": "finish", "skip_task_ids": None, "reason": "已足够回答"},
        PLAN, cursor=1, control_enabled=True, skipped_ids=[],
    )
    assert update["early_finish"] is True
    assert update["skipped_task_ids"] == ["task_3", "task_4"]


def test_disabled_switch_ignores_control():
    """开关关闭 → 不产生任何调度更新（退回"跑完整计划"）。"""
    update = _apply_subtask_control(
        {}, {"next_action": "finish", "skip_task_ids": ["task_3"]},
        PLAN, cursor=1, control_enabled=False, skipped_ids=[],
    )
    assert update == {}


def test_no_outcome_means_no_update():
    update = _apply_subtask_control(
        {}, None, PLAN, cursor=1, control_enabled=True, skipped_ids=[],
    )
    assert update == {}


def test_no_duplicate_ids_when_merging():
    """合并去重：同一 id 重复声明不应产生重复项。"""
    update = _apply_subtask_control(
        {}, {"next_action": "continue", "skip_task_ids": ["task_3"]},
        PLAN, cursor=0, control_enabled=True, skipped_ids=["task_3"],
    )
    assert update["skipped_task_ids"] == ["task_3"]
