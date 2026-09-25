# -*- coding: utf-8 -*-
"""审批恢复重放：``take_resume_approval_arguments`` 单测。

保证 plan 路径在 interrupt 重放时取回的是审批载荷里的**原始入参**，
而不是重新解析/重新 FC 出来的漂移值（2026-09-23 实测事故）。
"""

from app.core.agent.graph.approval import take_resume_approval_arguments


def _config(payloads):
    return {"configurable": {"thread_id": "s:1", "resume_approvals": payloads}}


def test_hit_by_tool_and_subtask_returns_full_arguments_copy() -> None:
    args = take_resume_approval_arguments(
        _config([{
            "type": "tool_approval",
            "tool_name": "sales_report_export_tool",
            "subtask_id": "t3",
            "arguments": {"report_title": "T", "content": "审批看到的正文"},
        }]),
        tool_name="sales_report_export_tool",
        subtask_id="t3",
    )
    assert args == {"report_title": "T", "content": "审批看到的正文"}


def test_returns_copy_not_payload_reference() -> None:
    payloads = [{
        "tool_name": "danger_tool", "subtask_id": "t1",
        "arguments": {"q": "a"},
    }]
    args = take_resume_approval_arguments(
        _config(payloads), tool_name="danger_tool", subtask_id="t1",
    )
    args["q"] = "mutated"
    assert payloads[0]["arguments"]["q"] == "a"


def test_miss_when_subtask_or_tool_differs() -> None:
    cfg = _config([{
        "tool_name": "sales_report_export_tool", "subtask_id": "t3",
        "arguments": {"q": "a"},
    }])
    assert take_resume_approval_arguments(
        cfg, tool_name="sales_report_export_tool", subtask_id="t4",
    ) is None
    assert take_resume_approval_arguments(
        cfg, tool_name="other_tool", subtask_id="t3",
    ) is None


def test_miss_on_empty_arguments_and_missing_injection() -> None:
    # 注入了但 arguments 为空（异常载荷）→ 不信任，走正常重解析
    cfg = _config([{"tool_name": "danger_tool", "subtask_id": "t1", "arguments": {}}])
    assert take_resume_approval_arguments(
        cfg, tool_name="danger_tool", subtask_id="t1",
    ) is None
    # 未注入（内存后端旧检查点 / 非 resume 路径）→ None
    assert take_resume_approval_arguments(
        {"configurable": {"thread_id": "s:1"}},
        tool_name="danger_tool", subtask_id="t1",
    ) is None
    assert take_resume_approval_arguments(
        None, tool_name="danger_tool", subtask_id="t1",
    ) is None
