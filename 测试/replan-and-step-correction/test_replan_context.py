# -*- coding: utf-8 -*-
"""重规划输入完整性的单测。

对应四条需求：
1. 每条执行结果必须携带子任务归属（实测 bug：读 `id` 导致全是 null）；
2. 已试过的路径 / 查询必须结构化传递；
3. 工具失效判定必须看"是否取得有效数据"，不能只看 status；
4. 失败原因不得包含已被本轮记录证否的方向。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.agent.graph.nodes.replan_node import (  # noqa: E402
    strip_contradicting_suggestions,
)
from app.core.agent.planner import (  # noqa: E402
    NO_OWNER_MARK,
    PlannerAgent,
    compact_attempt,
    result_is_ineffective,
)


def _rec(**overrides) -> dict:
    base: dict = {
        "subtask_id": "t1",
        "tool_name": "file_list_tool",
        "status": "ok",
        "llm_output": "拿到数据了",
        "observation": "",
        "action_input": {"path": "/data", "pattern": "*sales*"},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 4.1 归属
# ---------------------------------------------------------------------------
def test_results_carry_subtask_owner():
    compact = PlannerAgent._compact_results_for_replan([_rec(subtask_id="t3")])
    assert compact[0]["subtask_id"] == "t3"


def test_missing_owner_is_explicitly_marked_not_blank():
    """缺失时必须是显式标记：空值无法与"未设置"区分。"""
    compact = PlannerAgent._compact_results_for_replan([_rec(subtask_id=None)])
    assert compact[0]["subtask_id"] == NO_OWNER_MARK
    assert compact[0]["subtask_id"] != ""


def test_owner_key_is_subtask_id_not_id():
    """历史 bug：读 `id` 时结果全是 null（权威键是 subtask_id）。"""
    record = _rec(subtask_id="t2")
    record.pop("subtask_id")
    record["id"] = "t2"  # 只有旧键
    compact = PlannerAgent._compact_results_for_replan([record])
    assert compact[0]["subtask_id"] == NO_OWNER_MARK


# ---------------------------------------------------------------------------
# 4.2 已试路径 / 查询
# ---------------------------------------------------------------------------
def test_attempted_is_compacted_into_result():
    compact = PlannerAgent._compact_results_for_replan([_rec()])
    assert "/data" in compact[0]["attempted"]
    assert "path=" in compact[0]["attempted"]


def test_attempted_empty_when_no_args():
    assert compact_attempt({}) == ""
    assert compact_attempt({"action_input": "not-a-dict"}) == ""


def test_attempted_falls_back_to_first_string_value():
    assert "*sales*" in compact_attempt({"action_input": {"unknown_key": "*sales*"}})


# ---------------------------------------------------------------------------
# 4.3 失效判定
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", ["empty_data", "error", "budget_denied", "approval_denied"])
def test_bad_status_is_ineffective(status):
    assert result_is_ineffective(_rec(status=status)) is True


def test_ok_status_with_empty_output_is_ineffective():
    """状态为 ok 但没有任何输出 → 实际上没拿到数据。"""
    assert result_is_ineffective(_rec(status="ok", llm_output="", observation="")) is True


def test_ok_status_with_no_match_sentinel_is_ineffective():
    """实测：知识图谱检索未返回数据但 status 记为 ok，旧判据会漏掉它。"""
    record = _rec(
        status="ok",
        llm_output="",
        observation="知识图谱检索失败，未返回任何关于行业与公司核心产品之间实体关系的数据。",
    )
    # 该句不含既有 sentinel，但应当被"取到空文本"或后续扩展判据覆盖
    assert result_is_ineffective(record) is True


def test_ok_status_with_real_data_is_effective():
    assert result_is_ineffective(_rec(status="ok", llm_output="行业优先级排序如下：A > B")) is False


def test_non_dict_is_not_ineffective():
    assert result_is_ineffective(None) is False
    assert result_is_ineffective("x") is False


# ---------------------------------------------------------------------------
# 4.4 矛盾拦截
# ---------------------------------------------------------------------------
def test_contradicting_suggestion_is_removed():
    """error 建议检查 /data，而 /data 已被本轮证明为空 → 必须移除。"""
    error = (
        "汇总阶段判定现有证据不足以回答用户问题。\n"
        "- 建议方向：建议检查 /data 目录下是否存在其他命名的销售数据文件。\n"
    )
    prior = [_rec(status="empty_data", llm_output="在 /data 目录下未找到任何文件")]

    stripped, removed = strip_contradicting_suggestions(error, prior)

    assert "/data" not in stripped
    assert removed, "被移除的建议必须留痕"


def test_non_contradicting_suggestion_is_kept():
    error = "建议方向：改用知识库检索行业报告。\n"
    prior = [_rec(status="empty_data", llm_output="在 /data 目录下未找到任何文件")]

    stripped, removed = strip_contradicting_suggestions(error, prior)
    assert stripped == error
    assert removed == []


def test_no_error_text_returns_empty():
    assert strip_contradicting_suggestions("", [_rec()]) == ("", [])
