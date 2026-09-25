# -*- coding: utf-8 -*-
"""**资产差集专项测试**（任务 7.1）+ 候选生成 / 注入时机 / 翻译 / 闸门的单测。

这个算法是本变更的关键之一，因此单独成套、可直接看出输入输出。

背景（2026-09 实测）：一条 trace 里两次重规划共占 20,041 输入字符（53.5%），
要解决的却是步级问题——t1 读了技能文档、摘要丢掉路径，重规划只能盲扫 `/data`，
连续两次为空。病根是"让模型在没有候选集的情况下自己发明方向"。
本算法负责把候选**由代码确定性算出**，并让已证否的方向结构性不进列表。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.agent.graph.nodes.execute_node import _apply_step_correction  # noqa: E402
from app.core.agent.step_correction import (  # noqa: E402
    CANDIDATE_BLOCK_HEADER,
    CANDIDATE_MAX_CHARS,
    Candidate,
    attempted_asset_locations,
    build_candidates,
    injection_reason,
    parse_candidate_id,
    remaining_quota,
    render_candidates,
    translate_candidate,
)

# ── 取自真实 SKILL.md「数据资产地图」的结构 ──────────────────────────────────
FACTS = [
    {"name": "客户线索台账.xlsx",
     "location": "raw_data/sales_intel/客户线索台账.xlsx",
     "tool": "sales_sql_query"},
    {"name": "产品与报价表.xlsx",
     "location": "raw_data/sales_intel/产品与报价表.xlsx",
     "tool": "sales_sql_query"},
    {"name": "销售业绩月度表.xlsx",
     "location": "raw_data/sales_intel/销售业绩月度表.xlsx",
     "tool": "sales_sql_query"},
]

ALLOWED_WITH_EXCEL = ["sales_sql_query", "rag_knowledge_search", "web_search"]

_DEPS = SimpleNamespace(tracer=None)  # trace_event 对 tracer=None 是安全的


def _state(**overrides) -> dict:
    base: dict = {
        "user_input": "我们优先做哪些行业？哪些行业算次优先？",
        "intent": {"slots": {"agent_goal": "交付行业优先级排序结论"}},
        "extracted_facts": [dict(item) for item in FACTS],
        "active_tool_names": list(ALLOWED_WITH_EXCEL),
        "subtask_results": [],
        "step_corrections": [],
        "budget": {},
    }
    base.update(overrides)
    return base


def _rec(**overrides) -> dict:
    base: dict = {
        "subtask_id": "t1", "tool_name": "file_read_tool", "status": "ok",
        "llm_output": "已读取技能文档", "observation": "", "action_input": {"file_path": "SKILL.md"},
    }
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════
# 7.1 ① 未出现在任何 action_input 中的资产全部进入候选
# ═══════════════════════════════════════════════════════════════════════════
def test_untried_assets_all_become_candidates():
    result = build_candidates(_state())
    ids = [item.id for item in result.candidates]
    for fact in FACTS:
        assert f"asset:{fact['name']}" in ids


def test_asset_description_carries_verbatim_location():
    """位置必须与文档逐字一致——模型要照着它调工具。"""
    result = build_candidates(_state())
    rendered = render_candidates(result.candidates)
    for fact in FACTS:
        assert fact["location"] in rendered


# ═══════════════════════════════════════════════════════════════════════════
# 7.1 ② 位置被调用过的资产被剔除
# ═══════════════════════════════════════════════════════════════════════════
def test_attempted_asset_is_excluded():
    state = _state(subtask_results=[_rec(
        tool_name="sales_sql_query",
        action_input={"file_path": "raw_data/sales_intel/客户线索台账.xlsx"},
    )])
    ids = [item.id for item in build_candidates(state).candidates]
    assert "asset:客户线索台账.xlsx" not in ids
    # 其余未尝试的资产仍在
    assert "asset:产品与报价表.xlsx" in ids


# ═══════════════════════════════════════════════════════════════════════════
# 7.1 ③ 仅文件名命中也被判为已尝试
# ═══════════════════════════════════════════════════════════════════════════
def test_basename_only_match_counts_as_attempted():
    """覆盖"模型自己拼了目录、但文件名写对"的情形。"""
    state = _state(subtask_results=[_rec(
        tool_name="sales_sql_query",
        action_input={"file_path": "/some/other/dir/客户线索台账.xlsx"},
    )])
    hit = attempted_asset_locations(FACTS, state["subtask_results"])
    assert "raw_data/sales_intel/客户线索台账.xlsx" in hit

    ids = [item.id for item in build_candidates(state).candidates]
    assert "asset:客户线索台账.xlsx" not in ids


def test_path_normalization_ignores_separator_and_case():
    state = _state(subtask_results=[_rec(
        action_input={"file_path": r"RAW_DATA\Sales_Intel\客户线索台账.xlsx"},
    )])
    ids = [item.id for item in build_candidates(state).candidates]
    assert "asset:客户线索台账.xlsx" not in ids


# ═══════════════════════════════════════════════════════════════════════════
# 7.1 ④ 白名单不含消费该资产的工具 → 排除 + 告警
# ═══════════════════════════════════════════════════════════════════════════
def test_asset_requiring_unlisted_tool_is_excluded_with_warning():
    """本 trace 的真实情形：白名单只有 6 个工具、且不含任何 Excel 工具。"""
    state = _state(active_tool_names=["rag_knowledge_search", "web_search"])
    result = build_candidates(state)

    assert [item.id for item in result.candidates if item.id.startswith("asset:")] == []
    assert len(result.excluded) == len(FACTS)
    first = result.excluded[0]
    assert first["asset"] and first["location"] and "白名单" in first["reason"]


def test_asset_without_declared_tool_is_excluded():
    state = _state(extracted_facts=[{"name": "无工具表.xlsx", "location": "raw_data/x.xlsx", "tool": ""}])
    result = build_candidates(state)
    assert result.candidates == [] or all(not c.id.startswith("asset:") for c in result.candidates)
    assert result.excluded and "未声明" in result.excluded[0]["reason"]


# ═══════════════════════════════════════════════════════════════════════════
# 差集①：工具侧
# ═══════════════════════════════════════════════════════════════════════════
def test_used_tools_are_removed_from_tool_side_diff():
    state = _state(subtask_results=[_rec(tool_name="rag_knowledge_search")])
    ids = [item.id for item in build_candidates(state).candidates]
    assert "tool:rag_knowledge_search" not in ids
    assert "tool:web_search" in ids


def test_inflight_current_result_is_removed_from_candidates():
    """当前步尚未入账，但它调用的工具/尝试的资产已"在飞"，MUST 从候选剔除。

    否则纠偏步自身的工具会被当成"尚未尝试的候选"重新注入，模型再选即连锁插步。
    """
    state = _state(subtask_results=[_rec(tool_name="rag_knowledge_search")])
    inflight = _rec(
        tool_name="web_search",
        action_input={"file_path": "raw_data/sales_intel/客户线索台账.xlsx"},
    )
    ids = [item.id for item in build_candidates(state, current_result=inflight).candidates]
    assert "tool:web_search" not in ids
    assert "tool:sales_sql_query" in ids  # 其它未用工具仍在
    # 当前步刚尝试过的资产同样剔除（按位置/文件名反查）
    assert "asset:客户线索台账.xlsx" not in ids


# ═══════════════════════════════════════════════════════════════════════════
# 7.1 ⑤ 两个差集均为空 → 不注入且不中断
# ═══════════════════════════════════════════════════════════════════════════
def test_empty_union_renders_nothing_and_does_not_raise():
    state = _state(
        active_tool_names=["rag_knowledge_search"],
        extracted_facts=[],
        subtask_results=[_rec(tool_name="rag_knowledge_search")],
    )
    result = build_candidates(state)
    assert result.candidates == []
    assert render_candidates(result.candidates) == ""
    assert injection_reason(state) is None


def test_tolerates_malformed_input():
    assert build_candidates({}).candidates == []
    assert build_candidates({"extracted_facts": "not-a-list"}).candidates == []
    assert attempted_asset_locations(None, None) == set()
    assert attempted_asset_locations(FACTS, [None, "x", 1]) == set()


# ═══════════════════════════════════════════════════════════════════════════
# §2.5 渲染与长度上限
# ═══════════════════════════════════════════════════════════════════════════
def test_render_respects_length_cap():
    many = [Candidate(id=f"tool:t{i}", description=f"尚未尝试的工具：t{i}") for i in range(80)]
    rendered = render_candidates(many)
    assert len(rendered) <= CANDIDATE_MAX_CHARS
    assert rendered.startswith(CANDIDATE_BLOCK_HEADER)


def test_render_truncates_consistently():
    """被长度上限截掉的项不得以任何形式残留——否则模型会选到看不见的项。"""
    many = [Candidate(id=f"tool:t{i}", description="尚未尝试的工具" + "x" * 30) for i in range(20)]
    rendered = render_candidates(many)

    assert len(rendered) <= CANDIDATE_MAX_CHARS
    shown_ids = [item.id for item in many if f"[{item.id}]" in rendered]
    lines = [line for line in rendered.splitlines() if line.startswith("[")]
    assert len(shown_ids) == len(lines), "渲染行与出现的标识必须一一对应"
    assert len(shown_ids) < len(many), "本例应当确实发生了截断"


def test_render_contains_no_param_schema():
    """只注入索引：不得把参数 schema 塞进来（否则成本反超重规划）。"""
    rendered = render_candidates(build_candidates(_state()).candidates)
    for banned in ("action_input", "parameters", "{", "input_schema"):
        assert banned not in rendered


# ═══════════════════════════════════════════════════════════════════════════
# §3 注入时机
# ═══════════════════════════════════════════════════════════════════════════
def test_inject_on_failed_step():
    state = _state(subtask_results=[_rec(status="empty_data", llm_output="")])
    assert injection_reason(state) == "取数失败"


@pytest.mark.parametrize("solved", ["no", "partial"])
def test_inject_when_previous_step_unsolved(solved):
    state = _state(subtask_results=[_rec(status="ok", solved=solved)])
    assert injection_reason(state) == "上一步未解决"


def test_no_injection_when_everything_normal():
    state = _state(subtask_results=[_rec(status="ok", solved="yes")])
    assert injection_reason(state) is None


def test_evidence_gap_can_trigger_injection():
    state = _state(subtask_results=[_rec(status="ok", solved="yes")])
    assert injection_reason(state, evidence_gap=True) == "证据缺口"


# ═══════════════════════════════════════════════════════════════════════════
# §4 模型只选不造
# ═══════════════════════════════════════════════════════════════════════════
def test_parse_candidate_id_accepts_both_prefixes():
    assert parse_candidate_id("asset:客户线索台账.xlsx") == "asset:客户线索台账.xlsx"
    assert parse_candidate_id("tool:web_search") == "tool:web_search"
    assert parse_candidate_id("  TOOL:web_search  ") == "tool:web_search"


@pytest.mark.parametrize("raw", ["[tool:web_search]", "【asset:表.xlsx】", " [tool:web_search] "])
def test_parse_candidate_id_tolerates_brackets(raw):
    """模型常把渲染出来的方括号一起抄回来。"""
    assert parse_candidate_id(raw) is not None
    assert not parse_candidate_id(raw).startswith("[")


@pytest.mark.parametrize("raw", [None, "", "   ", "web_search", "完全瞎编的"])
def test_parse_candidate_id_rejects_garbage(raw):
    assert parse_candidate_id(raw) is None


def test_translate_asset_builds_tool_and_verbatim_path():
    candidates = build_candidates(_state()).candidates
    action = translate_candidate(
        "asset:客户线索台账.xlsx", candidates=candidates,
        allowed_tools=ALLOWED_WITH_EXCEL, facts=FACTS,
    )
    assert action is not None
    assert action["tool_name"] == "sales_sql_query"
    assert action["action_input"]["file_path"] == "raw_data/sales_intel/客户线索台账.xlsx"


def test_translate_tool_candidate():
    candidates = build_candidates(_state()).candidates
    action = translate_candidate(
        "tool:web_search", candidates=candidates, allowed_tools=ALLOWED_WITH_EXCEL, facts=FACTS,
    )
    assert action == {"tool_name": "web_search", "action_input": {}, "title": "调用 web_search"}


def test_translate_rejects_id_not_in_current_list():
    """模型编造 / 上一轮的旧标识一律拒绝。"""
    candidates = [Candidate(id="tool:web_search", description="尚未尝试的工具：web_search")]
    assert translate_candidate(
        "tool:rag_knowledge_search", candidates=candidates,
        allowed_tools=ALLOWED_WITH_EXCEL, facts=FACTS,
    ) is None


def test_translate_rejects_tool_outside_whitelist():
    candidates = [Candidate(id="tool:secret_tool", description="x")]
    assert translate_candidate(
        "tool:secret_tool", candidates=candidates,
        allowed_tools=ALLOWED_WITH_EXCEL, facts=FACTS,
    ) is None


def test_translate_rejects_asset_whose_tool_left_the_whitelist():
    candidates = [Candidate(id="asset:客户线索台账.xlsx", description="x")]
    assert translate_candidate(
        "asset:客户线索台账.xlsx", candidates=candidates,
        allowed_tools=["web_search"], facts=FACTS,
    ) is None


# ═══════════════════════════════════════════════════════════════════════════
# §6 闸门：配额 / 作用域
# ═══════════════════════════════════════════════════════════════════════════
def _apply(state, selected, candidates, *, cursor=0, skipped=None, plan=None, budget=None):
    return _apply_step_correction(
        state=state,
        plan=plan if plan is not None else [
            {"id": "t1", "title": "已执行", "action_type": "tool", "tool_name": "file_read_tool"},
            {"id": "t2", "title": "待执行", "action_type": "tool", "tool_name": "rag_knowledge_search",
             "description": "原描述"},
        ],
        cursor=cursor,
        skipped_ids=skipped or [],
        candidates=candidates,
        selected=selected,
        deps=_DEPS,
        trace_id="trace",
        subtask_id="t1",
        trigger="证据缺口",
    )


def test_correction_applied_within_quota():
    state = _state(subtask_results=[_rec()])
    update = _apply(state, "tool:web_search", build_candidates(state).candidates)

    assert update, "配额内应当生效"
    new_plan = update["plan"]
    assert len(new_plan) == 3, "纠偏是插入新任务，原计划项全部保留"
    # 插入位置 = cursor + 1：下一步立即执行纠偏步
    inserted = new_plan[1]
    assert inserted["id"] == "correction_t1"
    assert inserted["tool_name"] == "web_search"
    assert inserted["action_type"] == "tool"
    assert "执行期就地纠偏" in inserted["description"]
    # 原任务原样保留、顺延一位，不被改写
    assert new_plan[0]["id"] == "t1"
    assert new_plan[2]["id"] == "t2"
    assert new_plan[2]["tool_name"] == "rag_knowledge_search"
    assert new_plan[2]["description"] == "原描述"
    assert update["step_corrections"][0]["applied"] is True
    assert update["step_corrections"][0]["remaining_quota"] > 0


def test_correction_ignored_when_quota_exhausted():
    state = _state(subtask_results=[_rec()])
    candidates = build_candidates(state).candidates
    quota = remaining_quota(state)
    state["step_corrections"] = [{"applied": True}] * quota  # 配额用尽
    assert remaining_quota(state) == 0

    assert _apply(state, "tool:web_search", candidates) == {}


def test_no_chained_correction_on_correction_task():
    """纠偏任务（id 以 correction_ 开头）不再触发二层纠偏。

    复现 2026-09 事故：correction_task_4（file_grep）未解决后又插入
    correction_correction_task_4（file_read），空 hint 靠 FC 自由组参，
    模型编出不存在的"线索管理规范.md"。即使模型给了合法候选标识也必须忽略。
    """
    state = _state(subtask_results=[_rec()])
    candidates = build_candidates(state).candidates

    update = _apply_step_correction(
        state=state,
        plan=[
            {"id": "task_4", "title": "原任务", "action_type": "reasoning"},
            {"id": "correction_task_4", "title": "调用 file_grep_tool",
             "action_type": "tool", "tool_name": "file_grep_tool"},
        ],
        cursor=1,
        skipped_ids=[],
        candidates=candidates,
        selected="tool:web_search",
        deps=_DEPS,
        trace_id="trace",
        subtask_id="correction_task_4",
        trigger="上一步未解决",
    )
    assert update == {}, "纠偏任务的二次纠偏指令必须被整体忽略"
    # 跳过不写 step_corrections、不消耗配额
    assert state["step_corrections"] == []
    assert remaining_quota(state) >= 2


def test_normal_task_id_with_same_prefix_text_is_not_blocked():
    """普通任务只要不正好命中 correction_ 前缀就不受闸门影响。"""
    state = _state(subtask_results=[_rec()])
    update = _apply(state, "tool:web_search", build_candidates(state).candidates)
    assert update and update["plan"][1]["id"] == "correction_t1"


def test_correction_ignored_when_id_invalid():
    state = _state(subtask_results=[_rec()])
    assert _apply(state, "tool:完全不存在的工具", build_candidates(state).candidates) == {}
    assert _apply(state, None, build_candidates(state).candidates) == {}


def test_applying_correction_keeps_cursor_and_skips():
    """作用域闸门：cursor 不动、已跳过记录与提前收尾标记不清空。"""
    state = _state(subtask_results=[_rec()])
    update = _apply(
        state, "tool:web_search", build_candidates(state).candidates,
        cursor=0, skipped=["t9"],
    )
    assert update
    assert "cursor" not in update, "cursor MUST 保持不变"
    assert "skipped_task_ids" not in update, "已跳过记录 MUST NOT 被清空"
    assert "early_finish" not in update, "提前收尾标记 MUST NOT 被清空"


def test_correction_records_all_required_fields():
    """§6.3 留痕四项：触发条件 / 选中标识 / 目标动作 / 剩余配额。"""
    state = _state(subtask_results=[_rec()])
    update = _apply(state, "tool:web_search", build_candidates(state).candidates)
    record = update["step_corrections"][0]

    assert record["trigger"]
    assert record["selected"] == "tool:web_search"
    assert record["target_tool"] == "web_search"
    assert isinstance(record["remaining_quota"], int)


def test_correction_inserts_when_cursor_at_tail():
    """cursor 已在末尾 → 不再有"无可替换目标"闸门：直接在末尾追加纠偏步。"""
    state = _state(subtask_results=[_rec()])
    plan = [{"id": "t1", "title": "唯一一步", "action_type": "tool", "tool_name": "file_read_tool"}]
    update = _apply(
        state, "tool:web_search", build_candidates(state).candidates,
        cursor=0, plan=plan,
    )
    assert update, "末尾追加也应当生效"
    new_plan = update["plan"]
    assert len(new_plan) == 2
    assert new_plan[0] == plan[0], "已执行/已有计划项 MUST NOT 被改写"
    assert new_plan[1]["id"] == "correction_t1"
    assert new_plan[1]["tool_name"] == "web_search"
    record = update["step_corrections"][0]
    assert record["applied"] is True
    assert record["inserted"] is True
    assert record["insert_at"] == 1
    assert record["target_subtask_id"] == "correction_t1"


def test_repeated_correction_for_same_subtask_generates_unique_id():
    """同一 subtask_id 触发第二次纠偏（顺延到后续步时）→ id 自动去重。"""
    state = _state(subtask_results=[_rec()])
    candidates = build_candidates(state).candidates

    first = _apply(state, "tool:web_search", candidates)
    assert first["plan"][1]["id"] == "correction_t1"

    # 模拟第一次纠偏已写回 state / plan 后再次触发
    state["plan"] = first["plan"]
    state["step_corrections"] = first["step_corrections"]
    second = _apply(
        state, "tool:web_search", candidates,
        cursor=1, plan=first["plan"],
    )
    assert second, "默认配额 2 次，第二次仍应生效"
    assert second["plan"][2]["id"] == "correction_t1_2"
    assert len(second["plan"]) == 4
