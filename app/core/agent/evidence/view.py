# -*- coding: utf-8 -*-
"""证据板发送视图渲染（FC / 文本协议 / plan 提炼）。

渲染只读取 state 中的全量单元并以与入管同口径的规则重算证据板，不做任何
LLM 调用、不改写证据原文（除确定性的一行索引/截断标注外）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.agent.evidence.models import (
    KIND_CONTENT,
    KIND_ERROR,
    KIND_STATUS,
    KIND_TABLE,
    EvidenceUnit,
)
from app.core.agent.evidence.pipeline import (
    EXEMPT_KEEP_ROUNDS,
    EXEMPT_OBS_CHARS,
    EXEMPT_STUB_HEAD_CHARS,
    SCORE_MIN,
    board_budget_chars,
    select_board,
)
from app.core.agent.evidence.fetch import FETCH_TOOL_NAME

_ERROR_LINE_CHARS = 120
_STEP_HEADER = re.compile(r"Step\s+(\d+)")
_OBSERVATION_SPLIT = re.compile(r"(Observation:\s*)(.*)", re.DOTALL)


# ── 单元行渲染 ─────────────────────────────────────────────────────────────
def _source_label(unit: EvidenceUnit) -> str:
    return unit.source or unit.tool_name or "未知来源"


def _selected_line(unit: EvidenceUnit, *, with_fetch_hint: bool) -> str:
    head = f"[{unit.uid}｜来源：{_source_label(unit)}]"
    parts = [head, unit.text]
    if unit.truncated and with_fetch_hint:
        parts.append("…(已截断，可用 fetch_evidence 取回完整表格/原文)")
    if unit.also_from:
        parts.append(f"（另见：{'、'.join(unit.also_from)}）")
    return "\n".join(parts)


def _dupe_line(unit: EvidenceUnit) -> str:
    extra = f"（另见：{'、'.join(unit.also_from)}）" if unit.also_from else ""
    return f"↳ [{unit.uid}] 与 [{unit.dupe_of}] 内容重复，已合并{extra}"


def _index_line(unit: EvidenceUnit) -> str:
    return f"[{unit.uid}] {_source_label(unit)} · 相关性:低"


def _exempt_within_window(unit: EvidenceUnit, current_round: int) -> bool:
    return current_round - unit.round_idx < EXEMPT_KEEP_ROUNDS


def _exempt_stub_line(unit: EvidenceUnit) -> str:
    """超期短观测桩：编号 + 工具名 + 原始字符数 + 首句预览 + 取回提示。"""
    head = unit.text.strip().splitlines()[0] if unit.text.strip() else ""
    if len(head) > EXEMPT_STUB_HEAD_CHARS:
        head = head[:EXEMPT_STUB_HEAD_CHARS] + "…"
    tool = unit.tool_name or _source_label(unit)
    return (
        f"[{unit.uid}] {tool} · 原始 {len(unit.text)} 字符 · {head}"
        f"（需要全文可调取证据编号）"
    )


def _error_line(unit: EvidenceUnit) -> str:
    first = unit.text.strip().splitlines()[0] if unit.text.strip() else ""
    if len(first) > _ERROR_LINE_CHARS:
        first = first[:_ERROR_LINE_CHARS] + "…"
    return f"[{unit.uid}] ⚠️ {_source_label(unit)}：{first}"


def _render_units(
    units: Sequence[EvidenceUnit],
    board_uids: set,
    *,
    with_fetch_hint: bool,
    current_round: int,
) -> List[str]:
    """按属主（同一次工具调用）渲染其全部单元。"""
    lines: List[str] = []
    for unit in units:
        if unit.kind in (KIND_ERROR, KIND_STATUS):
            lines.append(_error_line(unit))
        elif unit.dupe_of:
            lines.append(_dupe_line(unit))
        elif unit.exempt:
            # 短观测豁免：3 个轮次内原文直出，超期一行桩
            if _exempt_within_window(unit, current_round):
                lines.append(_selected_line(unit, with_fetch_hint=with_fetch_hint))
            else:
                lines.append(_exempt_stub_line(unit))
        elif unit.uid in board_uids:
            lines.append(_selected_line(unit, with_fetch_hint=with_fetch_hint))
        else:
            lines.append(_index_line(unit))
    return lines


def _exempt_counts(
    units: Sequence[EvidenceUnit], current_round: int
) -> Tuple[int, int]:
    full = stub = 0
    for unit in units:
        if not unit.exempt or unit.dupe_of:
            continue
        if _exempt_within_window(unit, current_round):
            full += 1
        else:
            stub += 1
    return full, stub


def _cumulative_counts(
    units: Sequence[EvidenceUnit],
    rounds: Sequence[Dict[str, Any]],
) -> Tuple[int, int, int]:
    total = len(units)
    duplicated = sum(int(r.get("duplicated") or 0) for r in rounds)
    low = sum(
        1 for u in units
        if not u.dupe_of
        and not u.exempt
        and u.kind in (KIND_CONTENT, KIND_TABLE)
        and u.score < SCORE_MIN
    )
    return total, duplicated, low


def build_footer(
    units: Sequence[EvidenceUnit],
    board_uids: set,
    rounds: Sequence[Dict[str, Any]],
    *,
    with_fetch_hint: bool,
    current_round: int,
) -> str:
    total, duplicated, low = _cumulative_counts(units, rounds)
    on_board = len(board_uids)
    exempt_full, exempt_stub = _exempt_counts(units, current_round)
    lines = [
        f"─── 证据索引：共 {total} 条，板上 {on_board} 条，"
        f"低相关 {low} 条，重复 {duplicated} 条，"
        f"短观测全文 {exempt_full} 条/桩 {exempt_stub} 条。"
    ]
    if with_fetch_hint:
        lines.append("需要原文可调用 fetch_evidence(e编号) 或 "
                     "fetch_evidence(e编号, window=2) 取回相邻块。")
    latest_no_new = bool(rounds[-1].get("no_new_evidence")) if rounds else False
    if latest_no_new:
        lines.append(
            "⚠ 最近一次检索未带来新证据：若证据已足够请直接作答，"
            "否则请更换关键词或换用其他工具。"
        )
    return "\n".join(lines)


# ── FC 视图 ────────────────────────────────────────────────────────────────
def _call_name_map(messages: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            call_id = call.get("id")
            if call_id and isinstance(fn, dict):
                names[str(call_id)] = str(fn.get("name") or "")
    return names


def build_board_view(
    unit_dicts: Sequence[Dict[str, Any]],
    round_idx: int,
) -> Tuple[List[EvidenceUnit], set]:
    """重算指定轮次的证据板，返回 (板上单元, 板上 uid 集合)。"""
    units = [EvidenceUnit.from_dict(d) for d in unit_dicts]
    board = select_board(units, round_idx)
    return board, {u.uid for u in board}


def render_fc_messages(
    messages: Sequence[Dict[str, Any]],
    unit_dicts: Sequence[Dict[str, Any]],
    rounds: Sequence[Dict[str, Any]],
    *,
    round_idx: int,
) -> List[Dict[str, Any]]:
    """FC 发送视图：按 tool_call_id 属主替换 tool 消息内容，尾注挂最后一条。

    非 tool 消息原样透传（保 system/首轮 user 前缀字节不变）；
    fetch_evidence 的回填消息与没有任何单元归属的 tool 消息保留原文。
    """
    units = [EvidenceUnit.from_dict(d) for d in unit_dicts]
    _, board_uids = build_board_view(unit_dicts, round_idx)
    call_names = _call_name_map(messages)

    by_call: Dict[str, List[EvidenceUnit]] = {}
    for unit in units:
        call_id = unit.ref.get("call_id")
        if call_id:
            by_call.setdefault(str(call_id), []).append(unit)

    tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    rendered: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            rendered.append(message)
            continue
        call_id = str(message.get("tool_call_id") or "")
        if call_names.get(call_id) == FETCH_TOOL_NAME:
            rendered.append(message)  # 模型显式回取的原文，不二次压缩
            continue
        owners = by_call.get(call_id)
        if not owners:
            rendered.append(message)  # 无归属（如入管前的旧消息）：不动
            continue
        lines = _render_units(
            owners, board_uids,
            with_fetch_hint=True, current_round=round_idx,
        )
        content = "\n\n".join(lines)
        if index == tool_indexes[-1]:
            content = content + "\n\n" + build_footer(
                units, board_uids, rounds,
                with_fetch_hint=True, current_round=round_idx,
            )
        rendered.append({**message, "content": content})
    return rendered


# ── 文本协议视图 ───────────────────────────────────────────────────────────
def render_text_history_lines(
    history_lines: Sequence[str],
    unit_dicts: Sequence[Dict[str, Any]],
    rounds: Sequence[Dict[str, Any]],
    *,
    round_idx: int,
) -> List[str]:
    """文本协议发送视图：替换每个 Step 块的 Observation 段，尾注挂最后一块。"""
    units = [EvidenceUnit.from_dict(d) for d in unit_dicts]
    _, board_uids = build_board_view(unit_dicts, round_idx)
    by_round: Dict[int, List[EvidenceUnit]] = {}
    for unit in units:
        by_round.setdefault(unit.round_idx, []).append(unit)

    rendered_lines: List[str] = []
    for line in history_lines:
        step_match = _STEP_HEADER.search(line)
        owners = by_round.get(int(step_match.group(1)) - 1) if step_match else None
        if not owners:
            rendered_lines.append(line)  # 系统校验 FAIL 等无工具块：原样
            continue
        block = "\n\n".join(
            _render_units(
                owners, board_uids,
                with_fetch_hint=True, current_round=round_idx,
            )
        )
        replaced = _OBSERVATION_SPLIT.sub(
            lambda m: f"{m.group(1)}{block}", line, count=1
        )
        rendered_lines.append(replaced)

    if rendered_lines:
        rendered_lines[-1] = (
            rendered_lines[-1]
            + "\n"
            + build_footer(
                units, board_uids, rounds,
                with_fetch_hint=True, current_round=round_idx,
            )
        )
    return rendered_lines


# ── plan 子任务提炼视图 ────────────────────────────────────────────────────
def plan_view_uids(
    unit_dicts: Sequence[Dict[str, Any]],
    *,
    round_idx: int,
) -> Tuple[set, set]:
    """计算某 plan 子任务步的（已展示 uid 集合，已省略 uid 集合）。

    与 render_plan_observation 同口径，供节点层校验延迟回取请求使用：
    只有 omitted 集合中的 uid 才允许被 requested_evidence_uids 受理。
    """
    units = [EvidenceUnit.from_dict(d) for d in unit_dicts]
    current = [u for u in units if u.round_idx == round_idx]
    contents = [u for u in current if u.kind in (KIND_CONTENT, KIND_TABLE)]
    if not contents:
        return set(), set()

    _, board_uids = build_board_view(unit_dicts, round_idx)
    exempts = [u for u in contents if u.exempt and not u.dupe_of]
    ranked = [
        u for u in contents
        if not u.exempt and not u.dupe_of and u.uid in board_uids
    ]
    if not ranked:
        # 全部低于阈值时退回分数最高的非豁免块填满预算，避免提炼段失明
        ranked = sorted(
            (u for u in contents if not u.exempt and not u.dupe_of),
            key=lambda u: u.score,
            reverse=True,
        )

    budget = board_budget_chars(round_idx)
    shown = list(exempts)
    used = sum(len(u.text) for u in shown)
    for unit in ranked:
        if used + len(unit.text) > budget and shown:
            break
        shown.append(unit)
        used += len(unit.text)
    shown_uids = {u.uid for u in shown}
    omitted_uids = {
        u.uid for u in contents
        if u.uid not in shown_uids and not u.dupe_of
    }
    return shown_uids, omitted_uids


def render_plan_observation(
    observation: str,
    unit_dicts: Sequence[Dict[str, Any]],
    *,
    round_idx: int,
) -> str:
    """plan 提炼段的本步观测呈现：≤300 字短观测原文；超长则按相关性选择 +
    已省略索引（可在 requested_evidence_uids 填编号延迟一步索取全文）。

    若本步内容单元全部低于阈值，退回分数最高的若干块填满预算，保证模型
    仍能看到本步数据（阈值是板上筛选，不该让提炼段完全失明）。
    """
    text = observation or ""
    if len(text.strip()) <= EXEMPT_OBS_CHARS:
        return text

    units = [EvidenceUnit.from_dict(d) for d in unit_dicts]
    current = [u for u in units if u.round_idx == round_idx]
    if not current:
        return text[:6000]

    errors = [u for u in current if u.kind in (KIND_ERROR, KIND_STATUS)]
    contents = [u for u in current if u.kind in (KIND_CONTENT, KIND_TABLE)]
    shown_uids, omitted_uids = plan_view_uids(unit_dicts, round_idx=round_idx)
    shown = sorted(
        (u for u in contents if u.uid in shown_uids),
        key=lambda u: u.block_idx,
    )
    omitted = [u for u in contents if u.uid in omitted_uids]

    parts: List[str] = [
        f"【本步工具数据已按与子任务的相关性展示 {len(shown)}/{len(contents)} 条】"
    ]
    for unit in shown:
        parts.append(_selected_line(unit, with_fetch_hint=False))
    for unit in errors:
        parts.append(_error_line(unit))
    if omitted:
        parts.append(
            "已省略（低相关；如需全文，在 requested_evidence_uids 填入对应编号）："
            + "；".join(f"[{u.uid}] {_source_label(u)}" for u in omitted[:8])
        )
    return "\n\n".join(parts)
