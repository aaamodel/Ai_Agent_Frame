# -*- coding: utf-8 -*-
"""fetch_evidence：模型显式回取证据原文的确定性工具（不经工具注册表）。

- 只在 ReAct 循环节点内被拦截处理，不注册进 tool registry、不产生新依赖；
- 原文坐标 = 单元 ref.call_id + block_idx，指向 react_messages 里的原始
  tool 消息；取回时对原消息**重跑一次 chunker**定位，而不是另存原文；
- 取回单元在下一轮证据板强制保留并加 10 分，仅生效一轮。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from app.core.agent.evidence.chunkers import chunk_observation
from app.core.agent.evidence.models import (
    KIND_CONTENT,
    KIND_TABLE,
    EvidenceUnit,
)

if TYPE_CHECKING:  # 仅类型标注；运行时不引入工具层依赖
    from app.core.agent.toolcall import ToolCall

FETCH_TOOL_NAME = "fetch_evidence"
# plan_execute 延迟一步回取的内部伪工具：不注册进工具注册表，
# 仅在 plan 工具分发前被节点拦截，回填内容来自已保存观测。
EVIDENCE_RESTORE_TOOL = "evidence_restore"
WINDOW_MIN = 0
WINDOW_MAX = 5
_MAX_LISTED_UIDS = 20


def fetch_tool_definition() -> Dict[str, Any]:
    """OpenAI function-calling 工具定义。"""
    return {
        "type": "function",
        "function": {
            "name": FETCH_TOOL_NAME,
            "description": (
                "取回证据板上某条证据的完整原文，或其相邻片段。"
                "uid 取自观测中 [eN｜来源:...] 的编号；window=0 只取本条完整"
                "原文（含被截断的长表格），window=1..5 同时返回前后各 N 个片段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {
                        "type": "string",
                        "description": "证据编号，形如 e3",
                    },
                    "window": {
                        "type": "integer",
                        "minimum": WINDOW_MIN,
                        "maximum": WINDOW_MAX,
                        "description": "相邻片段数量，默认 0",
                    },
                },
                "required": ["uid"],
            },
        },
    }


def is_fetch_call(call: "ToolCall") -> bool:
    """兼容 ToolCall 对象 / 文本协议解析出的 dict。"""
    tool_name = getattr(call, "tool_name", None)
    if tool_name is None and isinstance(call, dict):
        tool_name = call.get("tool_name") or call.get("name")
    return tool_name == FETCH_TOOL_NAME


def parse_fetch_arguments(arguments: Any) -> Tuple[Optional[str], int, Optional[str]]:
    """返回 (uid, window, error)。error 非空时前两项不可用。"""
    data: Any = arguments
    if isinstance(arguments, str):
        try:
            data = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return None, 0, "fetch_evidence 参数不是合法 JSON"
    if not isinstance(data, dict):
        return None, 0, "fetch_evidence 参数必须是对象"
    uid = data.get("uid")
    if not isinstance(uid, str) or not uid.strip():
        return None, 0, "fetch_evidence 缺少必填参数 uid（形如 e3）"
    window = data.get("window", 0)
    if isinstance(window, bool) or not isinstance(window, int):
        return None, 0, "window 必须是 0..5 的整数"
    if not (WINDOW_MIN <= window <= WINDOW_MAX):
        return None, 0, f"window 必须在 {WINDOW_MIN}..{WINDOW_MAX} 之间"
    return uid.strip(), window, None


def _available_uids_text(units: Sequence[EvidenceUnit]) -> str:
    ids = [
        u.uid for u in units
        if u.kind in (KIND_CONTENT, KIND_TABLE) and not u.dupe_of
    ][:_MAX_LISTED_UIDS]
    tail = "……" if len(units) > len(ids) else ""
    return "可用编号：" + "、".join(ids) + tail


def _find_original_observation(
    state: Dict[str, Any], call_id: str
) -> Optional[str]:
    for message in state.get("react_messages") or []:
        if (
            isinstance(message, dict)
            and message.get("role") == "tool"
            and str(message.get("tool_call_id") or "") == call_id
        ):
            content = message.get("content")
            return content if isinstance(content, str) else None
    return None


def _render_blocks(
    unit: EvidenceUnit,
    blocks: Sequence[Dict[str, Any]],
    window: int,
) -> str:
    center = min(max(unit.block_idx, 0), max(0, len(blocks) - 1))
    start = max(0, center - window)
    end = min(len(blocks), center + window + 1)
    parts: List[str] = []
    for index in range(start, end):
        block = blocks[index]
        marker = " >>>" if index == center else ""
        source = block.get("source") or unit.tool_name
        parts.append(
            f"── 片段 {index + 1}/{len(blocks)}｜来源：{source}{marker} ──\n"
            f"{block.get('text', '')}"
        )
    return "\n\n".join(parts)


def handle_fetch_evidence(
    state: Dict[str, Any],
    *,
    uid: str,
    window: int = 0,
    round_idx: int = 0,
) -> Tuple[str, List[Dict[str, Any]]]:
    """处理一次回取。

    Returns:
        (给模型的观测文本, 更新后的全量 evidence_units dict 列表)
        不触碰工具注册表，不修改 react_messages。
    """
    units = [EvidenceUnit.from_dict(d) for d in state.get("evidence_units") or []]
    target = next((u for u in units if u.uid == uid), None)
    if target is None:
        return (
            f"未找到证据 {uid}。{_available_uids_text(units)}",
            [u.to_dict() for u in units],
        )
    if target.dupe_of:
        return (
            f"[{uid}] 已与 [{target.dupe_of}] 合并，请回取 {target.dupe_of}。",
            [u.to_dict() for u in units],
        )

    observation_text: str
    call_id = str(target.ref.get("call_id") or "")
    original = _find_original_observation(state, call_id) if call_id else None
    if original is None:
        # 原消息已不可得（如文本协议/极端裁剪）：退回板内文本
        observation_text = (
            f"[{target.uid}｜来源：{target.source or target.tool_name}]\n"
            f"{target.text}"
        )
    else:
        blocks = chunk_observation(
            tool_name=target.tool_name,
            observation=original,
            action_input=None,
            call_id=call_id,
        )
        if not blocks:
            observation_text = target.text
        else:
            observation_text = _render_blocks(target, blocks, window)

    # 标记取回：下一轮强制保留 +10（仅一轮）；同 uid 重复回取不重复占预算
    target.fetched = True
    target.ref = dict(target.ref)
    target.ref["fetched_round"] = int(round_idx)

    header = f"已为你取回 [{target.uid}] 的原文"
    if window:
        header += f"（前后各 {window} 个相邻片段）"
    return header + "：\n\n" + observation_text, [u.to_dict() for u in units]


def _find_plan_original_observation(
    subtask_results: Sequence[Dict[str, Any]], call_id: str
) -> Optional[str]:
    """plan 单元的 call_id 形如 ``plan_{cursor}_{subtask_id}``，据此定位已保存观测。"""
    marker = ""
    if call_id.startswith("plan_"):
        parts = call_id.split("_", 2)
        if len(parts) == 3:
            marker = parts[2]
    for rec in subtask_results:
        sid = str(rec.get("subtask_id") or "")
        if marker and sid == marker:
            obs = rec.get("observation")
            return obs if isinstance(obs, str) else None
    return None


def build_plan_restore_text(
    state: Dict[str, Any],
    uids: Sequence[str],
) -> Tuple[str, int]:
    """合成 plan 内部恢复步的观测：按 uid 从已保存证据/观测回填原文。

    零外部调用、零注册表接触；不修改 state。
    Returns:
        (观测文本, 实际命中的单元数)
    """
    units = [EvidenceUnit.from_dict(d) for d in state.get("evidence_units") or []]
    results = state.get("subtask_results") or []
    by_uid = {u.uid: u for u in units}

    parts: List[str] = []
    hits = 0
    for uid in uids:
        target = by_uid.get(str(uid))
        if target is None or target.kind not in (KIND_CONTENT, KIND_TABLE):
            parts.append(f"[{uid}] 未找到该证据编号，已跳过。")
            continue
        hits += 1
        body = target.text
        call_id = str(target.ref.get("call_id") or "")
        if target.truncated and call_id:
            # 截断块从已保存的原始观测（≤8000 字）重切取完整块
            original = _find_plan_original_observation(results, call_id)
            if original:
                blocks = chunk_observation(
                    tool_name=target.tool_name,
                    observation=original,
                    action_input=None,
                    call_id=call_id,
                )
                idx = min(max(target.block_idx, 0), max(0, len(blocks) - 1))
                if blocks:
                    body = str(blocks[idx].get("text") or body)
        source = target.source or target.tool_name or "未知来源"
        parts.append(f"── 回填 [{target.uid}｜来源：{source}] ──\n{body}")

    if not parts:
        return "证据回填失败：没有可用的证据编号。", 0
    return "\n\n".join(parts), hits
