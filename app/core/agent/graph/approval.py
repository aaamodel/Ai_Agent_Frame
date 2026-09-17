# -*- coding: utf-8 -*-
"""危险工具人工审批（HITL）策略与 interrupt 闸门。

- 危险工具名单**配置驱动**（``agent_danger_tools`` 逗号分隔），不改 tools/base.py；
- 总开关 ``agent_approval_enabled``：关闭时 gate 直接放行，不挂 interrupt
  （Settings 默认开启，.env 可覆盖）；
- 暂停判定铁律在 runner/API 侧：只看 ``aget_state(cfg).next`` +
  ``snapshot.tasks[].interrupts[]``，本模块只负责 payload 构造与 resume 值解析；
- resume 值契约：``Command(resume={"approved": bool, "comment": str})``。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from app.core.agent.toolcall import ToolCall


def danger_tool_names(deps: Any) -> set:
    """从配置解析危险工具名单（逗号分隔，空白/空串忽略）。"""
    raw: str = str(deps.cfg("agent_danger_tools", "") or "")
    return {piece.strip() for piece in raw.split(",") if piece.strip()}


def approval_enabled(deps: Any) -> bool:
    """审批总开关（默认关闭，保持现网行为）。"""
    return bool(deps.cfg("agent_approval_enabled", False))


def build_approval_payload(
    call: ToolCall,
    *,
    run_id: str,
    subtask_id: Optional[str] = None,
) -> Dict[str, Any]:
    """构造 interrupt 载荷（可 JSON 序列化，API 层原样回传前端）。"""
    try:
        args_preview: str = json.dumps(call.arguments, ensure_ascii=False)[:2000]
    except (TypeError, ValueError):
        args_preview = str(call.arguments)[:2000]
    return {
        "type": "tool_approval",
        "run_id": run_id,
        "tool_name": call.tool_name,
        "arguments": call.arguments,
        "arguments_preview": args_preview,
        "source": call.source,
        "subtask_id": subtask_id,
        "thought": call.thought or None,
    }


def parse_resume_decision(value: Any) -> Tuple[bool, str]:
    """解析 Command(resume=...) 的值。

    兼容 {"approved": bool, "comment": str} 与裸 bool；异常/缺失一律按拒绝处理
    （fail-closed：拿不到明确批准就不执行危险工具）。
    """
    if isinstance(value, bool):
        return value, ""
    if isinstance(value, dict):
        approved: bool = bool(value.get("approved", False))
        comment: str = str(value.get("comment", "") or "")
        return approved, comment
    return False, f"无法识别的审批恢复值: {value!r}"


def denied_observation(tool_name: str, comment: str = "") -> str:
    """审批拒绝后回注给模型的 Observation（走正常推理收尾，不触发 replan）。"""
    comment_part: str = f" 审批意见：{comment}" if comment else ""
    return (
        f"【人工审批拒绝】工具 [{tool_name}] 的本次调用未获人工批准。{comment_part}\n"
        "请不要再次尝试调用该工具；请基于已有信息直接回答，或改用其他不需要审批的工具。"
    )


async def gate_tool_approval(
    deps: Any,
    call: ToolCall,
    *,
    run_id: str,
    subtask_id: Optional[str] = None,
) -> Optional[str]:
    """危险工具 interrupt 闸门。

    在 ``execute_tool_call`` **之前** await 调用：
    - 审批关闭 / 工具不在名单 → 返回 None（放行，正常执行）；
    - 命中且人工批准 → 返回 None（放行）；
    - 命中且拒绝 / resume 值非法 → 返回拒绝 Observation 文本（调用方跳过执行，
      把该文本作为 Observation 回注，**不消耗预算**）。

    注意：LangGraph interrupt 恢复后整个节点会重放，重放时 interrupt 直接返回
    resume 值，不会二次暂停。
    """
    if not approval_enabled(deps):
        return None
    if call.tool_name not in danger_tool_names(deps):
        return None

    # 函数内 import：未启用审批的运行不承担额外导入耦合
    from langgraph.types import interrupt

    payload = build_approval_payload(call, run_id=run_id, subtask_id=subtask_id)
    approved, comment = parse_resume_decision(interrupt(payload))
    if approved:
        return None
    return denied_observation(call.tool_name, comment)
