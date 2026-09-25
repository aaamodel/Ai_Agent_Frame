# -*- coding: utf-8 -*-
"""证据板（Evidence Board）：请求内工具观测的确定性归一化 / 打分 / 去重 /
按轮预算选择 / 发送视图渲染 / 按编号回取。

全流程零 LLM 调用、零外部工具重调、零新存储（原始观测仍在 graph state 中）。
"""

from app.core.agent.evidence.models import (
    KIND_CONTENT,
    KIND_ERROR,
    KIND_STATUS,
    KIND_TABLE,
    EvidenceUnit,
    RoundReport,
)
from app.core.agent.evidence.pipeline import (
    BOARD_BASE_CHARS,
    BOARD_CAP_CHARS,
    BOARD_STEP_CHARS,
    SCORE_MIN,
    ingest_observation,
)
from app.core.agent.evidence.fetch import (
    FETCH_TOOL_NAME,
    fetch_tool_definition,
    handle_fetch_evidence,
    is_fetch_call,
)
from app.core.agent.evidence.view import (
    build_board_view,
    render_fc_messages,
    render_plan_observation,
    render_text_history_lines,
)

__all__ = [
    "KIND_CONTENT",
    "KIND_ERROR",
    "KIND_STATUS",
    "KIND_TABLE",
    "EvidenceUnit",
    "RoundReport",
    "BOARD_BASE_CHARS",
    "BOARD_CAP_CHARS",
    "BOARD_STEP_CHARS",
    "SCORE_MIN",
    "ingest_observation",
    "build_board_view",
    "render_fc_messages",
    "render_text_history_lines",
    "render_plan_observation",
    "FETCH_TOOL_NAME",
    "fetch_tool_definition",
    "handle_fetch_evidence",
    "is_fetch_call",
]
