# -*- coding: utf-8 -*-
"""证据板数据模型：Unit（证据单元）与轮次报告。

Unit 是请求级状态（``state["evidence_units"]``）中的纯 dict 友好结构；
证据板本身不落 state——它是每轮从全量 Unit 重算的发送视图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

# 单元种类
KIND_CONTENT = "content"   # 普通内容
KIND_TABLE = "table"       # 结构化表格（整表不可拆）
KIND_ERROR = "error"       # 工具失败（恒保留，不参与打分）
KIND_STATUS = "status"     # 无结果/空/导航性状态（恒保留，不参与打分）


@dataclass
class EvidenceUnit:
    """一条可独立引用、可独立回取的证据原子。

    ``text`` 只做确定性切分，绝不允许 LLM 改写。
    """

    uid: str
    tool_name: str
    round_idx: int
    block_idx: int
    text: str
    source: str = ""
    ref: Dict[str, Any] = field(default_factory=dict)
    kind: str = KIND_CONTENT
    score: float = 0.0
    simhash: int = 0
    selected: bool = False
    truncated: bool = False
    fetched: bool = False
    exempt: bool = False
    dupe_of: str = ""
    also_from: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "tool_name": self.tool_name,
            "round_idx": self.round_idx,
            "block_idx": self.block_idx,
            "text": self.text,
            "source": self.source,
            "ref": dict(self.ref),
            "kind": self.kind,
            "score": round(float(self.score), 4),
            "simhash": int(self.simhash),
            "selected": bool(self.selected),
            "truncated": bool(self.truncated),
            "fetched": bool(self.fetched),
            "exempt": bool(self.exempt),
            "dupe_of": str(self.dupe_of or ""),
            "also_from": list(self.also_from),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceUnit":
        return cls(
            uid=str(data.get("uid") or ""),
            tool_name=str(data.get("tool_name") or ""),
            round_idx=int(data.get("round_idx") or 0),
            block_idx=int(data.get("block_idx") or 0),
            text=str(data.get("text") or ""),
            source=str(data.get("source") or ""),
            ref=dict(data.get("ref") or {}),
            kind=str(data.get("kind") or KIND_CONTENT),
            score=float(data.get("score") or 0.0),
            simhash=int(data.get("simhash") or 0),
            selected=bool(data.get("selected")),
            truncated=bool(data.get("truncated")),
            fetched=bool(data.get("fetched")),
            exempt=bool(data.get("exempt")),
            dupe_of=str(data.get("dupe_of") or ""),
            also_from=list(data.get("also_from") or []),
        )


@dataclass
class RoundReport:
    """一轮证据处理的计数（trace 与尾注用）。"""

    round_idx: int
    new: int = 0
    duplicated: int = 0
    low_score: int = 0
    on_board: int = 0
    board_chars: int = 0
    exempt: int = 0
    no_new_evidence: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_idx,
            "new": self.new,
            "duplicated": self.duplicated,
            "low_score": self.low_score,
            "on_board": self.on_board,
            "board_chars": self.board_chars,
            "exempt": self.exempt,
            "no_new_evidence": self.no_new_evidence,
        }
