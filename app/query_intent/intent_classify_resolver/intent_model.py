# ============================================================================
# intent_models.py - 核心数据模型
# ============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.query_intent.intent_data_base import IntentLevel, IntentKind


@dataclass
class IntentNode:
    """意图树节点。

    Agent 编排语义（本轮改造新增）：
      - agent_tool_names：该意图场景下"大概率能拿到结果"的注册工具名
        （与 rag_constant.REGISTERED_ENABLED_TOOL_NAMES 对齐），按优先级排序；
      - tool_usage_hint：给编排层 / React / Planner 的调用提示（传参要点、时机）；
      - prefer_mode：该意图偏好的推理模式（"react" / "plan_execute"），
        仅作 ModeDecider 静态意图偏好层的参考信号，不覆盖规则层硬判断。

    兼容性说明：kb_id / collection_name(s) / mcp_tool_id / prompt_template 等
    旧 RAG 字段保留——DB mapper（_do_to_node）与旧 RAG 链路仍在使用；
    新工厂树叶子节点不再填充 kb_id / collection_name。
    """

    id: Optional[str] = None
    kb_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    level: Optional[IntentLevel] = None
    parent_id: Optional[str] = None
    examples: list[str] = field(default_factory=list)
    children: list["IntentNode"] = field(default_factory=list)
    embedding: Optional[list[float]] = None
    full_path: str = ""
    kind: Optional[IntentKind] = IntentKind.KB
    collection_name: Optional[str] = None
    collection_names: list[str] = field(default_factory=list)
    mcp_tool_id: Optional[str] = None
    top_k: Optional[int] = None
    prompt_snippet: Optional[str] = None
    prompt_template: Optional[str] = None
    param_prompt_template: Optional[str] = None
    # ---- Agent 编排工具路由字段（新增） ----
    agent_tool_names: list[str] = field(default_factory=list)
    tool_usage_hint: Optional[str] = None
    prefer_mode: Optional[str] = None

    def is_leaf(self) -> bool:
        return self.children is None or len(self.children) == 0

    def is_kb(self) -> bool:
        return self.kind is None or self.kind == IntentKind.KB

    def is_mcp(self) -> bool:
        return self.kind == IntentKind.MCP

    def is_system(self) -> bool:
        return self.kind == IntentKind.SYSTEM

    def get_effective_agent_tool_names(self) -> list[str]:
        """该节点最终生效的 Agent 工具名列表（去重、去空白）。

        优先级：agent_tool_names > mcp_tool_id（DB 旧数据兜底）。
        """
        normalized: list[str] = []
        seen: set[str] = set()
        if self.agent_tool_names is not None:
            for value in self.agent_tool_names:
                if value is None:
                    continue
                trimmed = value.strip()
                if trimmed and trimmed not in seen:
                    seen.add(trimmed)
                    normalized.append(trimmed)
        if (
            len(normalized) == 0
            and self.mcp_tool_id is not None
            and self.mcp_tool_id.strip()
        ):
            normalized.append(self.mcp_tool_id.strip())
        return normalized

    def get_effective_collection_names(self) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        if self.collection_names is not None:
            for value in self.collection_names:
                if value is None:
                    continue
                trimmed = value.strip()
                if trimmed and trimmed not in seen:
                    seen.add(trimmed)
                    normalized.append(trimmed)
        if len(normalized) == 0 and self.collection_name is not None and self.collection_name.strip():
            trimmed = self.collection_name.strip()
            if trimmed not in seen:
                normalized.append(trimmed)
        return normalized


@dataclass
class NodeScore:
    node: Optional[IntentNode] = None
    score: float = 0.0