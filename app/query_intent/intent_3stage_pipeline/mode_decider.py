"""Agent 编排模式决策器（ModeDecider）—— 完全规则化版本。

决策逻辑（自顶向下，命中则直接返回，不再往下走）：

  1) explicit_plan_hint（显式步骤提示）优先级最高
        → 走 plan_execute，decision_source = "explicit_hint"

  2) 规则阈值层（Rule Threshold）：
        estimated_steps >= MODE_DECISION_STEP_THRESHOLD
        或 estimated_tool_calls >= MODE_DECISION_TOOL_THRESHOLD
        或 has_multi_step_dependency=True
        → 走 plan_execute，decision_source = "rule_threshold"

  3) 意图层模式倾向信号（intent_prefer_mode，来自意图树节点静态字段，
     非 LLM 产出）：若强阈值未命中且意图主节点 prefer_mode="plan_execute"
        → 走 plan_execute，decision_source = "intent_prefer_mode"

  4) 兜底：走 react，decision_source = "rule_threshold"

性能说明：本决策器已彻底移除 LLM 兜底层（原 _decide_by_llm_with_fallback），
生产环境命中率低于 5% 且引入 500ms~1s 的额外首字延迟，收益为负；
现全部决策为纯内存计算，零网络调用。

本模块与 AgentOrchestrator 保持完全零侵入：不读取 orchestrator 内部状态，
仅依赖入参里的 AgentRewriteResult / AgentIntents。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from app.query_intent.intent_dto import (
    AgentIntents,
    AgentRewriteResult,
    ModeDecision,
    OrchestrationModeLiteral,
)
from app.query_intent.rag_constant import (
    MODE_DECISION_DEFAULT_MODE,
    MODE_DECISION_STEP_THRESHOLD,
    MODE_DECISION_TOOL_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class ModeDecider:
    """Agent 编排模式决策器：纯规则决策，零 LLM 调用。

    Attributes:
        step_threshold: 规则层步骤数阈值（默认 = rag_constant）。
        tool_threshold: 规则层工具调用次数阈值（默认 = rag_constant）。
        default_mode: 兜底模式（默认 react）。
    """

    step_threshold: int = MODE_DECISION_STEP_THRESHOLD
    tool_threshold: int = MODE_DECISION_TOOL_THRESHOLD
    default_mode: OrchestrationModeLiteral = MODE_DECISION_DEFAULT_MODE  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def decide_orchestration_mode(
        self,
        rewrite_result: AgentRewriteResult,
        intents_result: Optional[AgentIntents] = None,
        available_tool_ids: Optional[List[str]] = None,
    ) -> ModeDecision:
        """在 Agent 改写结果（+ 意图聚合结果）之上做纯规则模式决策。

        Args:
            rewrite_result: 改写层输出（复杂度 + explicit_plan_hint 是关键）。
            intents_result: 意图聚合层输出（仅读取 raw_slots.intent_prefer_mode
                静态信号；可为 None）。
            available_tool_ids: 兼容保留参数（历史签名），当前决策不使用。

        Returns:
            ModeDecision：最终模式、置信度、决策来源、reason、可选 hint。
        """
        # 第一层：显式步骤提示（最高优先级）
        decision = self._try_match_explicit_plan_hint(rewrite_result)
        if decision is not None:
            return decision

        # 第二层：规则阈值层
        decision = self._decide_by_rule_threshold(rewrite_result)
        if decision is not None and decision.mode == "plan_execute":
            return decision

        # 第三层：意图树静态模式倾向（非 LLM；task-todo-plan 等节点带 prefer_mode）
        prefer_mode_decision = self._decide_by_intent_prefer_mode(intents_result)
        if prefer_mode_decision is not None:
            return prefer_mode_decision

        # 兜底：react（_decide_by_rule_threshold 的 react 分支已带完整 reason）
        if decision is not None:
            return decision
        return ModeDecision(
            mode=self.default_mode,  # type: ignore[arg-type]
            confidence=0.6,
            reason="规则层无复杂度输入，走默认 react。",
            decision_source="rule_threshold",
            initial_plan_hint=None,
            first_tool_hint=None,
        )

    # ------------------------------------------------------------------
    # Layer 1: explicit_plan_hint
    # ------------------------------------------------------------------
    def _try_match_explicit_plan_hint(
        self,
        rewrite_result: AgentRewriteResult,
    ) -> Optional[ModeDecision]:
        explicit_hint_text: Optional[str] = rewrite_result.explicit_plan_hint
        if not explicit_hint_text or not str(explicit_hint_text).strip():
            return None
        # 用户已经明确说出了"先...再...最后..."/步骤 1..N 等表述：
        # 一律计划模式，置信度给 0.92（规则硬命中）。
        return ModeDecision(
            mode="plan_execute",  # type: ignore[typeddict-item]
            confidence=0.92,
            reason=(
                f"用户问题包含显式步骤提示：{str(explicit_hint_text)[:60]!r}，"
                "必须走 plan_execute（PlannerAgent plan→execute→replan 循环）。"
            ),
            decision_source="explicit_hint",
            initial_plan_hint=str(explicit_hint_text).strip(),
            first_tool_hint=None,
        )

    # ------------------------------------------------------------------
    # Layer 2: Rule Threshold
    # ------------------------------------------------------------------
    def _decide_by_rule_threshold(
        self,
        rewrite_result: AgentRewriteResult,
    ) -> Optional[ModeDecision]:
        complexity = rewrite_result.complexity_analysis
        if complexity is None:
            return None
        steps_value: int = complexity.estimated_steps or 1
        tools_value: int = complexity.estimated_tool_calls or 0
        has_dependency_flag: bool = bool(complexity.has_multi_step_dependency)
        has_external_flag: bool = bool(complexity.has_external_data_dependency)
        need_creative_flag: bool = bool(complexity.need_creative_output)

        # 强触发条件（任一项命中即 plan_execute）
        strong_hit_reasons: List[str] = []
        if steps_value >= self.step_threshold:
            strong_hit_reasons.append(
                f"预估步骤数={steps_value} >= 阈值={self.step_threshold}"
            )
        if tools_value >= self.tool_threshold:
            strong_hit_reasons.append(
                f"预估工具调用次数={tools_value} >= 阈值={self.tool_threshold}"
            )
        if has_dependency_flag:
            strong_hit_reasons.append("存在明确的多步骤依赖关系")

        if strong_hit_reasons:
            return ModeDecision(
                mode="plan_execute",  # type: ignore[typeddict-item]
                confidence=0.82,
                reason="规则层命中：" + "；".join(strong_hit_reasons),
                decision_source="rule_threshold",
                initial_plan_hint=None,
                first_tool_hint=None,
            )

        # 弱触发 / 默认 react
        weak_reasons: List[str] = []
        if steps_value <= 1 and tools_value <= 0 and not has_external_flag:
            weak_reasons.append("纯对话/无需工具，一步即可完成")
        if need_creative_flag and steps_value <= 1:
            weak_reasons.append("创造性输出但不依赖多步骤编排，ReAct 自然发散即可")
        react_reason: str = (
            "；".join(weak_reasons)
            if weak_reasons
            else f"规则层未命中 plan 强触发：steps={steps_value}, tools={tools_value}"
        )
        return ModeDecision(
            mode="react",  # type: ignore[typeddict-item]
            confidence=0.7,
            reason=react_reason,
            decision_source="rule_threshold",
            initial_plan_hint=None,
            first_tool_hint=None,
        )

    # ------------------------------------------------------------------
    # Layer 3: 意图层静态 prefer_mode 信号（非 LLM）
    # ------------------------------------------------------------------
    def _decide_by_intent_prefer_mode(
        self,
        intents_result: Optional[AgentIntents],
    ) -> Optional[ModeDecision]:
        """读取意图聚合 raw_slots["intent_prefer_mode"]（意图树节点静态字段）。

        该信号源自 DB/工厂树叶子节点的 prefer_mode 属性（如 task-todo-plan），
        属于静态规则数据，与被移除的 LLM 兜底层无关。
        """
        if intents_result is None:
            return None
        raw_slots = getattr(intents_result, "raw_slots", None) or {}
        prefer_mode = str(raw_slots.get("intent_prefer_mode") or "").strip()
        if prefer_mode != "plan_execute":
            return None
        primary_intent_text = str(
            getattr(intents_result, "primary_intent_text", "") or ""
        )
        return ModeDecision(
            mode="plan_execute",  # type: ignore[typeddict-item]
            confidence=0.75,
            reason=(
                f"意图主节点（{primary_intent_text}）静态标记 prefer_mode=plan_execute，"
                "走计划模式。"
            ),
            decision_source="intent_prefer_mode",
            initial_plan_hint=None,
            first_tool_hint=None,
        )
