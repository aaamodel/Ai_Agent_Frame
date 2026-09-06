"""Agent 编排侧 Pipeline 总入口。

将「问题改写 → 意图聚合 → 模式决策」三段串成稳定可复用的同步调用链，
并产出最终的 AgentQueryIntentPipelineOutput：供 /chat/with_agent 路由
直接读 mode / final_user_input / allowed_tools_final / merged_slots，再
组装成 AgentOrchestrator.run(mode=..., intent=IntentContext(...), ...) 的
真实调用参数。

设计原则：
  - **同步**：与 query_intent 现有大量服务保持一致；在 async 路由中通过
    asyncio.to_thread(pipeline.run, ...) 包装调用即可。
  - **零侵入执行层**：不依赖 / 不读取 / 不修改 AgentOrchestrator 及
    ReAct/Planner/Reflection 三个引擎的内部状态。
  - **严格容错**：每一段都有 fallback，永远不抛异常向上冒泡。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.query_intent.intent_classify_resolver.intent_model import NodeScore
from app.query_intent.intent_classify_resolver.intent_classify import AgentIntentAggregator
from app.query_intent.intent_classify_resolver.intent_resolver import IntentResolver
from app.query_intent.intent_data_base import AgentChatContext, IntentChatMessage
from app.query_intent.intent_dto import (
    AgentIntents,
    AgentQueryIntentPipelineOutput,
    AgentRewriteResult,
    ModeDecision,
    OrchestrationModeLiteral,
)
from app.query_intent.intent_3stage_pipeline.mode_decider import ModeDecider
from app.query_intent.rag_constant import (
    PIPELINE_INFRASTRUCTURE_TOOL_SET,
    REGISTERED_ENABLED_TOOL_NAMES,
)
from app.query_intent.rewrite.query_rewrite import AgentQueryRewriteService
from trace_to_markdown import trace_to_markdown

logger = logging.getLogger(__name__)


@dataclass
class AgentQueryIntentPipeline:
    """Agent 编排 Pipeline：改写 → 意图 → 决策，三段串联。

    Attributes:
        rewrite_service: Agent 改写服务实现。
        intent_resolver: 意图解析器（同时负责 resolve_for_agent 入口）。
        intent_aggregator: Agent 专用意图聚合器；intent_resolver 内部若未传
            aggregator，将优先复用此实例。
        mode_decider: 模式决策器（调整三后完全规则化：explicit_hint →
            rule_threshold → intent_prefer_mode → react 兜底，无 LLM）。
    """

    rewrite_service: AgentQueryRewriteService
    intent_resolver: IntentResolver
    intent_aggregator: AgentIntentAggregator
    mode_decider: ModeDecider

    # ------------------------------------------------------------------
    # Public Core API
    # ------------------------------------------------------------------
    def run(
        self,
        user_question: str,
        available_tool_ids: List[str],
        session_id: str,
        history: Optional[List[IntentChatMessage]] = None,
        available_skills: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentQueryIntentPipelineOutput:
        """完整 Pipeline 同步调用入口。

        Args:
            user_question: 用户原始问题。
            available_tool_ids: 当前请求 ToolRegistry.list_tool_names() 快照。
            session_id: Agent 会话 ID。
            history: 最近多轮对话历史（来自 ShortTermMemory）。
            available_skills: 当前请求高级技能快照（{name, description, ...}），
                用于改写阶段注入技能清单让 LLM 挑选编排层可用技能。

        Returns:
            AgentQueryIntentPipelineOutput：三段产物 + 最终喂给
            AgentOrchestrator.run 的参数（final_user_input / mode_decision.mode
            / allowed_tools_final / merged_slots）。
        """
        agent_context: AgentChatContext = AgentChatContext(
            original_user_question=user_question or "",
            session_id=session_id or "",
            available_tool_ids=list(available_tool_ids or []),
            conversation_history=list(history or []),
            available_skills=list(available_skills or []),
        )
        return self._execute_three_stage_pipeline(agent_context)

    @trace_to_markdown(output_file="build_orchestrator_input.md")
    def build_orchestrator_input(
        self,
        pipeline_output: AgentQueryIntentPipelineOutput,
        force_override_mode: Optional[OrchestrationModeLiteral] = None,
    ) -> Tuple[OrchestrationModeLiteral, Dict[str, Any], str]:
        """把 Pipeline 输出 → AgentOrchestrator.run 所需要的三元组。

        目的：让 chat.py 路由只需要做"拆三元组"而不关心 IntentContext 内部
        字段命名，避免改动 orchestrator.py。

        Args:
            pipeline_output: run() 返回值。
            force_override_mode: （对应疑问-G-1）若前端 strategy 显式指定模式，
                这里强覆盖 Pipeline 决策（但仍会把原始决策写入 slots 便于
                debug）。

        Returns:
            (mode: str, intent_context_payload: dict, effective_user_input: str)
              mode:               直接喂 orchestrator.run(mode=...)
              intent_context_payload: 满足 IntentContext(preferred_mode,
                                   intent, confidence, allowed_tools, slots) 字段名，直接
                                   解包构造 IntentContext(**intent_context_payload)
              effective_user_input: orchestrator.run(user_input=...)
        """
        original_mode: OrchestrationModeLiteral = pipeline_output.mode_decision.mode
        final_mode: OrchestrationModeLiteral = (
            force_override_mode
            if force_override_mode in ("react", "plan_execute")
            else original_mode
        )
        merged_slots: Dict[str, Any] = dict(pipeline_output.merged_slots or {})
        merged_slots.setdefault(
            "pipeline_original_mode_decision",
            {
                "mode": original_mode,
                "confidence": pipeline_output.mode_decision.confidence,
                "reason": pipeline_output.mode_decision.reason,
                "decision_source": pipeline_output.mode_decision.decision_source,
            },
        )
        if force_override_mode:
            merged_slots["mode_override_by_strategy"] = force_override_mode

        # ---- 【待重写-⑦-A · 顺序-1】G-1 模式强覆盖后：清理与新模式不匹配的 hint 字段 ----
        # 例：Pipeline 原本决策 plan_execute → slots 有 initial_plan_hint；
        # 现在被前端 strategy=react 强覆盖，再保留 initial_plan_hint 会让
        # ReAct 读到 Planner 冷启动 hint，产生语义误导。
        if force_override_mode == "react":
            # ReAct 不需要 initial_plan_hint（Planner 专属），删除；保留 first_tool_hint
            merged_slots.pop("initial_plan_hint", None)
        elif force_override_mode == "plan_execute":
            # PlanExecute 不需要 first_tool_hint（ReAct 专属），删除；保留 initial_plan_hint
            merged_slots.pop("first_tool_hint", None)
        # -----------------------------------------------------------------

        # 【待补充-⑥-A · 顺序-5】显式填充 confidence：
        # IntentContext 有 confidence 字段（默认 1.0），这里使用意图聚合的置信度，
        # 避免追踪面板永远显示 1.0 误导人。值非法/空时兜底 1.0。
        raw_confidence: Any = getattr(
            pipeline_output.intents_result, "aggregated_confidence", None
        )
        try:
            effective_confidence: float = float(raw_confidence) if raw_confidence is not None else 1.0
        except (TypeError, ValueError):
            effective_confidence = 1.0
        if not 0.0 <= effective_confidence <= 1.0:
            # 防御归一化：超过 [0,1] 区间时 clamp 到合法范围
            effective_confidence = max(0.0, min(1.0, effective_confidence)) or 1.0

        intent_context_payload: Dict[str, Any] = {
            "preferred_mode": final_mode,
            "intent": pipeline_output.intents_result.primary_intent_text
            or "general",
            "confidence": effective_confidence,
            "allowed_tools": list(pipeline_output.allowed_tools_final or []),
            "slots": merged_slots,
        }
        # 【新增 · 技能感知】把改写阶段最终选定的技能名显式落到 slots，
        # 供编排层用得意技能号令可用工具（若 _merge_slots 已写入则幂等覆盖）。
        final_skills: List[str] = list(
            pipeline_output.selected_skills_final
            or pipeline_output.rewrite_result.suggested_skills
            or []
        )
        if final_skills:
            merged_slots.setdefault("selected_skills_final", final_skills)
        effective_user_input: str = (
            pipeline_output.final_user_input
            or pipeline_output.original_user_question
        )
        return final_mode, intent_context_payload, effective_user_input

    # ------------------------------------------------------------------
    # 三段执行总控
    # ------------------------------------------------------------------

    @trace_to_markdown(output_file="_execute_three_stage_pipeline.md")
    def _execute_three_stage_pipeline(
        self,
        agent_context: AgentChatContext,
    ) -> AgentQueryIntentPipelineOutput:
        original_question_text: str = agent_context.original_user_question or ""
        fallback_output: AgentQueryIntentPipelineOutput = (
            AgentQueryIntentPipelineOutput(
                rewrite_result=AgentRewriteResult(
                    rewritten_question=original_question_text,
                    should_split=False,
                    sub_questions=[original_question_text] if original_question_text else [],
                ),
                intents_result=AgentIntents(primary_intent_text="general"),
                mode_decision=ModeDecision(
                    mode="react",  # type: ignore[typeddict-item]
                    confidence=0.0,
                    reason="Pipeline 全链路异常兜底：默认 react",
                    decision_source="fallback_default",
                ),
                final_user_input=original_question_text,
                allowed_tools_final=self._finalize_allowed_tools(
                    rewrite_suggested_tools=[],
                    intent_hit_tool_names=[],
                    request_snapshot_tools=list(
                        agent_context.available_tool_ids or []
                    ),
                ),
                merged_slots={},
                original_user_question=original_question_text,
                session_id=agent_context.session_id,
            )
        )

        # Stage 1: 改写
        rewrite_result: AgentRewriteResult = self._stage_1_rewrite(agent_context)

        # Stage 2: 意图聚合（改写后的主问题 + 子问题）
        # 调整一：组合调用（AgentCombinedRewriteIntentService）已在 Stage1 单次
        # LLM 中产出逐问题意图打分（precomputed_intent_scores），此处零 LLM
        # 直接聚合；None 时回退旧链路（intent_resolver 独立 LLM 意图识别）。
        try:
            precomputed_scores: Optional[Dict[str, Any]] = getattr(
                rewrite_result, "precomputed_intent_scores", None
            )
            if precomputed_scores:
                intents_result: AgentIntents = (
                    self.intent_aggregator.aggregate_for_agent_precomputed(
                        primary_question=rewrite_result.rewritten_question,
                        sub_questions=(
                            list(rewrite_result.sub_questions or [])
                            if rewrite_result.should_split
                            else []
                        ),
                        precomputed_scores=precomputed_scores,
                    )
                )
            else:
                # 零 LLM 兜底：组合调用未给出可用意图打分时，直接复用已召回的
                # 向量候选按位次打分聚合，避免再发一次 intent_analysis LLM 调用。
                intents_result = self._aggregate_via_vector_recall(rewrite_result)
                if intents_result is None:
                    intents_result = self.intent_resolver.resolve_for_agent(
                        rewrite_result=rewrite_result,
                        aggregator=self.intent_aggregator,
                    )
        except Exception as intent_error:
            logger.warning(
                "意图解析阶段异常，降级为 general 意图。question=%s，异常=%s",
                original_question_text,
                intent_error,
                exc_info=True,
            )
            intents_result = AgentIntents(primary_intent_text="general")

        # Stage 3: 模式决策
        try:
            mode_decision: ModeDecision = self.mode_decider.decide_orchestration_mode(
                rewrite_result=rewrite_result,
                intents_result=intents_result,
                available_tool_ids=list(agent_context.available_tool_ids or []),
            )
        except Exception as mode_error:
            logger.warning(
                "模式决策阶段异常，回退到默认 react。question=%s，异常=%s",
                original_question_text,
                mode_error,
                exc_info=True,
            )
            mode_decision = ModeDecision(
                mode=fallback_output.mode_decision.mode,
                confidence=0.0,
                reason=f"模式决策异常兜底：{mode_error}",
                decision_source="fallback_default",
            )

        # 【新增 · 技能感知】改写阶段 LLM 选出的技能名 → Pipeline 各阶段统一传播。
        # 去重、去空，作为 intents_result.related_skill_names /
        # mode_decision.eligible_skill_names / selected_skills_final 的统一数据源，
        # 供编排层用得意技能号令可用工具。
        selected_skills_final: List[str] = []
        for skill_name in (rewrite_result.suggested_skills or []):
            candidate_skill: str = str(skill_name).strip()
            if candidate_skill and candidate_skill not in selected_skills_final:
                selected_skills_final.append(candidate_skill)
        intents_result.related_skill_names = list(selected_skills_final)
        mode_decision.eligible_skill_names = list(selected_skills_final)

        # Stage 4: 合并白名单 & merged_slots
        try:
            intent_hit_tool_names: List[str] = (
                self.intent_aggregator.collect_intent_tool_names_from_intents(
                    intents_result
                )
            )
        except Exception as intent_tool_error:
            logger.warning(
                "抽取意图树命中工具名失败，忽略（已作为空列表处理）：%s",
                intent_tool_error,
            )
            intent_hit_tool_names = []

        allowed_tools_final: List[str] = self._finalize_allowed_tools(
            rewrite_suggested_tools=list(rewrite_result.suggested_tools or []),
            intent_hit_tool_names=intent_hit_tool_names,
            request_snapshot_tools=list(agent_context.available_tool_ids or []),
        )

        merged_slots: Dict[str, Any] = self._merge_slots(
            intents_result=intents_result,
            mode_decision=mode_decision,
            rewrite_result=rewrite_result,
            intent_hit_tool_names=intent_hit_tool_names,
        )

        final_user_input: str = (
            rewrite_result.rewritten_question.strip()
            if rewrite_result.rewritten_question and rewrite_result.rewritten_question.strip()
            else original_question_text
        )

        logger.info(
            "Agent Pipeline 执行完成：session_id=%s\n  原始问题：%s\n  "
            "最终输入：%s\n  主意图：%s (conf=%.3f)\n  "
            "模式决策：mode=%s，来源=%s，置信度=%.2f，理由=%s\n  "
            "允许工具数=%d：%s",
            agent_context.session_id,
            original_question_text,
            final_user_input,
            intents_result.primary_intent_text,
            intents_result.aggregated_confidence,
            mode_decision.mode,
            mode_decision.decision_source,
            mode_decision.confidence,
            mode_decision.reason,
            len(allowed_tools_final),
            allowed_tools_final,
        )

        return AgentQueryIntentPipelineOutput(
            rewrite_result=rewrite_result,
            intents_result=intents_result,
            mode_decision=mode_decision,
            final_user_input=final_user_input,
            allowed_tools_final=allowed_tools_final,
            merged_slots=merged_slots,
            original_user_question=original_question_text,
            session_id=agent_context.session_id,
            selected_skills_final=selected_skills_final,
        )

    # ------------------------------------------------------------------
    # Stage 1 Wrapper（异常 → fallback rewrite）
    # ------------------------------------------------------------------
    def _stage_1_rewrite(
        self,
        agent_context: AgentChatContext,
    ) -> AgentRewriteResult:
        original_text: str = agent_context.original_user_question or ""
        fallback_rewrite = AgentRewriteResult(
            rewritten_question=original_text,
            should_split=False,
            sub_questions=[original_text] if original_text else [],
        )
        try:
            rewrite_outcome: AgentRewriteResult = self.rewrite_service.rewrite_for_agent(
                agent_context
            )
            # 结果完整性兜底
            if not rewrite_outcome.rewritten_question and original_text:
                rewrite_outcome.rewritten_question = original_text
            if not rewrite_outcome.sub_questions:
                rewrite_outcome.should_split = False
                rewrite_outcome.sub_questions = [
                    rewrite_outcome.rewritten_question or original_text
                ]
            return rewrite_outcome
        except Exception as rewrite_error:
            logger.warning(
                "改写阶段异常，使用 original 问题兜底。question=%r，异常=%s",
                original_text,
                rewrite_error,
                exc_info=True,
            )
            return fallback_rewrite

    # ------------------------------------------------------------------
    # 零 LLM 意图聚合兜底（调整：省去第二次串行意图识别 LLM 调用）
    # ------------------------------------------------------------------
    def _aggregate_via_vector_recall(
        self,
        rewrite_result: AgentRewriteResult,
    ) -> Optional[AgentIntents]:
        """零 LLM 意图聚合：复用向量 Top-K 召回候选，按位次打分后聚合为 AgentIntents。

        组合调用（Stage1）若未产出可用意图打分（日志表现为「预计算意图问题数=0」），
        原实现会再发一次 intent_analysis LLM（约 4s，日志中结果仍为 general）。
        此处改为：优先复用组合调用已召回的候选节点（vector_candidate_nodes），
        否则按需重新检索；把候选按位次映射为 NodeScore（score = 1 - rank/top_k），
        再走 aggregate_for_agent_precomputed 完成零 LLM 聚合。

        Returns:
            AgentIntents；向量召回不可用 / 聚合失败时返回 None，调用方回退旧
            resolve_for_agent（LLM）链路。
        """
        retriever = getattr(self.rewrite_service, "vector_retriever", None)
        primary_question = (rewrite_result.rewritten_question or "").strip()
        if not primary_question:
            return None

        nodes = getattr(rewrite_result, "vector_candidate_nodes", None)
        if not nodes:
            # 组合调用被跳过/未召回（父类两段链路）——此处按需检索（1 次 query embedding）
            if retriever is None:
                return None
            try:
                nodes = retriever.retrieve(primary_question)
            except Exception as retr_err:
                logger.warning(
                    "意图向量零LLM聚合：检索异常，回退旧链路。question=%r err=%s",
                    primary_question,
                    retr_err,
                )
                return None
        if not nodes:
            return None

        top_k: int = max(1, int(getattr(retriever, "_top_k", 8)))
        node_scores: List[NodeScore] = [
            NodeScore(node=node, score=max(0.0, 1.0 - (rank / top_k)))
            for rank, node in enumerate(nodes)
        ]
        try:
            return self.intent_aggregator.aggregate_for_agent_precomputed(
                primary_question=primary_question,
                sub_questions=[],
                precomputed_scores={primary_question: node_scores},
            )
        except Exception as agg_err:
            logger.warning(
                "意图向量零LLM聚合失败，回退旧链路。question=%r err=%s",
                primary_question,
                agg_err,
            )
            return None

    # ------------------------------------------------------------------
    # 白名单合并（三段 union + infrastructure 兜底 + 与 REGISTERED 交集）
    # ------------------------------------------------------------------
    def _finalize_allowed_tools(
        self,
        rewrite_suggested_tools: List[str],
        intent_hit_tool_names: List[str],
        request_snapshot_tools: List[str],
    ) -> List[str]:
        """三段白名单合并（顺序稳定 + 交集过滤 + 基础设施保底）。

        合并策略（从高到低）：
          1) 改写阶段 suggested_tools（LLM 对当前问题的判断）
          2) 意图树命中节点的 agent_tool_names（编排层大概率能拿到结果的工具）
          3) infrastructure 3 工具兜底（Skill System 读 SKILL.md）
        过滤：最终结果必须是 "REGISTERED_ENABLED ∪ request 快照" 的集合成员，
             防止 LLM 幻觉或意图树脏数据。
        """
        allowed_tool_universe: set[str] = set(REGISTERED_ENABLED_TOOL_NAMES) | {
            tool_name.strip()
            for tool_name in (request_snapshot_tools or [])
            if isinstance(tool_name, str) and tool_name.strip()
        }

        ordered_result: Dict[str, None] = {}
        for source in (
            rewrite_suggested_tools,
            intent_hit_tool_names,
            PIPELINE_INFRASTRUCTURE_TOOL_SET,
        ):
            for tool_name in source or []:
                if not isinstance(tool_name, str):
                    continue
                cleaned_name: str = tool_name.strip()
                if not cleaned_name:
                    continue
                if cleaned_name not in allowed_tool_universe:
                    continue
                if cleaned_name in ordered_result:
                    continue
                ordered_result[cleaned_name] = None

        # 防御：极端情况下（例如 universe 为空），至少保留 infrastructure 3 项
        if not ordered_result:
            for infra_name in PIPELINE_INFRASTRUCTURE_TOOL_SET:
                ordered_result[infra_name] = None

        return list(ordered_result.keys())

    @staticmethod
    def _merge_slots(
        intents_result: AgentIntents,
        mode_decision: ModeDecision,
        rewrite_result: AgentRewriteResult,
        intent_hit_tool_names: List[str],
    ) -> Dict[str, Any]:
        """汇总 slots，供 IntentContext.slots 使用。

        注意：疑问-F 推荐 F-1：initial_plan_hint 写进 slots，不侵入 PlannerAgent
        形参。Planner 在渲染 user_message 时可通过
        slots["initial_plan_hint"] 读取后拼在末尾。
        """
        merged: Dict[str, Any] = dict(intents_result.raw_slots or {})

        # 改写阶段 hint（显式步骤）
        if rewrite_result.explicit_plan_hint:
            merged["explicit_plan_hint"] = rewrite_result.explicit_plan_hint

        # 模式决策：plan 的步骤 hint / react 的首工具 hint
        if mode_decision.mode == "plan_execute" and mode_decision.initial_plan_hint:
            # 疑问-F F-1：写 intent_context.slots.initial_plan_hint
            merged["initial_plan_hint"] = mode_decision.initial_plan_hint
        if mode_decision.mode == "react" and mode_decision.first_tool_hint:
            merged["first_tool_hint"] = mode_decision.first_tool_hint

        # debug / trace 信息：保留原始决策产物，便于前端 debug 面板
        merged["pipeline_mode_meta"] = {
            "mode": mode_decision.mode,
            "confidence": mode_decision.confidence,
            "reason": mode_decision.reason,
            "decision_source": mode_decision.decision_source,
        }
        # 【待重写-⑤-A · 顺序-6】键名对齐链路图：pipeline_complexity → pipeline_complexity_meta
        # 【待重写·可选加固 · 顺序-7】同时加一层 rewrite_result.complexity_analysis None 防御：
        #   AgentRewriteResult 默认 factory 不会是 None，但上游如有手工赋值路径，
        #   用 getattr + 默认值兜底，避免 AttributeError 中断 Pipeline。
        # 字段名严格对齐 TaskComplexityAnalysis DTO：
        #   estimated_steps / estimated_tool_calls / has_multi_step_dependency /
        #   has_external_data_dependency / need_creative_output / reasoning_notes
        ca_obj: Any = getattr(rewrite_result, "complexity_analysis", None)
        merged["pipeline_complexity_meta"] = {
            "estimated_steps": int(
                getattr(ca_obj, "estimated_steps", 1) or 1
            ),
            "estimated_tool_calls": int(
                getattr(ca_obj, "estimated_tool_calls", 0) or 0
            ),
            "has_multi_step_dependency": bool(
                getattr(ca_obj, "has_multi_step_dependency", False)
            ),
            "has_external_data_dependency": bool(
                getattr(ca_obj, "has_external_data_dependency", False)
            ),
            "need_creative_output": bool(
                getattr(ca_obj, "need_creative_output", False)
            ),
            "reasoning_notes": str(
                getattr(ca_obj, "reasoning_notes", "") or ""
            ),
        }
        merged["pipeline_rewrite_meta"] = {
            "should_split": rewrite_result.should_split,
            "sub_questions_count": len(rewrite_result.sub_questions or []),
            "suggested_tools": list(rewrite_result.suggested_tools or []),
            "suggested_skills": list(rewrite_result.suggested_skills or []),
        }
        # 【新增 · 技能感知】改写阶段选定的技能名透传到 slots，供编排层号令工具。
        if rewrite_result.suggested_skills:
            merged["suggested_skills"] = list(rewrite_result.suggested_skills or [])
        if mode_decision.eligible_skill_names:
            merged["eligible_skill_names"] = list(mode_decision.eligible_skill_names or [])
        merged["pipeline_intent_hit_tool_names"] = list(intent_hit_tool_names or [])

        # 【改进点 3 · 子问题意图感知】把按子问题维度的意图明细也透传进 slots，
        # 供编排层 Plan-Execute 做真正的"分而治之"约束注入（此前该字段仅在
        # intents_result 上闲置，未被任何下游消费）。
        sub_intent_detail_list: List[Any] = list(intents_result.per_sub_question_intents or [])
        if sub_intent_detail_list:
            merged["per_sub_questions"] = [
                str(getattr(sq, "sub_question", "") or "") for sq in sub_intent_detail_list
            ]
            merged["sub_intent_scores"] = [
                {
                    "sub_question": str(getattr(sq, "sub_question", "") or ""),
                    "scores": [
                        {
                            "id": getattr(ns.node, "id", None) if ns.node is not None else None,
                            "name": getattr(ns.node, "name", None) if ns.node is not None else None,
                            "score": float(ns.score),
                        }
                        for ns in (sq.node_scores or [])
                    ],
                }
                for sq in sub_intent_detail_list
            ]
        return merged
