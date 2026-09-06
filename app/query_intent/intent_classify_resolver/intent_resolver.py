# ============================================================================
# intent_resolver.py - 意图解析器与过滤器
# ============================================================================
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional

from app.query_intent.intent_classify_resolver.intent_classify import (
    AgentIntentAggregator,
    DefaultIntentClassifier,
    IntentClassifier,
)
from app.query_intent.intent_classify_resolver.intent_model import NodeScore

from app.query_intent.intent_data_base import IntentKind
from app.query_intent.intent_dto import (
    AgentIntents,
    AgentRewriteResult,
    IntentCandidate,
    IntentGroup,
    SubQuestionIntent,
)
from app.query_intent.rag_constant import INTENT_MIN_SCORE, MAX_INTENT_COUNT

from app.query_intent.rewrite.query_rewrite import RewriteResult

logger = logging.getLogger(__name__)


# -------------------- 节点分数过滤器 --------------------
class NodeScoreFilters:
    def __init__(self) -> None:
        raise RuntimeError("This class cannot be instantiated")

    @staticmethod
    def mcp(scores: list[NodeScore]) -> list[NodeScore]:
        return [
            ns for ns in scores
            if ns.node is not None
            and ns.node.is_mcp()
            and ns.node.mcp_tool_id is not None
            and ns.node.mcp_tool_id.strip()
        ]

    @staticmethod
    def kb(scores: list[NodeScore], min_score: float | None = None) -> list[NodeScore]:
        result = [
            ns for ns in scores
            if ns.node is not None and ns.node.is_kb()
        ]
        if min_score is not None:
            result = [ns for ns in result if ns.score >= min_score]
        return result

    @staticmethod
    def kb_collections(scores: list[NodeScore]) -> list[str]:
        kb_scores = NodeScoreFilters.kb(scores)
        result: list[str] = []
        seen: set[str] = set()
        for ns in kb_scores:
            for name in ns.node.get_effective_collection_names():
                if name not in seen:
                    seen.add(name)
                    result.append(name)
        return result


# -------------------- 意图解析器 --------------------
class IntentResolver:
    def __init__(
        self,
        intent_classifier: IntentClassifier | None = None,
        executor: Any | None = None,
    ) -> None:

        self._intent_classifier = intent_classifier or DefaultIntentClassifier()
        self._executor = executor

    def resolve(self, rewrite_result: RewriteResult) -> list[SubQuestionIntent]:
        sub_questions = (
            rewrite_result.sub_questions
            if rewrite_result.sub_questions is not None and len(rewrite_result.sub_questions) > 0
            else [rewrite_result.rewritten_question]
        )

        sub_intents: list[SubQuestionIntent] = []

        if self._executor is not None:
            futures = []
            for q in sub_questions:
                futures.append(
                    self._executor.submit(self._classify_safe, q)
                )
            for future, q in zip(futures, sub_questions):
                try:
                    result = future.result()
                except Exception as e:
                    logger.error("子问题意图分类失败，降级为空意图，question：%s", q, exc_info=e)
                    result = []
                sub_intents.append(SubQuestionIntent(sub_question=q, node_scores=result))
        else:
            with ThreadPoolExecutor(max_workers=max(1, len(sub_questions))) as pool:
                future_to_q = {pool.submit(self._classify_safe, q): q for q in sub_questions}
                q_to_result: dict[str, list[NodeScore]] = {}
                for future in as_completed(future_to_q):
                    q = future_to_q[future]
                    try:
                        q_to_result[q] = future.result()
                    except Exception as e:
                        logger.error("子问题意图分类失败，降级为空意图，question：%s", q, exc_info=e)
                        q_to_result[q] = []
            for q in sub_questions:
                sub_intents.append(SubQuestionIntent(sub_question=q, node_scores=q_to_result.get(q, [])))

        return self._cap_total_intents(sub_intents)

    def merge_intent_group(self, sub_intents: list[SubQuestionIntent]) -> IntentGroup:
        mcp_intents: list[NodeScore] = []
        kb_intents: list[NodeScore] = []
        for si in sub_intents:
            mcp_intents.extend(NodeScoreFilters.mcp(si.node_scores))
            kb_intents.extend(NodeScoreFilters.kb(si.node_scores))
        return IntentGroup(mcp_intents=mcp_intents, kb_intents=kb_intents)

    @staticmethod
    def is_system_only(node_scores: list[NodeScore]) -> bool:
        return (
            len(node_scores) == 1
            and node_scores[0].node is not None
            and node_scores[0].node.kind == IntentKind.SYSTEM
        )

    def _classify_safe(self, question: str) -> list[NodeScore]:
        try:
            return self._classify_intents(question)
        except Exception as e:
            logger.error("子问题意图分类失败，降级为空意图，question：%s", question, exc_info=e)
            return []

    def _classify_intents(self, question: str) -> list[NodeScore]:
        scores = self._intent_classifier.classify_targets(question)
        return [
            ns for ns in scores
            if ns.score >= INTENT_MIN_SCORE
        ][:MAX_INTENT_COUNT]

    @staticmethod
    def _cap_total_intents(sub_intents: list[SubQuestionIntent]) -> list[SubQuestionIntent]:
        total_intents = sum(len(si.node_scores) for si in sub_intents)

        if total_intents <= MAX_INTENT_COUNT:
            return sub_intents

        all_candidates = IntentResolver._collect_all_candidates(sub_intents)

        guaranteed_intents = IntentResolver._select_top_intent_per_sub_question(
            all_candidates, len(sub_intents)
        )

        remaining = MAX_INTENT_COUNT - len(guaranteed_intents)

        additional_intents = IntentResolver._select_additional_intents(
            all_candidates, guaranteed_intents, remaining
        )

        return IntentResolver._rebuild_sub_intents(sub_intents, guaranteed_intents, additional_intents)

    @staticmethod
    def _collect_all_candidates(sub_intents: list[SubQuestionIntent]) -> list[IntentCandidate]:
        candidates: list[IntentCandidate] = []
        for i, si in enumerate(sub_intents):
            node_scores = si.node_scores
            if node_scores is None or len(node_scores) == 0:
                continue
            for ns in node_scores:
                candidates.append(IntentCandidate(sub_question_index=i, node_score=ns))
        candidates.sort(key=lambda c: c.node_score.score, reverse=True)
        return candidates

    @staticmethod
    def _select_top_intent_per_sub_question(
        all_candidates: list[IntentCandidate],
        sub_question_count: int,
    ) -> list[IntentCandidate]:
        top_intents: list[IntentCandidate] = []
        selected: list[bool] = [False] * sub_question_count

        for candidate in all_candidates:
            index = candidate.sub_question_index
            if not selected[index]:
                top_intents.append(candidate)
                selected[index] = True
            if len(top_intents) == sub_question_count:
                break
        return top_intents

    @staticmethod
    def _select_additional_intents(
        all_candidates: list[IntentCandidate],
        guaranteed_intents: list[IntentCandidate],
        remaining: int,
    ) -> list[IntentCandidate]:
        if remaining <= 0:
            return []

        additional: list[IntentCandidate] = []
        guaranteed_set = {id(g) for g in guaranteed_intents}
        for candidate in all_candidates:
            if id(candidate) in guaranteed_set:
                continue
            additional.append(candidate)
            if len(additional) >= remaining:
                break
        return additional

    @staticmethod
    def _rebuild_sub_intents(
        original_sub_intents: list[SubQuestionIntent],
        guaranteed_intents: list[IntentCandidate],
        additional_intents: list[IntentCandidate],
    ) -> list[SubQuestionIntent]:
        all_selected: list[IntentCandidate] = list(guaranteed_intents)
        all_selected.extend(additional_intents)

        grouped_by_index: dict[int, list[NodeScore]] = {}
        for candidate in all_selected:
            if candidate.sub_question_index not in grouped_by_index:
                grouped_by_index[candidate.sub_question_index] = []
            grouped_by_index[candidate.sub_question_index].append(candidate.node_score)

        result: list[SubQuestionIntent] = []
        for i, original in enumerate(original_sub_intents):
            retained = grouped_by_index.get(i, [])
            result.append(SubQuestionIntent(sub_question=original.sub_question, node_scores=retained))
        return result

    # ==========================================================================
    # Agent 编排专用便捷方法（⑧ 新增）
    # ==========================================================================
    def resolve_for_agent(
        self,
        rewrite_result: AgentRewriteResult,
        aggregator: Optional[AgentIntentAggregator] = None,
    ) -> AgentIntents:
        """面向 Agent 编排 Pipeline 的便捷封装：AgentRewriteResult → AgentIntents。

        1) 若传入 aggregator：优先复用 aggregator（其 base_classifier 可能是
           外部已配置好 PromptTemplateLoader / 意图树缓存的实例）。
        2) 否则兜底：基于 self._intent_classifier 做一次兼容转换。若
           self._intent_classifier 不是 DefaultIntentClassifier，再临时 new
           一个 aggregator。

        这样既保持与 Pipeline 统一入参一致，也避免在 IntentResolver 内
        部直接造 aggregator 产生状态不一致。
        """
        effective_aggregator: AgentIntentAggregator
        if aggregator is not None:
            effective_aggregator = aggregator
        elif isinstance(self._intent_classifier, DefaultIntentClassifier):
            effective_aggregator = AgentIntentAggregator(
                base_classifier=self._intent_classifier,
                min_intent_score=INTENT_MIN_SCORE,
                top_k_per_question=MAX_INTENT_COUNT,
            )
        else:
            effective_aggregator = AgentIntentAggregator(
                base_classifier=DefaultIntentClassifier(
                    llm_service=getattr(self._intent_classifier, "_llm_service", None),
                    intent_node_mapper=getattr(
                        self._intent_classifier, "_intent_node_mapper", None
                    ),
                    prompt_template_loader=getattr(
                        self._intent_classifier, "_prompt_template_loader", None
                    ),
                ),
                min_intent_score=INTENT_MIN_SCORE,
                top_k_per_question=MAX_INTENT_COUNT,
            )

        primary_question_text: str = (
            rewrite_result.rewritten_question
            if rewrite_result.rewritten_question
            else ""
        )
        sub_questions_list: List[str] = (
            list(rewrite_result.sub_questions)
            if rewrite_result.should_split and rewrite_result.sub_questions
            else []
        )

        aggregated_intents: AgentIntents = effective_aggregator.aggregate_for_agent(
            primary_question=primary_question_text,
            sub_questions=sub_questions_list,
        )
        logger.info(
            "Agent 意图解析完成：主意图='%s'，综合置信度=%.3f，KB=%d / MCP=%d / SYS=%d",
            aggregated_intents.primary_intent_text,
            aggregated_intents.aggregated_confidence,
            aggregated_intents.kb_hit_count,
            aggregated_intents.mcp_hit_count,
            aggregated_intents.sys_hit_count,
        )
        return aggregated_intents