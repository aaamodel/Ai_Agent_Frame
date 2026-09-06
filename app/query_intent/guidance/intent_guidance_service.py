import logging
import re

from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field


from app.query_intent.guidance.guidance_decision_checker import AmbiguityLLMChecker, GuidanceDecision
from app.query_intent.intent_classify_resolver.intent_classify import IntentNodeRegistry
from app.query_intent.intent_classify_resolver.intent_model import NodeScore, IntentNode
from app.query_intent.intent_classify_resolver.intent_resolver import NodeScoreFilters
from app.query_intent.intent_dto import SubQuestionIntent


from app.query_intent.intent_prompt.prompt_template_loader import PromptTemplateLoader
from app.query_intent.rag_constant import GUIDANCE_PROMPT_PATH, INTENT_MIN_SCORE
from query_intent import trace_to_markdown

logger = logging.getLogger(__name__)

MIN_COMPARABLE_NAME_LENGTH = 2


def _is_empty(collection) -> bool:
    return collection is None or len(collection) == 0


def _is_not_blank(s: str) -> bool:
    return s is not None and s.strip() != ""


def _blank_to_default(s: str, default: str) -> str:
    return s if _is_not_blank(s) else default



@dataclass
class GuidanceProperties:
    enabled: bool = field(default=True)
    ambiguity_score_ratio: float = field(default=0.8)
    ambiguity_margin: float = field(default=0.15)
    max_options: int = field(default=6)

@dataclass
class PathConflict:
    topic_name: str
    ranked: List[NodeScore]


@dataclass
class AmbiguityGroup:
    topic_name: str
    ranked: List[NodeScore]


@dataclass
class IntentGuidanceService:
    guidance_properties: GuidanceProperties
    intent_node_registry: IntentNodeRegistry
    prompt_template_loader: PromptTemplateLoader
    ambiguity_llm_checker: AmbiguityLLMChecker



    def detect_ambiguity(self, question: str, sub_intents: List[SubQuestionIntent]) -> GuidanceDecision:
        if not (self.guidance_properties.enabled is True):
            return GuidanceDecision.none()

        group = self._find_ambiguity_group(question, sub_intents)
        if group is None or _is_empty(group.ranked):
            return GuidanceDecision.none()

        prompt = self._build_prompt(group.topic_name, group.ranked)
        return GuidanceDecision.prompt(prompt)

    def _find_ambiguity_group(self, question: str, sub_intents: List[SubQuestionIntent]) -> Optional[AmbiguityGroup]:
        if _is_empty(sub_intents) or len(sub_intents) != 1:
            return None

        ranked = self._rank_candidates(self._filter_candidates(sub_intents[0].node_scores))
        if len(ranked) < 2:
            return None

        conflict = self._collect_path_conflicts(question, ranked)
        if conflict is None:
            return None

        if not self.ambiguity_llm_checker.check_ambiguity(question, conflict.ranked):
            logger.info("LLM 判定候选路径不构成歧义, 跳过澄清, question={}".format(question))
            return None

        return AmbiguityGroup(conflict.topic_name, self._trim_ranked_options(conflict.ranked))

    def _filter_candidates(self, scores: List[NodeScore]) -> List[NodeScore]:
        if _is_empty(scores):
            return []
        return NodeScoreFilters.kb(scores, INTENT_MIN_SCORE)

    def _rank_candidates(self, candidates: List[NodeScore]) -> List[NodeScore]:
        best_by_node: Dict[str, NodeScore] = {}
        for candidate in candidates:
            node = candidate.node
            key = _blank_to_default(node.id, node.name if node.name is not None else "")
            if key in best_by_node:
                kept = best_by_node[key]
                best_by_node[key] = kept if kept.score >= candidate.score else candidate
            else:
                best_by_node[key] = candidate
        result = list(best_by_node.values())
        result.sort(key=lambda x: x.get_score(), reverse=True)
        return result

    def _collect_path_conflicts(self, question: str, ranked: List[NodeScore]) -> Optional[PathConflict]:
        node_cache: Dict[str, IntentNode] = {}
        primary = ranked[0]
        primary_path = self._build_node_path(primary.node, node_cache)
        normalized_question = self._normalize_name(question)

        conflicts: List[NodeScore] = [primary]
        topic_name = None
        for other in ranked[1:]:
            other_path = self._build_node_path(other.node, node_cache)
            hit_name = self._detect_conflict_name(primary_path, other_path, normalized_question)
            if not _is_not_blank(hit_name):
                continue
            conflicts.append(other)
            if topic_name is None:
                topic_name = hit_name

        if len(conflicts) < 2:
            return None
        logger.info("候选意图路径重名[{}], 调 LLM 确认是否需要澄清, question={}".format(topic_name, question))
        return PathConflict(topic_name, conflicts)

    def _detect_conflict_name(self, primary_path: List[IntentNode], other_path: List[IntentNode], normalized_question: str) -> Optional[str]:
        if _is_empty(primary_path) or _is_empty(other_path):
            return None

        primary_leaf = primary_path[-1]
        leaf_name = self._normalize_name(primary_leaf.name)
        if self._is_comparable_name(leaf_name) and leaf_name == self._normalize_name(other_path[-1].name):
            return primary_leaf.name

        common = self._common_prefix_length(primary_path, other_path)
        other_names: Set[str] = set()
        for node in other_path[common:]:
            name = self._normalize_name(node.name)
            if self._is_comparable_name(name):
                other_names.add(name)
        for node in primary_path[common:]:
            name = self._normalize_name(node.name)
            if self._is_comparable_name(name) and name in other_names and name in normalized_question:
                return node.name
        return None

    def _build_node_path(self, node: IntentNode, node_cache: Dict[str, IntentNode]) -> List[IntentNode]:
        path: List[IntentNode] = []
        visited: Set[str] = set()
        current = node
        while current is not None:
            path.insert(0, current)
            visited.add(current.id)
            parent_id = current.parent_id
            if not _is_not_blank(parent_id) or parent_id in visited:
                break
            current = self._fetch_node(parent_id, node_cache)
        return path

    def _common_prefix_length(self, left: List[IntentNode], right: List[IntentNode]) -> int:
        max_len = min(len(left), len(right))
        index = 0
        while index < max_len and self._is_same_node(left[index], right[index]):
            index += 1
        return index

    def _is_same_node(self, left: IntentNode, right: IntentNode) -> bool:
        return _is_not_blank(left.id) and left.id == right.id

    def _is_comparable_name(self, normalized_name: str) -> bool:
        return _is_not_blank(normalized_name) and len(normalized_name) >= MIN_COMPARABLE_NAME_LENGTH

    def _fetch_node(self, node_id: str, node_cache: Dict[str, IntentNode]) -> Optional[IntentNode]:
        if node_id in node_cache:
            return node_cache[node_id]
        node = self.intent_node_registry.get_node_by_id(node_id)
        node_cache[node_id] = node
        return node

    def _trim_ranked_options(self, ranked: List[NodeScore]) -> List[NodeScore]:
        max_options = self.guidance_properties.max_options if self.guidance_properties.get_max_options() is not None else len(ranked)
        if len(ranked) <= max_options:
            return ranked
        return ranked[:max_options]

    def _build_prompt(self, topic_name: str, ranked: List[NodeScore]) -> str:
        options = self._render_options(ranked)
        return self.prompt_template_loader.render(
            GUIDANCE_PROMPT_PATH,
            {
                "topic_name": _blank_to_default(topic_name, ""),
                "options": options
            }
        )

    def _render_options(self, ranked: List[NodeScore]) -> str:
        sb = []
        for i in range(len(ranked)):
            node = ranked[i].node
            display = self._resolve_option_display(node)
            sb.append("{}) {}\n".format(i + 1, display))
        return "".join(sb).strip()

    def _resolve_option_display(self, node: Optional[IntentNode]) -> str:
        if node is None:
            return ""
        if _is_not_blank(node.full_path):
            return node.full_path
        return _blank_to_default(node.name, node.id)

    def _normalize_name(self, name: str) -> str:
        if name is None:
            return ""
        cleaned = name.strip().lower()
        return re.sub(r"[\p{P}\s]+", "", cleaned)
