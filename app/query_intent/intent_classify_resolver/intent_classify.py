# ============================================================================
# intent_classifier.py - 分类器接口与默认实现
# ============================================================================
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from app.query_intent.intent_classify_resolver.intent_model import IntentNode, NodeScore
from app.query_intent.intent_classify_resolver.intent_tree import IntentTreeCacheManager

from app.query_intent.intent_data_base import IntentKind, IntentLevel
from app.query_intent.rag_constant import INTENT_CLASSIFIER_PROMPT_PATH
from app.query_intent.intent_dto import AgentIntents, SubQuestionIntent
# 本轮结构化输出：S2 意图分类 schema + 协议转换 + 数组形状兜底
from app.query_intent.llm_schemas import (
    IntentClassifyListSchema,
    coerce_llm_json_to_schema,
    pydantic_to_openai_response_format,
)

logger = logging.getLogger(__name__)


# -------------------- 抽象接口 --------------------
class IntentClassifier(ABC):
    @abstractmethod
    def classify_targets(self, question: str) -> list[NodeScore]:
        ...

    def top_k_above_threshold(self, question: str, top_n: int, min_score: float) -> list[NodeScore]:
        return [
            ns for ns in self.classify_targets(question)
            if ns.score >= min_score
        ][:top_n]


class IntentNodeRegistry(ABC):
    @abstractmethod
    def get_node_by_id(self, node_id: str) -> IntentNode | None:
        ...

    @abstractmethod
    def list_mcp_tool_nodes(self) -> list[IntentNode]:
        ...


# -------------------- 默认实现 --------------------
@dataclass
class _IntentTreeData:
    all_nodes: list[IntentNode]
    leaf_nodes: list[IntentNode]
    id_to_node: dict[str, IntentNode]


class DefaultIntentClassifier(IntentClassifier, IntentNodeRegistry):
    def __init__(
        self,
        llm_service: Any | None = None,
        intent_node_mapper: Any | None = None,
        prompt_template_loader: Any | None = None,
        intent_tree_cache_manager: IntentTreeCacheManager | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._intent_node_mapper = intent_node_mapper
        self._prompt_template_loader = prompt_template_loader
        self._intent_tree_cache_manager = intent_tree_cache_manager or IntentTreeCacheManager()

    def _load_intent_tree_data(self) -> _IntentTreeData:
        roots = self._intent_tree_cache_manager.get_intent_tree_from_cache()

        if roots is None or len(roots) == 0:
            roots = self._load_intent_tree_from_db()
            if roots is not None and len(roots) > 0:
                self._intent_tree_cache_manager.save_intent_tree_to_cache(roots)

        if roots is None or len(roots) == 0:
            return _IntentTreeData(all_nodes=[], leaf_nodes=[], id_to_node={})

        # 用户上传的动态知识库集合（带功能描述/检索时机）合并为 knowledge 域下
        # 的 KB 叶子节点——合并发生在缓存读取之后，因此不会污染缓存，且集合
        # 增删后无需清树缓存即可生效（向量索引侧另由上传/删除接口负责 reset）。
        # 必须深拷贝：缓存（或 DB mapper）持有的是同一批根节点对象引用，原地
        # append 会让动态节点残留在缓存里，注册表清空后也摘不下来。
        try:
            import copy

            from app.query_intent.kb_collection_registry import KbCollectionRegistry

            roots = KbCollectionRegistry.merge_into_tree(copy.deepcopy(roots))
        except Exception as merge_error:  # noqa: BLE001 - 动态集合是增强而非硬依赖
            logger.warning("动态 KB 集合合并失败（忽略，不影响静态意图树）：%s", merge_error)

        all_nodes = self._flatten(roots)
        leaf_nodes = [n for n in all_nodes if n.is_leaf()]
        id_to_node = {n.id: n for n in all_nodes if n.id is not None}

        logger.debug("意图树数据加载完成, 总节点数: %d, 叶子节点数: %d", len(all_nodes), len(leaf_nodes))
        return _IntentTreeData(all_nodes=all_nodes, leaf_nodes=leaf_nodes, id_to_node=id_to_node)

    def load_intent_tree_data(self) -> _IntentTreeData:
        """公开意图树数据访问入口（供向量检索器等外部组件复用缓存逻辑）。"""
        return self._load_intent_tree_data()

    def invalidate_tree_cache(self) -> None:
        """清除意图树缓存（动态 KB 集合增删后，向量索引重建前调用）。"""
        try:
            self._intent_tree_cache_manager.clear_intent_tree_cache()
        except Exception as cache_error:  # noqa: BLE001 - 清缓存失败不阻断刷新
            logger.warning("意图树缓存清除失败（忽略）：%s", cache_error)

    def get_node_by_id(self, node_id: str) -> IntentNode | None:
        if node_id is None or not node_id.strip():
            return None
        data = self._load_intent_tree_data()
        return data.id_to_node.get(node_id)

    def list_mcp_tool_nodes(self) -> list[IntentNode]:
        data = self._load_intent_tree_data()
        result = [
            n for n in data.leaf_nodes
            if n.is_mcp()
            and n.mcp_tool_id is not None
            and n.mcp_tool_id.strip()
        ]
        result.sort(key=lambda n: n.id or "")
        return result

    @staticmethod
    def _flatten(roots: list[IntentNode]) -> list[IntentNode]:
        result: list[IntentNode] = []
        stack: deque[IntentNode] = deque(roots)
        while stack:
            n = stack.pop()
            result.append(n)
            if n.children is not None:
                for child in n.children:
                    stack.append(child)
        return result

    def classify_targets(self, question: str) -> list[NodeScore]:
        """对单条用户问题 → 叶子意图节点打分列表。

        【疑问-B 结论：B-1 保留 dict 请求格式】——线上适配层支持直接传 dict
        给 llm_service.chat()，因此 classify_targets 的请求构造方式不调整。
        如后续 llm_service 仅收 IntentChatRequest，请切 B-2：把 request 改为
        IntentChatRequest.builder() 链式构造，接口签名不变。
        """
        data = self._load_intent_tree_data()
        if len(data.leaf_nodes) == 0:
            logger.debug("意图树没有可用叶子节点，跳过 LLM 意图识别")
            return []

        system_prompt = self._build_prompt(data.leaf_nodes)

        # 【B-1 现状保留】直接构造 dict messages；此处故意不用 IntentChatRequest
        request: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": 0.1,
            "top_p": 0.3,
            "thinking": False,
            # S2 意图分类：关闭思考 + 严格 JSON schema（results 包装的打分数组）
            "response_format": pydantic_to_openai_response_format(IntentClassifyListSchema),
        }

        try:
            if self._llm_service is not None:
                raw_response = self._llm_service.chat(request)
            else:
                raw_response = "[]"
        except Exception as error:
            logger.warning("意图识别 LLM 调用失败，返回空意图", exc_info=error)
            return []

        return self._parse_scores(raw_response, data, question)

    def _parse_scores(self, raw: str, data: _IntentTreeData, question: str) -> list[NodeScore]:
        """S2 意图打分解析：response_format 已保证 IntentClassifyListSchema 格式。

        流程：strip_markdown_code_fence 兼容旧模型围栏 → IntentClassifyListSchema.model_validate_json
        → 按每条 item.id 查真实节点 → float(score) 夹 0~1 → 组装 NodeScore 排序。
        兜底：qwen 系模型 response_format 未严格生效时可能按旧版 Prompt 输出
        顶层数组（[{...}, ...]），此时用 coerce_llm_json_to_schema(wrap_key="results")
        包装为 {"results": [...]} 二次解析，避免打分被静默丢弃。
        """
        cleaned_raw = self._strip_markdown_code_fence(raw)
        items_list: list = []
        try:
            parsed_container = IntentClassifyListSchema.model_validate_json(cleaned_raw)
            items_list = parsed_container.results or []
        except Exception as parse_error:
            fallback_container = coerce_llm_json_to_schema(
                IntentClassifyListSchema, cleaned_raw, wrap_key="results"
            )
            if fallback_container is not None:
                items_list = fallback_container.results or []
                logger.warning(
                    "意图分类输出为顶层数组（response_format 未严格执行），"
                    "已按 results 包装兼容解析，question=%s",
                    question,
                )
            else:
                logger.warning(
                    "LLM 返回了非预期的 JSON 格式（IntentClassifyListSchema），"
                    "原始响应预览: %s，err=%s",
                    self._log_safe_preview(raw),
                    parse_error,
                )
                return []

        scores: list[NodeScore] = []
        for el in items_list:
            node_id = str(el.id)
            node = data.id_to_node.get(node_id)
            if node is None:
                logger.warning("LLM 返回了未知的意图节点 ID: %s, 已跳过", node_id)
                continue
            try:
                score = float(el.score)
                score = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                continue
            scores.append(NodeScore(node=node, score=score))

        scores.sort(key=lambda ns: ns.score, reverse=True)

        preview_scores = []
        for s in scores:
            preview_node = IntentNode(
                id=s.node.id,
                name=s.node.name,
                full_path=s.node.full_path,
                kind=s.node.kind,
            )
            preview_scores.append({"node_id": preview_node.id, "score": s.score})

        logger.info(
            "当前问题：%s\n意图识别前 %d 名如下所示：\n%s",
            question,
            len(preview_scores),
            json.dumps(preview_scores, ensure_ascii=False, indent=2),
        )
        return scores

    def top_k_above_threshold(self, question: str, top_n: int, min_score: float) -> list[NodeScore]:
        return [
            ns for ns in self.classify_targets(question)
            if ns.score >= min_score
        ][:top_n]

    def _build_prompt(self, leaf_nodes: list[IntentNode]) -> str:
        sb: list[str] = []

        for node in leaf_nodes:
            sb.append(f"- id={node.id}\n")
            sb.append(f"  path={node.full_path}\n")
            sb.append(f"  description={node.description}\n")

            if node.is_mcp():
                sb.append("  type=MCP\n")
                if node.mcp_tool_id is not None:
                    sb.append(f"  toolId={node.mcp_tool_id}\n")
            elif node.is_system():
                sb.append("  type=SYSTEM\n")
            else:
                sb.append("  type=KB\n")

            # Agent 编排语义：命中该节点后，编排层大概率会调用这些工具拿结果
            effective_tools = node.get_effective_agent_tool_names()
            if effective_tools:
                sb.append(f"  tools={','.join(effective_tools)}\n")

            if node.examples is not None and len(node.examples) > 0:
                sb.append("  examples=")
                sb.append(" / ".join(node.examples))
                sb.append("\n")
            sb.append("\n")

        intent_list = "".join(sb)

        if self._prompt_template_loader is not None:
            return self._prompt_template_loader.render(
                INTENT_CLASSIFIER_PROMPT_PATH,
                {"intent_list": intent_list},
            )
        else:
            return (
                "你是一个意图分类器。请根据以下意图列表，对用户问题进行分类打分。\n\n"
                "【意图列表】\n"
                f"{intent_list}\n"
                "请输出一个 JSON 对象：{\"results\": [{\"id\": \"...\", \"score\": 0.9}]}，"
                "顶层必须是对象（results 为打分列表），不要输出顶层数组。"
            )

    def _load_intent_tree_from_db(self) -> list[IntentNode]:
        if self._intent_node_mapper is None:
            from .intent_tree import IntentTreeFactory
            return IntentTreeFactory.build_intent_tree()

        try:
            intent_node_do_list = self._intent_node_mapper.select_list(
                {"deleted": 0, "enabled": 1}
            )
        except Exception:
            from .intent_tree import IntentTreeFactory
            return IntentTreeFactory.build_intent_tree()

        if intent_node_do_list is None or len(intent_node_do_list) == 0:
            return []

        id_to_node: dict[str, IntentNode] = {}
        for each in intent_node_do_list:
            node = self._do_to_node(each)
            if node.children is None:
                node.children = []
            if node.id is not None:
                id_to_node[node.id] = node

        roots: list[IntentNode] = []
        for node in id_to_node.values():
            parent_id = node.parent_id
            if parent_id is None or not parent_id.strip():
                roots.append(node)
                continue

            parent = id_to_node.get(parent_id)
            if parent is None:
                roots.append(node)
                continue

            if parent.children is None:
                parent.children = []
            parent.children.append(node)

        self._fill_full_path(roots, None)
        return roots

    @staticmethod
    def _do_to_node(each: Any) -> IntentNode:


        node = IntentNode()
        node.id = getattr(each, "intent_code", None)
        node.parent_id = getattr(each, "parent_code", None)
        node.kb_id = getattr(each, "kb_id", None)
        node.name = getattr(each, "name", None)
        node.description = getattr(each, "description", None)

        level_code = getattr(each, "level", None)
        node.level = IntentLevel.from_code(level_code) if level_code is not None else None

        kind_code = getattr(each, "kind", None)
        node.kind = IntentKind.from_code(kind_code) if kind_code is not None else IntentKind.KB

        node.mcp_tool_id = getattr(each, "mcp_tool_id", None)
        node.param_prompt_template = getattr(each, "param_prompt_template", None)
        node.top_k = getattr(each, "top_k", None)
        node.prompt_snippet = getattr(each, "prompt_snippet", None)
        node.prompt_template = getattr(each, "prompt_template", None)
        node.collection_name = getattr(each, "collection_name", None)

        raw_examples = getattr(each, "examples", None)
        node.examples = DefaultIntentClassifier._parse_examples(raw_examples)

        raw_collection_names = getattr(each, "collection_names", None)
        if raw_collection_names is None or len(raw_collection_names) == 0:
            if node.collection_name is not None and node.collection_name.strip():
                node.collection_names = [node.collection_name]
            else:
                node.collection_names = []
        else:
            node.collection_names = list(raw_collection_names)

        return node

    @staticmethod
    def _parse_examples(examples: str | None) -> list[str]:
        if examples is None or not examples.strip():
            return []
        try:
            parsed = json.loads(examples)
            if not isinstance(parsed, list):
                logger.warning("意图节点 examples 不是 JSON 数组, 原始值预览: %s", DefaultIntentClassifier._log_safe_preview(examples))
                return []
            result: list[str] = []
            for el in parsed:
                if isinstance(el, str):
                    result.append(el)
            return result
        except Exception as e:
            logger.warning("意图节点 examples 解析失败, 原始值预览: %s", DefaultIntentClassifier._log_safe_preview(examples), exc_info=e)
            return []

    @staticmethod
    def _fill_full_path(nodes: list[IntentNode], parent: IntentNode | None) -> None:
        if nodes is None:
            return

        for node in nodes:
            if parent is None:
                node.full_path = node.name or ""
            else:
                node.full_path = (parent.full_path or "") + " > " + (node.name or "")

            if node.children is not None and len(node.children) > 0:
                DefaultIntentClassifier._fill_full_path(node.children, node)

    @staticmethod
    def _strip_markdown_code_fence(raw: str) -> str:
        if raw is None:
            return ""
        result = raw.strip()
        if result.startswith("```"):
            first_newline = result.find("\n")
            if first_newline != -1:
                result = result[first_newline + 1:]
            if result.endswith("```"):
                result = result[:-3]
        return result.strip()

    @staticmethod
    def _log_safe_preview(text: str, max_len: int = 200) -> str:
        if text is None:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."


# ==============================================================================
# AgentIntentAggregator（Agent 编排侧新增 · ⑦-2）
# ==============================================================================
@dataclass
class AgentIntentAggregator:
    """Agent 编排专用的"多子问题 → 统一意图汇总"服务。

    与 RAG 侧 IntentResolver.resolve（只返回 list[NodeScore]）不同，
    这里的汇总目标是产出 AgentIntents，包含：
      - KB / MCP / SYSTEM 三通道分别的命中计数与 NodeScore 明细；
      - 聚合后综合置信度、主意图 display_name；
      - （若拆分）按子问题维度的 SubQuestionIntent 列表，一一对应；
      - 原始 slots 字典（简单取 TOP Node 的 description 作为主 hint，复杂
        槽位抽取可在此后扩展 param_prompt_template）。

    Attributes:
        base_classifier: 实际执行单条问题打分的分类器实现（通常是
            DefaultIntentClassifier）。
        min_intent_score: 过滤 NodeScore 的最低分数（默认 0.35，与
            rag_constant.INTENT_MIN_SCORE 一致）。
        top_k_per_question: 单条问题最多保留的高分节点数。
    """

    base_classifier: DefaultIntentClassifier
    min_intent_score: float = field(default=0.35)
    top_k_per_question: int = field(default=3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def aggregate_for_agent(
        self,
        primary_question: str,
        sub_questions: List[str] | None = None,
    ) -> AgentIntents:
        """Agent 专用意图汇总。

        Args:
            primary_question: 主问题（通常 = rewrite.rewritten_question）。
            sub_questions: 可选拆分后的子问题。若提供则会逐问题分类，最终
                再做全局聚合。

        Returns:
            AgentIntents：三通道汇总 + 主意图 + slots + 逐子问题明细。
        """
        primary_scores: List[NodeScore] = self._classify_and_filter(primary_question)
        sub_score_lists: List[List[NodeScore]] = [
            self._classify_and_filter(sub_question_text)
            for sub_question_text in (sub_questions or [])
        ]
        return self._aggregate_from_score_lists(
            primary_question=primary_question,
            sub_questions=list(sub_questions or []),
            primary_scores=primary_scores,
            sub_score_lists=sub_score_lists,
        )

    def aggregate_for_agent_precomputed(
        self,
        primary_question: str,
        sub_questions: List[str] | None = None,
        precomputed_scores: Dict[str, List[NodeScore]] | None = None,
    ) -> AgentIntents:
        """Agent 专用意图汇总（预计算版 · 调整一）。

        与 aggregate_for_agent 的唯一区别：不调用 base_classifier（即零 LLM
        调用），而是直接消费组合调用（AgentCombinedRewriteIntentService）
        已经产出的逐问题打分结果。

        Args:
            primary_question: 主问题文本（作为 precomputed_scores 的 key）。
            sub_questions: 拆分子问题列表（key 同上）。
            precomputed_scores: 问题文本 → NodeScore 列表 的预计算映射。
                缺失的 key 回退为空打分（等价于该问题无意图命中）。

        Returns:
            AgentIntents：与 aggregate_for_agent 完全同构的聚合产物。
        """
        precomputed = precomputed_scores or {}

        def _filter_precomputed(question_text: str) -> List[NodeScore]:
            raw_scores = list(precomputed.get(question_text, []))
            filtered = [
                ns for ns in raw_scores if ns.score >= self.min_intent_score
            ]
            filtered.sort(key=lambda ns: ns.score, reverse=True)
            return filtered[: self.top_k_per_question]

        primary_scores: List[NodeScore] = _filter_precomputed(primary_question)
        sub_questions_list: List[str] = list(sub_questions or [])
        sub_score_lists: List[List[NodeScore]] = [
            _filter_precomputed(sub_text) for sub_text in sub_questions_list
        ]
        return self._aggregate_from_score_lists(
            primary_question=primary_question,
            sub_questions=sub_questions_list,
            primary_scores=primary_scores,
            sub_score_lists=sub_score_lists,
        )

    def _aggregate_from_score_lists(
        self,
        primary_question: str,
        sub_questions: List[str],
        primary_scores: List[NodeScore],
        sub_score_lists: List[List[NodeScore]],
    ) -> AgentIntents:
        """聚合主问题 + 各子问题的打分列表 → AgentIntents（共用聚合主体）。"""
        per_sub_question_results: List[SubQuestionIntent] = []

        # 1) 先汇总 primary_question 作为基线
        kb_scores, mcp_scores, sys_scores = self._partition_by_kind(primary_scores)
        all_poll_scores: List[NodeScore] = list(primary_scores)

        # 2) 若有拆分：逐子问题填充 per_sub_question_intents
        for sub_question_text, sub_scores in zip(sub_questions, sub_score_lists):
            per_sub_question_results.append(
                SubQuestionIntent(
                    sub_question=sub_question_text, node_scores=sub_scores
                )
            )
            all_poll_scores.extend(sub_scores)
            (
                sub_kb_scores,
                sub_mcp_scores,
                sub_sys_scores,
            ) = self._partition_by_kind(sub_scores)
            kb_scores.extend(sub_kb_scores)
            mcp_scores.extend(sub_mcp_scores)
            sys_scores.extend(sub_sys_scores)

        # 3) 按 NodeScore.node.id 去重并加权合并分数
        merged_kb: List[NodeScore] = self._merge_node_scores(kb_scores)
        merged_mcp: List[NodeScore] = self._merge_node_scores(mcp_scores)
        merged_sys: List[NodeScore] = self._merge_node_scores(sys_scores)

        # 4) 主意图 & 聚合置信度
        primary_intent_text, aggregated_confidence = self._compute_primary_intent(
            merged_kb, merged_mcp, merged_sys
        )

        # 5) 简单槽位抽取（首版：把主节点的关键属性传 slots；后续可扩展）
        raw_slots: Dict[str, Any] = self._extract_simple_slots(
            merged_kb, merged_mcp, merged_sys
        )

        return AgentIntents(
            primary_intent_text=primary_intent_text,
            aggregated_confidence=aggregated_confidence,
            kb_hit_count=len(merged_kb),
            mcp_hit_count=len(merged_mcp),
            sys_hit_count=len(merged_sys),
            kb_node_scores=merged_kb,
            mcp_node_scores=merged_mcp,
            sys_node_scores=merged_sys,
            per_sub_question_intents=per_sub_question_results,
            raw_slots=raw_slots,
        )

    def collect_intent_tool_names_from_intents(
        self,
        agent_intents: AgentIntents,
    ) -> List[str]:
        """从 AgentIntents 的全通道命中节点抽取意图层工具名并集。

        Agent 编排语义：返回"意图树认为大概率能拿到结果"的注册工具列表，
        供 Pipeline 第三段 allowed_tools 合并（来源优先级：分数高者靠前）。
        兼容性：节点无 agent_tool_names 时回退 mcp_tool_id（DB 旧数据），
        因此本方法完整替代旧的 collect_mcp_tool_ids_from_intents。
        """
        all_scores: List[NodeScore] = [
            *agent_intents.kb_node_scores,
            *agent_intents.mcp_node_scores,
            *agent_intents.sys_node_scores,
        ]
        # 跨通道按分数降序 → 高分意图的工具优先进入白名单
        all_scores.sort(key=lambda ns: getattr(ns, "score", 0.0), reverse=True)

        tool_set: Dict[str, None] = {}
        for node_score in all_scores:
            node = node_score.node
            if node is None:
                continue
            for tool_name in node.get_effective_agent_tool_names():
                tool_set.setdefault(tool_name, None)
        return list(tool_set.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _classify_and_filter(self, question_text: str) -> List[NodeScore]:
        if not question_text or not str(question_text).strip():
            return []
        return self.base_classifier.top_k_above_threshold(
            question_text,
            top_n=self.top_k_per_question,
            min_score=self.min_intent_score,
        )

    @staticmethod
    def _partition_by_kind(
        score_list: List[NodeScore],
    ) -> Tuple[List[NodeScore], List[NodeScore], List[NodeScore]]:
        kb_part: List[NodeScore] = []
        mcp_part: List[NodeScore] = []
        sys_part: List[NodeScore] = []
        for node_score in score_list:
            node_kind = getattr(node_score.node, "kind", None)
            if node_kind == IntentKind.KB or (
                isinstance(node_kind, int) and node_kind == IntentKind.KB.value
            ):
                kb_part.append(node_score)
            elif node_kind == IntentKind.MCP or (
                isinstance(node_kind, int) and node_kind == IntentKind.MCP.value
            ):
                mcp_part.append(node_score)
            else:
                # 兜底：非 KB/MCP 的都视作 SYSTEM 通道
                sys_part.append(node_score)
        return kb_part, mcp_part, sys_part

    @staticmethod
    def _merge_node_scores(score_list: List[NodeScore]) -> List[NodeScore]:
        """同一 node_id 出现多次 → 分数取 max（保守）。"""
        merged_map: Dict[str, NodeScore] = {}
        for node_score in score_list:
            node_id = getattr(node_score.node, "id", None)
            if node_id is None:
                continue
            cached_score = merged_map.get(node_id)
            if cached_score is None or node_score.score > cached_score.score:
                merged_map[node_id] = node_score
        sorted_scores: List[NodeScore] = sorted(
            merged_map.values(), key=lambda ns: ns.score, reverse=True
        )
        return sorted_scores

    @staticmethod
    def _compute_primary_intent(
        kb_scores: List[NodeScore],
        mcp_scores: List[NodeScore],
        sys_scores: List[NodeScore],
    ) -> Tuple[str, float]:
        """在三通道内选全局最高分节点作为主意图，并返回其 display_name。"""

        all_candidates: List[NodeScore] = [*kb_scores, *mcp_scores, *sys_scores]
        if not all_candidates:
            return "general", 0.0
        top_candidate: NodeScore = max(
            all_candidates, key=lambda ns: ns.score
        )
        display_name: str = (
            getattr(top_candidate.node, "full_path", None)
            or getattr(top_candidate.node, "name", None)
            or getattr(top_candidate.node, "id", "general")
            or "general"
        )
        return str(display_name), float(top_candidate.score)

    @staticmethod
    def _extract_simple_slots(
        kb_scores: List[NodeScore],
        mcp_scores: List[NodeScore],
        sys_scores: List[NodeScore],
    ) -> Dict[str, Any]:
        """Agent 编排槽位抽取。

        产出：
          - top_kb_node / top_mcp_node / top_system_node：各通道 TOP1 关键属性
            （含 agent_tool_names / tool_usage_hint / prefer_mode）；
          - intent_hit_tool_names：全通道命中节点工具名并集（分数序），
            供 Pipeline._finalize_allowed_tools 直接合并进 allowed_tools；
          - intent_prefer_mode：全局最高分节点的模式倾向（react/plan_execute），
            供 ModeDecider 静态意图偏好层参考（调整三后已无 LLM 层）。
        """
        slots: Dict[str, Any] = {}

        def _pack_top_one(top_scores: List[NodeScore]) -> Dict[str, Any]:
            if not top_scores:
                return {}
            top_node = top_scores[0].node
            return {
                "node_id": getattr(top_node, "id", None),
                "full_path": getattr(top_node, "full_path", None),
                "kb_id": getattr(top_node, "kb_id", None),
                "collection_name": getattr(top_node, "collection_name", None),
                "collection_names": getattr(top_node, "collection_names", None),
                "mcp_tool_id": getattr(top_node, "mcp_tool_id", None),
                "param_prompt_template": getattr(
                    top_node, "param_prompt_template", None
                ),
                "top_k": getattr(top_node, "top_k", None),
                # ---- Agent 编排工具路由字段（新增） ----
                "agent_tool_names": getattr(top_node, "agent_tool_names", None),
                "tool_usage_hint": getattr(top_node, "tool_usage_hint", None),
                "prefer_mode": getattr(top_node, "prefer_mode", None),
            }

        if kb_scores:
            slots["top_kb_node"] = _pack_top_one(kb_scores)
        if mcp_scores:
            slots["top_mcp_node"] = _pack_top_one(mcp_scores)
        if sys_scores:
            slots["top_system_node"] = _pack_top_one(sys_scores)

        # 全通道命中工具并集（分数序）：意图树对 allowed_tools 的核心贡献
        all_scores: List[NodeScore] = [*kb_scores, *mcp_scores, *sys_scores]
        all_scores.sort(key=lambda ns: ns.score, reverse=True)
        intent_tool_set: Dict[str, None] = {}
        for node_score in all_scores:
            node = node_score.node
            if node is None:
                continue
            for tool_name in node.get_effective_agent_tool_names():
                intent_tool_set.setdefault(tool_name, None)
        slots["intent_hit_tool_names"] = list(intent_tool_set.keys())

        # 全局最高分节点的模式倾向（仅参考信号，不覆盖规则层）
        if all_scores:
            primary_prefer_mode = getattr(all_scores[0].node, "prefer_mode", None)
            if primary_prefer_mode:
                slots["intent_prefer_mode"] = primary_prefer_mode

        return slots