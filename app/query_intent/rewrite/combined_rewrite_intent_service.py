# ============================================================================
# combined_rewrite_intent_service.py - 「改写 + 意图识别」组合服务（调整一核心）
# ============================================================================
"""Agent 编排侧「问题改写 + 意图识别」一体化服务。

背景（调整一 + 调整二）：
  原三段 Pipeline 中 Stage1（改写）与 Stage2（意图识别）是两次串行 LLM
  调用（合计约 1.5s~3s），且 Stage2 每次把整棵意图树全量塞进 Prompt。

本服务：
  1) 用 IntentTreeVectorRetriever（调整二）对用户问题做 Embedding Top-K
     召回，只把最相关的候选意图节点序列化进 Prompt（召回失败时降级为
     全量叶子清单，保住「单次调用」收益）；
  2) 单次 LLM 调用同时产出：改写字段（6 顶层字段，与 AgentRewriteSchema
     一致）+ 逐问题意图打分（intent_classifications，按 question_index
     索引主/子问题）；
  3) 改写字段解析完全复用父类 _parse_agent_rewrite（clamp + 工具白名单
     过滤 + hint 兜底），意图打分只做「id → 真实 IntentNode」映射，
     清单外 id 直接丢弃；
  4) 产物 AgentRewriteResult 携带 precomputed_intent_scores，Pipeline
     Stage2 据此走 aggregate_for_agent_precomputed 零 LLM 聚合。

降级链路（可用性优先，行为永不劣于旧链路）：
  向量召回失败 → 全量叶子清单继续单次调用；
  组合调用 / 解析失败 → super().rewrite_for_agent()（旧两次调用链路）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.query_intent.intent_data_base import (
    AgentChatContext,
    IntentChatMessage,
    IntentChatRequest,
    IntentChoiceTier,
)
from app.query_intent.intent_classify_resolver.intent_model import NodeScore
from app.query_intent.intent_utils import LLMResponseCleaner
from app.query_intent.llm_schemas import (
    AgentRewriteIntentCombinedSchema,
    coerce_llm_json_to_schema,
    pydantic_to_openai_response_format,
)
from app.query_intent.rag_constant import AGENT_REWRITE_INTENT_COMBINED_PROMPT_PATH
from app.query_intent.rewrite.multi_question_rewrite_service import (
    AgentMultiQuestionRewriteService,
)
from app.query_intent.intent_dto import AgentRewriteResult
from trace_to_markdown import trace_to_markdown

logger = logging.getLogger(__name__)


@dataclass
class AgentCombinedRewriteIntentService(AgentMultiQuestionRewriteService):
    """「改写 + 意图识别」单次 LLM 调用的组合服务（继承 Agent 改写服务）。

    Attributes（新增两个字段，其余全部继承自父类）:
        vector_retriever: 意图树向量检索器（调整二）。None 时直接走全量
            叶子清单降级（仍保持单次调用）。
        intent_classifier: DefaultIntentClassifier 实例，仅复用其
            load_intent_tree_data()（Redis 缓存的意图树访问逻辑）；
            不再用于 LLM 打分。
    """

    vector_retriever: Any = None
    intent_classifier: Any = None

    # ------------------------------------------------------------------
    # AgentQueryRewriteService 抽象实现（覆写：组合调用优先，失败回退父类）
    # ------------------------------------------------------------------
    @trace_to_markdown(output_file="rewrite_for_agent.md")
    def rewrite_for_agent(
        self,
        agent_chat_context: AgentChatContext,
    ) -> AgentRewriteResult:
        """Pipeline Stage1 入口：优先走组合调用，任何失败回退父类旧链路。"""
        try:
            combined_result: Optional[AgentRewriteResult] = (
                self.rewrite_and_classify_for_agent(agent_chat_context)
            )
            if combined_result is not None:
                return combined_result
        except Exception as combined_error:
            logger.warning(
                "组合「改写+意图」调用失败，回退独立两段链路。original=%r，异常=%s",
                agent_chat_context.original_user_question,
                combined_error,
                exc_info=True,
            )
        return super().rewrite_for_agent(agent_chat_context)

    # ------------------------------------------------------------------
    # 组合核心：单次 LLM 调用同时产出改写 + 逐问题意图打分
    # ------------------------------------------------------------------
    def rewrite_and_classify_for_agent(
        self,
        agent_chat_context: AgentChatContext,
    ) -> Optional[AgentRewriteResult]:
        """组合调用主体。

        Returns:
            AgentRewriteResult：含 precomputed_intent_scores（调整一）；
            None 表示组合链路不可用（调用方应回退父类 rewrite_for_agent）。
        """
        original_question: str = agent_chat_context.original_user_question or ""
        normalized_question: str = (
            self.query_term_mapping_service.normalize(original_question)
            if self.query_term_mapping_service is not None
            else original_question
        )
        pre_rule_plan_hint: Optional[str] = self._extract_plan_hint_if_present(
            original_question
        )

        if (
            self.rag_config_properties is not None
            and self.rag_config_properties.query_rewrite_enabled is False
        ):
            # 配置层禁用改写：交回父类走 rule-only 分支
            return None

        # ---- 1) 候选意图召回（调整二：向量 Top-K；失败降级全量叶子清单）----
        candidate_nodes: Optional[List[Any]] = None
        if self.vector_retriever is not None:
            try:
                candidate_nodes = self.vector_retriever.retrieve(
                    normalized_question or original_question
                )
            except Exception as retriever_error:
                logger.warning(
                    "意图向量检索异常，降级全量叶子清单：%s", retriever_error
                )
                candidate_nodes = None
        if candidate_nodes is None:
            candidate_nodes = self._load_full_leaf_nodes()
            if not candidate_nodes:
                return None

        id_to_node: Dict[str, Any] = {
            str(node.id): node
            for node in candidate_nodes
            if getattr(node, "id", None)
        }

        # ---- 2) 加载并渲染组合 Prompt（4 变量）----
        try:
            system_prompt: str = self.prompt_template_loader.load(
                AGENT_REWRITE_INTENT_COMBINED_PROMPT_PATH
            )
        except Exception as prompt_load_error:
            logger.warning(
                "加载组合 Prompt 模板失败，路径=%s，异常=%s。",
                AGENT_REWRITE_INTENT_COMBINED_PROMPT_PATH,
                prompt_load_error,
            )
            return None

        intent_list_text: str = self._render_intent_list(candidate_nodes)
        rendered_system_prompt: str = self._render_agent_rewrite_template(
            system_prompt_template=system_prompt,
            conversation_history=list(agent_chat_context.conversation_history or []),
            available_tool_ids=list(agent_chat_context.available_tool_ids or []),
            pre_rule_plan_hint=pre_rule_plan_hint,
            extra_variables={"intent_list": intent_list_text},
            available_skills=list(agent_chat_context.available_skills or []),
        )

        # ---- 3) 单次 LLM 调用（system + 最近 8 条 USER 历史 + user）----
        request_payload: IntentChatRequest = self._build_combined_request(
            rendered_system_prompt=rendered_system_prompt,
            normalized_question=normalized_question,
            conversation_history=list(agent_chat_context.conversation_history or []),
        )
        raw_response_text: str = self.llm_service.chat(
            request_payload, IntentChoiceTier.FAST
        )

        # ---- 4) 解析：改写字段复用父类（clamp + 白名单过滤 + hint 兜底）----
        final_result: Optional[AgentRewriteResult] = self._parse_agent_rewrite(
            raw_response_text=raw_response_text,
            fallback_question=(normalized_question or original_question),
            available_tool_ids=list(agent_chat_context.available_tool_ids or []),
            pre_rule_plan_hint=pre_rule_plan_hint,
            available_skill_names=[
                str(skill_name).strip()
                for skill_name in (
                    skill.get("name")
                    for skill in (agent_chat_context.available_skills or [])
                )
                if isinstance(skill_name, str) and skill_name.strip()
            ],
        )
        if final_result is None:
            return None

        # ---- 5) 意图打分提取：question_index → 问题文本 → List[NodeScore] ----
        precomputed_scores: Dict[str, List[NodeScore]] = (
            self._extract_precomputed_scores(
                raw_response_text=raw_response_text,
                primary_question=final_result.rewritten_question,
                sub_questions=list(final_result.sub_questions or []),
                id_to_node=id_to_node,
            )
        )
        final_result.precomputed_intent_scores = precomputed_scores or None
        # 附加已召回的候选意图节点（供 Pipeline Stage2 零 LLM 聚合直接复用，
        # 避免为同一问题再次检索/embedding）。对象为 per-request 临时产物，无污染。
        try:
            final_result.vector_candidate_nodes = candidate_nodes
        except Exception:  # pragma: no cover - 防御：非 slots dataclass 必然可赋值
            pass

        logger.info(
            "组合「改写+意图」单次调用完成：\n  原问题：%s\n  改写后：%s\n  "
            "是否拆分：%s，子问题数=%d\n  预计算意图问题数=%d\n  候选意图节点数=%d",
            original_question,
            final_result.rewritten_question,
            final_result.should_split,
            len(final_result.sub_questions),
            len(precomputed_scores),
            len(candidate_nodes),
        )
        return final_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_full_leaf_nodes(self) -> List[Any]:
        """全量叶子节点降级路径（向量检索不可用 / 召回失败时）。"""
        if self.intent_classifier is None:
            return []
        try:
            tree_data = self.intent_classifier.load_intent_tree_data()
            return list(getattr(tree_data, "leaf_nodes", None) or [])
        except Exception as tree_error:
            logger.warning(
                "意图树全量叶子清单加载失败，组合链路不可用：%s", tree_error
            )
            return []

    def _build_combined_request(
        self,
        rendered_system_prompt: str,
        normalized_question: str,
        conversation_history: List[IntentChatMessage],
    ) -> IntentChatRequest:
        """构造组合调用请求（消息结构对齐父类 _build_agent_rewrite_request，
        仅 response_format 换成 AgentRewriteIntentCombinedSchema）。"""
        messages: List[IntentChatMessage] = []
        if rendered_system_prompt and rendered_system_prompt.strip():
            messages.append(IntentChatMessage.system(rendered_system_prompt))

        recent_user_messages: List[IntentChatMessage] = [
            msg for msg in conversation_history
            if str(msg.role).lower() == "user"
        ]
        if len(recent_user_messages) > 8:
            recent_user_messages = recent_user_messages[-8:]
        messages.extend(recent_user_messages)

        messages.append(IntentChatMessage.user(normalized_question))

        return IntentChatRequest(
            messages=messages,
            temperature=0.1,
            top_p=0.3,
            thinking=False,
            # 调整一：严格 JSON schema（6 改写字段 + intent_classifications）
            response_format=pydantic_to_openai_response_format(
                AgentRewriteIntentCombinedSchema
            ),
        )

    @staticmethod
    def _render_intent_list(candidate_nodes: List[Any]) -> str:
        """候选意图节点 → Prompt 清单文本（渲染风格对齐
        DefaultIntentClassifier._build_prompt，保持 LLM 见到的格式一致）。"""
        sb: List[str] = []
        for node in candidate_nodes:
            sb.append(f"- id={getattr(node, 'id', None)}\n")
            sb.append(f"  path={getattr(node, 'full_path', None)}\n")
            sb.append(f"  description={getattr(node, 'description', None)}\n")

            if node.is_mcp():
                sb.append("  type=MCP\n")
                if getattr(node, "mcp_tool_id", None):
                    sb.append(f"  toolId={node.mcp_tool_id}\n")
            elif node.is_system():
                sb.append("  type=SYSTEM\n")
            else:
                sb.append("  type=KB\n")

            effective_tools = node.get_effective_agent_tool_names()
            if effective_tools:
                sb.append(f"  tools={','.join(effective_tools)}\n")

            examples = getattr(node, "examples", None) or []
            if examples:
                sb.append("  examples=")
                sb.append(" / ".join(str(e) for e in examples))
                sb.append("\n")
            sb.append("\n")
        return "".join(sb)

    def _extract_precomputed_scores(
        self,
        raw_response_text: str,
        primary_question: str,
        sub_questions: List[str],
        id_to_node: Dict[str, Any],
    ) -> Dict[str, List[NodeScore]]:
        """解析组合输出的 intent_classifications → 预计算打分映射。

        - question_index=0 → 主问题；1~N → 按序对应 sub_questions；
        - id 必须能在 id_to_node 命中（清单外 id 直接丢弃，防幻觉）；
        - 同一问题文本出现多个批次（异常场景）时按 node_id 合并取最高分；
        - 解析失败返回空 dict（调用方置 None → Stage2 回退旧链路）。
        兜底：qwen 系模型 response_format 未严格生效时可能输出「数组包裹对象」
        （[{...}]），此时用 coerce_llm_json_to_schema 取首个对象元素二次解析。
        """
        cleaned_text: str = LLMResponseCleaner.strip_markdown_code_fence(
            raw_response_text or ""
        )
        try:
            combined_struct: AgentRewriteIntentCombinedSchema = (
                AgentRewriteIntentCombinedSchema.model_validate_json(cleaned_text)
            )
        except Exception as parse_error:
            fallback_struct = coerce_llm_json_to_schema(
                AgentRewriteIntentCombinedSchema, cleaned_text
            )
            if fallback_struct is not None and isinstance(
                fallback_struct, AgentRewriteIntentCombinedSchema
            ):
                combined_struct = fallback_struct
                logger.warning(
                    "组合输出为数组包裹对象（response_format 未严格执行），"
                    "已取首元素兼容解析。raw=%s",
                    (raw_response_text or "")[:300],
                )
            else:
                logger.warning(
                    "解析组合意图打分失败（AgentRewriteIntentCombinedSchema），"
                    "raw=%s，err=%s",
                    (raw_response_text or "")[:300],
                    parse_error,
                )
                return {}

        index_to_text: Dict[int, str] = {0: primary_question}
        for offset, sub_text in enumerate(sub_questions):
            index_to_text[offset + 1] = sub_text

        collected: Dict[str, Dict[str, NodeScore]] = {}
        for batch in combined_struct.intent_classifications or []:
            question_text = index_to_text.get(int(batch.question_index))
            if not question_text or not question_text.strip():
                continue
            bucket: Dict[str, NodeScore] = collected.setdefault(question_text, {})
            for item in batch.results or []:
                node = id_to_node.get(str(item.id))
                if node is None:
                    continue
                score_value: float = float(item.score)
                existing = bucket.get(str(item.id))
                if existing is None or score_value > existing.score:
                    bucket[str(item.id)] = NodeScore(node=node, score=score_value)

        result: Dict[str, List[NodeScore]] = {}
        for question_text, bucket in collected.items():
            if not bucket:
                continue
            result[question_text] = sorted(
                bucket.values(), key=lambda ns: ns.score, reverse=True
            )
        return result
