import logging
import re

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from app.query_intent.intent_data_base import (
    AgentChatContext,
    IntentChatMessage,
    IntentChoiceTier,
    IntentChatRequest,
)
from app.query_intent.intent_service import IntentLLMService
from app.query_intent.intent_utils import  LLMResponseCleaner
from app.query_intent.rag_constant import (
    AGENT_QUESTION_REWRITE_PROMPT_PATH,
    QUERY_REWRITE_AND_SPLIT_PROMPT_PATH,
    REGISTERED_ENABLED_TOOL_NAMES,
)
from app.query_intent.intent_prompt.prompt_template_loader import PromptTemplateLoader
from app.query_intent.rewrite.query_rewrite import (
    AgentQueryRewriteService,
    QueryRewriteService,
    QueryTermMappingService,
    RewriteResult,
)
from app.query_intent.intent_dto import (
    AgentRewriteResult,
    TaskComplexityAnalysis,
    normalize_agent_goal,
)
# 本轮结构化输出：Pydantic schema + OpenAI 协议转换工具 + 数组形状兜底
from app.query_intent.llm_schemas import (
    AgentRewriteSchema,
    RagRewriteSchema,
    coerce_llm_json_to_schema,
    pydantic_to_openai_response_format,
)

from trace_to_markdown import  trace_to_markdown
logger = logging.getLogger(__name__)




@dataclass
class RAGConfigProperties:
    query_rewrite_enabled: Optional[bool] = field(default=None)
    rerank_enabled: Optional[bool] = field(default=None)
    context_enrich_enabled: Optional[bool] = field(default=None)
    citation_enabled: Optional[bool] = field(default=None)

@dataclass
class MultiQuestionRewriteService(QueryRewriteService):
    llm_service: IntentLLMService
    rag_config_properties: RAGConfigProperties
    query_term_mapping_service: QueryTermMappingService
    prompt_template_loader: PromptTemplateLoader


    def rewrite(self, user_question: str) -> str:
        return self._rewrite_and_split(user_question).rewritten_question

    def rewrite_with_split(self, user_question: str, history: Optional[List[IntentChatMessage]] = None) -> RewriteResult:
        if history is None:
            return self._rewrite_and_split(user_question)
        return self._rewrite_with_split_history(user_question, history)

    @trace_to_markdown(output_file="rewrite_with_split_history.md")
    def _rewrite_with_split_history(self, user_question: str, history: List[IntentChatMessage]) -> RewriteResult:
        if not self.rag_config_properties.query_rewrite_enabled:
            normalized = self.query_term_mapping_service.normalize(user_question)
            subs = self._rule_based_split(normalized)
            return RewriteResult(rewritten_question=normalized, sub_questions=subs)

        normalized_question = self.query_term_mapping_service.normalize(user_question)
        return self._call_llm_rewrite_and_split(normalized_question, user_question, history)

    def _rewrite_and_split(self, user_question: str) -> RewriteResult:
        if not self.rag_config_properties.query_rewrite_enabled:
            normalized = self.query_term_mapping_service.normalize(user_question)
            subs = self._rule_based_split(normalized)
            return RewriteResult(rewritten_question=normalized, sub_questions=subs)

        normalized_question = self.query_term_mapping_service.normalize(user_question)
        return self._call_llm_rewrite_and_split(normalized_question, user_question, [])

    def _call_llm_rewrite_and_split(self, normalized_question: str, original_question: str, history: List[IntentChatMessage]) -> RewriteResult:
        system_prompt = self.prompt_template_loader.load(QUERY_REWRITE_AND_SPLIT_PROMPT_PATH)
        req = self._build_rewrite_request(system_prompt, normalized_question, history)

        fallback = RewriteResult(rewritten_question=normalized_question, sub_questions=[normalized_question])
        try:
            parsed = self._parse_rewrite_and_split(self.llm_service.chat(req, IntentChoiceTier.FAST))
            result = parsed if parsed is not None else fallback
        except Exception as e:
            logger.warning("查询改写 LLM 调用失败，使用归一化问题兜底", e)
            result = fallback

        logger.info(
            "RAG用户问题查询改写+拆分：\n原始问题：{}\n归一化后：{}\n改写结果：{}\n子问题：{}",
            original_question, normalized_question, result.rewritten_question, result.sub_questions
        )
        return result

    def _build_rewrite_request(self, system_prompt: str, question: str, history: List[IntentChatMessage]) -> IntentChatRequest:
        messages = []
        if system_prompt and system_prompt.strip() != "":
            messages.append(IntentChatMessage.system(system_prompt))

        if len(history) > 0:
            recent_history = [
                msg for msg in history
                if msg.role in ("user", "assistant", "USER", "ASSISTANT")
            ]
            recent_history = recent_history[max(0, len(recent_history) - 4):]
            messages.extend(recent_history)

        messages.append(IntentChatMessage.user(question))

        return IntentChatRequest(
            messages=messages,
            temperature=0.1,
            top_p=0.3,
            thinking=False,
            # S4 RAG 改写：关闭思考 + 严格 JSON schema 输出（3 字段：rewrite/should_split/sub_questions）
            response_format=pydantic_to_openai_response_format(RagRewriteSchema),
        )

    def _parse_rewrite_and_split(self, raw: str) -> Optional[RewriteResult]:
        """S4 RAG 改写 JSON 解析：协议层 response_format 已保证合法 Pydantic，简化解析。

        仍保留 strip_markdown_code_fence 一层兜底（防止老模型偶尔返回 ```json 围栏的异常），
        之后直接走 RagRewriteSchema.model_validate_json 完成字段校验。
        """
        try:
            cleaned_text: str = LLMResponseCleaner.strip_markdown_code_fence(raw or "")
            parsed_struct = RagRewriteSchema.model_validate_json(cleaned_text)

            rewrite_value: str = parsed_struct.rewrite.strip()
            if not rewrite_value:
                return None
            sub_questions_value: List[str] = [
                s.strip() for s in (parsed_struct.sub_questions or [])
                if isinstance(s, str) and s.strip()
            ]
            if len(sub_questions_value) == 0:
                sub_questions_value = [rewrite_value]
            return RewriteResult(
                rewritten_question=rewrite_value,
                sub_questions=sub_questions_value,
            )
        except Exception as parse_error:
            logger.warning(
                "解析改写+拆分结果失败（RagRewriteSchema），raw=%s，err=%s",
                (raw or "")[:200],
                parse_error,
            )
            return None

    def _rule_based_split(self, question: str) -> List[str]:
        parts = re.split(r"[?？。；;\n]+", question)
        parts = [s.strip() for s in parts if s.strip() != ""]

        if len(parts) == 0:
            return [question]
        return [s if s.endswith("？") or s.endswith("?") else s + "？" for s in parts]


# ==============================================================================
# AgentMultiQuestionRewriteService（Agent 编排侧专用子类 · 新增）
# ==============================================================================
@dataclass
class AgentMultiQuestionRewriteService(
    MultiQuestionRewriteService, AgentQueryRewriteService
):
    """Agent 编排侧改写服务实现。

    在 RAG 版 MultiQuestionRewriteService 基础上：
      - 不破坏父类所有对外方法签名（保证 RAG 链路零侵入）；
      - 新增 rewrite_for_agent(agent_chat_context) 接口，
        使用独立的 agent-question-rewrite.st Prompt，返回 AgentRewriteResult
        并额外做"工具名白名单过滤"与"复杂度字段上下界兜底"。

    Attributes（全部继承自父类，无需再声明）:
        llm_service: 统一 IntentLLMService 门面（已通过 dependencies 适配）。
        rag_config_properties: RAG 相关配置开关。
        query_term_mapping_service: 术语映射服务（可 None/空实现）。
        prompt_template_loader: .st 模板加载器。
    """

    # 显式步骤提示的正则：匹配"先A再B然后C"、"步骤1 ... 2 ..."等表述
    _EXPLICIT_STEP_HINT_REGEX: re.Pattern = re.compile(
        r"(先|步骤|第\s*\d+\s*步|\d+[\.\)、]\s*)"
    )

    # ------------------------------------------------------------------
    # AgentQueryRewriteService 抽象实现
    # ------------------------------------------------------------------
    @trace_to_markdown(output_file="rewrite_for_agent.md")
    def rewrite_for_agent(
        self,
        agent_chat_context: AgentChatContext,
    ) -> AgentRewriteResult:
        """Agent 编排侧改写：产出改写问题 + 拆分 + 复杂度分析 + 工具建议。

        流程：
          1) 术语映射 normalize；
          2) 从用户问题原文提取 explicit_plan_hint（规则层面快速抽，供后续 LLM
             校验）；
          3) 加载 agent-question-rewrite.st，注入 available_tool_list 和
             conversation_history；
          4) 调 LLM；
          5) 解析 JSON，做白名单/上下界兜底 → 返回 AgentRewriteResult。
        """
        original_question: str = agent_chat_context.original_user_question or ""
        normalized_question: str = self.query_term_mapping_service.normalize(
            original_question
        ) if self.query_term_mapping_service is not None else original_question

        # 规则侧步骤提示（作为 fallback + Prompt hint 双重用途）
        pre_rule_plan_hint: Optional[str] = self._extract_plan_hint_if_present(
            original_question
        )

        fallback_result: AgentRewriteResult = AgentRewriteResult(
            rewritten_question=normalized_question or original_question,
            # 规则兜底同样带上目标：否则 Pipeline 写 slots 时该键缺失，
            # 下游读取侧只能退到 user_input，与主链路的取值口径不一致。
            agent_goal=normalize_agent_goal(
                "", normalized_question or original_question
            ),
            should_split=False,
            sub_questions=[normalized_question or original_question],
            complexity_analysis=TaskComplexityAnalysis(
                estimated_steps=1,
                estimated_tool_calls=0,
                has_multi_step_dependency=False,
                has_external_data_dependency=False,
                need_creative_output=False,
                reasoning_notes=(
                    "Agent 改写 LLM 失败，走归一化兜底；未命中 explicit_plan_hint。"
                ),
            ),
            suggested_tools=[],
            explicit_plan_hint=pre_rule_plan_hint,
        )

        if (
            self.rag_config_properties is not None
            and self.rag_config_properties.query_rewrite_enabled is False
        ):
            # 配置层禁用了改写：直接返回 rule-only fallback，不调 LLM
            return fallback_result

        try:
            system_prompt: str = self.prompt_template_loader.load(
                AGENT_QUESTION_REWRITE_PROMPT_PATH
            )
        except Exception as prompt_load_error:
            logger.warning(
                "加载 Agent 改写 Prompt 模板失败，路径=%s，异常=%s。走 fallback。",
                AGENT_QUESTION_REWRITE_PROMPT_PATH,
                prompt_load_error,
            )
            return fallback_result

        try:
            request_payload: IntentChatRequest = self._build_agent_rewrite_request(
                system_prompt=system_prompt,
                normalized_question=normalized_question,
                conversation_history=list(agent_chat_context.conversation_history or []),
                available_tool_ids=list(agent_chat_context.available_tool_ids or []),
                pre_rule_plan_hint=pre_rule_plan_hint,
                available_skills=list(agent_chat_context.available_skills or []),
            )
            raw_response_text: str = self.llm_service.chat(
                request_payload, IntentChoiceTier.FAST
            )
            parsed_result: Optional[AgentRewriteResult] = self._parse_agent_rewrite(
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
        except Exception as llm_error:
            logger.warning(
                "Agent 改写 LLM 调用失败，original='{}'，异常={}。使用 fallback。",
                original_question,
                llm_error,
            )
            parsed_result = None

        final_result: AgentRewriteResult = parsed_result or fallback_result

        logger.info(
            "Agent 改写完成：\n  原问题：%s\n  改写后：%s\n  是否拆分：%s，子问题数=%d\n  "
            "预估步骤=%d，预估工具调用=%d\n  建议工具=%s\n  explicit_plan_hint=%s",
            original_question,
            final_result.rewritten_question,
            final_result.should_split,
            len(final_result.sub_questions),
            final_result.complexity_analysis.estimated_steps,
            final_result.complexity_analysis.estimated_tool_calls,
            final_result.suggested_tools,
            final_result.explicit_plan_hint,
        )
        return final_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_agent_rewrite_request(
        self,
        system_prompt: str,
        normalized_question: str,
        conversation_history: List[IntentChatMessage],
        available_tool_ids: List[str],
        pre_rule_plan_hint: Optional[str],
        available_skills: Optional[List[Dict[str, Any]]] = None,
    ) -> IntentChatRequest:
        """构造 Agent 改写请求。

        Prompt 中的 3 个变量 {conversation_history} / {available_tool_list} /
        {rule_level_plan_hint}（+ 可选 {skill_list}）在这里用显式替换完成，
        避免依赖 PromptTemplateLoader 的高级能力。
        """

        # 1) 渲染 system_prompt 中的模板变量
        rendered_system_prompt: str = self._render_agent_rewrite_template(
            system_prompt_template=system_prompt,
            conversation_history=conversation_history,
            available_tool_ids=available_tool_ids,
            pre_rule_plan_hint=pre_rule_plan_hint,
            available_skills=available_skills,
        )

        # 2) 按 system + 最近 8 条历史(只看 user) + user 顺序组装消息
        messages: List[IntentChatMessage] = []
        if rendered_system_prompt and rendered_system_prompt.strip():
            messages.append(IntentChatMessage.system(rendered_system_prompt))

        recent_user_messages: List[IntentChatMessage] = [
            msg for msg in conversation_history
            if str(msg.role).lower() == "user"
        ]
        # 只取最近 8 条 USER 消息用于指代消解；避免上下文过长。
        if len(recent_user_messages) > 8:
            recent_user_messages = recent_user_messages[-8:]
        messages.extend(recent_user_messages)

        messages.append(IntentChatMessage.user(normalized_question))

        return IntentChatRequest(
            messages=messages,
            temperature=0.1,
            top_p=0.3,
            thinking=False,
            # S1 Agent 改写：关闭思考 + 严格 JSON schema 输出（6 顶层字段 + 嵌套 complexity）
            response_format=pydantic_to_openai_response_format(AgentRewriteSchema),
        )

    def _render_agent_rewrite_template(
        self,
        system_prompt_template: str,
        conversation_history: List[IntentChatMessage],
        available_tool_ids: List[str],
        pre_rule_plan_hint: Optional[str],
        extra_variables: Optional[Dict[str, Any]] = None,
        available_skills: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """对 Prompt 模板的 3 个变量做简单的 str.format_map 渲染。

        extra_variables：（可选）子类追加的额外模板变量（如组合 Prompt 的
        {intent_list}），会与 3 个基础变量合并后一起渲染。
        available_skills：（可选）技能清单（{name, description} 列表），渲染为
        {skill_list} 注入改写 Prompt，让 LLM 依据它挑选 suggested_skills。
        """

        # conversation_history：只保留最近 10 条 user/assistant，防止 prompt 爆炸。
        trimmed_history: List[IntentChatMessage] = [
            msg for msg in conversation_history
            if str(msg.role).lower() in ("user", "assistant")
        ]
        if len(trimmed_history) > 10:
            trimmed_history = trimmed_history[-10:]
        if trimmed_history:
            rendered_history = "\n".join(
                f"{idx + 1}. [{msg.role}]: {msg.content}"
                for idx, msg in enumerate(trimmed_history)
            )
        else:
            rendered_history = "（该会话没有多轮历史）"

        # available_tool_list：与注册中心实际名字 1:1 对齐
        merged_tool_set: List[str] = self._merge_tool_lists(
            available_tool_ids, REGISTERED_ENABLED_TOOL_NAMES
        )
        if merged_tool_set:
            rendered_tool_list = "\n".join(
                f"- {tool_name}" for tool_name in merged_tool_set
            )
        else:
            rendered_tool_list = "- （无可用工具）"

        variables: Dict[str, Any] = {
            "conversation_history": rendered_history,
            "available_tool_list": rendered_tool_list,
            "rule_level_plan_hint": pre_rule_plan_hint or "（未检测到显式步骤提示）",
            "skill_list": self._render_skill_list(available_skills),
        }
        if extra_variables:
            variables.update(extra_variables)

        try:
            return system_prompt_template.format_map(variables)
        except (KeyError, IndexError, ValueError) as fmt_error:
            logger.warning(
                "Agent 改写 Prompt format_map 失败（可能存在花括号冲突），"
                "改为仅追加变量尾部。异常=%s",
                fmt_error,
            )
            suffix = (
                "\n\n--- 以下变量未能自动注入，请作为参考文字使用 ---\n"
                "## 多轮对话历史（如有）\n{conversation_history}\n\n"
                "## Agent 当前可用工具列表（供工具建议时参考匹配）\n{available_tool_list}\n\n"
                "## 规则层显式步骤提示\n- {rule_level_plan_hint}\n"
            ).format(**variables)
            return system_prompt_template + suffix

    @staticmethod
    def _render_skill_list(available_skills: Optional[List[Dict[str, Any]]]) -> str:
        """把技能清单（{name, description} 列表）渲染为 {skill_list} 文本。

        供 LLM 依据它挑选 suggested_skills；空清单渲染为「无」。
        """
        skill_entries: List[Dict[str, Any]] = [
            s for s in (available_skills or []) if isinstance(s, dict)
        ]
        if not skill_entries:
            return "- （当前没有可用的高级技能）"
        lines: List[str] = []
        for index, skill in enumerate(skill_entries, start=1):
            skill_name: str = str(skill.get("name") or "").strip()
            if not skill_name:
                continue
            skill_desc: str = str(skill.get("description") or "").strip()
            if skill_desc:
                lines.append(f"{index}. name={skill_name} | description={skill_desc}")
            else:
                lines.append(f"{index}. name={skill_name}")
        return "\n".join(lines) if lines else "- （当前没有可用的高级技能）"

    @staticmethod
    def _merge_tool_lists(*tool_lists: List[str]) -> List[str]:
        """多个工具列表合并去重，并按 REGISTERED_ENABLED_TOOL_NAMES 顺序稳定排序。"""
        union_set: Dict[str, None] = {}
        for one_list in tool_lists:
            for name in one_list or []:
                if isinstance(name, str) and name.strip():
                    union_set.setdefault(name.strip(), None)

        ordered_keys: List[str] = []
        for canonical_name in REGISTERED_ENABLED_TOOL_NAMES:
            if canonical_name in union_set:
                ordered_keys.append(canonical_name)
                union_set.pop(canonical_name)
        ordered_keys.extend(sorted(union_set.keys()))
        return ordered_keys

    def _extract_plan_hint_if_present(self, raw_question: str) -> Optional[str]:
        """规则层面抽显式步骤提示。

        仅用于兜底 & Prompt hint，最终以 LLM 返回为准。
        """
        if raw_question is None or not raw_question.strip():
            return None

        stripped: str = raw_question.strip()
        if self._EXPLICIT_STEP_HINT_REGEX.search(stripped) is None:
            return None
        # 简单裁剪到最多 120 字，避免污染 Prompt
        return stripped[:120]

    def _parse_agent_rewrite(
        self,
        raw_response_text: str,
        fallback_question: str,
        available_tool_ids: List[str],
        pre_rule_plan_hint: Optional[str],
        available_skill_names: Optional[List[str]] = None,
    ) -> Optional[AgentRewriteResult]:
        """S1 Agent 改写解析：response_format 已保证合法 AgentRewriteSchema。

        流程：strip_markdown_code_fence 兼容 → AgentRewriteSchema.model_validate_json 结构化
        → 业务侧再做一遍上下界 clamp（防御 Pydantic schema 未来变更漏约束）+ 工具名
        白名单交集过滤（schema 无法表达「动态白名单」的枚举）→ 返回 AgentRewriteResult。
        兜底：qwen 系模型 response_format 未严格生效时可能输出「数组包裹对象」
        （[{...}]），此时用 coerce_llm_json_to_schema 取首个对象元素二次解析。
        """
        cleaned_text: str = LLMResponseCleaner.strip_markdown_code_fence(
            raw_response_text or ""
        )
        try:
            parsed_struct: AgentRewriteSchema = AgentRewriteSchema.model_validate_json(
                cleaned_text
            )
        except Exception as parse_error:
            fallback_struct = coerce_llm_json_to_schema(AgentRewriteSchema, cleaned_text)
            if fallback_struct is not None and isinstance(fallback_struct, AgentRewriteSchema):
                parsed_struct = fallback_struct
                logger.warning(
                    "Agent 改写输出为数组包裹对象（response_format 未严格执行），"
                    "已取首元素兼容解析。raw=%s",
                    (raw_response_text or "")[:300],
                )
            else:
                logger.warning(
                    "解析 Agent 改写 JSON 失败（AgentRewriteSchema）：%s，raw=%s",
                    parse_error,
                    (raw_response_text or "")[:300],
                )
                return None

        rewrite_value: str = str(parsed_struct.rewrite or "").strip()
        final_rewritten_question: str = rewrite_value or fallback_question

        # 子问题拆分
        should_split_value = bool(parsed_struct.should_split)
        sub_questions_value: List[str] = [
            s.strip() for s in (parsed_struct.sub_questions or [])
            if isinstance(s, str) and s.strip()
        ]
        if not sub_questions_value:
            should_split_value = False
            sub_questions_value = [final_rewritten_question]

        # 复杂度分析：Pydantic schema 已强制 1<=steps<=10 / 0<=tools<=10，
        # 再做一遍 clamp 保持防御风格。
        complexity_struct = parsed_struct.complexity_analysis
        estimated_steps_value: int = max(
            1, min(10, int(complexity_struct.estimated_steps))
        )
        estimated_tool_calls_value: int = max(
            0, min(10, int(complexity_struct.estimated_tool_calls))
        )
        complexity_result = TaskComplexityAnalysis(
            estimated_steps=estimated_steps_value,
            estimated_tool_calls=estimated_tool_calls_value,
            has_multi_step_dependency=bool(complexity_struct.has_multi_step_dependency),
            has_external_data_dependency=bool(complexity_struct.has_external_data_dependency),
            need_creative_output=bool(complexity_struct.need_creative_output),
            reasoning_notes=str(complexity_struct.reasoning_notes or "").strip(),
        )

        # 工具名白名单过滤（动态 ∪ REGISTERED_ENABLED，schema 枚举不了动态值，必须代码层）
        allowed_tool_union: set[str] = set(REGISTERED_ENABLED_TOOL_NAMES)
        for name in available_tool_ids or []:
            if isinstance(name, str):
                allowed_tool_union.add(name.strip())
        suggested_tools_value: List[str] = []
        for tool_name in parsed_struct.suggested_tools or []:
            candidate: str = str(tool_name).strip()
            if (
                candidate
                and candidate in allowed_tool_union
                and candidate not in suggested_tools_value
            ):
                suggested_tools_value.append(candidate)

        # explicit_plan_hint：优先取 LLM；空或 null-like 回退到规则层抽
        plan_hint_raw: Optional[str] = parsed_struct.explicit_plan_hint
        if (
            plan_hint_raw is None
            or (isinstance(plan_hint_raw, str) and plan_hint_raw.lower() in ("", "null", "none"))
        ):
            explicit_plan_hint_value = pre_rule_plan_hint
        else:
            explicit_plan_hint_value = str(plan_hint_raw).strip() or None

        # agent_goal：本轮目标锚点。归一化**只在这里做一次**（strip → 截断 →
        # 空值回退为改写后问题），下游三个注入点只读不各自兜底——否则会出现
        # "一处回退到改写后问题、另一处注入空串"的漂移（design.md D5）。
        agent_goal_value: str = normalize_agent_goal(
            parsed_struct.agent_goal, final_rewritten_question
        )

        # 技能名白名单过滤（与注入清单 name 逐字比对，防 LLM 幻觉自造技能名）
        skill_whitelist: set[str] = {
            str(skill_name).strip()
            for skill_name in (available_skill_names or [])
            if isinstance(skill_name, str) and skill_name.strip()
        }
        suggested_skills_value: List[str] = []
        for skill_name in parsed_struct.suggested_skills or []:
            candidate_skill: str = str(skill_name).strip()
            if (
                candidate_skill
                and candidate_skill in skill_whitelist
                and candidate_skill not in suggested_skills_value
            ):
                suggested_skills_value.append(candidate_skill)

        return AgentRewriteResult(
            rewritten_question=final_rewritten_question,
            agent_goal=agent_goal_value,
            should_split=should_split_value,
            sub_questions=sub_questions_value,
            complexity_analysis=complexity_result,
            suggested_tools=suggested_tools_value,
            explicit_plan_hint=explicit_plan_hint_value,
            suggested_skills=suggested_skills_value,
        )
