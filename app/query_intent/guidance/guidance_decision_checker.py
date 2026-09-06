import logging
from typing import List
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from app.query_intent.intent_classify_resolver.intent_model import NodeScore
from app.query_intent.intent_data_base import IntentChatRequest, IntentChatMessage, IntentChoiceTier
from app.query_intent.intent_service import IntentLLMService

from app.query_intent.intent_utils import LLMResponseCleaner

from app.query_intent.intent_prompt.prompt_template_loader import PromptTemplateLoader
from app.query_intent.rag_constant import GUIDANCE_AMBIGUITY_CHECK_PROMPT_PATH
# 本轮结构化输出：S5 歧义澄清检查 schema + 协议转换
from app.query_intent.llm_schemas import (
    AmbiguityCheckSchema,
    pydantic_to_openai_response_format,
)

logger = logging.getLogger(__name__)


def _blank_to_default(s: str, default: str) -> str:
    return s if (s is not None and s.strip() != "") else default


def _is_not_blank(s: str) -> bool:
    return s is not None and s.strip() != ""


@dataclass
class AmbiguityLLMChecker:
    llm_service: IntentLLMService
    prompt_template_loader: PromptTemplateLoader

    def check_ambiguity(self, question: str, ranked: List[NodeScore]) -> bool:
        candidates_text = self._build_candidates_text(ranked)
        prompt = self.prompt_template_loader.render(
            GUIDANCE_AMBIGUITY_CHECK_PROMPT_PATH,
            {
                "question": question,
                "candidates": candidates_text
            }
        )

        request = IntentChatRequest(
            messages=[IntentChatMessage.user(prompt)],
            temperature=0.1,
            top_p=0.3,
            thinking=False,
            # S5 歧义澄清：关闭思考 + 严格 JSON schema（ambiguous/category_ids/reason）
            response_format=pydantic_to_openai_response_format(AmbiguityCheckSchema),
        )

        try:
            raw = self.llm_service.chat(request, IntentChoiceTier.FAST)
            cleaned = LLMResponseCleaner.strip_markdown_code_fence(raw)
            parsed_struct: AmbiguityCheckSchema = AmbiguityCheckSchema.model_validate_json(
                cleaned
            )

            ambiguous_value = bool(parsed_struct.ambiguous)
            reason_value = str(parsed_struct.reason or "") if parsed_struct.reason else ""
            logger.info(
                "LLM 歧义确认结果: ambiguous={}, reason={}, question={}",
                ambiguous_value,
                reason_value,
                question,
            )
            return ambiguous_value

        except Exception as e:
            logger.warning(
                "歧义确认 LLM 调用或解析失败（AmbiguityCheckSchema）, 降级为跳过澄清, "
                "question={}, err={}",
                question,
                e,
            )
            return False

    def _build_candidates_text(self, ranked: List[NodeScore]) -> str:
        lines = []
        for ns in ranked:
            node = ns.node
            full_path = _blank_to_default(node.full_path, node.name if node.name is not None else "")
            line = "- 意图ID: {}, 名称: {}, 完整路径: {}".format(node.id, node.name, full_path)
            if _is_not_blank(node.description):
                line += ", 说明: {}".format(node.description)
            line += ", 匹配分数: {:.2f}".format(ns.score)
            lines.append(line)
        return "\n".join(lines)



class Action(Enum):
    NONE = "NONE"
    PROMPT = "PROMPT"


@dataclass
class GuidanceDecision:
    action: Action
    prompt: Optional[str] = None

    @staticmethod
    def none() -> "GuidanceDecision":
        return GuidanceDecision(action=Action.NONE, prompt=None)

    @staticmethod
    def prompt(prompt: str) -> "GuidanceDecision":
        return GuidanceDecision(action=Action.PROMPT, prompt=prompt)

    def is_prompt(self) -> bool:
        return self.action == Action.PROMPT

