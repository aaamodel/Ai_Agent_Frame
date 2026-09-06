# -*- coding: utf-8 -*-
"""
反思 Agent：对输出做质量检查、幻觉与完整性评估，并给出改进建议。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

# 本轮结构化输出：S7 Reflection 审查 schema + 协议转换
from app.query_intent.llm_schemas import (
    ReflectionReportSchema,
    pydantic_to_openai_response_format,
)


REFLECTION_SYSTEM_PROMPT = """你是严格的输出质量审查员。请对用户问题与助手回答进行审查。

## 你必须只输出一个 JSON 对象，格式如下：
{
  "quality_score": 0-100 的整数,
  "is_complete": true/false,
  "likely_hallucination": true/false,
  "hallucination_reasons": ["若怀疑幻觉，列出具体疑点；否则为空数组"],
  "completeness_notes": "是否遗漏关键要点",
  "suggestions": ["可执行的改进建议，面向助手"],
  "summary": "一句中文总结审查结论"
}

审查标准：
- 幻觉：回答是否包含无依据的具体事实、虚构来源或与用户问题无关的断言。
- 完整性：是否覆盖用户问题的核心子问题。
- 质量分：综合考虑正确性、清晰度与有用性。
"""


@dataclass
class ReflectionReport:
    """反思审查报告。"""

    quality_score: int
    is_complete: bool
    likely_hallucination: bool
    hallucination_reasons: List[str]
    completeness_notes: str
    suggestions: List[str]
    summary: str
    raw_model_output: Optional[str] = None
    parse_error: Optional[str] = None


class ReflectionLLM(Protocol):
    """反思阶段使用的 LLM。"""

    async def acomplete(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> str:
        ...


def _extract_json(text: str) -> ReflectionReportSchema:
    """S7 Reflection 解析：response_format 已保证 ReflectionReportSchema 合法 JSON。

    使用 Pydantic model_validate_json 一步完成校验+类型转换；不再需要 json.loads/正则抓大括号。
    为兼容极少数旧模型仍输出 Markdown 围栏，先用简单字符串 strip 去围栏。
    """
    cleaned_text: str = (text or "").strip()
    # 去 Markdown 围栏（```json ... ``` / ``` ... ```）
    if cleaned_text.startswith("```"):
        first_line_end = cleaned_text.find("\n")
        if first_line_end >= 0:
            cleaned_text = cleaned_text[first_line_end + 1:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3].rstrip()
    return ReflectionReportSchema.model_validate_json(cleaned_text)


class ReflectionAgent:
    """对 Agent 最终输出进行反思与质量把关。

    P1① 扁平化：优先直接持有 ModelRouter（Agent → ModelRouter.chat 2 层链路），
    兼容旧 `llm`（ReflectionLLM Protocol）作为回退。
    旧嵌套：Agent → orchestrator._LLMAdapter → _PurposeLLMAdapter → ModelRouter（4 层）
    新链路：Agent → ModelRouter.chat（2 层，减少 2 层包装）
    """

    def __init__(
        self,
        llm: Optional[ReflectionLLM] = None,
        min_quality_to_pass: int = 60,
        *,
        model_router: Optional[Any] = None,
        purpose_hint: str = "reflection",
    ) -> None:
        if llm is None and model_router is None:
            raise ValueError("ReflectionAgent 需要提供 llm 或 model_router 至少其一")
        self._llm = llm
        self._model_router = model_router
        self._purpose: str = purpose_hint
        self.min_quality_to_pass = max(0, min(100, min_quality_to_pass))

    async def _llm_chat(self, messages, **kwargs) -> str:
        """统一 LLM 调用入口：优先 ModelRouter，否则回退旧 ReflectionLLM.acomplete。"""
        if self._model_router is not None:
            resp = await self._model_router.chat(
                messages=list(messages), purpose_hint=self._purpose, **kwargs
            )
            return (getattr(resp, "content", None) or "").strip()
        return await self._llm.acomplete(messages, **kwargs)

    async def reflect(
        self,
        user_query: str,
        agent_answer: str,
        evidence_snippets: Optional[List[str]] = None,
        trace_summary: Optional[str] = None,
    ) -> ReflectionReport:
        """
        对助手答案进行质量检查。

        evidence_snippets：可选的检索或工具观察片段，用于对照幻觉。
        trace_summary：可选的执行轨迹摘要，用于判断推理是否支撑结论。
        """
        ev_block = "\n".join(f"- {s}" for s in (evidence_snippets or [])) or "（无外部证据）"
        trace_block = trace_summary or "（无轨迹）"
        user_content = f"""## 用户问题
{user_query}

## 助手回答
{agent_answer}

## 外部证据/观察片段
{ev_block}

## 执行轨迹摘要
{trace_block}

请输出 JSON 审查结果。"""

        messages: Sequence[Dict[str, str]] = [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            # S7 Reflection：关闭思考 + 严格 JSON schema（quality_score 0~100 + 6 字段），
            # 协议层保证合法，减少解析失败率。
            raw = await self._llm_chat(
                messages,
                temperature=0.1,
                thinking=False,
                response_format=pydantic_to_openai_response_format(ReflectionReportSchema),
            )
            parsed_struct: ReflectionReportSchema = _extract_json(raw)
        except Exception as e:  # noqa: BLE001
            logger.exception("反思模型调用或解析失败")
            return ReflectionReport(
                quality_score=50,
                is_complete=False,
                likely_hallucination=False,
                hallucination_reasons=[],
                completeness_notes="反思阶段解析失败，无法完成自动审查",
                suggestions=["请人工复核该回答"],
                summary="反思流程异常，已降级为保守评分",
                raw_model_output=None,
                parse_error=str(e),
            )

        report = ReflectionReport(
            # schema 已强制 int 0-100；转 int + 再夹一遍防御 float/非 int
            quality_score=max(0, min(100, int(parsed_struct.quality_score))),
            is_complete=bool(parsed_struct.is_complete),
            likely_hallucination=bool(parsed_struct.likely_hallucination),
            hallucination_reasons=list(parsed_struct.hallucination_reasons or []),
            completeness_notes=str(parsed_struct.completeness_notes or ""),
            suggestions=list(parsed_struct.suggestions or []),
            summary=str(parsed_struct.summary or ""),
            raw_model_output=raw[:8000],
        )
        return report

    def should_retry_or_warn(self, report: ReflectionReport) -> Dict[str, Any]:
        """
        根据报告给出是否建议重试/告警的企业级决策结构。
        """
        warn = report.quality_score < self.min_quality_to_pass
        retry = report.likely_hallucination or not report.is_complete
        return {
            "warn_low_quality": warn,
            "suggest_retry": retry,
            "min_quality_threshold": self.min_quality_to_pass,
        }
