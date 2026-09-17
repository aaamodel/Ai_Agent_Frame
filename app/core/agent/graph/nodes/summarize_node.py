# -*- coding: utf-8 -*-
"""summarize 节点：plan 子任务结论汇总 + L3 证据充分性自判；react 终局归一。

一次 LLM 调用同时产出「最终答案」与「证据是否充分」结构化判定（0 额外调用）：

- sufficient=True  → final_answer → persist；
- sufficient=False 且仍有 replan 余量 → 写 insufficiency_signal/draft_answer，
  经 route_after_summarize 去 replan 补取（携带缺口说明与建议数据源）；
- sufficient=False 但余量耗尽 → 用草稿收尾（无草稿用 GRACEFUL），degraded=True。

任何异常都不抛出图：解析失败 fail-open（把原文当答案），LLM 失败给友好文案。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field

from app.core.agent.graph.approval import danger_tool_names
from app.core.agent.graph.deps import get_deps
from app.core.agent.graph.nodes._common import trace_event
from app.core.agent.graph.state import AgentGraphState, replan_capacity
from app.core.agent.react_agent import GRACEFUL_TIMEOUT_MESSAGE, _mini_json_repair
from app.query_intent.llm_schemas import pydantic_to_openai_response_format

# 与 execute_node.PLAN_BAD_STATUSES 同源（本地副本避免 nodes 包间循环导入）：
# 写操作子任务只要落在这些状态里，就代表"动作没完成"。
_WRITE_BAD_STATUSES = frozenset({"empty_data", "error", "budget_denied", "approval_denied"})


class SummaryVerdictSchema(BaseModel):
    """汇总 + 证据闸门合并输出（response_format 约束）。"""

    sufficient: bool = Field(
        description="现有子任务结论是否足以直接、完整回答用户原始问题。"
                    "查询/分析类任务：能给出部分可靠答案也为 true；"
                    "修改/写入/导出等操作类任务：对应写操作没有成功落盘时必须为 false"
    )
    answer: str = Field(
        description="sufficient=true 时的最终答案；false 时可给草稿或留空"
    )
    missing_info: str = Field(
        default="", description="sufficient=false 时：具体缺少什么信息"
    )
    suggestion: str = Field(
        default="", description="sufficient=false 时：建议尝试的其他数据源/工具方向"
    )


_SUMMARY_SYSTEM_PROMPT = """你是最终总结助手，同时负责"证据充分性"判定。请严格按以下顺序工作：

1. 逐条审视各子任务结论，先在心里给每条打标签：相关 / 空数据 / 错误 / 跑题
   （跑题=返回了内容但与用户问题的核心实体、指标、时间无关）。
2. 汇总所有"相关"结论，回答用户原始问题。
3. 判定 sufficient：
   - sufficient=false 的**唯一**情形：没有任何一条子任务结论能支撑回答（全部为空数据/错误/跑题）。
   - 只要能给出部分可靠答案，就必须 sufficient=true，并在答案中说明已知部分与缺口。
   - **操作类任务硬规则**：用户要的是"修改/更新/写入/导出/删除"等动作（不是查信息）时，
     只要对应的写操作子任务（local_excel_write_tool / sales_report_export_tool 等）状态
     不是 ok（error、审批拒绝、empty_data 等），sufficient 一律为 false——
     **严禁把"更新失败/未执行，请手动修改"包装成成功答复**；应说明哪一步失败、为什么失败。
   - 严禁偷懒（动辄 false 回避作答），也严禁拿着跑题数据编造答案。

只输出一个 JSON 对象，不要输出 JSON 以外的任何文字：
{
  "sufficient": true 或 false,
  "answer": "最终答案（sufficient=false 时可给草稿或空字符串）",
  "missing_info": "false 时具体缺少什么，true 时空字符串",
  "suggestion": "false 时建议尝试的其他数据源/工具，true 时空字符串"
}"""


def _build_evidence(subtask_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把每步记录整理成给汇总模型的证据清单（坏步带状态标签与原始观测摘要）。"""
    evidence: List[Dict[str, Any]] = []
    for r in subtask_results:
        item: Dict[str, Any] = {
            "step": r.get("title") or r.get("subtask_id"),
            "status": r.get("status") or "ok",
        }
        if r.get("llm_output"):
            item["conclusion"] = r["llm_output"]
        elif r.get("observation"):
            item["raw_observation"] = str(r["observation"])[:500]
        evidence.append(item)
    return evidence


def _failed_write_ops(
    subtask_results: List[Dict[str, Any]], danger_names: set
) -> List[Dict[str, Any]]:
    """确定性闸门：找出"调用了写/导出类危险工具但没成功"的子任务。

    只看结构化的 tool_name + status，**不信任 LLM 自己判的 sufficient**：
    实测 trace 中写操作 status=error，汇总层却输出 sufficient=true 并答复
    "请手动修改"。操作类任务的完成度必须由执行状态决定，而不是由文案决定。
    """
    failed: List[Dict[str, Any]] = []
    for r in subtask_results:
        tool_name = r.get("tool_name")
        if not tool_name or tool_name not in danger_names:
            continue
        status = str(r.get("status") or "ok")
        if status in _WRITE_BAD_STATUSES:
            failed.append({
                "tool": tool_name,
                "status": status,
                "step": r.get("title") or r.get("subtask_id"),
            })
    return failed


def _parse_verdict(raw_text: str) -> Dict[str, Any]:
    """解析结构化判定；内容不是合法 JSON 时 fail-open 视为 sufficient 原文答案。

    不走 planner._extract_json_object：那个解析器末尾绑定 PlanGenerateSchema
    强类型校验，会把判定 JSON 判为非法。这里用通用的去围栏 + repair + json.loads。
    """
    cleaned: str = (raw_text or "").strip()
    if cleaned.startswith("```"):
        first_line_end = cleaned.find("\n")
        if first_line_end >= 0:
            cleaned = cleaned[first_line_end + 1:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
    try:
        parsed: Any = json.loads(_mini_json_repair(cleaned))
    except Exception:  # noqa: BLE001 - 非 JSON 输出：当普通答案处理，绝不丢答案
        parsed = None
    if not isinstance(parsed, dict):
        return {"sufficient": True, "answer": raw_text.strip(),
                "missing_info": "", "suggestion": ""}
    return {
        "sufficient": bool(parsed.get("sufficient", True)),
        "answer": str(parsed.get("answer") or "").strip(),
        "missing_info": str(parsed.get("missing_info") or "").strip(),
        "suggestion": str(parsed.get("suggestion") or "").strip(),
    }


async def summarize_node(state: AgentGraphState, config: RunnableConfig) -> dict:
    deps = get_deps(config)
    trace_id: str = state["trace_id"]
    query: str = state["user_input"]

    # 1. react Final Answer / 已有终局答案：仅归一，零额外调用
    existing_answer: str = (state.get("final_answer") or "").strip()
    if existing_answer:
        return {"final_answer": existing_answer, "insufficiency_signal": None}

    # 2. plan 路径：汇总全部子任务结论（含坏步状态标签）
    subtask_results: List[Dict[str, Any]] = state.get("subtask_results") or []
    evidence: List[Dict[str, Any]] = _build_evidence(subtask_results)

    if not evidence:
        # 空计划/起步即失败且无任何留痕
        logger.warning("summarize 无任何子任务记录可汇总，返回友好降级文案。")
        trace_event(
            deps.tracer, trace_id, "summarize.graceful",
            {"last_error": state.get("last_error"),
             "insufficiency_signal": state.get("insufficiency_signal")},
        )
        return {"final_answer": GRACEFUL_TIMEOUT_MESSAGE, "success": False,
                "insufficiency_signal": None}

    summary_msgs = [
        {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"原始问题：{query}\n各步骤执行记录：\n"
                       f"{json.dumps(evidence, ensure_ascii=False, indent=2)}",
        },
    ]
    try:
        resp = await deps.model_router.chat(
            messages=summary_msgs,
            purpose_hint="planner",
            thinking=False,
            temperature=0.2,
            response_format=pydantic_to_openai_response_format(SummaryVerdictSchema),
        )
        raw_text: str = (getattr(resp, "content", None) or "").strip()
    except Exception as exc:  # noqa: BLE001 - 汇总 LLM 失败：友好兜底，不抛图
        logger.exception("plan 汇总阶段 LLM 调用失败")
        return {
            "final_answer": GRACEFUL_TIMEOUT_MESSAGE,
            "success": False,
            "last_error": f"汇总阶段失败: {exc}",
            "insufficiency_signal": None,
        }

    verdict: Dict[str, Any] = _parse_verdict(raw_text)
    gate_enabled: bool = bool(deps.cfg("agent_evidence_gate_enabled", True))

    # 2.5 确定性后置闸门：写/导出类子任务有失败 → 强制 sufficient=false。
    # 这一步不信任上面 LLM 的判定（prompt 规则可能不被遵守），只看执行状态。
    failed_writes: List[Dict[str, Any]] = _failed_write_ops(
        subtask_results, danger_tool_names(deps)
    )
    if failed_writes and verdict["sufficient"]:
        detail: str = "；".join(
            f"{f['step'] or f['tool']} 的 {f['tool']}（状态 {f['status']}）"
            for f in failed_writes
        )
        logger.warning("写操作子任务失败但 LLM 判定 sufficient=true，后置闸门强制改为 false：{}", detail)
        trace_event(deps.tracer, trace_id, "summarize.write_gate",
                    {"failed_writes": failed_writes})
        verdict["sufficient"] = False
        verdict["missing_info"] = (
            f"用户要求的写/导出操作没有成功完成：{detail}。"
            "操作类任务只有写操作真正落盘成功才算完成，不能以'请手动修改/请自行操作'收尾。"
        )
        if not verdict["suggestion"]:
            verdict["suggestion"] = (
                "根据错误信息修正参数后重试写操作；修改某条记录优先使用 "
                "local_excel_write_tool 的 filter_column+filter_value+target_column+new_value "
                "语义模式（不要自己算 A1 坐标）；审批被拒则不要重复提交。"
            )

    # 3a. 证据充分（或闸门关闭）：直接收尾。
    # 注意：写操作硬闸门不受"证据闸门开关"影响——动作没完成就是没完成。
    if (not gate_enabled or verdict["sufficient"]) and not failed_writes:
        answer: str = verdict["answer"] or raw_text or GRACEFUL_TIMEOUT_MESSAGE
        trace_event(deps.tracer, trace_id, "summarize.done", {"answer_len": len(answer)})
        return {
            "final_answer": answer,
            "success": bool(answer),
            "insufficiency_signal": None,
            "draft_answer": None,
        }

    # 3b. 证据不足：有 replan 余量 → 带缺口说明回炉；无余量 → 草稿/GRACEFUL 降级收尾
    draft: str = verdict["answer"]
    if failed_writes:
        signal: str = (
            f"用户要求的写/导出操作尚未成功完成。\n"
            f"- 失败详情：{verdict['missing_info']}\n"
            f"- 修正方向：{verdict['suggestion'] or '根据错误信息修正后重试写操作'}\n"
            "请安排子任务修正后重新执行写操作；不要重复已知会失败的调用，"
            "更不要把'请用户手动修改'作为答复。"
        )
    else:
        signal = (
            f"汇总阶段判定现有证据不足以回答用户问题。\n"
            f"- 缺少的信息：{verdict['missing_info'] or '（模型未具体说明）'}\n"
            f"- 建议方向：{verdict['suggestion'] or '尝试其他可用数据源'}\n"
            "请据此换用其他数据源补充取数；已尝试且无效的工具不要重复调用；"
            "若确无其他数据源，则基于已有信息给出诚实的部分答案，严禁编造。"
        )
    if replan_capacity(state):
        logger.info("summarize L3 判定证据不足，带缺口说明回炉 replan。")
        trace_event(deps.tracer, trace_id, "summarize.insufficient",
                    {"missing_info": verdict["missing_info"],
                     "suggestion": verdict["suggestion"],
                     "failed_writes": failed_writes})
        return {
            "insufficiency_signal": signal,
            "draft_answer": draft or None,
            "final_answer": "",
            "success": False,
        }

    if failed_writes:
        # 写操作失败且余量耗尽：明确告知未完成，success 必须是 False（trace 事故里
        # 旧逻辑把"更新失败，请手动修改"当成功答案返回，这是对用户的误导）。
        final_text: str = (
            f"抱歉，该操作未能成功完成。{verdict['missing_info']}"
            + (f"\n\n以下是执行过程中得到的信息，供参考：\n{draft}" if draft else "")
        )
    else:
        final_text = draft or GRACEFUL_TIMEOUT_MESSAGE
    logger.warning("summarize 判定证据不足但 replan 余量已耗尽，用草稿/友好文案降级收尾。")
    trace_event(deps.tracer, trace_id, "summarize.exhausted",
                {"has_draft": bool(draft), "failed_writes": failed_writes})
    return {
        "final_answer": final_text,
        "success": bool(draft) and not failed_writes,
        "degraded": True,
        "insufficiency_signal": None,
        "draft_answer": draft or None,
    }
