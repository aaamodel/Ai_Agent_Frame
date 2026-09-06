# -*- coding: utf-8 -*-
"""
规划 Agent：Plan-and-Execute，含任务分解、执行与重规划。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.core.agent.react_agent import AgentResult
from app.core.agent.react_agent import ToolInvoker, LLMCallable, postprocess_tool_result
from app.core.tools.base import tool_to_function_call_definition

# 本轮结构化输出：MID-1 Planner 初始计划/重计划 schema + 协议转换
from app.query_intent.llm_schemas import (
    PlanGenerateSchema,
    pydantic_to_openai_response_format,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 【方案B】空业务结果触发重规划（非异常型）
# ---------------------------------------------------------------------------
# 该错误前缀用于标记“数据源工具返回空数据（非调用错误）”导致的 replan 原因。
# 与 API 异常/网络抖动/熔断不同，它不是为了修复错误，而是为了让模型换一个
# 可用数据源重新取数，避免拿着空数据硬造后续步骤或生成占位结果。
EMPTY_DATASOURCE_REPLAN_PREFIX: str = "EMPTY_DATASOURCE_RESULT"


def _is_empty_data(obs_text: str) -> bool:
    """保守判定一次工具观测结果是否属于“空业务数据”（非异常）。返回 True 时触发 replan。

    判定策略（宁缺毋滥，避免误伤正常执行）：
      1. 空白 / 空字符串 → 判空；
      2. 可解析 JSON 且为【空数组】或【空对象】→ 判空；
      3. 短文本（<=60 字符）且含明确的“无数据”语义关键词 → 判空。

    Args:
        obs_text: 工具返回并已字符串化的观测内容。

    Returns:
        True 表示应视为空业务数据。
    """
    text: str = (obs_text or "").strip()
    if not text:
        return True

    try:
        parsed_value: Any = json.loads(text)
        if isinstance(parsed_value, (list, dict)) and len(parsed_value) == 0:
            return True
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    lowered: str = text.lower()
    if len(text) <= 60 and any(
            marker in lowered for marker in (
                "无数据", "没有数据", "未找到", "返回为空", "无有效数据", "空列表",
                "暂无数据", "未查询到", "0条", "0 条", "no data", "no data found",
                "not found", "empty list",
            )
    ):
        return True

    return False

# 保持原有 Prompt 结构不变...
PLAN_SYSTEM_PROMPT = """你是一个高瞻远瞩的规划专家。你需要将用户的复杂目标拆解为具体的、可执行的子任务（subtasks）。
请务必结合用户提供的多轮对话历史、长期记忆事实，以及当前可用的高级技能树，进行合理的任务拆解。

【严格输出规则 · 必须遵守】
  1. 你的整个回复只能是一个合法的 JSON 对象，不得输出任何 JSON 以外的前缀、后缀、
     解释文字、Markdown 围栏（```json / ```）或思考过程。
  2. JSON 顶层必须包含且仅包含一个字段 "subtasks"，其值为子任务对象数组。
  3. 每个子任务对象字段严格如下：
       - id            : string  子任务唯一 ID
       - title         : string  子任务一句话标题
       - description   : string  子任务详细说明
       - action_type   : string  枚举 "tool" 或 "reasoning"
       - tool_name     : string  （仅 action_type="tool" 必填）工具名称，必须属于给定可用工具
       - tool_args_hint: string  （可选）建议传入工具的参数结构或具体值
  4. 不得出现尾随逗号、单引号作为字段引号、未闭合的花/方括号等 JSON 语法错误。
  5.（备用数据源约束）在规划取数类工具时，请评估该数据源可能不可用/返回空数据（例如鉴权失败、
     无权限、无记录）的风险；如确有该风险，请在计划中保留一个可行的备用数据源备选，
     避免把整个计划的成败压在一个数据源上。

输出示例：
{
  "subtasks": [
    {"id": "task_1", "title": "...", "description": "...", "action_type": "tool/reasoning", "tool_name": "...", "tool_args_hint": "..."}
  ]
}
"""

REPLAN_SYSTEM_PROMPT = """你是一个动态调整与重规划专家。当执行过程中遭遇异常或无法达成预期时，你需要根据当前已有的执行结果以及发生的错误，对剩余的子任务进行修订和重新编排。

【严格输出规则 · 必须遵守】
  1. 你的整个回复只能是一个合法的 JSON 对象，不得输出任何 JSON 以外的前缀、后缀、
     解释文字、Markdown 围栏（```json / ```）或思考过程。
  2. JSON 顶层必须包含且仅包含一个字段 "subtasks"，其值为修订后的子任务对象数组。
     已判定成功完成的子任务请从新计划中移除，只保留需要重做 / 调整顺序 / 新增的子任务。
  3. 每个子任务字段同初始计划规则：id / title / description / action_type（tool|reasoning）/
     tool_name（tool 必填）/ tool_args_hint（可选）。
  4. 不得出现尾随逗号、单引号作为字段引号、未闭合的花/方括号等 JSON 语法错误。
  5.（空数据回退）若既有的执行结果显示某数据源工具返回了【空数据】（例如空数组 / 空对象 /
     空字符串，而非报错），则修订计划中必须改用另一个可用数据源工具（例如从飞书多维表切换到
     本地 Excel / RAG 知识库）重新获取数据，严禁基于空数据继续编造后续步骤或凭空生成占位结果。

输出格式同样为严格的 JSON 对象。"""


@dataclass
class SubTask:
    id: str
    title: str
    description: str
    action_type: str
    tool_name: Optional[str] = None
    tool_args_hint: Optional[str] = None


@dataclass
class PlanExecuteState:
    plan: List[SubTask] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)


def _mini_json_repair(raw: str) -> str:
    """LLM JSON 场景专属的轻量修复（不依赖第三方 json_repair 库）。

    针对 LLM 输出 JSON 时最高频的 4 类病态做纯标准库修复：
      1. 顶层杂文字剥离："Sure, here you go: {...}" / "... \n{...}" → 取第一个 { 到最后一个 } 切片
      2. Markdown 围栏残余：`````` / ````json / ```` 残留
      3. 尾随逗号（trailing comma）：`{ "a": 1, }` / `[1,2,3,]` → 去掉 } / ] 之前的最后一个逗号
      4. 单引号对象字段：`{'key': 'value'}` → 粗暴替换所有非内容场景的单引号为双引号
         （对英文 contractions 误伤风险极低：Plan JSON / Action Input 字段一般不出现 don't / can't）

    如果修复失败，原样返回（由下游 json.loads / Pydantic 再抛异常交给 Retry / fallback 处理）。

    Args:
        raw: 待修复的原始字符串（可能含 JSON）

    Returns:
        修复后的字符串（若无法判定则原样返回）
    """
    working: str = raw or ""
    if not working:
        return working

    # Step 1: 去 Markdown 围栏（包括 ```json / ``` / ```` 混合）
    if "```" in working:
        # 去除所有 ``` 开头的行
        lines: List[str] = working.splitlines()
        clean_lines: List[str] = [ln for ln in lines if not ln.lstrip().startswith("```")]
        working = "\n".join(clean_lines)

    # Step 2: 顶层花/方括号切片，剥离前后杂文字（优先对象 {}，其次数组 []）
    first_obj_open: int = working.find("{")
    first_arr_open: int = working.find("[")
    candidate_text: str = working
    if first_obj_open >= 0:
        last_obj_close: int = working.rfind("}")
        if last_obj_close > first_obj_open:
            candidate_text = working[first_obj_open : last_obj_close + 1]
    elif first_arr_open >= 0:
        last_arr_close: int = working.rfind("]")
        if last_arr_close > first_arr_open:
            candidate_text = working[first_arr_open : last_arr_close + 1]
    working = candidate_text

    if not working.strip():
        return raw  # 切片结果为空，退回原样

    # Step 3: 粗暴单引号→双引号（仅用于外层字段/值，不处理双引号嵌套里的单引号）
    # 先用双引号做保护：把已经是双引号字符串的内容暂时替换成占位符太复杂；
    # 直接做一次简单替换：所有单引号 → 双引号。
    # 风险：如果 JSON 字符串值里包含 "He said 'hi'" 这种嵌套会坏，LLM 输出很少见，可接受。
    if "'" in working and '"' not in working[:50]:  # heuristic：如果前 50 字符没有双引号基本是单引号风格
        working = working.replace("'", '"')

    # Step 4: 尾随逗号修复 — 从右向左扫描，遇到 `,` 紧邻 `}` / `]`（跳过空白）就删除
    chars: List[str] = list(working)
    n: int = len(chars)
    # 多次扫描处理 `[[1,2,],]` 这类嵌套尾逗号场景（最多 5 次收敛）
    for _ in range(5):
        changed = False
        for i in range(n - 1, -1, -1):
            if chars[i] in ("}", "]"):
                j = i - 1
                # 跳过空白（space / tab / \r / \n）
                while j >= 0 and chars[j] in (" ", "\t", "\r", "\n"):
                    j -= 1
                if j >= 0 and chars[j] == ",":
                    # 删除逗号
                    del chars[j]
                    n -= 1
                    changed = True
                    break
        if not changed:
            break
    return "".join(chars)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """MID-1 Planner / REPLAN 解析：对照 8 层框架分层处理。

    分层顺序：
      LLM 输出 → 去围栏（内置）→ _mini_json_repair（通用 Parser 兜底）
               → PlanGenerateSchema.model_validate_json（Pydantic 强类型校验）
               → model_dump() → 下游 _parse_subtasks（业务规则解析）

    Args:
        text: LLM 原始输出字符串

    Returns:
        dict：保证带 "subtasks" 顶层键的 Python 字典（类型/字段已被 Pydantic 校验过）

    Raises:
        pydantic.ValidationError：修复后 Pydantic 仍校验失败（交给调用方 Retry / fallback）
        ValueError：mini_json_repair 后仍然不是合法 JSON 形态（交给调用方 Retry / fallback）
    """
    cleaned_text: str = (text or "").strip()
    if cleaned_text.startswith("```"):
        first_line_end = cleaned_text.find("\n")
        if first_line_end >= 0:
            cleaned_text = cleaned_text[first_line_end + 1:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3].rstrip()

    repaired_text: str = _mini_json_repair(cleaned_text)

    # 先用 json.loads 做"通用 Parser"验证，保证传给 Pydantic 的至少是可解析的合法 JSON 字符流
    try:
        json.loads(repaired_text)
    except json.JSONDecodeError as decode_error:
        raise ValueError(
            f"Planner JSON 通用 Parser 失败（json.loads）：{decode_error}. "
            f"Repaired text head 500: {repaired_text[:500]!r}"
        ) from decode_error

    parsed_struct = PlanGenerateSchema.model_validate_json(repaired_text)
    return parsed_struct.model_dump()


def _parse_subtasks(data: Dict[str, Any]) -> List[SubTask]:
    """对照 8 层框架：Pydantic OK → 通用语义校验 → 业务规则校验 → 最终 SubTask 列表。

    分层校验顺序（在已经通过 Pydantic model_validate_json 的基础上做业务语义过滤）：
      【通用语义校验层】（纯结构约束，与具体业务规则无关）：
        ① subtasks 必须是非空 list，不能为 None/空数组
        ② 每个元素必须是 dict（Pydantic 已经要求了 item_schema，但这里做防御式再查）
        ③ 每个 subtask 的 action_type 标准化：大小写任意 → 强制小写，tool/reasoning 二选一，其他值降级为 reasoning

      【业务规则校验层】（Agent 编排的硬业务约束）：
        ④ 若 action_type == 'tool'：tool_name 必须是非空字符串；否则强制降级为 reasoning 并记录原因
        ⑤ tool_name 如果声明了，tool_args_hint 允许为空（可以在执行阶段再解析参数）
        ⑥ id 若缺失或重复：自动补 task_0/task_1... + 附加 _dup 后缀

    全部校验通过 → 返回 SubTask 列表。
    若校验后 subtasks 为空 → 抛出 ValueError，交给调用方 Retry / fallback 兜底。
    """
    raw_list: Any = data.get("subtasks")
    # ---- 语义校验 ①：subtasks 必须是 list（空数组判为无效，触发 Retry）----
    if not isinstance(raw_list, list) or len(raw_list) == 0:
        raise ValueError(
            f"Planner 通用语义校验失败：subtasks 必须是非空数组，实际为 {type(raw_list).__name__} "
            f"(len={len(raw_list) if isinstance(raw_list, list) else 'n/a'})"
        )

    seen_ids: set[str] = set()
    result: List[SubTask] = []

    for idx, item in enumerate(raw_list):
        # ---- 语义校验 ②：每一项必须是 dict ----
        if not isinstance(item, dict):
            logger.warning(
                "Planner 语义校验跳过 subtask[%d]：不是 dict 类型，实际 type=%s，raw head=%r",
                idx, type(item).__name__, str(item)[:200],
            )
            continue

        # ---- 语义校验 ③：action_type 标准化（二选一，其他值降级 reasoning）----
        raw_action_type: str = str(item.get("action_type", "reasoning") or "reasoning").strip().lower()
        if raw_action_type not in ("tool", "reasoning"):
            logger.warning(
                "Planner 语义校验：subtask[%d] action_type=%r 不在 {tool, reasoning}，降级为 reasoning",
                idx, raw_action_type,
            )
            raw_action_type = "reasoning"

        # ---- 业务规则校验 ④：tool 类型必须有非空 tool_name ----
        declared_tool_name: Any = item.get("tool_name")
        final_tool_name: Optional[str] = None
        # ---- 业务规则校验 ⑤：tool_args_hint 允许为空，空值标准化为 "{}"（显式补 JSON 对象字面量）----
        raw_tool_args_hint: Any = item.get("tool_args_hint")
        if isinstance(raw_tool_args_hint, str) and raw_tool_args_hint.strip():
            final_tool_args_hint: Optional[str] = raw_tool_args_hint.strip()
        elif raw_tool_args_hint is None or (isinstance(raw_tool_args_hint, str) and not raw_tool_args_hint.strip()):
            final_tool_args_hint = "{}"
        else:
            # 非字符串类型（dict/list/number）一律做 JSON 化，保证下游 Router 总能拿到合法 JSON 字符串
            try:
                final_tool_args_hint = json.dumps(raw_tool_args_hint, ensure_ascii=False)
            except (TypeError, ValueError):
                final_tool_args_hint = "{}"
        if raw_action_type == "tool":
            if not isinstance(declared_tool_name, str) or not declared_tool_name.strip():
                logger.warning(
                    "Planner 业务规则[④]：subtask[%d] 声明为 tool 但 tool_name 缺失，降级为 reasoning 子任务",
                    idx,
                )
                raw_action_type = "reasoning"
            else:
                final_tool_name = declared_tool_name.strip()

        # ---- 业务规则校验 ⑥：id 去重 + 自动补 ----
        raw_id: str = str(item.get("id", "") or "").strip()
        final_id: str = raw_id or f"task_{idx}"
        if final_id in seen_ids:
            final_id = f"{final_id}_dup{len(result)}"
        seen_ids.add(final_id)

        final_title: str = str(item.get("title", "") or "").strip() or f"子任务{idx + 1}"
        final_description: str = str(item.get("description", "") or "").strip() or final_title

        # ---- PASS/FAIL 语义门：组装结果前再做 8 层框架的显式语义兜底 ----
        # 若 reasoning 类型但 description/title 都空 → 本条按 FAIL 跳过
        # （其它已经在上面各校验条里过滤，此处不重复）
        if raw_action_type == "reasoning" and not final_description.strip():
            logger.warning(
                "Planner 语义校验[⑥门]：subtask[%d] reasoning 类型但 description 为空，按 FAIL 跳过",
                idx,
            )
            continue

        result.append(SubTask(
            id=final_id,
            title=final_title,
            description=final_description,
            action_type=raw_action_type,
            tool_name=final_tool_name,
            tool_args_hint=final_tool_args_hint,
        ))

    if not result:
        raise ValueError("Planner 业务规则校验失败：所有 subtask 均被过滤，最终为空列表，触发 Retry")

    return result


class PlannerAgent:
    """P1① 扁平化：优先直接持有 ModelRouter（2 层链路：Agent → ModelRouter.chat → 引擎），
    兼容旧 `llm` 协议参数作为回退兜底。

    旧嵌套（改造前）：Agent → orchestrator._LLMAdapter → _PurposeLLMAdapter → ModelRouter（4 层）
    新链路（改造后）：Agent → ModelRouter.chat（2 层，减少 2 个适配器包装层）
    """

    def __init__(
            self,
            llm: Optional[LLMCallable] = None,
            tools: Optional[ToolInvoker] = None,
            memory: Optional[Any] = None,  # 废弃底层单体耦合
            max_replan_attempts: int = 2,
            *,
            model_router: Optional[Any] = None,
            purpose_hint: str = "planner",
            call_budget: Optional[Any] = None,
            enable_empty_result_replan: bool = True,
    ) -> None:
        if llm is None and model_router is None:
            raise ValueError("PlannerAgent 需要提供 llm 或 model_router 至少其一")
        self._llm = llm
        self._model_router = model_router
        self._purpose: str = purpose_hint
        self._tools = tools
        # 工具调用预算：单次 Agent 请求作用域（plan/replan 提示词动态规划 + execute 硬熔断共用）
        self._call_budget: Optional[Any] = call_budget
        # 【方案B】空业务结果 → 触发 replan（非异常型），默认开启
        self._enable_empty_result_replan: bool = enable_empty_result_replan
        self.max_replan_attempts = max(0, max_replan_attempts)

    async def _llm_chat(self, messages, **kwargs) -> str:
        """统一 LLM 调用入口：优先 ModelRouter，否则回退旧 acomplete 协议。"""
        if self._model_router is not None:
            resp = await self._model_router.chat(
                messages=list(messages), purpose_hint=self._purpose, **kwargs
            )
            return (getattr(resp, "content", None) or "").strip()
        return await self._llm.acomplete(messages, **kwargs)

    def _budget_prompt_block(self) -> str:
        """生成注入 plan/replan 提示词的预算说明段（静态基线 + 剩余额度动态规划）。

        Returns:
            预算提示文本；未启用预算时返回空串。
        """
        budget = self._call_budget
        if budget is None:
            return ""
        lines = budget.snapshot_lines()
        if not lines:
            return ""
        tool_names_hint: str = ""
        if budget.total_budget > 0:
            tool_names_hint = (
                f"\n请把本次计划中需要调用工具的子任务总量控制在剩余总额度（{budget.total_remaining()} 次）内；"
                "对于已接近/达到单工具上限的工具不要再拆出调用它的子任务。"
            )
        return (
            "## 工具调用预算（任务执行硬约束）\n"
            + "\n".join(lines)
            + tool_names_hint
            + "\n规划 tool 子任务时请据此收敛调用次数；纯推理子任务不受额度限制。\n\n"
        )

    # 整合skills
    async def plan(self, query: str, memory_context: Dict[str, Any], skills_block: str = '') -> List[SubTask]:
        """MID-1：初始计划生成（执行 1 次 LLM + Parse 失败后 Retry 1 次）。

        对应 8 层框架完整执行链路：
          LLM 调用（thinking=False + response_format=json_schema）
              → _extract_json_object（JSON Extraction → mini_json_repair → json.loads 通用 Parser
                                        → Pydantic PlanGenerateSchema）
              → _parse_subtasks（Schema OK → 通用语义校验 → 业务规则校验 → PASS / FAIL）
              → [PASS] 返回 SubTask[] / [FAIL] → Retry 1 次（temperature 升至 0.4）
              → 仍 FAIL → fallback 单 reasoning 子任务。
        """
        short_term_list = memory_context.get("short_term") or []
        long_term_snippets = memory_context.get("long_term") or []

        # 组装参考背景
        history_str = ""
        for old_msg in short_term_list:
            role = getattr(old_msg, "role", None) or old_msg.get("role", "user")
            content = getattr(old_msg, "content", None) or old_msg.get("content", "")
            history_str += f"[{role}]: {content}\n"

        mem_block = "\n".join(f"- {s}" for s in long_term_snippets) if long_term_snippets else "（无）"

        # 整合skills
        skills_section = f"## 可用高级技能 (渐进式披露)\n{skills_block}\n\n" if skills_block else ""

        messages: Sequence[Dict[str, str]] = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"## 历史多轮对话上下文：\n{history_str}\n"
                           f"## 长期事实参考：\n{mem_block}\n\n"
                           # 整合skills
                           f"{skills_section}"
                           # 工具预算：让计划阶段的 tool 子任务数量主动收敛在额度内
                           f"{self._budget_prompt_block()}"
                           f"## 当前用户新目标：\n{query}\n\n"
                           "【再次提醒】请严格遵守系统提示词中的 JSON 规则："
                           "整个回复必须且只能是合法 JSON 对象，不得输出任何解释/围栏。请输出合理的 JSON 计划。"
            },
        ]

        fallback_plan: List[SubTask] = [
            SubTask(id="fallback_1", title="直接回答", description=query, action_type="reasoning")
        ]

        # 按 8 层框架：最多 2 次 LLM 调用（首次 + Retry 1 次）
        for attempt_index in range(2):
            temperature_value: float = 0.3 if attempt_index == 0 else 0.4
            try:
                # MID-1 Planner 初始计划：关闭思考 + 严格 JSON schema（subtasks 数组）
                raw = await self._llm_chat(
                    messages,
                    temperature=temperature_value,
                    thinking=False,
                    response_format=pydantic_to_openai_response_format(PlanGenerateSchema),
                )
                parsed_dict = _extract_json_object(raw)
                subtasks_list = _parse_subtasks(parsed_dict)
                return subtasks_list
            except Exception as plan_error:
                if attempt_index == 0:
                    logger.warning(
                        "Planner.plan() 第 1 次 parse 失败，准备发起第 2 次 Retry（temperature=%.1f）：%s",
                        temperature_value + 0.1, plan_error,
                    )
                    continue
                # 第 2 次也失败 → 降级 fallback
                logger.exception("Planner.plan() 第 2 次（最后一次）仍失败，降级为 fallback 单任务")
                return fallback_plan

        return fallback_plan

    async def _resolve_tool_args_via_function_call(
            self,
            task: SubTask,
            query: str,
            prior_context_str: str,
    ) -> Optional[Dict[str, Any]]:
        """【原生 Function Calling】为工具子任务强制取参。

        强制 tool_choice=子任务声明的工具，让模型按该工具的 JSON Schema 生成
        100% 合法的参数 dict，根治原先 ``tool_args_hint`` 自由文本与工具 schema
        错配导致的 ``参数 xxx 不能为空`` 类错误。

        Args:
            task: 待执行的 tool 子任务
            query: 原始总问题（供模型理解上下文）
            prior_context_str: 前序子任务精炼结论（截断字符串）

        Returns:
            合法的参数字典；任一环节失败（无 model_router / 无法 introspection /
            LLM 未产出 tool_calls / 参数非 dict）返回 None，由调用方降级走老逻辑。
        """
        # FC 依赖 ModelRouter.chat_with_tools；纯 llm 回退模式无法走协议，直接放弃
        if self._model_router is None:
            return None

        try:
            if hasattr(self._tools, "get_tool"):
                tool_instance: Any = self._tools.get_tool(task.tool_name)  # type: ignore[attr-defined]
            else:
                # 仅暴露 invoke 的执行代理无法反射 schema，走降级
                tool_instance = None
        except KeyError:
            tool_instance = None
        if tool_instance is None:
            return None

        function_def: Dict[str, Any] = tool_to_function_call_definition(tool_instance)
        arg_fill_system_prompt: str = (
            "你是 Agent 子任务执行前的「工具参数填充器」。"
            "你必须调用系统提供的那个唯一工具，并按其参数 JSON Schema 生成调用所需的参数字段；"
            "参数名、类型与必填项必须严格与 Schema 一致，禁止虚构 Schema 中不存在的字段。"
            "若可选参数不影响任务推进可省略。你的输出只会被当成工具参数解析，禁止输出任何解释文字。"
        )
        user_content: str = (
            f"原始总问题：{query}\n"
            f"当前子任务：{task.title}\n"
            f"子任务详细要求：{task.description}\n"
            f"规划器备注（仅供参考，可能为空）：{task.tool_args_hint or '（无）'}\n"
            f"可参考的前序子任务上下文：\n{(prior_context_str or '')[:3000]}"
        )
        messages: Sequence[Dict[str, str]] = [
            {"role": "system", "content": arg_fill_system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            resp: Any = await self._model_router.chat_with_tools(
                messages=list(messages),
                tools=[function_def],
                tool_choice={"type": "function", "function": {"name": task.tool_name}},
                purpose_hint=self._purpose,
                temperature=0.1,
                thinking=False,
            )
        except Exception as fc_exc:  # noqa: BLE001 - 通道/模型不支持 tools 时降级老逻辑
            logger.warning(
                "Planner FC 强制取参调用失败（tool=%s），降级走 tool_args_hint 解析: %s",
                task.tool_name, fc_exc,
            )
            return None
        if not getattr(resp, "tool_calls", None):
            logger.warning(
                "Planner FC 强制取参未返回 tool_calls（tool=%s），降级走 tool_args_hint 解析",
                task.tool_name,
            )
            return None

        raw_args: Any = resp.tool_calls[0].get("function", {}).get("arguments")
        if not isinstance(raw_args, str):
            return None
        try:
            parsed_args: Any = json.loads(raw_args)
        except json.JSONDecodeError as decode_error:
            logger.warning(
                "Planner FC 强制取参返回非法 JSON（tool=%s）：%s", task.tool_name, decode_error
            )
            return None
        if not isinstance(parsed_args, dict):
            return None
        return parsed_args

    async def execute(
            self,
            plan: List[SubTask],
            query: str,
            session_id: str,
            tool_names: Optional[Sequence[str]] = None,
            trace_callback: Optional[Any] = None,
    ) -> AgentResult:
        allowed = set(tool_names) if tool_names else None
        results: List[Dict[str, Any]] = []

        for idx, task in enumerate(plan):
            rec: Dict[str, Any] = {
                "subtask_id": task.id,
                "title": task.title,
                "action_type": task.action_type
            }
            try:
                # 前移：先构建前序子任务精炼上下文（工具强制取参与提炼阶段共用同一份）
                simplified_context = [
                    {
                        "subtask_id": r["subtask_id"],
                        "title": r["title"],
                        "conclusion": r.get("llm_output", r.get("observation", ""))
                    }
                    for r in results
                ]
                ctx_str = json.dumps(simplified_context, ensure_ascii=False, indent=2)[:8000]

                # 1. 如果需要工具，先获取工具的原始观察值 (Observation)
                obs_str: Optional[str] = None
                if task.action_type == "tool":
                    if not task.tool_name:
                        raise ValueError(f"子任务 [{task.id}] 声明为 tool 类型，但未指定 tool_name")
                    if allowed is not None and task.tool_name not in allowed:
                        raise RuntimeError(f"工具 {task.tool_name} 不在允许列表中")

                    # 预算熔断（先检查后计数）：单工具上限或全局总硬上限任一已达 → 不再发起调用
                    if self._call_budget is not None and not self._call_budget.can_call(task.tool_name):
                        deny_text: str = self._call_budget.deny_text(task.tool_name)
                        obs_str = deny_text
                        rec["budget_denied"] = True
                        logger.info(
                            "Planner 子任务 [%s] 工具 [%s] 触发额度熔断，转为纯推理收尾：%s",
                            task.id, task.tool_name, deny_text[:160],
                        )
                    else:
                        if self._call_budget is not None:
                            self._call_budget.consume(task.tool_name)

                        # 一律优先【原生 Function Calling】强制取参：模型按该工具 JSON Schema 生成合法参数
                        args: Dict[str, Any] = await self._resolve_tool_args_via_function_call(
                            task, query, ctx_str,
                        )
                        if args is None:
                            # 降级（纯 llm 无 model_router / 工具不可 introspection / LLM 异常）：
                            # 沿用 tool_args_hint 文本解析，保证不因 FC 失败而中断流程
                            args = {}
                            if task.tool_args_hint:
                                try:
                                    parsed = json.loads(task.tool_args_hint)
                                    args = parsed if isinstance(parsed, dict) else {"hint": str(parsed)}
                                except Exception:
                                    args = {"hint": task.tool_args_hint}
                            if "user_query" not in args:
                                args["user_query"] = query

                        # 注意：planner 用的是标准库 logging（%-风格），不要用 {} 占位
                        logger.info(
                            "Planner 子任务 [%s] 调用工具 [%s]，实际参数: %s",
                            task.id, task.tool_name, args,
                        )

                        # 调用工具并安全转为字符串
                        obs = await self._tools.invoke(task.tool_name, args)
                        if isinstance(obs, str) and ("【系统拒绝执行】" in obs or "error" in obs.lower()):
                            raise RuntimeError(f"工具执行返回错误: {obs[:200]}")
                        if isinstance(obs, (dict, list)):
                            obs_str = json.dumps(obs, ensure_ascii=False, indent=2)
                        else:
                            obs_str = str(obs)

                        # 预算后处理：空/异常累计无效、第 N 次相关性抽查（可能触发硬熔断）
                        obs_str = await postprocess_tool_result(
                            self._call_budget, task.tool_name,
                            str(args)[:1500], obs_str, self._model_router,
                        )

                        rec["observation"] = obs_str[:8000] # 保留原始观测记录备查

                        # 【方案B】空业务结果 → 停止剩余剧本并触发 replan（非异常型）。
                        # 与通过 except 抛异常触发 replan 的本质区别：这不是“调用出错”需要修复，
                        # 而是“数据源为空”需要临时换一个可用数据源重新取数，避免拿着空数据
                        # 硬造后续步骤或生成占位结果（例如用空列表凭空编“线索A~E”）。
                        if (
                            self._enable_empty_result_replan
                            and isinstance(obs_str, str)
                            and _is_empty_data(obs_str)
                        ):
                            rec["status"] = "empty_data"
                            rec["replan_reason"] = (
                                f"{EMPTY_DATASOURCE_REPLAN_PREFIX}: {task.tool_name} 返回空业务数据"
                            )
                            results.append(rec)
                            if trace_callback:
                                await trace_callback({"phase": "execute", "record": rec})
                            logger.info(
                                "Planner 子任务 [%s] 工具 [%s] 返回空业务数据，触发备选数据源重规划",
                                task.id, task.tool_name,
                            )
                            return AgentResult(
                                success=False,
                                final_answer="",
                                steps=results,
                                error=(
                                    f"{EMPTY_DATASOURCE_REPLAN_PREFIX}: 工具 [{task.tool_name}] "
                                    "未返回可用业务数据，请改用其他可用数据源工具重新获取，严禁编造占位数据。"
                                ),
                            )

                # 2. 通用思考/提炼阶段：不管是工具任务还是纯推理任务，统一由 LLM 生成当前子任务的明确结论
                # 组装当前子任务 Prompt
                prompt_content = f"原始总问题：{query}\n当前子任务：{task.title}\n详细要求：{task.description}\n历史子任务结论：\n{ctx_str}\n"
                if obs_str:
                    prompt_content += f"\n本步骤工具调用返回的原始数据：\n{obs_str[:6000]}\n请结合工具数据完成本子任务。"
                else:
                    prompt_content += "\n请根据历史上下文推理并完成本子任务。"

                subtask_msgs: Sequence[Dict[str, str]] = [
                    {
                        "role": "system",
                        "content": "你是高效的子任务执行专家。请根据上下文（及工具数据），针对当前子任务给出简洁、准确的最终结论或分析结果。"
                    },
                    {"role": "user", "content": prompt_content},
                ]

                # 执行 LLM 提炼
                text = await self._llm_chat(subtask_msgs,thinking=False, temperature=0.3)
                rec["llm_output"] = text.strip()
                rec["status"] = "ok"

            except Exception as e:
                rec["status"] = "error"
                rec["error"] = str(e)
                results.append(rec)
                if trace_callback:
                    await trace_callback({"phase": "execute", "record": rec})
                return AgentResult(success=False, final_answer="", steps=results, error=str(e))

            results.append(rec)
            if trace_callback:
                await trace_callback({"phase": "execute", "record": rec})

        # 3. 最终汇总阶段：直接基于每一个子任务提炼后的精炼结论（llm_output）生成面向用户的回答
        try:
            final_context = [
                {"step": r["title"], "result": r.get("llm_output", "")}
                for r in results
            ]
            summary_msgs: Sequence[Dict[str, str]] = [
                {"role": "system", "content": "你是最终总结助手。请根据各子任务的执行结论，整合并回答用户的原始问题。"},
                {
                    "role": "user",
                    "content": f"原始问题：{query}\n各步骤执行结论：\n{json.dumps(final_context, ensure_ascii=False, indent=2)}"
                },
            ]
            final = await self._llm_chat(summary_msgs,thinking=False, temperature=0.2)
        except Exception as e:
            return AgentResult(success=False, final_answer="", steps=results, error=f"汇总阶段失败: {e}")

        return AgentResult(success=True, final_answer=final.strip(), steps=results)

    async def replan(
        self,
        plan: List[SubTask],
        results: List[Dict[str, Any]],
        error: Optional[str],
    ) -> List[SubTask]:
        """A-C2：Planner 重计划（执行过程遭遇异常后的计划修订）。

        与 plan() 差异点：
          - 保留 thinking（A-C2 是循环恢复点的推理任务，需要模型诊断"哪些已完成/哪里错了"，
            关闭会导致"把已完成任务又塞回去"的失误）
          - 复用 PlanGenerateSchema response_format（严格 JSON schema 保证格式正确）
          - 按 8 层框架：同样走 1 次 LLM → Parse 失败 Retry 1 次 的完整分层校验
        """  # 虽然调用大模型，这里本质是一个小模型在编排任务，并且是吧json注入到系统提示词的笨办法
        payload = {
            "previous_plan": [task.__dict__ for task in plan],
            "results_so_far": results,
            "error": error,
        }
        messages: Sequence[Dict[str, str]] = [
            {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "上下文：\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)[:16000]
                + "\n\n"
                + self._budget_prompt_block()  # 剩余额度提醒：重规划时收敛新增 tool 子任务
                + "【再次提醒】严格遵守 REPLAN 的 JSON 输出规则：整个回复只能是合法 JSON 对象，"
                "顶层为 {subtasks:[...]}；已成功完成的子任务不要再出现在新计划中。请输出修订后的 JSON 计划。",
            },
        ]

        fallback_replan: List[SubTask] = [
            SubTask(
                id="replan_fallback",
                title="降级为单步推理",
                description="基于已有结果直接整合",
                action_type="reasoning",
            )
        ]

        # 按 8 层框架：最多 2 次 LLM 调用（首次 + Retry 1 次）—— REPLAN 不关闭 thinking
        for attempt_index in range(2):
            temperature_value: float = 0.3 if attempt_index == 0 else 0.45
            try:
                # REPLAN：A-C2 属于 Plan 循环恢复点的推理环节（不关闭 thinking 保留推理），
                # 但复用 PlanGenerateSchema response_format 保证输出格式 100% 正确。
                raw = await self._llm_chat(
                    messages,
                    temperature=temperature_value,
                    response_format=pydantic_to_openai_response_format(PlanGenerateSchema),
                )
                parsed_dict = _extract_json_object(raw)
                subtasks_list = _parse_subtasks(parsed_dict)
                return subtasks_list
            except Exception as replan_error:  # noqa: BLE001
                if attempt_index == 0:
                    logger.warning(
                        "Planner.replan() 第 1 次 parse 失败，准备发起第 2 次 Retry（temperature=%.2f）：%s",
                        temperature_value + 0.15, replan_error,
                    )
                    continue
                # 第 2 次也失败 → 降级 fallback
                logger.exception("Planner.replan() 第 2 次（最后一次）仍失败，降级为 replan_fallback 单任务")
                return fallback_replan

        return fallback_replan

    # 整合skills
    async def run_with_plan(
            self,
            query: str,
            session_id: str,
            tool_names: Optional[Sequence[str]] = None,
            trace_callback: Optional[Any] = None,
            memory_context: Dict[str, Any] = None,  # 💡 穿透接收
            skills_block: str = '',
    ) -> AgentResult:
        # 整合skills
        current_plan = await self.plan(
            query,
            memory_context or {"short_term": [], "long_term": []},
            skills_block=skills_block
        )
        last_error: Optional[str] = None
        aggregate_results: List[Dict[str, Any]] = []

        for attempt in range(self.max_replan_attempts + 1):
            res = await self.execute(
                current_plan, query, session_id, tool_names=tool_names, trace_callback=trace_callback
            )
            if res.success:
                return res

            last_error = res.error
            aggregate_results.extend(res.steps)
            if attempt >= self.max_replan_attempts:
                break

            current_plan = await self.replan(current_plan, res.steps, last_error)
            if not current_plan:
                return AgentResult(success=False, final_answer="", steps=aggregate_results, error="重规划返回空计划")

        return AgentResult(success=False, final_answer="", steps=aggregate_results, error=last_error)