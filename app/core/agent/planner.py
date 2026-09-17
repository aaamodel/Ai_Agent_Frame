# -*- coding: utf-8 -*-
"""
规划 Agent：Plan-and-Execute，含任务分解、执行与重规划。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.core.agent.toolcall import (
    ToolCall,
    ToolInvoker,
    execute_tool_call,
    observation_has_error_marker,
)
from app.core.tools.base import tool_to_function_call_definition

# 本轮结构化输出：MID-1 Planner 初始计划/重计划 schema + 协议转换
from app.query_intent.llm_schemas import (
    PlanGenerateSchema,
    build_dynamic_plan_schema,
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

# 模型声明了白名单外的工具名时的兜底映射。
# 实测故障：planner 由技能名 sales-intelligence-assistant 自行派生出了
# sales_intelligence_query，执行阶段被白名单拒绝 → 触发一次完整 replan
# （单笔 7149 tokens）。这类"编造的查询类工具"真实意图几乎都是知识库检索，
# 因此直接映射到兜底检索工具，**不触发重规划**；兜底工具本身不在白名单时才降级。
UNKNOWN_TOOL_FALLBACK: str = "rag_knowledge_search"

# replan 上下文裁剪阈值：单条子任务摘要 / 整体 payload 上限。
# 旧实现整体截断 16000 字符，而单条 observation 本身就有 8000 字符，
# 于是整篇 SKILL.md 会被原样喂进 replan（这是 replan 单笔 7149 token 的主因）。
REPLAN_SUMMARY_MAX_CHARS: int = 400
REPLAN_PAYLOAD_MAX_CHARS: int = 4000


# 一刀切用 <=60 字符的长度闸门会把它们漏掉，导致"空数据"被当成"正常结果"继续提炼。
_EMPTY_RESULT_SENTINELS: tuple = (
    "未匹配到任何高相关性的文档片段",   # rag_knowledge_search 空召回
    "未匹配到任何高相关性",
    "知识库未匹配到",
)


def _is_empty_data(obs_text: str) -> bool:
    """保守判定一次工具观测结果是否属于“空业务数据”（非异常）。返回 True 时触发 replan。

    判定策略（宁缺毋滥，避免误伤正常执行）：
      1. 空白 / 空字符串 → 判空；
      2. 可解析 JSON 且为【空数组】或【空对象】→ 判空；
      3. 命中"空结果哨兵短语"（工具固定话术，不限长度）→ 判空；
      4. 短文本（<=60 字符）且含明确的“无数据”语义关键词 → 判空。

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

    if any(sentinel in text for sentinel in _EMPTY_RESULT_SENTINELS):
        return True

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


PLAN_SYSTEM_PROMPT = """你是规划专家。结合对话历史、长期记忆与可用技能，把用户目标拆解为可执行子任务。
只输出计划本身，不要解释、Markdown 围栏或思考过程。
工具选型与参数用法严格依据下方「可用工具清单」中各工具自己的描述，禁止使用清单外的工具。
取数类工具有失败或返回空数据的可能（鉴权/权限/无记录）：风险明显时在计划中保留一个备用数据源，不要把成败压在单一来源上。
"""

REPLAN_SYSTEM_PROMPT = """你是重规划专家。根据已有执行结果与错误信息，修订剩余子任务。
已成功完成的子任务从新计划移除，只保留需要重做 / 调整顺序 / 新增的部分。
只输出计划本身，不要解释、Markdown 围栏或思考过程。
工具选型与参数用法严格依据下方「可用工具清单」中各工具自己的描述。
- 工具返回【空数据】（非报错）：换另一个可用数据源重新取数，禁止基于空数据编造后续步骤。
- 工具明确报错：按错误信息修正参数后重试，禁止原样重放已知会失败的调用。
"""


'''replan备份：提示词你是一个动态调整与重规划专家。当执行过程中遭遇异常或无法达成预期时，你需要根据当前已有的执行结果以及发生的错误，对剩余的子任务进行修订和重新编排。

【输出格式】
只输出一个 JSON 对象（顶层字段 subtasks），不要输出 JSON 之外的任何文字、Markdown 围栏或思考过程。
已判定成功完成的子任务请从新计划中移除，只保留需要重做 / 调整顺序 / 新增的子任务。
每个子任务字段同初始计划：id / title / description / action_type（只允许 "tool" 或 "reasoning"）/
tool_name（tool 必填）/ tool_args_hint（可选）。

【重规划要求】
若既有执行结果显示某数据源工具返回了【空数据】（空数组 / 空对象 / 空字符串，而非报错），
修订计划中必须改用另一个可用数据源工具（例如从飞书多维表切换到本地 Excel / RAG 知识库）重新取数，
严禁基于空数据继续编造后续步骤或凭空生成占位结果。

⚠️ 但**不要把"部分数据"误判成"没有数据"**（这一条是实测事故补的）：
若工具返回中出现"仅展示前 N 行 / 已截断 / 预览 / 结构摘要"等**局部视图**提示，
说明数据源是好的、只是查询条件不够精确，**严禁**判定数据不存在、**严禁**更换数据源。
此时必须改写取数方式：改用 filter_column+filter_value 按条件精确定位，
或补上 sheet_name / 调大 head_rows（检索类工具则调大 top_k），
并在子任务描述里写清"要用哪个字段的哪个值来筛"；
涉及计算/统计的，优先改用 local_excel_query_tool 直接在全量数据上算。
只有工具**明确报错**或**明确返回空数据**时，才允许更换工具或数据源。

⚠️ 若失败的是写操作（local_excel_write_tool 等），重试子任务必须改用语义参数
filter_column + filter_value + target_column + new_value，并先安排一次读取定位拿到
真实列名与唯一条件值；**禁止**再次让执行方自己算 A1 单元格坐标，也禁止重复提交
已知会被拒绝的相同调用（如审批已拒绝）。

⚠️ 若之前的失败原因是参数里出现了 "<从 task_N 获取的…>" 这类**占位符**，
新计划必须直接写出该参数的真实取值（路径/文件名照抄工具返回里的真实值），
严禁在新计划的 tool_args_hint 中再次输出任何尖括号占位符。'''

@dataclass
class SubTask:
    id: str
    title: str
    description: str
    action_type: str
    tool_name: Optional[str] = None
    tool_args_hint: Optional[str] = None
    covers_sub_questions: Optional[List[int]] = None
    """本子任务覆盖的子问题序号（从 1 开始）。用于校验子问题是否被全覆盖。"""


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


def _resolve_tool_name(
    candidate: str,
    allowed_tool_names: Optional[Sequence[str]],
) -> Optional[str]:
    """把模型声明的工具名收敛到当前白名单内。

    ⚠️ 这是"模型编造工具名"的最后一道程序化护栏（schema enum 只是软约束：
    不同厂商对 ``anyOf`` 内 enum 的执行力度不一致，不能只靠它）。

    Returns:
        - 命中白名单 → 原样返回；
        - 未命中且兜底工具（``UNKNOWN_TOOL_FALLBACK``）在白名单 → 返回兜底工具名；
        - 否则 → ``None``（调用方降级为 reasoning 子任务）。

    返回非 None 意味着**不需要重规划**：这一步修掉的是"工具名写错"，
    计划本身的结构与目标仍然有效，走 replan 纯属浪费一整轮全量上下文。
    """
    if not allowed_tool_names:
        return candidate
    if candidate in allowed_tool_names:
        return candidate
    if UNKNOWN_TOOL_FALLBACK in allowed_tool_names:
        return UNKNOWN_TOOL_FALLBACK
    return None


def _parse_subtasks(
    data: Dict[str, Any],
    allowed_tool_names: Optional[Sequence[str]] = None,
    sub_question_count: int = 0,
    strict_coverage: bool = False,
) -> List[SubTask]:
    """``sub_question_count`` / ``strict_coverage`` 服务于业务规则 ⑧（子问题覆盖校验）。

    Args:
        sub_question_count: 意图层实际拆分出的子问题数（未拆分时为 0 → 不做校验）。
        strict_coverage: True 表示这是**首次**解析——覆盖缺失时抛错，让调用方
            再要一次计划；False 表示已到重试上限，改为告警后放行（fail-open）。
            两档是刻意的：只靠提示词要求"每子问题一个子任务"是软约束，模型仍会
            合并；而无限重试又会把链路卡死，所以第二次选择告警放行。
    """
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
        ⑦（白名单护栏）tool_name 必须在 allowed_tool_names 内：不在则映射到兜底检索工具
           rag_knowledge_search；兜底工具也不在白名单时才降级为 reasoning。
           这一条是模型编造工具名（如 sales_intelligence_query）的兜底，
           命中即原地纠正，**不触发重规划**。
        ⑧（子问题覆盖校验）若 sub_question_count > 0：每个子问题序号（1..N）
           都必须被至少一个子任务的 covers_sub_questions 覆盖。
           首次解析缺失 → 抛错触发重新规划；重试后仍缺失 → 告警并放行。

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
                candidate_tool: str = declared_tool_name.strip()
                resolved_tool: Optional[str] = _resolve_tool_name(
                    candidate_tool, allowed_tool_names
                )
                if resolved_tool is None:
                    logger.warning(
                        "Planner 业务规则[⑦]：subtask[%d] 工具 %r 不在白名单 %s 且无兜底映射，"
                        "降级为 reasoning 子任务（不触发重规划）",
                        idx, candidate_tool, list(allowed_tool_names or []),
                    )
                    raw_action_type = "reasoning"
                else:
                    if resolved_tool != candidate_tool:
                        logger.warning(
                            "Planner 业务规则[⑦]：subtask[%d] 工具 %r 不在白名单，"
                            "已自动映射为 %r（不触发重规划）",
                            idx, candidate_tool, resolved_tool,
                        )
                    final_tool_name = resolved_tool

        # ---- 业务规则校验 ⑥：id 去重 + 自动补 ----
        raw_id: str = str(item.get("id", "") or "").strip()
        final_id: str = raw_id or f"task_{idx}"
        if final_id in seen_ids:
            final_id = f"{final_id}_dup{len(result)}"
        seen_ids.add(final_id)

        final_title: str = str(item.get("title", "") or "").strip() or f"子任务{idx + 1}"
        final_description: str = str(item.get("description", "") or "").strip() or final_title

        # ---- 业务规则 ⑧ 的输入：本子任务声明覆盖哪些子问题（序号从 1 开始）----
        raw_covers: Any = item.get("covers_sub_questions")
        final_covers: Optional[List[int]] = None
        if isinstance(raw_covers, (list, tuple)):
            parsed_covers: List[int] = []
            for value in raw_covers:
                try:
                    parsed_covers.append(int(value))
                except (TypeError, ValueError):
                    continue
            final_covers = parsed_covers or None

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
            covers_sub_questions=final_covers,
        ))

    if not result:
        raise ValueError("Planner 业务规则校验失败：所有 subtask 均被过滤，最终为空列表，触发 Retry")

    # ---- 业务规则 ⑧：子问题覆盖校验 ----
    if sub_question_count > 0:
        covered: set = set()
        for task in result:
            if task.covers_sub_questions:
                covered.update(task.covers_sub_questions)
        missing: List[int] = [
            index for index in range(1, sub_question_count + 1) if index not in covered
        ]
        if missing:
            detail: str = (
                f"Planner 业务规则[⑧]：子问题 {missing} 没有任何子任务覆盖"
                f"（共 {sub_question_count} 个，已覆盖 {sorted(covered) or '无'}）"
            )
            if strict_coverage:
                raise ValueError(f"{detail}，触发重新规划")
            # 已到重试上限 → fail-open：无限重试会把链路卡死，静默丢弃又会漏答，
            # 所以选择「告警 + 继续执行」，把缺失暴露在日志与 trace 里，
            # 由汇总阶段的证据充分性判定兜底。
            # ⚠️ 本模块用标准库 logging（%s 风格），不是 loguru（{} 风格）：
            # 写成 "{}"... 会抛 "not all arguments converted during string formatting"，
            # 与之前 execute_node 踩过的坑同源。
            logger.warning("%s，已达重试上限，fail-open 继续执行", detail)

    return result


class PlannerAgent:
    """P1① 扁平化：直接持有 ModelRouter（2 层链路：Agent → ModelRouter.chat → 引擎）。

    旧嵌套（改造前）：Agent → orchestrator._LLMAdapter → _PurposeLLMAdapter → ModelRouter（4 层）
    新链路（改造后）：Agent → ModelRouter.chat（2 层，减少 2 个适配器包装层）

    注：旧 ``llm`` / ``acomplete`` 协议兼容分支已移除——当前接线恒走 ModelRouter，
    该分支不可达（orchestrator 只以 model_router= 构造本类）。
    """

    def __init__(
            self,
            tools: Optional[ToolInvoker] = None,
            memory: Optional[Any] = None,  # 废弃底层单体耦合
            max_replan_attempts: int = 2,
            *,
            model_router: Any,
            purpose_hint: str = "planner",
            call_budget: Optional[Any] = None,
            enable_empty_result_replan: bool = True,
            allowed_tool_names: Optional[List[str]] = None,
    ) -> None:
        if model_router is None:
            raise ValueError("PlannerAgent 需要提供 model_router（旧 llm/acomplete 协议已移除）")
        self._model_router = model_router
        self._purpose: str = purpose_hint
        self._tools = tools
        # 运行时工具白名单（意图 ∩ 注册中心 ∩ 技能号令）：既用于提示词注入清单，
        # 也用于给 response_format 的 tool_name 生成 enum，以及 _parse_subtasks 的兜底映射。
        self._allowed_tool_names: List[str] = [
            str(name).strip()
            for name in (allowed_tool_names or [])
            if isinstance(name, str) and str(name).strip()
        ]
        # 工具调用预算：单次 Agent 请求作用域（plan/replan 提示词动态规划 + execute 硬熔断共用）
        self._call_budget: Optional[Any] = call_budget
        # 【方案B】空业务结果 → 触发 replan（非异常型），默认开启
        self._enable_empty_result_replan: bool = enable_empty_result_replan
        self.max_replan_attempts = max(0, max_replan_attempts)

    async def _llm_chat(self, messages, **kwargs) -> str:
        """统一 LLM 调用入口：走 ModelRouter，返回纯文本 content。"""
        resp = await self._model_router.chat(
            messages=list(messages), purpose_hint=self._purpose, **kwargs
        )
        return (getattr(resp, "content", None) or "").strip()

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

    def _resolve_allowed_tools(self) -> List[str]:
        """当前允许调用的工具名（构造参数优先，缺省回落到注册中心全量）。"""
        if self._allowed_tool_names:
            return list(self._allowed_tool_names)
        getter = getattr(self._tools, "list_tool_names", None)
        if callable(getter):
            try:
                return [str(name) for name in getter()]
            except Exception as exc:  # noqa: BLE001 - 反射失败不应阻断规划
                logger.warning("Planner 反射工具清单失败，跳过工具白名单注入：%s", exc)
        return []

    def _tool_catalog_block(self) -> str:
        """生成「可用工具清单」提示段：工具名 + 能力描述 + 必填/可选参数。

        ⚠️ 这一段是让 planner 不再编造工具名的**主手段**（schema enum 只是辅助约束：
        各家对 ``anyOf`` 分支内 enum 的执行力度并不一致）。

        旧实现里 planner 在 plan 路径上根本拿不到工具名，只能从预算段看到
        "其余 N 个工具额度充足"，于是按技能名派生出了 ``sales_intelligence_query``
        这种不存在的工具 → 执行被白名单拒绝 → 触发一次完整 replan。
        """
        allowed: List[str] = self._resolve_allowed_tools()
        if not allowed:
            return ""
        getter = getattr(self._tools, "get_tool", None)
        lines: List[str] = ["## 可用工具清单（tool_name 必须原样取自下列名称，禁止编造）"]
        for name in allowed:
            desc: str = ""
            param_desc: str = ""
            if callable(getter):
                try:
                    tool: Any = getter(name)
                except Exception:  # noqa: BLE001 - 个别工具反射失败不影响其余工具
                    tool = None
                if tool is not None:
                    desc = str(getattr(tool, "description", "") or "").strip()
                    parts: List[str] = []
                    for param in (getattr(tool, "parameters", None) or []):
                        mark: str = "*" if getattr(param, "required", False) else ""
                        parts.append(f"{getattr(param, 'name', '?')}{mark}")
                    if parts:
                        param_desc = "，".join(parts)
            entry: str = f"- {name}" + (f"：{desc}" if desc else "")
            if param_desc:
                entry += f"\n    参数: {param_desc}（* 为必填）"
            lines.append(entry[:400])
        lines.append(
            "禁止把技能名或业务词直接当工具名使用（例如 sales_intelligence_query 这类工具不存在）；"
            "知识/业务类问答一律用 rag_knowledge_search；确无合适工具时把子任务设为 reasoning。"
        )
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _compact_results_for_replan(results: Any) -> List[Dict[str, Any]]:
        """把完整执行记录压成〔id / 工具 / 状态 / 摘要〕四元组。

        重规划只需要知道"哪些做完了、结论是什么、哪些失败了"，**不需要**原始观测。
        旧实现直接把 ``results`` 全量 json.dumps 后截断 16000 字符，而单条
        observation 在上游就被截到了 8000 字符 —— 于是整篇 SKILL.md 会原样进入
        replan 提示词，这是 replan 单笔 7149 token 的主要来源。

        摘要优先取子任务 LLM 提炼结论（``llm_output``），没有才退回原始观测。
        """
        compact: List[Dict[str, Any]] = []
        for item in results or []:
            if not isinstance(item, dict):
                continue
            summary: str = str(
                item.get("llm_output") or item.get("observation") or ""
            )[:REPLAN_SUMMARY_MAX_CHARS]
            compact.append({
                "id": item.get("id"),
                "tool_name": item.get("tool_name"),
                "status": item.get("status"),
                "summary": summary,
            })
        return compact

    def _plan_response_formats(self) -> tuple:
        """返回 (动态带 enum 的 response_format, 静态兜底 response_format)。

        第 1 次尝试用动态 schema（模型无法输出白名单外的工具名）；
        第 2 次（retry）退回静态 schema —— 万一厂商拒绝动态 schema，
        链路仍可继续，不会因 schema 问题整体降级成 fallback 计划。
        """
        allowed: List[str] = self._resolve_allowed_tools()
        dynamic: Optional[Dict[str, Any]] = (
            pydantic_to_openai_response_format(build_dynamic_plan_schema(allowed))
            if allowed
            else None
        )
        static: Dict[str, Any] = pydantic_to_openai_response_format(PlanGenerateSchema)
        return dynamic, static

    # 整合skills
    async def plan(
            self, query: str, memory_context: Dict[str, Any], skills_block: str = '',
            sub_question_count: int = 0,
    ) -> List[SubTask]:
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
                           # 可用工具清单：plan 路径上唯一能看到真实工具名的地方
                           f"{self._tool_catalog_block()}"
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

        allowed_tool_names: List[str] = self._resolve_allowed_tools()
        dynamic_format, static_format = self._plan_response_formats()

        # 按 8 层框架：最多 2 次 LLM 调用（首次 + Retry 1 次）
        for attempt_index in range(2):
            temperature_value: float = 0.3 if attempt_index == 0 else 0.4
            # 首次用带 tool_name enum 的动态 schema；retry 退回静态 schema
            #（服务端若不支持动态 schema，仍能拿到计划，不会整体降级）
            response_format = dynamic_format if attempt_index == 0 else static_format
            try:
                # MID-1 Planner 初始计划：关闭思考 + 严格 JSON schema（subtasks 数组）
                raw = await self._llm_chat(
                    messages,
                    temperature=temperature_value,
                    thinking=False,
                    response_format=response_format or static_format,
                )
                parsed_dict = _extract_json_object(raw)
                subtasks_list = _parse_subtasks(
                    parsed_dict,
                    allowed_tool_names,
                    sub_question_count=sub_question_count,
                    # 只有首次解析才因覆盖缺失而重试；重试后仍缺失则告警放行
                    strict_coverage=(attempt_index == 0),
                )
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
        # 2.2：规范实现已迁至 graph/nodes/_fc_args.py（execute 节点共用），此处薄委托。
        from app.core.agent.graph.nodes._fc_args import resolve_tool_args_via_function_call

        return await resolve_tool_args_via_function_call(
            self._model_router,
            self._tools,
            tool_name=task.tool_name,
            title=task.title,
            description=task.description,
            tool_args_hint=task.tool_args_hint,
            query=query,
            prior_context_str=prior_context_str,
            purpose_hint=self._purpose,
        )

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
        # ⚠️ 只喂"摘要 + 失败原因 + 白名单"：全量 results_so_far 会把整篇
        # SKILL.md（单条 observation 上游已截到 8000 字符）原样带进 replan。
        payload = {
            "allowed_tools": self._resolve_allowed_tools(),
            "previous_plan": [
                {
                    "id": task.id,
                    "title": task.title,
                    "action_type": task.action_type,
                    "tool_name": task.tool_name,
                }
                for task in plan
            ],
            "results_so_far": self._compact_results_for_replan(results),
            "error": (error or "")[:500],
        }
        messages: Sequence[Dict[str, str]] = [
            {"role": "system", "content": REPLAN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "上下文：\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)[:REPLAN_PAYLOAD_MAX_CHARS]
                + "\n\n"
                + self._tool_catalog_block()
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

        allowed_tool_names: List[str] = self._resolve_allowed_tools()
        dynamic_format, static_format = self._plan_response_formats()

        # 按 8 层框架：最多 2 次 LLM 调用（首次 + Retry 1 次）—— REPLAN 不关闭 thinking
        for attempt_index in range(2):
            temperature_value: float = 0.3 if attempt_index == 0 else 0.45
            response_format = dynamic_format if attempt_index == 0 else static_format
            try:
                # REPLAN：A-C2 属于 Plan 循环恢复点的推理环节（不关闭 thinking 保留推理），
                # 但复用 Plan schema 的 response_format 保证输出格式 100% 正确。
                raw = await self._llm_chat(
                    messages,
                    temperature=temperature_value,
                    thinking=False,
                    response_format=response_format or static_format,
                )
                parsed_dict = _extract_json_object(raw)
                subtasks_list = _parse_subtasks(parsed_dict, allowed_tool_names)
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
