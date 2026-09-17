# -*- coding: utf-8 -*-
"""文件所在目录：app/core/agent/react_agent.py
ReAct Agent：Thought → Action → Observation 循环与专属技能集注入。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 2.1：工具调用统一抽象（ToolCall / 执行流水线 / 预算后处理），FC 与文本两路共用
from app.core.agent.toolcall import (
    ToolInvoker,
    execute_tool_call,
    judge_tool_result_relevance,
    postprocess_tool_result,
    tool_call_from_text,
    tool_calls_from_fc,
    tool_observation_looks_invalid,
)

__all__ = [
    "AgentResult",
    "ToolInvoker",
    "GRACEFUL_TIMEOUT_MESSAGE",
    "build_react_user_prompt",
    "execute_tool_call",
    "judge_tool_result_relevance",
    "postprocess_tool_result",
    "tool_call_from_text",
    "tool_calls_from_fc",
    "tool_observation_looks_invalid",
]


def _mini_json_repair(raw: str) -> str:
    """ReAct Action Input 场景专属 JSON 轻量修复（纯标准库，无第三方依赖）。

    针对 LLM 输出"Action Input: <JSON>"片段时最高频的 4 类病态修复：
      1. 去围栏残余（```json / ``` 开头或收尾的行）
      2. 顶层杂文字剥离（输出前后的"Sure: {...}" / "如下所示：\n{...}\n希望有帮助"）
      3. 单引号字段/值 → 双引号（前 50 字符无双引号时触发，避免误修英文 contractions）
      4. 尾随逗号（trailing comma）修复：} / ] 之前跳过空白后的逗号删除（最多 5 次扫描收敛嵌套尾逗号）

    Args:
        raw: 正则抓出来的疑似 Action Input JSON 原始字符串

    Returns:
        修复后的字符串；若无法修复原样返回（由下游 json.loads 再报 parse_error 兜底）。
    """
    working: str = raw or ""
    if not working:
        return working

    # Step 1: 去 Markdown 围栏行（LLM 有时会把 Action Input 包进围栏，尽管 Prompt 禁止）
    if "```" in working:
        lines: List[str] = working.splitlines()
        clean_lines: List[str] = [ln for ln in lines if not ln.lstrip().startswith("```")]
        working = "\n".join(clean_lines)

    # Step 2: 顶层花括号切片（优先取第一个 { 到最后一个 }，剥离 Action Input 前后的文字）
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
        return raw

    # Step 3: 单引号 → 双引号（heuristic：前 50 字符没有双引号时才启用，避免误伤嵌套双引号场景）
    if "'" in working and '"' not in working[:50]:
        working = working.replace("'", '"')

    # Step 4: 尾随逗号修复（从右向左扫描，最多 5 次扫描收敛嵌套尾逗号）
    chars: List[str] = list(working)
    n: int = len(chars)
    for _ in range(5):
        changed = False
        for i in range(n - 1, -1, -1):
            if chars[i] in ("}", "]"):
                j = i - 1
                while j >= 0 and chars[j] in (" ", "\t", "\r", "\n"):
                    j -= 1
                if j >= 0 and chars[j] == ",":
                    del chars[j]
                    n -= 1
                    changed = True
                    break
        if not changed:
            break
    return "".join(chars)


REACT_SYSTEM_PROMPT = """你是一个严谨的智能助手，必须使用 ReAct（推理+行动）方式回答问题。

## 输出格式（严格遵守，每一步只输出一块内容）

### 若需要调用工具
先写思考，再写动作：
Thought: <用中文简要说明你为什么需要下一步、打算做什么>
Action: <工具名称，必须是可用工具列表中之一>
Action Input: <严格合法的 JSON 对象，工具的参数>

### 若已有足够信息可直接作答
Thought: <简要总结依据>
Final Answer: <面向用户的完整最终答案，使用用户使用的语言>

## 规则
- 不要编造工具名称或 Observation；Observation 由系统在你输出 Action 后自动追加。
- Thought、Final Answer 允许是自由中文/英文段落；但 **Action Input 必须是严格合法的 JSON**：
  1. 必须以双引号作为对象字段名和字符串值的引号（不得使用单引号）
  2. 不得出现最后一个字段/元素之后的尾随逗号（trailing comma）
  3. 不得输出 Markdown 围栏（```json / ```）、注释、前缀解释或后缀说明
  4. 若工具无参数，必须输出空对象 `{}`（不得省略 Action Input 或输出 null）
"""


# 原生 Function Calling 主链路系统提示词：工具定义经 OpenAI tools[] 下发，模型
# 通过 tool_calls 协议自主选择/调用；仅当信息已足够时才输出最终答案文本。
REACT_FC_SYSTEM_PROMPT = """你是一个严谨的多步骤智能助手，需要通过原生函数调用（Function Calling）完成任务。

## 工作方式
- 系统会通过 `tools` 参数提供一批可用工具及其参数 JSON Schema。
- 每轮你只有两条路径：
  1. **调用工具**：当需要查询外部信息/执行动作才能推进时，直接发起工具调用（tool_calls）。
     系统会返回工具执行结果（Observation），你基于结果继续推理。
  2. **直接作答**：当信息已足够回答用户问题时，停止调用工具，直接用自然语言输出面向用户的最终答案。
- 严禁编造工具执行结果。工具结果一律由系统真实返回，不得虚构。
- 需要分步骤完成时请一步步调用工具推进，不要试图一步到位。"""


class _FCFallbackRequired(Exception):
    """ReAct 原生 FC 主链路遇到不可用异常（模型/通道不支持 tools 等）时抛出，
    外层 run_react_agent 捕获后切回原有文本 Thought/Action 协议兜底。"""


# 模型候选全部失败（通常是超时/通道故障）时的统一友好提示，避免整条链路
# 硬失败对着 500，而是给用户一个可理解、可重试的降级答复。
GRACEFUL_TIMEOUT_MESSAGE: str = (
    "模型服务暂时没有在限定时间内回答（可能是高峰期负载较高）。"
    "你可以稍后重试，或把问题拆分、简化后再问，我会尽力帮你完成。"
)


def _is_candidate_exhaustion(fallback_error: BaseException) -> bool:
    """判断 FC 链路失败是否属于「模型候选耗尽/超时」（此时文本兜底会再次对
    同一批模型超时，重试无意义），而非「模型不支持 tools」的可降级场景。"""
    message: str = str(fallback_error)
    if "candidates failed" in message.lower():
        return True
    if "timeout" in message.lower():
        return True
    cause: Optional[BaseException] = getattr(fallback_error, "__cause__", None)
    while cause is not None:
        if isinstance(cause, TimeoutError):
            return True
        if "candidates failed" in str(cause).lower() or "timeout" in str(cause).lower():
            return True
        cause = getattr(cause, "__cause__", None)
    return False


# 注：ToolInvoker / 无效判定 / 相关性抽查 / 预算后处理已下沉至
# app.core.agent.toolcall（2.1 统一工具调用抽象），文件顶部已 re-export。


def build_react_user_prompt(query: str, tool_descriptions: str, history_block: str, skills_block: str = '') -> str:
    """构造包含高级技能树、可用工具与执行轨迹的终极用户提示词"""
    skills_section = f"## 可用高级技能 (渐进式披露)\n{skills_block}\n\n" if skills_block else ""

    return f"""## 用户问题
{query}

{skills_section}## 可用工具（含参数说明）
{tool_descriptions}

## 已执行的步骤与观察（如有）
{history_block}

请根据当前信息，输出下一步：要么 Action + Action Input，要么 Final Answer。"""




@dataclass
class AgentResult:
    success: bool
    final_answer: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    trace_id: Optional[str] = None


_ACTION_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,99}$")
_MAX_ACTION_INPUT_BYTES: int = 32 * 1024  # 32KB 上限，避免 LLM 吐巨型 JSON 阻塞下游 Router


def _parse_react_step(text: str) -> Dict[str, Any]:
    """ReAct 步骤解析：严格按 8 层框架只约束 Action Input JSON。

    Thought / Final Answer 是自由文本，**完全不走 JSON 分层约束**。
    Action Input 分层（与 8 层标准框架一一对应）：
      L2 JSON Extraction         ：正则只截取 Action Input 的 {xxx} 片段
      L3 Generic Parser          ：_mini_json_repair（4 类病态修复）→ json.loads（通用合法性校验）
      L4 Pydantic                ：N/A（不做，ReAct 业务性质不允许强套 Pydantic — Thought/Final 自由文本）
      L5 Schema Error→Retry      ：N/A（不做 LLM 级 Retry，交给 ReAct 主循环下一步天然 Retry）
      L6 Generic Semantic Validation【显式落地，3 条】：
          ① action_input 必须是 dict（JSON object），不是 list/str/number/null
          ② dict key 必须是字符串（虽然合法 JSON 要求 key 本来就是字符串，但 repair 后仍要二次确认）
          ③ dict 整体体积不得超过 32KB（防止 LLM 恶意/失误塞大文本造成 Router OOM）
      L7 Business Rules【显式落地，2 条】：
          ④ Action 名：如果声明了 action，必须匹配 _ACTION_NAME_PATTERN（字母开头，允许字母/数字/下划线/点/冒号/横杠，≤100 字），否则判 FAIL
          ⑤ Action Input：如果 action 声明了，且 action_input={}（空 dict），不是 ERROR，标 WARN 但允许 PASS（工具执行层再按默认参数跑）
      L8 PASS/FAIL 路由：
          PASS（上面 3+2 条校验通过）→ 输出 validated_action/action_input
          FAIL（任何一条不通过）→ semantic_error/business_error 结构化写进 parsed，
                  主循环把 error 文本注入 Observation 走下一轮循环（天然 Retry，不中断 Pipeline）
    """
    out: Dict[str, Any] = {"raw": text.strip(), "semantic_error": None, "business_error": None}

    thought_m = re.search(r"Thought:\s*(.+?)(?=\n(?:Action:|Final Answer:)|\Z)", text, re.S | re.I)
    if thought_m:
        out["thought"] = thought_m.group(1).strip()

    if re.search(r"Final Answer:\s*", text, re.I):
        fa_m = re.search(r"Final Answer:\s*(.+)\Z", text, re.S | re.I)
        if fa_m:
            out["final_answer"] = fa_m.group(1).strip()
            out["done"] = True
        return out

    # ---------- L2 JSON Extraction：只截取 Action / Action Input 片段 ----------
    action_m = re.search(r"Action:\s*(\S+)", text, re.I)
    input_m = re.search(r"Action Input:\s*(\{[\s\S]*\})", text)
    if action_m:
        raw_action_name: str = action_m.group(1).strip()
        # 去除可能跟在 action 名后面的尾随冒号/JSON 片段（极端 case 正则切不干净）
        raw_action_name = raw_action_name.rstrip(":：,，;；")
        out["action"] = raw_action_name
    else:
        raw_action_name = ""

    # ---------- L3 Generic Parser：repair + json.loads 双路校验 ----------
    repaired_action_input: str = ""
    if input_m:
        raw_action_input_text: str = input_m.group(1)
        repaired_action_input = _mini_json_repair(raw_action_input_text)
        try:
            parsed_json_value: Any = json.loads(repaired_action_input)
            out["action_input"] = parsed_json_value
        except json.JSONDecodeError as decode_err:
            out["action_input"] = {}
            out["parse_error"] = (
                f"L3 Parser FAIL：Action Input JSON 修复后仍解析失败: {decode_err}. "
                f"Repaired head 300: {repaired_action_input[:300]!r}"
            )
            # L3 层 parse_error 直接返回，不继续走 L6/L7，交由主循环注入 Observation
            return out
    else:
        # 没有 Action Input：允许 PASS，但显式赋空 dict，不走 L6 对 action_input 的结构校验
        out["action_input"] = {}

    # ---------- L6 Generic Semantic Validation（3 条） ----------
    parsed_action_input: Any = out.get("action_input", {})

    # L6-① action_input 必须是 dict（JSON object）
    if not isinstance(parsed_action_input, dict):
        out["semantic_error"] = (
            f"L6 Semantic ① FAIL：Action Input 解析结果不是 JSON object（dict），"
            f"实际 type={type(parsed_action_input).__name__}。已自动降级为 action_input={{}}"
        )
        out["action_input"] = {}
        return out

    # L6-② dict key 必须全部是 str
    bad_keys: List[str] = [str(k) for k in parsed_action_input.keys() if not isinstance(k, str)]
    if bad_keys:
        out["semantic_error"] = (
            f"L6 Semantic ② FAIL：Action Input 存在非字符串 key，head 5 bad={bad_keys[:5]!r}。"
            f"已自动降级为 action_input={{}}"
        )
        out["action_input"] = {}
        return out

    # L6-③ action_input 整体序列化体积 ≤ 32KB
    try:
        serialized_input: str = json.dumps(parsed_action_input, ensure_ascii=False)
    except (TypeError, ValueError) as serialize_err:
        out["semantic_error"] = (
            f"L6 Semantic ③ FAIL：Action Input 无法重新序列化为合法 JSON（{serialize_err}）。"
            f"已自动降级为 action_input={{}}"
        )
        out["action_input"] = {}
        return out
    if len(serialized_input.encode("utf-8")) > _MAX_ACTION_INPUT_BYTES:
        out["semantic_error"] = (
            f"L6 Semantic ③ FAIL：Action Input 体积超过 {_MAX_ACTION_INPUT_BYTES} 字节（"
            f"实际 {len(serialized_input.encode('utf-8'))} 字节）。已自动降级为 action_input={{}}"
        )
        out["action_input"] = {}
        return out

    # ---------- L7 Business Rules（2 条） ----------
    if raw_action_name:
        # L7-④ Action 名格式校验：字母开头，合法字符集 + 长度 ≤ 100
        if not _ACTION_NAME_PATTERN.match(raw_action_name):
            out["business_error"] = (
                f"L7 Business ④ FAIL：Action 名 [{raw_action_name!r}] 不符合工具命名规范"
                f"（字母开头，仅允许字母/数字/_/./:/-，长度≤100）。"
            )
            out["action"] = None
            return out
        # L7-⑤ action 声明了但 action_input={}：WARN 但 PASS（工具层会处理默认参数）
        if parsed_action_input == {} and input_m:
            out["business_warn"] = (
                f"L7 Business ⑤ WARN：声明了 Action [{raw_action_name}] 但 Action Input 是空对象 {{}}。"
                f"仍放行 PASS，由 Router / 工具执行层按默认参数处理。"
            )
    else:
        # 没声明 action 且不是 Final Answer：判 business_error，主循环注入 Observation Retry
        out["business_error"] = (
            "L7 Business ④ FAIL：既没有声明 Action，也没有返回 Final Answer。"
            "请严格按 ReAct 格式（Thought → Action: xxx → Action Input: {...}）或直接给出 Final Answer。"
        )
        out["action"] = None

    # L8 PASS / FAIL 路由标记（供主循环识别，不影响 downstream 取值）
    if out.get("semantic_error") or out.get("business_error") or out.get("parse_error"):
        out["validation_result"] = "FAIL"
    else:
        out["validation_result"] = "PASS"

    return out
