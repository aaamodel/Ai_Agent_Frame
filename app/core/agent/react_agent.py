# -*- coding: utf-8 -*-
"""文件所在目录：app/core/agent/react_agent.py
ReAct Agent：Thought → Action → Observation 循环与专属技能集注入。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


#from app.core.tools import ToolRegistry

logger = logging.getLogger(__name__)


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


class LLMCallable:
    async def acomplete(self, messages: Sequence[Dict[str, str]], **kwargs: Any): ...


class ToolInvoker:
    async def invoke(self, name: str, arguments: Dict[str, Any]): ...


# 代码层即可判定的"无效返回"特征串：空结果 / 工具系统异常 / 官方兜底提示等。
# （内容"不匹配但非空"无法用代码判定，交给 reach_checkpoint 时的 LLM 相关性抽查。）
_INVALID_OBSERVATION_MARKERS: tuple = (
    "【系统提示】联网搜索工具目前不可用",      # Tavily 兜底文案
    "执行期间发生异常",                        # registry 捕获的异常统一前缀
    "工具执行异常",
    "错误：试图调用未注册的工具",
    "参数 query 不能为空",
    "参数 不能为空",
    "【系统拒绝执行】",                        # 工具自身/其它熔断语义文案
)


def tool_observation_looks_invalid(obs: str) -> bool:
    """代码层判断一次工具返回是否算"无效结果"（空 / 异常 / 系统错误提示）。"""
    text: str = (obs or "").strip()
    if not text:
        return True
    for marker in _INVALID_OBSERVATION_MARKERS:
        if marker in text:
            return True
    return False


async def judge_tool_result_relevance(
    model_router: Any,
    tool_name: str,
    call_context: str,
    observation: str,
    purpose_hint: str = "react",
) -> bool:
    """轻量 LLM 抽查：第 N 次调用结果与调用意图是否相关且含实质信息。

    Args:
        model_router: ModelRouter（负责纯文本 chat）
        tool_name: 被抽查的工具名
        call_context: 该次调用的意图/参数上下文（如检索 query 文本）
        observation: 工具返回内容（会截断）
        purpose_hint: 路由场景提示（默认 react → FAST 档）

    Returns:
        True=相关有效（不干预）；False=完全不匹配（调用方应硬熔断）。
        判定失败时返回 True（宁可放行也不误杀工具）。
    """
    if model_router is None:
        return True
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "你是一个工具结果相关性审查员。请结合「调用意图/参数」与「工具实际返回内容」，"
                "判断该次调用是否返回了与意图相关且含有实质性信息的有效结果。"
                "仅输出一个词：相关 或 不相关。不要输出任何解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"工具名称：{tool_name}\n"
                f"调用意图/参数：\n{(call_context or '')[:1500]}\n\n"
                f"工具实际返回内容（截断）：\n{(observation or '')[:4000]}"
            ),
        },
    ]
    try:
        resp: Any = await model_router.chat(
            messages=messages,
            temperature=0.0,
            thinking=False,
            purpose_hint=purpose_hint,
        )
        answer: str = (getattr(resp, "content", None) or "").strip()
        return "不相关" not in answer
    except Exception:  # noqa: BLE001 - 抽查失败不影响主流程
        return True


async def postprocess_tool_result(
    call_budget: Any,
    tool_name: str,
    call_context: str,
    observation: str,
    model_router: Any,
) -> str:
    """工具真实执行后的预算后处理：登记无效结果 / 触发第 N 次相关性抽查。

    Returns:
        应回喂给 LLM 的观察文本（若该工具刚被熔断，会在头部附 lock_notice）。
    """
    if call_budget is None:
        return observation or ""
    raw: str = observation or ""

    # ① 代码层无效（空 / 异常 / 系统错误）→ 累计无效计数，达 3 次即熔断
    if tool_observation_looks_invalid(raw):
        just_locked: bool = call_budget.record_invalid(tool_name)
        if just_locked:
            return call_budget.lock_notice(tool_name) + raw
        return raw

    # ② 内容非空 → 若恰为该工具第 N 次调用（默认 5），做一次 LLM 相关性抽查
    if call_budget.reached_relevance_checkpoint(tool_name):
        relevant: bool = await judge_tool_result_relevance(
            model_router, tool_name, call_context, raw
        )
        if not relevant:
            call_budget.lock_tool(tool_name, "relevance")
            return call_budget.lock_notice(tool_name) + raw

    return raw


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


class ReActAgent:
    """P1① 扁平化：优先直接持有 ModelRouter（Agent → ModelRouter.chat 2 层链路），
    兼容旧 `llm` 参数作为回退。
    旧嵌套：Agent → orchestrator._LLMAdapter → _PurposeLLMAdapter → ModelRouter（4 层）
    新链路：Agent → ModelRouter.chat（2 层，减少 2 层包装）
    """

    def __init__(
            self,
            llm: Optional[LLMCallable] = None,
            tools: Optional[ToolInvoker] = None,
            memory: Optional[Any] = None,
            max_steps: int = 1,
            session_id: Optional[str] = None,
            *,
            model_router: Optional[Any] = None,
            purpose_hint: str = "react",
    ) -> None:
        if llm is None and model_router is None:
            raise ValueError("ReActAgent 需要提供 llm 或 model_router 至少其一")
        self._llm = llm
        self._model_router = model_router
        self._purpose: str = purpose_hint
        self._tools = tools
        self.max_steps = max(1, max_steps)
        self.session_id = session_id or "default"

    async def _llm_chat(self, messages, **kwargs) -> str:
        """统一 LLM 调用入口：优先 ModelRouter，否则回退旧 acomplete。"""
        if self._model_router is not None:
            resp = await self._model_router.chat(
                messages=list(messages), purpose_hint=self._purpose, **kwargs
            )
            return (getattr(resp, "content", None) or "").strip()
        return await self._llm.acomplete(messages, **kwargs)

    def _tool_catalog_text(self, tool_names: Sequence[str], tool_schemas: Dict[str, str] = {}) -> str:
        """
        构造详细的工具说明块，优先使用传入的 tool_schemas（含参数描述），
        若未提供则退化为简单的名称列表。
        """
        lines = []
        for name in tool_names:
            if name in tool_schemas:
                lines.append(f"### {name}\n{tool_schemas[name]}")
            else:
                lines.append(f"- {name}  （参数信息缺失，请根据常识谨慎填写）")
        return "\n\n".join(lines) if lines else "（无外部工具，请直接 Final Answer）"

    async def run_react_agent(self, query: str, context: Dict[str, Any], skills_block: str = '') -> AgentResult:
        """ReAct 执行入口：优先【原生 Function Calling】主链路；不可用时自动降级文本协议兜底。

        - 当编排层提供了 ``openai_tools``（工具池 JSON Schema）且持有 ModelRouter 时，
          走 ``_run_fc_agent``（模型通过 tool_calls 自主选工具调用，参数由协议保证合法）。
        - 若模型/通道不支持 tools（FC 调用抛异常 → ``_FCFallbackRequired``）或缺少工具池，
          整体回退到原有的文本 Thought/Action 协议（``_run_text_agent``），保证兼容性。
        """
        fc_tools: List[Dict[str, Any]] = list(context.get("openai_tools") or [])
        if fc_tools and self._model_router is not None:
            try:
                return await self._run_fc_agent(query, context, skills_block, fc_tools)
            except _FCFallbackRequired as fallback_err:
                logger.warning(
                    "ReAct 原生 FC 主链路不可用，降级走文本 Thought/Action 协议。原因: {}",
                    fallback_err,
                )
                # 模型候选已耗尽/超时：文本兜底会再次对同一批模型超时，重试无意义，
                # 直接返回友好降级答复，避免用户再等一个完整超时窗口后硬失败。
                if _is_candidate_exhaustion(fallback_err):
                    logger.warning(
                        "FC 失败源于模型候选耗尽/超时，跳过文本兜底，直接返回友好降级答复。"
                    )
                    return AgentResult(
                        success=False,
                        final_answer=GRACEFUL_TIMEOUT_MESSAGE,
                        steps=[],
                        error=str(fallback_err),
                    )
                # 清除 openai_tools，确保文本兜底不再尝试 FC
                context = {**context, "openai_tools": []}
        return await self._run_text_agent(query, context, skills_block)

    async def _run_fc_agent(
            self,
            query: str,
            context: Dict[str, Any],
            skills_block: str,
            fc_tools: List[Dict[str, Any]],
    ) -> AgentResult:
        """原生 Function Calling 主链路（tool_choice="auto" · thinking 保持开启）。

        遵循 OpenAI 兼容多轮协议：模型返回 tool_calls → 执行工具 → 追加 assistant(tool_calls)
        + 各 tool 结果消息 → 继续下一轮；直到模型返回纯文本 content 即视为最终答案。
        """
        trace_cb = context.get("trace_callback")
        tool_names: List[str] = list(context.get("tool_names") or [])
        allowed: Optional[set] = set(tool_names) if tool_names else None
        call_budget: Optional[Any] = context.get("call_budget")  # 工具调用预算（动态规划+熔断）
        extra_system = str(context.get("extra_system", ""))
        effective_skills = skills_block or context.get("skills_prompt", "")

        memory_context = context.get("memory_context") or {"short_term": [], "long_term": []}
        short_term_list = memory_context.get("short_term") or []
        long_term_snippets = memory_context.get("long_term") or []
        mem_block = "\n".join(f"- {s}" for s in long_term_snippets) if long_term_snippets else "（无）"

        steps: List[Dict[str, Any]] = []
        messages: List[Dict[str, Any]] = []
        tool_call_counts: Dict[str, int] = {}
        max_tool_attempts = 3
        empty_turns: int = 0

        system_content: str = (
            REACT_FC_SYSTEM_PROMPT
            + ("\n\n" + extra_system if extra_system else "")
            + f"\n\n## 检索长期事实记忆\n{mem_block}"
        )
        messages.append({"role": "system", "content": system_content})

        for old_msg in short_term_list:
            role = getattr(old_msg, "role", None) or old_msg.get("role", "user")
            if hasattr(role, "value"):
                role = role.value
            content = getattr(old_msg, "content", None) or old_msg.get("content", "")
            messages.append({"role": str(role), "content": str(content)})

        user_parts: List[str] = [f"## 用户问题\n{query}"]
        if effective_skills:
            user_parts.append(f"## 可用高级技能 (渐进式披露)\n{effective_skills}")
        user_parts.append("请基于以上信息推进任务：需要外部信息或动作时调用工具，信息足够时直接输出面向用户的最终答案。")
        messages.append({"role": "user", "content": "\n\n".join(user_parts)})

        for step_idx in range(self.max_steps):
            # ---- 模型自主决策：调用工具 or 输出最终答案 ----
            # 每次请求前临时注入实时工具额度（不持久化进 messages，避免历史膨胀）
            send_messages: List[Dict[str, Any]] = [*messages]
            if call_budget is not None:
                budget_live: str = call_budget.live_prompt()
                if budget_live:
                    send_messages.append({"role": "system", "content": budget_live})
            try:
                resp = await self._model_router.chat_with_tools(
                    messages=send_messages,
                    tools=fc_tools,
                    tool_choice="auto",
                    purpose_hint=self._purpose,
                    temperature=0.2,
                    thinking=False,
                )
            except Exception as fc_err:  # noqa: BLE001 - 通道/模型不支持 tools 等 → 整体文本兜底
                raise _FCFallbackRequired(f"FC 调用异常: {fc_err}") from fc_err

            reasoning_txt: str = getattr(resp, "reasoning_content", None) or ""
            tool_calls: List[Dict[str, Any]] = list(getattr(resp, "tool_calls", None) or [])

            if not tool_calls:
                answer: str = (getattr(resp, "content", None) or "").strip()
                if not answer:
                    # 既无工具调用也无最终答案（如思考模型思考完但被截断）：注入提示并重试
                    empty_turns += 1
                    if empty_turns > 2:
                        return AgentResult(
                            success=False, final_answer="", steps=steps,
                            error="FC 主链路连续多轮未产出工具调用或最终答案",
                        )
                    messages.append({
                        "role": "user",
                        "content": "[系统提示] 你上一轮既没有调用工具，也没有输出最终答案。"
                                   "请直接判断：若仍需数据请立即调用合适工具；若已足够请直接输出最终答案文本。",
                    })
                    continue

                rec: Dict[str, Any] = {
                    "step": step_idx,
                    "phase": "react",
                    "kind": "fc_final",
                    "raw_llm": answer[:4000],
                    "reasoning": reasoning_txt[:2000],
                    "parsed": {"done": True, "final_answer": answer},
                    "final": True,
                }
                steps.append(rec)
                if trace_cb:
                    await trace_cb(rec)
                return AgentResult(success=True, final_answer=answer, steps=steps)

            # ---- 本轮决定调用工具：先归一化 tool_call_id 并回填 assistant 消息 ----
            empty_turns = 0
            for call_index, tc in enumerate(tool_calls):
                if not tc.get("id"):
                    tc["id"] = f"call_{step_idx}_{call_index}"
            assistant_payload: Dict[str, Any] = {
                "role": "assistant",
                "content": None,  # OpenAI 协议要求：tool_calls 消息 content 置空/null
                "tool_calls": tool_calls,
            }
            messages.append(assistant_payload)

            for tc in tool_calls:
                func_name: str = str(tc.get("function", {}).get("name") or "")
                raw_arguments: str = str(tc.get("function", {}).get("arguments") or "")
                parsed_args: Dict[str, Any] = {}
                if raw_arguments:
                    try:
                        parsed_value: Any = json.loads(raw_arguments)
                        parsed_args = parsed_value if isinstance(parsed_value, dict) else {}
                    except json.JSONDecodeError:
                        parsed_args = {}

                rec = {
                    "step": step_idx,
                    "phase": "react",
                    "kind": "fc_tool_call",
                    "tool_call": tc,
                    "reasoning": reasoning_txt[:2000],
                    "parsed": {"action": func_name, "action_input": parsed_args},
                }

                # L7/业务门：白名单校验 → 额度熔断（预算优先，先检查后计数）→ 旧计数兜底
                if allowed is not None and func_name not in allowed:
                    obs_text = (
                        f"【系统校验 FAIL（工具白名单）】：工具 [{func_name}] 不在当前允许调用的工具列表中。\n"
                        f"允许的工具：{sorted(list(tool_names))}\n"
                        f"请从允许列表中选择工具，或若信息已足够请直接输出 Final Answer。"
                    )
                elif call_budget is not None:
                    if not call_budget.can_call(func_name):
                        # 硬熔断：单工具上限/无效累计/相关性抽查命中/总额度用尽
                        obs_text = call_budget.deny_text(func_name)
                    else:
                        call_budget.consume(func_name)
                        try:
                            obs_value: Any = await self._tools.invoke(func_name, parsed_args)
                            if isinstance(obs_value, str):
                                obs_text = obs_value
                            elif isinstance(obs_value, (dict, list)):
                                obs_text = json.dumps(obs_value, ensure_ascii=False, indent=2)
                            else:
                                obs_text = str(obs_value)
                        except Exception as tool_exc:  # noqa: BLE001
                            obs_text = f"工具执行异常: {tool_exc}"
                        # 后处理：无效结果累计 / 第 N 次相关性抽查
                        obs_text = await postprocess_tool_result(
                            call_budget, func_name, raw_arguments, obs_text, self._model_router
                        )
                else:
                    tool_call_counts[func_name] = tool_call_counts.get(func_name, 0) + 1
                    if tool_call_counts[func_name] > max_tool_attempts:
                        logger.warning(
                            "🚨 FC 工具 [%s] 调用次数已达上限（%d次），触发硬熔断！",
                            func_name, max_tool_attempts,
                        )
                        obs_text = (
                            f"【系统拒绝执行】：你已经连续/累计调用工具 [{func_name}] 达到 {max_tool_attempts} 次的最大上限。"
                            f"该工具已被系统锁定。这通常意味着你使用的参数在当前生产环境中不可用。"
                            f"请绝对不要再次尝试调用 [{func_name}]！请利用现有信息直接回答，或改用其他工具。"
                        )
                    else:
                        try:
                            obs_value: Any = await self._tools.invoke(func_name, parsed_args)
                            if isinstance(obs_value, str):
                                obs_text = obs_value
                            elif isinstance(obs_value, (dict, list)):
                                obs_text = json.dumps(obs_value, ensure_ascii=False, indent=2)
                            else:
                                obs_text = str(obs_value)
                        except Exception as tool_exc:  # noqa: BLE001
                            obs_text = f"工具执行异常: {tool_exc}"

                rec["action"] = func_name
                rec["action_input"] = parsed_args
                rec["observation"] = obs_text[:8000]
                steps.append(rec)
                if trace_cb:
                    await trace_cb(rec)

                # 协议闭环：把该工具真实结果作为 role=tool 消息喂回对话
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": obs_text[:8000],
                })

        return AgentResult(success=False, final_answer="", steps=steps, error="达到最大步数")

    async def _run_text_agent(self, query: str, context: Dict[str, Any], skills_block: str = '') -> AgentResult:
        trace_cb = context.get("trace_callback")
        tool_names: List[str] = list(context.get("tool_names") or [])
        extra_system = str(context.get("extra_system", ""))
        call_budget: Optional[Any] = context.get("call_budget")  # 工具调用预算（动态规划+熔断）

        # 🎯 核心新增：从上下文获取工具参数描述（若调用方未传入，则使用简单列表）
        tool_schemas: Dict[str, str] = context.get("tool_schemas", {})

        effective_skills = skills_block or context.get("skills_prompt", "")

        memory_context = context.get("memory_context") or {"short_term": [], "long_term": []}
        short_term_list = memory_context.get("short_term") or []
        long_term_snippets = memory_context.get("long_term") or []

        mem_block = "\n".join(f"- {s}" for s in long_term_snippets) if long_term_snippets else "（无）"

        steps: List[Dict[str, Any]] = []
        history_lines: List[str] = []
        tool_call_counts: Dict[str, int] = {}
        max_tool_attempts = 3

        for step_idx in range(self.max_steps):
            # 构造包含详细参数说明的工具描述文本
            tool_desc = self._tool_catalog_text(tool_names, tool_schemas)
            history_block = "\n".join(history_lines) if history_lines else "（尚无）"

            user_prompt = build_react_user_prompt(
                query=query,
                tool_descriptions=tool_desc,
                history_block=history_block,
                skills_block=effective_skills
            )
            # 每轮注入实时工具额度，供模型动态规划后续该调/不该调哪些工具
            if call_budget is not None:
                budget_live: str = call_budget.live_prompt()
                if budget_live:
                    user_prompt = f"{budget_live}\n\n{user_prompt}"

            messages: List[Dict[str, str]] = []
            messages.append({
                "role": "system",
                "content": REACT_SYSTEM_PROMPT
                           + ("\n\n" + extra_system if extra_system else "")
                           + f"\n\n## 检索长期事实记忆\n{mem_block}"
            })

            for old_msg in short_term_list:
                role = getattr(old_msg, "role", None) or old_msg.get("role", "user")
                if hasattr(role, "value"):
                    role = role.value
                content = getattr(old_msg, "content", None) or old_msg.get("content", "")
                messages.append({"role": str(role), "content": str(content)})

            messages.append({"role": "user", "content": user_prompt})



            try:
                raw = await self._llm_chat(messages, temperature=0.2)

            except Exception as e:
                err = f"LLM 调用失败: {e}"
                print(f"❌ 角色 [AI_MESSAGE] 调用异常: {err}\n" + "=" * 110 + "\n")
                return AgentResult(success=False, final_answer=GRACEFUL_TIMEOUT_MESSAGE, steps=steps, error=err)

            parsed = _parse_react_step(raw)
            rec: Dict[str, Any] = {
                "step": step_idx, "phase": "react", "raw_llm": raw[:4000],
                "parsed": {k: v for k, v in parsed.items() if k != "raw"},
            }

            if parsed.get("done") and parsed.get("final_answer"):
                answer = str(parsed["final_answer"])
                rec["final"] = True
                steps.append(rec)
                if trace_cb:
                    await trace_cb(rec)
                return AgentResult(success=True, final_answer=answer, steps=steps)

            action = parsed.get("action")
            action_input = parsed.get("action_input") or {}
            business_warn = parsed.get("business_warn")

            # ---------- L8 PASS / FAIL 路由：任何 FAIL（parse/semantic/business/action缺失）
            #            一律注入结构化 Observation → 下一轮循环 Retry，**不中断 Pipeline** ----------
            if not action or parsed.get("parse_error") or parsed.get("semantic_error") or parsed.get("business_error"):
                fail_reasons: List[str] = []
                if parsed.get("parse_error"):
                    fail_reasons.append(str(parsed["parse_error"]))
                if parsed.get("semantic_error"):
                    fail_reasons.append(str(parsed["semantic_error"]))
                if parsed.get("business_error"):
                    fail_reasons.append(str(parsed["business_error"]))
                if not action and not fail_reasons:
                    fail_reasons.append("未解析到 Action，也没有 Final Answer。")
                obs = (
                    "【系统校验 FAIL（进入下一轮循环 Retry）】：\n"
                    + "\n".join(f"- {reason}" for reason in fail_reasons)
                    + "\n\n【建议调整】："
                    + "请严格遵守 ReAct 输出格式。你只能选择两条路径之一：\n"
                    + "  (1) 调用工具：Thought: ... → Action: <合法工具名> → Action Input: <严格合法 JSON object>\n"
                    + "  (2) 直接回答：直接输出 Final Answer: <你的最终答复>\n"
                    + f"\n当前步骤的 Thought 内容是：{parsed.get('thought', '')}"
                )
                rec["error"] = fail_reasons
                rec["observation"] = obs[:8000]
                steps.append(rec)
                if trace_cb:
                    await trace_cb(rec)
                print(f"⚙️ 角色 [VALIDATION FAIL Observation]:\n输入: {action_input}\n返回: {obs}")
                print("=" * 110 + "\n")
                history_lines.append(
                    f"Step {step_idx + 1}\nThought: {parsed.get('thought', '')}\n"
                    f"[系统校验 FAIL，结构化回传以便下一轮 Retry]\nObservation: {obs}\n"
                )
                # 进入下一轮循环（max_steps 耗尽前都视作 L8 FAIL→Retry）
                continue

            if tool_names and action not in tool_names:
                obs = (
                    f"【系统校验 FAIL（工具白名单）】：工具 [{action}] 不在当前允许调用的工具列表中。\n"
                    f"允许的工具：{sorted(list(tool_names))}\n"
                    f"请从允许列表中选择工具，或若信息已足够请直接输出 Final Answer。"
                )
                rec["warn"] = f"工具 [{action}] 不在白名单，本轮按 FAIL 回注 Observation"
            elif call_budget is not None:
                # 额度熔断（预算优先，先检查后计数）
                if not call_budget.can_call(action):
                    obs = call_budget.deny_text(action)
                    rec["budget_denied"] = True
                else:
                    call_budget.consume(action)
                    try:
                        obs = await self._tools.invoke(action, action_input or {})
                        if business_warn:
                            obs = f"{business_warn}\n\n{obs}"
                    except Exception as e:
                        obs = f"工具执行异常: {e}"
                    # 后处理：无效结果累计 / 第 N 次相关性抽查
                    obs = await postprocess_tool_result(
                        call_budget, action,
                        json.dumps(action_input, ensure_ascii=False) if action_input else "{}",
                        obs, self._model_router,
                    )
            else:
                tool_call_counts[action] = tool_call_counts.get(action, 0) + 1
                if tool_call_counts[action] > max_tool_attempts:
                    logger.warning(f"🚨 工具 [{action}] 调用次数已达上限（{max_tool_attempts}次），触发硬熔断！")
                    obs = (
                        f"【系统拒绝执行】：你已经连续/累计调用工具 [{action}] 达到 {max_tool_attempts} 次的最大上限。 "
                        f"该工具已被系统锁定。这通常意味着你使用的参数（如表名、SQL语法、API入参）在当前生产环境中不可用。 "
                        f"请绝对不要再次尝试调用 [{action}]！请利用现有信息直接回答，或尝试转换思路改用其他工具。"
                    )
                else:
                    try:
                        obs = await self._tools.invoke(action, action_input or {})
                        if business_warn:
                            obs = f"{business_warn}\n\n{obs}"
                    except Exception as e:
                        obs = f"工具执行异常: {e}"


            rec["action"] = action
            rec["action_input"] = action_input
            rec["observation"] = obs[:8000]
            steps.append(rec)
            if trace_cb:
                await trace_cb(rec)

            history_lines.append(
                f"Step {step_idx + 1}\nThought: {parsed.get('thought', '')}\n"
                f"Action: {action}\nObservation: {obs}\n"
            )

        return AgentResult(success=False, final_answer="", steps=steps, error="达到最大步数")