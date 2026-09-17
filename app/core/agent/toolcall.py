# -*- coding: utf-8 -*-
"""文件所在目录：app/core/agent/toolcall.py
2.1 统一工具调用抽象（FC 协议 / 文本 ReAct 协议归一）。

两路协议在解析阶段各产生一个 :class:`ToolCall`，之后共用同一套执行流水线
:func:`execute_tool_call`：

    白名单校验 → 预算熔断（先检查后计数）→ invoker.invoke 执行
              → 观测值标准化 → 预算后处理（无效累计 / 相关性抽查）

行为一致性由结构保证，FC 与文本链路不再各自维护一份重复实现。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------
ToolCallSource = Literal["fc", "text"]
"""工具调用的协议来源：fc=原生 Function Calling；text=文本 Thought/Action 解析。"""

ToolCallStatus = Literal["ok", "denied", "error"]
"""执行结果状态：ok=已执行；denied=白名单/预算拒绝（未执行）；error=执行抛异常。"""

# 观测结果截断上限（回喂 LLM 的 Observation 最大字符数）
OBSERVATION_MAX_CHARS: int = 8000

# 工具返回 None 时的统一友好提示（与原 StandardizedToolInvokerProxy 文案一致）
_EMPTY_OBSERVATION_NOTICE: str = "【系统提示】工具执行完毕，未返回任何有效可视化数据。"


# ---------------------------------------------------------------------------
# 执行器协议：ToolRegistry 与 StandardizedToolInvokerProxy 均满足
# ---------------------------------------------------------------------------
@runtime_checkable
class ToolInvoker(Protocol):
    """工具执行器协议：输入工具名与参数 dict，返回任意观测值（通常为 str）。"""

    async def invoke(self, name: str, arguments: Dict[str, Any]) -> Any:
        """执行指定工具并返回观测结果。"""


# ---------------------------------------------------------------------------
# 统一数据结构
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    """归一后的单次工具调用（FC / 文本两路同构）。

    Attributes:
        tool_name: 工具注册名。
        arguments: 已解析的参数字典（FC 的 JSON arguments / 文本的 Action Input）。
        source: 协议来源（``fc`` 或 ``text``）。
        call_id: FC 协议的 tool_call_id（用于 role=tool 消息闭环）；文本路径自动补。
        thought: 文本协议下模型给出的 Thought（FC 路径为空串）。
        raw_arguments: FC 路径的原始 arguments JSON 字符串（供相关性抽查上下文）。
        raw: 原始载荷（FC 的 tool_call dict / 文本的 parsed dict），仅用于 trace。
        business_warn: 解析层 WARN（如空 Action Input 放行），执行前拼到 Observation 头部。
    """

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    source: ToolCallSource = "fc"
    call_id: str = ""
    thought: str = ""
    raw_arguments: str = ""
    raw: Any = None
    business_warn: str = ""

    @property
    def context_text(self) -> str:
        """供预算后处理（相关性抽查）使用的调用上下文文本。"""
        if self.raw_arguments:
            return self.raw_arguments
        if self.arguments:
            return json.dumps(self.arguments, ensure_ascii=False)
        return "{}"


@dataclass
class ToolResult:
    """统一执行流水线的产出。

    Attributes:
        call: 对应的 ToolCall。
        observation: 最终回喂给 LLM 的 Observation 文本（已标准化/后处理/截断）。
        status: ok / denied / error。
        invoked: 是否真正调用了 invoker（白名单/预算拒绝时为 False）。
        budget_denied: 是否被工具预算熔断拒绝。
    """

    call: ToolCall
    observation: str = ""
    status: ToolCallStatus = "ok"
    invoked: bool = False
    budget_denied: bool = False

    @property
    def denied(self) -> bool:
        """是否被系统侧拒绝执行（白名单或预算）。"""
        return self.status == "denied"


# ---------------------------------------------------------------------------
# 无效结果判定 + 预算后处理（原 react_agent.py 资产下沉，两路共用）
# ---------------------------------------------------------------------------
# 代码层即可判定的"无效返回"特征串：空结果 / 工具系统异常 / 官方兜底提示等。
# （内容"不匹配但非空"无法用代码判定，交给相关性抽查的 LLM 判定。）
_INVALID_OBSERVATION_MARKERS: tuple = (
    # ── 联网搜索（web_search / daubao 内部降级后仍然拿不到数据）─────────
    # 联网搜索现在是"单工具对外 + 代码层降级"：模型看到的失败文案只有这两种
    # （豆包与内部 Tavily 通道都没结果）。缺了它们时，搜索失败会被当成**有效结果**，
    # 模型会一路换 query 空转到配额上限——与下面 rag/图谱两条注释是同一类坑。
    "【系统提示】联网搜索工具发生故障",         # 两通道均故障（_SEARCH_TOOL_FAILURE_SUFFIX）
    "【系统提示】搜索接口调用成功，但未检索到匹配网页数据",  # 两通道均无命中（_SEARCH_EMPTY_RESULT）
    "执行期间发生异常",                        # registry 捕获的异常统一前缀
    "工具执行异常",
    "错误：试图调用未注册的工具",
    "参数 query 不能为空",
    "参数 不能为空",
    "【系统拒绝执行】",                        # 工具自身/其它熔断语义文案
    # ── 空召回（非异常但同样"没拿到数据"）─────────────────────────────
    # 缺了这一条时，检索工具的"未匹配到"文案不算无效 → 熔断器永不触发 →
    # 模型可以一路换同义 query 重试到单工具上限（实测一条用例白烧 3 轮 ≈5 万 token）。
    "未匹配到任何高相关性的文档片段",           # rag_knowledge_search 空召回文案
    "未匹配到任何高相关性",
    "未命中任何",
    # ── 图谱通道降级（同样是"没拿到数据"）─────────────────────────────
    # 缺了这两条时，knowledge_graph_search 失败会返回一段"让模型自己答"的文本：
    # 非空、也不含上面的错误前缀 → 被当成**有效结果** → 模型继续换 query 空转到配额上限
    # （实测原 T02 连调 10 次、86 秒；图谱语料补齐后这类失败才真正少见）。
    "在检索私有知识库时发生异常",               # knowledge_graph_search 异常兜底文案
    "本地私有知识库服务当前不可用",             # knowledge_graph_search 引擎未就绪文案
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


def observation_has_error_marker(obs: str) -> bool:
    """观测文本中是否含明确的错误/熔断标记（供 plan 路径决定是否 replan）。"""
    return tool_observation_looks_invalid(obs)


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
    purpose_hint: str = "react",
) -> str:
    """工具真实执行后的预算后处理：登记无效结果 / 触发第 N 次相关性抽查。

    Args:
        call_budget: ToolCallBudget（None 时原样返回，兼容无预算调用）
        tool_name: 工具名
        call_context: 调用意图/参数上下文
        observation: 标准化后的工具观测文本
        model_router: 相关性抽查用 LLM
        purpose_hint: LLM 路由场景提示

    Returns:
        应回喂给 LLM 的观察文本（若该工具刚被熔断，会在头部附 lock_notice）。
    """
    if call_budget is None:
        return observation or ""
    raw: str = observation or ""

    # ① 代码层无效（空 / 异常 / 系统错误提示 / 检索空召回）→ 登记无效计数
    if tool_observation_looks_invalid(raw):
        just_locked: bool = call_budget.record_invalid(tool_name)
        if just_locked:
            return call_budget.lock_notice(tool_name) + raw
        return raw

    # ①' 有效结果 → 连续无效计数清零（累计计数不冲销）
    if hasattr(call_budget, "record_effective"):
        call_budget.record_effective(tool_name)

    # ② 内容非空 → 若恰为该工具第 N 次调用（默认 5），做一次 LLM 相关性抽查
    if call_budget.reached_relevance_checkpoint(tool_name):
        relevant: bool = await judge_tool_result_relevance(
            model_router, tool_name, call_context, raw, purpose_hint=purpose_hint
        )
        if not relevant:
            call_budget.lock_tool(tool_name, "relevance")
            return call_budget.lock_notice(tool_name) + raw

    return raw


# ---------------------------------------------------------------------------
# 观测值标准化
# ---------------------------------------------------------------------------
def normalize_observation(raw_value: Any) -> str:
    """把工具执行的任意返回值标准化为字符串 Observation。

    - None → 统一"无有效数据"提示；
    - dict/list → 合法 JSON（缩进，非 ASCII 原样）；
    - str/其它 → str() 转换。
    """
    if raw_value is None:
        return _EMPTY_OBSERVATION_NOTICE
    if isinstance(raw_value, str):
        return raw_value
    if isinstance(raw_value, (dict, list)):
        try:
            return json.dumps(raw_value, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(raw_value)
    return str(raw_value)


# ---------------------------------------------------------------------------
# 两路协议 → ToolCall
# ---------------------------------------------------------------------------
def tool_calls_from_fc(
    tool_calls: List[Dict[str, Any]],
    step_idx: int,
) -> List[ToolCall]:
    """把 SDK 返回的 FC tool_calls 列表归一为 ToolCall 列表。

    Args:
        tool_calls: OpenAI 兼容协议的 tool_calls（dict 形态）。
        step_idx: 当前步数（用于补缺失的 tool_call_id）。

    Returns:
        与输入同序的 ToolCall 列表；非法/匿名条目跳过。
    """
    normalized: List[ToolCall] = []
    for call_index, tc in enumerate(tool_calls):
        func_payload: Dict[str, Any] = tc.get("function", {}) or {}
        tool_name: str = str(func_payload.get("name") or "")
        if not tool_name:
            continue
        raw_arguments: str = str(func_payload.get("arguments") or "")
        parsed_args: Dict[str, Any] = {}
        if raw_arguments:
            try:
                parsed_value: Any = json.loads(raw_arguments)
                parsed_args = parsed_value if isinstance(parsed_value, dict) else {}
            except json.JSONDecodeError:
                parsed_args = {}
        call_id: str = str(tc.get("id") or f"call_{step_idx}_{call_index}")
        normalized.append(
            ToolCall(
                tool_name=tool_name,
                arguments=parsed_args,
                source="fc",
                call_id=call_id,
                raw_arguments=raw_arguments,
                raw=tc,
            )
        )
    return normalized


def tool_call_from_text(parsed: Dict[str, Any]) -> Optional[ToolCall]:
    """把 ``_parse_react_step`` 的解析结果归一为 ToolCall。

    Args:
        parsed: 文本协议单步解析 dict（须含非空 action）。

    Returns:
        ToolCall；无 action（Final Answer / 解析失败）时返回 None。
    """
    tool_name: Any = parsed.get("action")
    if not tool_name:
        return None
    action_input: Any = parsed.get("action_input") or {}
    arguments: Dict[str, Any] = action_input if isinstance(action_input, dict) else {}
    business_warn: str = str(parsed.get("business_warn") or "")
    return ToolCall(
        tool_name=str(tool_name),
        arguments=arguments,
        source="text",
        call_id="",
        thought=str(parsed.get("thought") or ""),
        raw_arguments=json.dumps(arguments, ensure_ascii=False) if arguments else "",
        raw=parsed,
        business_warn=business_warn,
    )


# ---------------------------------------------------------------------------
# 统一执行流水线
# ---------------------------------------------------------------------------
def _whitelist_deny_text(tool_name: str, allowed_names: List[str]) -> str:
    """构造白名单拒绝 Observation（FC / 文本两路统一文案）。"""
    return (
        f"【系统校验 FAIL（工具白名单）】：工具 [{tool_name}] 不在当前允许调用的工具列表中。\n"
        f"允许的工具：{sorted(allowed_names)}\n"
        f"请从允许列表中选择工具，或若信息已足够请直接输出 Final Answer。"
    )


def _legacy_limit_deny_text(tool_name: str, max_tool_attempts: int) -> str:
    """无预算对象时的旧版每工具 3 次硬熔断文案（防御性兜底，正常接线不触发）。"""
    return (
        f"【系统拒绝执行】：你已经连续/累计调用工具 [{tool_name}] 达到 {max_tool_attempts} 次的最大上限。"
        f"该工具已被系统锁定。这通常意味着你使用的参数在当前生产环境中不可用。"
        f"请绝对不要再次尝试调用 [{tool_name}]！请利用现有信息直接回答，或改用其他工具。"
    )


async def execute_tool_call(
    call: ToolCall,
    invoker: ToolInvoker,
    *,
    call_budget: Optional[Any] = None,
    allowed_names: Optional[List[str]] = None,
    model_router: Any = None,
    purpose_hint: str = "react",
    legacy_counts: Optional[Dict[str, int]] = None,
    max_tool_attempts: int = 3,
) -> ToolResult:
    """统一工具执行流水线（FC / 文本两路共用）。

    顺序与旧双链路逐条对齐：
      1. 白名单校验（拒绝则不执行、不耗预算）；
      2. 预算：先 can_call 检查 → consume 计数 → invoke；无预算对象时走旧版每工具
         ``max_tool_attempts`` 计数兜底；
      3. invoker.invoke 执行，异常转文本（不抛出）；
      4. 观测标准化 + business_warn 拼接 + 预算后处理（无效累计 / 相关性抽查）。

    Args:
        call: 归一后的工具调用。
        invoker: 工具执行器（ToolRegistry 或标准化代理）。
        call_budget: 单次运行作用域的 ToolCallBudget。
        allowed_names: 允许调用的工具名白名单（None 表示不限制）。
        model_router: 相关性抽查用 LLM。
        purpose_hint: LLM 路由场景提示。
        legacy_counts: 无预算时跨步共享的调用计数 dict（防御性兜底）。
        max_tool_attempts: 无预算时每工具硬上限。

    Returns:
        ToolResult（含 observation / status / budget_denied）。
    """
    result: ToolResult = ToolResult(call=call)
    tool_name: str = call.tool_name
    allowed_set: Optional[set] = set(allowed_names) if allowed_names is not None else None

    # 1. 白名单
    if allowed_set is not None and tool_name not in allowed_set:
        result.status = "denied"
        result.observation = _whitelist_deny_text(tool_name, list(allowed_set))
        return result

    # 2. 预算（先检查后计数）
    if call_budget is not None:
        if not call_budget.can_call(tool_name):
            result.status = "denied"
            result.budget_denied = True
            result.observation = call_budget.deny_text(tool_name)
            return result
        call_budget.consume(tool_name)
    else:
        # 无预算对象的防御性兜底：每工具固定次数硬熔断（正常接线恒有预算）
        counters: Dict[str, int] = legacy_counts if legacy_counts is not None else {}
        counters[tool_name] = counters.get(tool_name, 0) + 1
        if counters[tool_name] > max_tool_attempts:
            logger.warning(
                "🚨 工具 [%s] 调用次数已达上限（%d 次），触发硬熔断！",
                tool_name, max_tool_attempts,
            )
            result.status = "denied"
            result.budget_denied = True
            result.observation = _legacy_limit_deny_text(tool_name, max_tool_attempts)
            return result

    # 3. 执行（异常不抛出，转 Observation 文本，保证 Agent 不崩溃）
    try:
        raw_value: Any = await invoker.invoke(tool_name, call.arguments)
        obs_text: str = normalize_observation(raw_value)
        result.invoked = True
    except Exception as tool_exc:  # noqa: BLE001 - 与旧链路一致：异常转 Observation
        result.status = "error"
        obs_text = f"工具执行异常: {tool_exc}"

    # 解析层 WARN 拼到 Observation 头部（文本协议旧行为）
    if call.business_warn:
        obs_text = f"{call.business_warn}\n\n{obs_text}"

    # 4. 预算后处理：无效结果累计 / 第 N 次相关性抽查
    obs_text = await postprocess_tool_result(
        call_budget,
        tool_name,
        call.context_text,
        obs_text,
        model_router,
        purpose_hint=purpose_hint,
    )

    result.observation = obs_text[:OBSERVATION_MAX_CHARS]
    return result
