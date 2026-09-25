# -*- coding: utf-8 -*-
"""plan 子任务执行前的【原生 Function Calling 强制取参】（从 planner.py 迁出）。

强制 tool_choice=子任务声明工具，让模型按工具 JSON Schema 生成 100% 合法参数，
根治 tool_args_hint 自由文本与 schema 错配导致的参数错误。
execute 节点（plan 形态）与过渡期 PlannerAgent.execute 共用同一实现。
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from app.core.tools.base import tool_to_function_call_definition

logger = logging.getLogger(__name__)


def _parse_hint_to_dict(hint: Any) -> Optional[Dict[str, Any]]:
    """把 ``tool_args_hint`` 解析成 dict，容忍 JSON 与 Python 字面量两种写法。

    ⚠️ qwen 系模型经常输出**单引号**的 Python 字面量
    （``{'query': '私有化部署 上线周期', 'top_k': 5}``），``json.loads`` 会直接失败。
    只认 JSON 的话，明明 planner 已经把参数给全了，却还要再花一次 LLM 取参。
    """
    if isinstance(hint, dict):
        return hint
    if not isinstance(hint, str) or not hint.strip():
        return None
    text: str = hint.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            value: Any = parser(text)
        except Exception:  # noqa: BLE001 - 逐个解析器试错，最后统一返回 None
            continue
        if isinstance(value, dict):
            return value
    return None


TASK_REF_PATTERN = re.compile(r"<[^<>]{0,60}?task[_\s-]?0*(\d+)[^<>]{0,60}?>", re.IGNORECASE)
"""前序任务产出占位符：``<从 task_2 获取的路径>`` 这类写法的识别式。

⚠️ 为什么必须程序化替换、不能指望模型自己填：
    planner 生成计划时**看不到前序任务的运行结果**，只能写这种占位符来表达依赖。
    实测故障链：hint 写成 ``{"file_path": "<从 task_2 获取的路径>", ...}`` →
    执行节点原样把占位符当 file_path 传给工具 →
    ``错误：找不到指定的 Excel 文件: <从 task_2 获取的路径>`` →
    模型理解为"路径解析不一致"，改用绝对路径重试，**但重试时丢掉了 sheet_name/filter 参数**
    → 又退回"读结构摘要" → 只看到前 5 行 → 误判"没有 8 月数据" → 任务彻底卡死。
    一处占位符没替换，后面全部环节都在为它空转。
"""


def substitute_task_refs(tool_args_hint: Any, artifacts_by_task: Mapping[str, Any]) -> Any:
    """把 hint 里的 ``<从 task_N 获取的…>`` 占位符替换成 task_N 的真实产出。

    Args:
        tool_args_hint: planner 给出的参数（JSON 文本 / dict / 其他形态）。
        artifacts_by_task: ``{"task_2": {"file_path": "raw_data/...xlsx", ...}}``——
            按子任务 ID 索引的"该步**实际使用**的入参"。
            取值优先级：``file_path`` → 唯一字符串值。

    Returns:
        替换后的 dict（hint 可解析为对象时）；不可解析则原样返回，
        由 :func:`resolve_tool_args_from_hint` 判定不可用并回退 FC 取参。

    注意：查不到对应产出时**保留原占位符**（而不是替换成空串）——
    空串会伪装成"参数合法"，而保留占位符能被 :func:`has_unresolved_task_ref`
    识别出来，从而安全地回退到 FC 取参。
    """

    def _lookup(task_id: str) -> Optional[str]:
        artifact: Any = artifacts_by_task.get(task_id)
        if isinstance(artifact, dict):
            value: Any = artifact.get("file_path")
            if isinstance(value, str) and value.strip():
                return value.strip()
            strings: List[str] = [
                v.strip() for v in artifact.values() if isinstance(v, str) and v.strip()
            ]
            if len(strings) == 1:
                return strings[0]
        elif isinstance(artifact, str) and artifact.strip():
            return artifact.strip()
        return None

    def _walk(node: Any) -> Any:
        if isinstance(node, str):
            def _sub(match: "re.Match[str]") -> str:
                target: Optional[str] = _lookup(f"task_{int(match.group(1))}")
                return target if target else match.group(0)
            return TASK_REF_PATTERN.sub(_sub, node)
        if isinstance(node, dict):
            return {key: _walk(value) for key, value in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    parsed: Optional[Dict[str, Any]] = _parse_hint_to_dict(tool_args_hint)
    if not parsed:
        return tool_args_hint
    return _walk(parsed)


def has_unresolved_task_ref(value: Any) -> bool:
    """参数里是否还残留未解析的 ``<…task_N…>`` 占位符（递归检查）。"""
    if isinstance(value, str):
        return bool(TASK_REF_PATTERN.search(value))
    if isinstance(value, dict):
        return any(has_unresolved_task_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(has_unresolved_task_ref(item) for item in value)
    return False


def resolve_tool_args_from_hint(
    tools: Any,
    tool_name: str,
    tool_args_hint: Any,
) -> Optional[Dict[str, Any]]:
    """直接复用 planner 给出的 ``tool_args_hint``，省掉一次「参数填充器」LLM 调用。

    实测：一次计划里 3 个 tool 子任务的参数填充合计约 3.8k tokens
    （769 / 1346 / 1756），而这些参数 planner 在 ``tool_args_hint`` 里已经给过了。

    ⚠️ 只在**能确定参数合法**时才返回，否则一律返回 None 让调用方回退到 FC 取参
    —— 不能为了省 token 把"参数 xxx 不能为空"的老问题放回来：
      1. hint 必须是可解析的 JSON / 字面量对象；
      2. 工具的全部**必填参数**都要有非空值；
      3. 顺带丢掉工具 schema 不认识的字段（例如历史遗留的 ``user_query``），
         避免 ``execute_tool_call`` 收到未知关键字。

    Returns:
        合法参数字典；hint 不可用 / 工具不可反射 / 必填缺失时返回 None。
    """
    parsed: Optional[Dict[str, Any]] = _parse_hint_to_dict(tool_args_hint)
    if not parsed:
        return None

    # ⚠️ 残留占位符必须在这里被拦下：占位符是**非空字符串**，能通过下面的必填校验，
    # 于是一路直达工具，变成"找不到文件: <从 task_2 获取的路径>"。
    # 宁可多花一次 FC 取参，也不能把占位符当参数传出去。
    if has_unresolved_task_ref(parsed):
        logger.warning(
            "工具 [%s] 的 tool_args_hint 含未解析的前序任务占位符，拒绝直用并回退 FC 取参",
            tool_name,
        )
        return None

    try:
        tool: Any = tools.get_tool(tool_name) if hasattr(tools, "get_tool") else None
    except Exception:  # noqa: BLE001 - 工具不存在时交给调用方走 FC / 兜底
        tool = None
    if tool is None:
        return None

    try:
        definition: Dict[str, Any] = tool_to_function_call_definition(tool)
        parameters: Dict[str, Any] = (
            definition.get("function", {}).get("parameters", {}) or {}
        )
    except Exception as def_error:  # noqa: BLE001
        logger.warning("工具 [%s] schema 反射失败，跳过 hint 直用：%s", tool_name, def_error)
        return None

    known: Dict[str, Any] = parameters.get("properties") or {}
    args: Dict[str, Any] = (
        {key: value for key, value in parsed.items() if key in known}
        if known
        else dict(parsed)
    )

    required: List[str] = list(parameters.get("required") or [])
    for name in required:
        value: Any = args.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            logger.info(
                "工具 [%s] 的 tool_args_hint 缺少必填参数 %r，回退 FC 强制取参", tool_name, name
            )
            return None

    if not args:
        return None
    logger.info("工具 [%s] 直接复用 planner 的 tool_args_hint，跳过参数填充 LLM", tool_name)
    return args


async def resolve_tool_args_via_function_call(
    model_router: Any,
    tools: Any,
    *,
    tool_name: str,
    title: str,
    description: str,
    tool_args_hint: Optional[str],
    query: str,
    prior_context_str: str,
    purpose_hint: str = "planner",
) -> Optional[Dict[str, Any]]:
    """为 tool 子任务强制取参。

    Returns:
        合法参数字典；无 model_router / 工具不可反射 / LLM 异常 / 未产出 tool_calls
        / 参数非 dict 时返回 None，调用方降级走 tool_args_hint 解析。
    """
    if model_router is None or tools is None:
        return None

    try:
        if hasattr(tools, "get_tool"):
            tool_instance: Any = tools.get_tool(tool_name)
        else:
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
        f"当前子任务：{title}\n"
        f"子任务详细要求：{description}\n"
        f"规划器备注（仅供参考，可能为空）：{tool_args_hint or '（无）'}\n"
        f"可参考的前序子任务上下文：\n{(prior_context_str or '')[:3000]}\n"
        "硬性要求：正文/内容类参数（如 content、text、body、summary 等承载业务"
        "数据的字段）必须基于上文中【前序子任务的真实结论/数据】组织成完整内容；"
        "前序上下文非空时，严禁只写标题或“××数据/××结果”之类的占位短语"
        "（规划器看不到执行结果，其备注里的这类文字不是数据本身）。"
    )
    messages: Sequence[Dict[str, str]] = [
        {"role": "system", "content": arg_fill_system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        resp: Any = await model_router.chat_with_tools(
            messages=list(messages),
            tools=[function_def],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            purpose_hint=purpose_hint,
            temperature=0.1,
            thinking=False,
        )
    except Exception as fc_exc:  # noqa: BLE001 - 通道/模型不支持 tools 时降级老逻辑
        logger.warning(
            "Planner FC 强制取参调用失败（tool=%s），降级走 tool_args_hint 解析: %s",
            tool_name, fc_exc,
        )
        return None

    if not getattr(resp, "tool_calls", None):
        logger.warning(
            "Planner FC 强制取参未返回 tool_calls（tool=%s），降级走 tool_args_hint 解析",
            tool_name,
        )
        return None

    raw_args: Any = resp.tool_calls[0].get("function", {}).get("arguments")
    if not isinstance(raw_args, str):
        return None
    try:
        parsed_args: Any = json.loads(raw_args)
    except json.JSONDecodeError as decode_error:
        logger.warning(
            "Planner FC 强制取参返回非法 JSON（tool=%s）：%s", tool_name, decode_error
        )
        return None
    if not isinstance(parsed_args, dict):
        return None
    return parsed_args
