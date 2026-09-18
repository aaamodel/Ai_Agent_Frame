# -*- coding: utf-8 -*-
"""节点共享小工具：提示词常量、trace/steps 埋点、记忆 chips 渲染、预算账本写回。

仅放无状态纯函数/常量；运行时依赖一律由调用方从 ``GraphDeps`` 传入。
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import re
from loguru import logger

from app.core.agent.graph.state import budget_to_ledger
from app.query_intent.intent_dto import normalize_agent_goal


# ─── SKILLS 渐进式披露系统级核心提示词（从 orchestrator 平移，orchestrator 侧 re-export）───
SKILLS_SYSTEM_PROMPT = """## Skills System

You have access to a specialized skills library to handle complex workflows and domain-specific tasks.

{skills_locations}{skills_load_warnings}
**Available Skills:**
{skills_list}

**How to Use Skills (Progressive Disclosure):**
To prevent context bloat, you only see the brief abstracts above. When a task matches a skill, you MUST fetch its full details before executing:

1. **Identify Relevance**: Check if the user's goal matches any skill description listed above.
2. **Read Full Instructions**: Use `file_read_tool` with the exact 'Source File' path. (The tool defaults to 2000 lines, which is enough to read the full file).
3. **Strictly Follow Workflows**: Follow the precise workflows, configurations, or script paths defined inside that file. Do not guess parameters.

Remember: Always read the corresponding skill file first if a relevant skill exists for the task!"""


def render_skills_prompt(skill_manager: Any) -> str:
    """扫描技能树状态，渲染渐进式披露提示词（失败返回空串，非致命）。"""
    try:
        skills_lines: List[str] = []
        for name, meta in skill_manager.state.available_skills.items():
            skills_lines.append(
                f"- **{name}**: {meta['description']} (Source File: `{meta['file_path']}`)"
            )
        if not skills_lines:
            return ""
        return SKILLS_SYSTEM_PROMPT.format(
            skills_locations="",
            skills_load_warnings="",
            skills_list="\n".join(skills_lines),
        )
    except Exception as skill_exc:  # noqa: BLE001 - 技能渲染失败不阻断主链路
        logger.warning("技能提示词渲染失败（按无技能运行）: {}", skill_exc)
        return ""


def render_skills_index(skill_manager: Any) -> str:
    """扫描技能树，渲染**规划侧**用的极简技能清单（失败返回空串，非致命）。

    与 :func:`render_skills_prompt` 的分工：

      - ``render_skills_prompt``：执行侧用的**完整规约**，含"如何读取技能文件、
        如何遵守其工作流"的操作指引；
      - ``render_skills_index``：规划侧只需要知道"有哪些技能、各自解决什么"。
        读取与遵守发生在执行期，不属于规划需要的信息。

    实测（2026-09）：完整规约约 1071 字符，占 planner 提示词的 36%；而里面
    "How to Use Skills"那一整段讲的是执行动作，规划器用不上——更矛盾的是，
    planner 读完这段后把"读 SKILL.md"排成了第一个子任务。
    """
    try:
        skills_lines: List[str] = []
        for name, meta in skill_manager.state.available_skills.items():
            description: str = str(meta.get("description") or "").strip().replace("\n", " ")
            # 只取第一句：规划需要的是"这个技能解决什么"，不是完整说明书
            brief: str = re.split(r"[。；;\n]", description)[0].strip()
            skills_lines.append(
                f"- {name}：{brief}（Source File: `{meta.get('file_path')}`）"
            )
        if not skills_lines:
            return ""
        return "\n".join(skills_lines)
    except Exception as skill_exc:  # noqa: BLE001 - 技能渲染失败不阻断主链路
        logger.warning("技能清单渲染失败（按无技能运行）: {}", skill_exc)
        return ""


def trace_event(tracer: Any, trace_id: str, event: str, payload: Dict[str, Any]) -> None:
    """安全写 tracer 事件（tracer 缺失/异常均不阻断节点）。"""
    if tracer is None:
        return
    try:
        tracer.log_event(trace_id, event, payload)
    except Exception as trace_exc:  # noqa: BLE001
        logger.debug("tracer 事件写入失败 {}: {}", event, trace_exc)


def emit_step(rec: Dict[str, Any]) -> Dict[str, Any]:
    """构造 steps reducer 的 partial dict（等价旧 callback 的 {"ts": ..., **rec} 形态）。"""
    return {"steps": [{"ts": time.time(), **rec}]}


def emit_plan_step(rec: Dict[str, Any]) -> Dict[str, Any]:
    """plan 路径旧 steps 形态：{"ts", "phase": "execute", "record": rec}。"""
    return {"steps": [{"ts": time.time(), "phase": "execute", "record": rec}]}


def write_back_budget(budget: Any) -> Dict[str, Any]:
    """节点激活结束把临时 ToolCallBudget 计数写回 state 账本。"""
    return {"budget": budget_to_ledger(budget)}


# ---------------------------------------------------------------------------
# plan 台账 + 游标推进（控制协议的公共真源）
# ---------------------------------------------------------------------------
# 为什么这两个函数必须放在同一处：**执行节点与路由都要用同一套"下一个待执行
# 下标"口径**。历史上工具白名单就因为"提示词 / schema / 执行闸门"三处各自实现
# 而漂移过一次，这里不能再复制第三份。
PLAN_STATUS_PENDING: str = "待执行"
PLAN_STATUS_RUNNING: str = "执行中"
PLAN_STATUS_DONE: str = "已完成"
PLAN_STATUS_SKIPPED: str = "已跳过"
PLAN_STATUS_FAILED: str = "失败"

# 与 execute_node.PLAN_BAD_STATUSES 同源（本地副本避免 nodes 包间循环导入）
_LEDGER_BAD_STATUSES = frozenset({"error", "empty_data", "budget_denied", "approval_denied"})

SOLVED_YES: str = "是"
SOLVED_PARTIAL: str = "部分"
SOLVED_NO: str = "否"
SOLVED_UNKNOWN: str = "-"
_SOLVED_ZH: Dict[str, str] = {"yes": SOLVED_YES, "partial": SOLVED_PARTIAL, "no": SOLVED_NO}
_LEDGER_GOAL_MAX: int = 24


def _display_width(text: str) -> int:
    """终端显示宽度：CJK 全角字符占 2 列，半角占 1 列。

    ⚠️ 台账是**给模型看的表格**，列必须对齐——直接用 ``f"{s:<28}"`` 会按字符数
    填充，而中文一个字符占两个显示列，结果就是 `rag_knowledge_search已完成`
    这种挤在一起的行，模型解析表格的准确率会下降。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度右填充空格。"""
    return text + " " * max(0, width - _display_width(text))


def next_pending_cursor(
    plan: Optional[Sequence[Any]],
    cursor: int,
    skipped_task_ids: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """从 ``cursor`` 起找第一个**未被跳过**的子任务下标；没有则返回 None。

    这是"跳过子任务"与"提前收尾"共用的唯一口径：
    finish 在实现上就是把剩余 id 全记入 skipped，于是本函数自然返回 None。
    """
    skipped: Set[str] = {str(item) for item in (skipped_task_ids or [])}
    index: int = int(cursor or 0)
    total: int = len(plan or [])
    while index < total:
        entry: Any = plan[index] or {}
        if isinstance(entry, Mapping) and str(entry.get("id") or "") not in skipped:
            return index
        index += 1
    return None


def render_plan_ledger(
    plan: Optional[Sequence[Any]],
    subtask_results: Optional[Sequence[Any]] = None,
    *,
    cursor: int = 0,
    skipped_task_ids: Optional[Sequence[str]] = None,
    agent_goal: str = "",
) -> str:
    """渲染计划台账（模型可见的计划全貌）。

    ⚠️ 状态**由 plan + subtask_results 推导**，不读取任何额外的状态副本——
    维护第二份台账必然与真实进度漂移，而漂移后的表现（模型按错误进度决策）
    比缺信息更危险。

    两列刻意分开：
      - 执行：客观的调用结果（程序判定，失败可确定）
      - 是否解决：语义判断（**只能由模型自评**，因为"工具成功但没答到点上"
        程序无从判断——这正是前一轮"看到前 5 行就判定没数据"事故的根因）
    执行失败的步骤强制"是否解决=否"：那些步骤没有走结论提炼，模型并未看到数据。

    ``agent_goal``（可选）渲染为**表格上方的独立首行**，而不是表格的一列：
    它是所有子任务共同的约束，放进表格会让 N 行重复同一段文字。
    """
    skipped: Set[str] = {str(item) for item in (skipped_task_ids or [])}
    results_by_id: Dict[str, Any] = {}
    for record in subtask_results or []:
        if not isinstance(record, Mapping):
            continue
        key: str = str(record.get("subtask_id") or "")
        if key:
            results_by_id[key] = record  # 后写覆盖：同一 id 重试以最后一次为准

    lines: List[str] = []
    # 目标：**紧邻表格但在表格之外**——它是全部子任务共同的约束，不是某个子任务的
    # 属性；作为表格的一列会让 N 行重复同一段文字（design.md D3）。放在块首是因为
    # "跳过子任务 / 提前收尾"的判断恰好发生在读取台账的这一刻。
    goal_line: str = str(agent_goal or "").strip().replace("\n", " ")
    if goal_line:
        lines.append(f"[本轮目标] {goal_line}")
    lines.extend(
        [
            "[计划台账] 执行=工具调用是否成功；是否解决=该子任务是否拿到它要的答案"
            "（两者不等价：工具成功也可能没答到点上）",
            _pad("#", 4) + _pad("id", 10) + _pad("解决的问题", 26)
            + _pad("工具", 22) + _pad("执行", 8) + "是否解决",
        ]
    )
    for index, entry in enumerate(plan or []):
        task: Any = entry if isinstance(entry, Mapping) else {}
        task_id: str = str(task.get("id") or f"task_{index + 1}")
        goal: str = str(task.get("description") or task.get("title") or "").replace("\n", " ")
        if len(goal) > _LEDGER_GOAL_MAX:
            goal = goal[:_LEDGER_GOAL_MAX] + "…"
        tool: str = str(task.get("tool_name") or "(无工具)")[:20]

        record: Any = results_by_id.get(task_id)
        if task_id in skipped:
            status: str = PLAN_STATUS_SKIPPED
        elif record is not None:
            status = (
                PLAN_STATUS_FAILED
                if str(record.get("status") or "") in _LEDGER_BAD_STATUSES
                else PLAN_STATUS_DONE
            )
        elif index == int(cursor or 0):
            status = PLAN_STATUS_RUNNING
        else:
            status = PLAN_STATUS_PENDING

        if status in (PLAN_STATUS_PENDING, PLAN_STATUS_RUNNING, PLAN_STATUS_SKIPPED):
            solved: str = SOLVED_UNKNOWN
        elif status == PLAN_STATUS_FAILED:
            solved = SOLVED_NO
        else:
            solved = _SOLVED_ZH.get(str(record.get("solved") or ""), SOLVED_UNKNOWN)

        lines.append(
            _pad(str(index + 1), 4) + _pad(task_id, 10) + _pad(goal, 26)
            + _pad(tool, 22) + _pad(status, 8) + solved
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 记忆上下文
# ---------------------------------------------------------------------------
def normalize_memory_context(raw: Any) -> Dict[str, Any]:
    """把 MemoryContext(Pydantic) / dict 归一为四键 dict（兼容两种读取方式）。

    旧 orchestrator 同时写 "short_term"/"long_term"（planner/react 读取）
    与 "short term memory"/"long term memory"（历史兼容），此处保持一致。
    """
    if isinstance(raw, dict):
        short_term: List[Any] = (
            raw.get("short_term")
            or raw.get("short_term_messages")
            or raw.get("short term memory")
            or []
        )
        long_term: List[Any] = (
            raw.get("long_term")
            or raw.get("long_term_items")
            or raw.get("long term memory")
            or []
        )
    else:
        short_term = (
            getattr(raw, "short_term_messages", None)
            or getattr(raw, "short_term", None)
            or []
        )
        long_term = (
            getattr(raw, "long_term_items", None)
            or getattr(raw, "long_term", None)
            or []
        )
    return {
        "short_term": list(short_term or []),
        "long_term": list(long_term or []),
        "short term memory": list(short_term or []),
        "long term memory": list(long_term or []),
    }


# 长期记忆注入预算：历史落库记录会被拼进**每一轮** system 提示词，
# 不设上限时单轮 token 随会话数线性膨胀（评测里 4 万 token/轮的主因之一）。
LONG_TERM_MAX_ITEMS: int = 6
"""注入 system 的长期记忆条数上限。"""

LONG_TERM_ITEM_MAX_CHARS: int = 400
"""单条长期记忆注入上限（超出截断并加省略号）。"""

LONG_TERM_MAX_CHARS: int = 2000
"""长期记忆注入总字符上限（累计触顶后停止追加）。"""


def _memory_item_text(item: Any) -> str:
    """长期记忆条目 → 纯文本（只取 content/text，丢弃 id/score/metadata 等 repr 噪音）。"""
    if item is None:
        return ""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("content") or item.get("text") or "").strip()
    text = getattr(item, "content", None)
    if text is None:
        text = getattr(item, "text", None)
    return str(text).strip() if text is not None else ""


_QUESTION_PREFIX = re.compile(r"^(?:用户问题|用户提问|问)\s*[:：]\s*")
# 去重指纹要能容忍"同一句话的标点差异"（"...里？" vs "...里"），否则同一批记忆
# 会因一个问号而被判为"新内容"重新注入。空白与标点在比对时一律视为噪声。
_DEDUP_NOISE = re.compile(r"[\s。！？!?；;，,：:、~～…\.\-—=*#`\"'“”‘’（）()\[\]【】<>《》/\\|]+")


def _normalize_dedup_text(text: str) -> str:
    """归一化文本用于重复判定（去空白与标点，大小写不敏感）。"""
    return _DEDUP_NOISE.sub("", text or "").lower()


def _ltm_question_of(text: str) -> str:
    """取出长期记忆条目的"问题"部分（写入侧格式为 ``用户问题: …\\n结论: …``）。"""
    first_line: str = (text or "").split("\n", 1)[0]
    return _QUESTION_PREFIX.sub("", first_line).strip()


def _short_term_dedup_keys(memory_context: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """收集短期历史已出现过的内容指纹：``(归一化文本, trace_id)``。

    用途是把**已经以真实 messages 形式进入请求**的内容，从长期记忆注入里剔除。
    当前 ``recall`` 按 ``session_id`` 过滤，召回的就是本会话历史，与短期记忆高度重叠；
    不去重的话同一批事实会在一轮请求里出现两次。
    """
    raw_items: List[Any] = (
        (memory_context or {}).get("short_term")
        or (memory_context or {}).get("short_term_messages")
        or []
    )
    texts: Set[str] = set()
    trace_ids: Set[str] = set()
    for item in raw_items:
        if isinstance(item, Mapping):
            content = item.get("content")
            metadata = item.get("metadata")
        else:
            content = getattr(item, "content", None)
            metadata = getattr(item, "metadata", None)
        if content:
            texts.add(_normalize_dedup_text(str(content)))
        if isinstance(metadata, Mapping):
            trace_id: Any = metadata.get("trace_id")
            if trace_id:
                trace_ids.add(str(trace_id))
    return texts, trace_ids


def long_term_mem_block(memory_context: Dict[str, Any]) -> str:
    """FC/文本 system 提示词中使用的长期事实纯文本块（空时返回"（无）"）。

    这里是**长期记忆唯一的注入点**（旧实现还会经 ``render_memory_chips`` 再注入一份，
    导致同一批事实在一轮请求里出现两次，属纯浪费，已删除）。

    四重收口，防止"历史问答记录"撑爆每轮 prompt：
      1. **与短期历史去重**：召回条目若与本会话已有的 messages 同源（trace_id 相同）
         或同问题（归一化文本相同），直接丢弃——这是最大的浪费点，因为按当前的
         ``session_id`` 召回策略，长期召回结果本来就大概率是本会话的旧轮次；
      2. 条数上限 ``LONG_TERM_MAX_ITEMS``；
      3. 单条上限 ``LONG_TERM_ITEM_MAX_CHARS``；
      4. 总字符上限 ``LONG_TERM_MAX_CHARS``。

    渲染只取 ``content`` 纯文本：旧实现 ``f"- {s}"`` 会把 ``MemoryItem`` 的 pydantic
    repr（``content='...' score=0.41 id='...' metadata={...}``）原样打进提示词，
    体积翻倍且全是噪音。
    """
    snippets: List[Any] = (
        (memory_context or {}).get("long_term")
        or (memory_context or {}).get("long_term_items")
        or []
    )
    short_texts, short_trace_ids = _short_term_dedup_keys(memory_context)

    lines: List[str] = []
    used_chars: int = 0
    seen_questions: Set[str] = set()
    for raw_item in snippets:
        text: str = _memory_item_text(raw_item)
        if not text:
            continue

        # ① 跨系统去重：trace_id 完全相同 = 同一次问答，短期记忆里已有原文
        metadata: Any = getattr(raw_item, "metadata", None)
        if not isinstance(metadata, Mapping) and isinstance(raw_item, Mapping):
            metadata = raw_item.get("metadata")
        if isinstance(metadata, Mapping):
            item_trace: Any = metadata.get("trace_id")
            if item_trace and str(item_trace) in short_trace_ids:
                continue

        # ① 问题级去重：同一个问题已经以 user 消息进入上下文，结论也无新增信息
        question: str = _ltm_question_of(text)
        if question and _normalize_dedup_text(question) in short_texts:
            continue

        # 长期记忆内部自我去重（同一问题被多次沉淀时只保留最先出现的一条）
        dedup_key: str = _normalize_dedup_text(question or text)
        if dedup_key in seen_questions:
            continue
        seen_questions.add(dedup_key)

        if len(text) > LONG_TERM_ITEM_MAX_CHARS:
            text = text[:LONG_TERM_ITEM_MAX_CHARS] + "…"
        if used_chars + len(text) > LONG_TERM_MAX_CHARS:
            break
        used_chars += len(text)
        lines.append(f"- {text}")

    # 条数上限作用于**去重之后**的有效条目，避免重复项白占名额
    if len(lines) > LONG_TERM_MAX_ITEMS:
        lines = lines[:LONG_TERM_MAX_ITEMS]
    return "\n".join(lines) if lines else "（无）"


def short_term_messages(memory_context: Dict[str, Any]) -> List[Dict[str, str]]:
    """短期历史 → OpenAI 消息 dict 列表（role/content）。"""
    raw: List[Any] = (
        (memory_context or {}).get("short_term")
        or (memory_context or {}).get("short_term_messages")
        or []
    )
    messages: List[Dict[str, str]] = []
    for old_msg in raw:
        role = getattr(old_msg, "role", None) or (
            old_msg.get("role", "user") if isinstance(old_msg, dict) else "user"
        )
        if hasattr(role, "value"):
            role = role.value
        content = getattr(old_msg, "content", None) or (
            old_msg.get("content", "") if isinstance(old_msg, dict) else ""
        )
        messages.append({"role": str(role), "content": str(content)})
    return messages


_SLOT_NODE_KEYS: Tuple[str, ...] = ("top_kb_node", "top_mcp_node", "top_system_node")


def compact_intent_slots(slots: Mapping[str, Any]) -> str:
    """把意图槽位压成"模型真的会用"的两类信息：工具调用提示 + 定向集合。

    原实现把整个 ``slots`` 用 ``repr`` 打进 system 提示词，而 slots 里绝大部分是噪音：
      - 一堆恒为 ``null`` 的路由字段（kb_id / mcp_tool_id / param_prompt_template / top_k…）；
      - **重复两份**的工具名列表（``intent_hit_tool_names`` 与
        ``pipeline_intent_hit_tool_names`` 内容相同）；
      - **重复两份**的模式决策（``pipeline_mode_meta`` 与
        ``pipeline_original_mode_decision`` 字段完全一致）；
      - LLM 自述的 ``pipeline_complexity_meta.reasoning_notes``（人看有用，模型不需要）；
      - 拆分场景下的 ``per_sub_questions`` / ``sub_intent_scores``（每个子问题 × 全部候选节点
        的打分明细，可达数 KB）。

    这些内容会**每一轮 ReAct 重发一遍**，实测基线里占单轮输入的可观比例。
    真正有信息量的是节点上的 ``tool_usage_hint``（传参要点/时机，**没有别的注入点**，
    删了会掉能力）与定向集合名。
    """
    lines: List[str] = []
    seen_hints: Set[str] = set()
    for key in _SLOT_NODE_KEYS:
        node = slots.get(key) or {}
        if not isinstance(node, Mapping):
            continue
        hint: str = str(node.get("tool_usage_hint") or "").strip()
        if hint and hint not in seen_hints:
            seen_hints.add(hint)
            lines.append(f"- {hint}")
        names: List[str] = [
            str(name).strip()
            for name in (node.get("collection_names") or [])
            if str(name).strip()
        ]
        if names:
            lines.append(f"- 定向集合：{'、'.join(names)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 本轮目标（agent_goal）读取 —— 所有注入点共用的唯一入口
# ---------------------------------------------------------------------------
AGENT_GOAL_SLOT_KEY: str = "agent_goal"
"""目标在 ``intent.slots`` 中的键名（由 Pipeline ``_merge_slots`` 写入）。"""


def agent_goal_from_state(state: Mapping[str, Any]) -> str:
    """从图状态读取本轮目标；缺失或为空时回退为**当前问题**（``user_input``）。

    ⚠️ 为什么读取必须收敛成一个函数，而不是三个注入点各自取、各自兜底：
        那样极易演变成"这里回退到 user_input、那里注入空串"，于是同一轮里
        模型在不同决策点看到的目标并不一致（design.md D4/D5 明确禁止这种漂移）。

    ⚠️ 兜底源必须是 ``user_input``（= **改写后的问题**）：上游
        ``effective_user_input = final_user_input or original_user_question``，
        而 ``final_user_input = rewritten_question or original``——与解析层
        :func:`normalize_agent_goal` 的回退值同源，从而保证"槽位缺失"与
        "模型输出空值"两条路径得到**完全相同**的结果。

    Args:
        state: 图状态。目标按 ``intent.slots[AGENT_GOAL_SLOT_KEY]`` 读取。

    Returns:
        目标文本；仅当槽位与 ``user_input`` 都为空时返回空串，
        此时按"无目标"处理，MUST NOT 中断链路。
    """
    intent: Any = state.get("intent") or {}
    slots: Any = intent.get("slots") if isinstance(intent, Mapping) else {}
    if not isinstance(slots, Mapping):
        slots = {}
    return normalize_agent_goal(
        slots.get(AGENT_GOAL_SLOT_KEY), str(state.get("user_input") or "")
    )


def agent_goal_trace_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    """``agent.goal`` 留痕事件的载荷（一轮只记一次，由入口节点 prepare 发出）。

    ``source`` 是排查注入不一致的关键字段：
      - ``slots``：目标由改写阶段产出并经 Pipeline 写入槽位（正常路径）；
      - ``fallback``：槽位没供上（未产出 / 键缺失 / 值为空），走的是回退值。
    若 source 长期为 fallback，说明改写阶段根本没产出可用目标——这正是要抽查的。
    """
    intent: Any = state.get("intent") or {}
    slots: Any = intent.get("slots") if isinstance(intent, Mapping) else {}
    raw_goal: Any = slots.get(AGENT_GOAL_SLOT_KEY) if isinstance(slots, Mapping) else None
    goal_value: str = agent_goal_from_state(state)
    return {
        "agent_goal": goal_value,
        "source": "slots" if str(raw_goal or "").strip() else "fallback",
        "chars": len(goal_value),
    }


def build_extra_system(state: Dict[str, Any]) -> str:
    """execute 节点统一系统附加段：意图锚点 + prepare 写入的 hints + 槽位要点。

    ⚠️ 这里**不再**追加记忆 chips（原 ``render_memory_chips``）。原因：
      - 短期历史已经由 ``short_term_messages()`` 以真实 messages 形式进入请求，
        chips 是同一批内容的第二份粘贴（纯重复）；
      - 长期事实唯一注入点是 ``long_term_mem_block()``，chips 是第二份。
    去掉后 system 体积显著下降且**零能力损失**（信息一份不少，只是不重复）。

    ⚠️ 同样地，这里**不再**把整个 ``slots`` repr 进去，改用
    :func:`compact_intent_slots` 的投影（同样零能力损失：丢掉的全是重复项与 null 字段）。
    """
    intent: Dict[str, Any] = state.get("intent") or {}
    parts: List[str] = []
    # 目标放在**最前**：它是本轮唯一的锚点，ReAct 每一轮都要重读一次来决定
    # "下一步该调什么 / 是否已经够了"。缺失（两侧都空）时不注入占位噪音。
    goal: str = agent_goal_from_state(state)
    if goal:
        parts.append(f"本轮目标：{goal}")
    parts.extend(state.get("extra_system_hints") or [])
    parts.append(f"当前识别意图：{intent.get('intent', 'general')}")
    slot_digest: str = compact_intent_slots(intent.get("slots") or {})
    if slot_digest:
        parts.append("## 工具调用要点（来自意图路由）\n" + slot_digest)
    return "\n".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# 工具观察压缩（ReAct 历史随步数线性膨胀的收口）
# ---------------------------------------------------------------------------
# 设计口径（与主流 Agent 框架一致，**不额外调用大模型**）：
#   1. 最近 ``TOOL_OBS_KEEP_RECENT`` 条工具观察保留限长原文——agent 的下一步
#      推理通常只依赖最近几步的证据，动它就伤正确率；
#   2. 更早的观察降级为"短桩"：工具名 + 返回体量 + 首段要点 + 重取提示。
#      丢掉但留指针：真需要原文时模型可以重新调用该工具，而不是被一段
#      LLM 改写过的摘要误导（摘要既花钱又可能丢掉关键证据）。
# 为什么不用 LLM 摘要旧观察：要多付一次完整 prompt 的钱，换来的是有损信息，
# 且摘要本身可能抹掉下一步恰恰需要的细节——负收益。
TOOL_OBS_KEEP_RECENT: int = 3
"""保留限长原文的最近工具观察条数。"""

TOOL_OBS_STUB_HEAD_CHARS: int = 200
"""降级短桩保留的原文首段字符数。"""

TOOL_OBS_STUB_TEMPLATE: str = (
    "【历史观察已压缩】工具 `{tool}` 曾返回 {size} 字符，此处仅保留首段要点；"
    "如需完整原文请重新调用该工具。\n{head}"
)


def compact_tool_observations(
    messages: Sequence[Dict[str, Any]],
    *,
    keep_recent: int = TOOL_OBS_KEEP_RECENT,
) -> List[Dict[str, Any]]:
    """生成"发送给 LLM 的消息视图"：压缩较早轮次的 role=tool 观察。

    只构造**新 dict**，不改动入参（state 里仍保留完整原文供 trace / 账本使用）。

    Args:
        messages: 完整消息列表（含 assistant 的 tool_calls 与 role=tool 回填）。
        keep_recent: 保留原文的最近工具观察条数。

    Returns:
        新的消息列表；非 tool 消息原样透传，被压缩的 tool 消息换成长短桩。
    """
    if keep_recent < 0:
        keep_recent = 0

    # call_id → 工具名（从 assistant.tool_calls 反查，桩里要写明是哪个工具）
    call_names: Dict[str, str] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            call_id: Any = call.get("id")
            function_payload: Any = call.get("function") or {}
            if call_id and isinstance(function_payload, Mapping):
                call_names[str(call_id)] = str(function_payload.get("name") or "unknown")

    tool_indexes: List[int] = [
        index for index, message in enumerate(messages) if message.get("role") == "tool"
    ]
    stub_indexes: set = set(tool_indexes[:-keep_recent] if keep_recent else tool_indexes)
    if not stub_indexes:
        return list(messages)

    compacted: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        if index not in stub_indexes:
            compacted.append(message)
            continue
        content: str = str(message.get("content") or "")
        tool_name: str = call_names.get(str(message.get("tool_call_id") or ""), "unknown")
        head: str = content[:TOOL_OBS_STUB_HEAD_CHARS]
        if len(content) > TOOL_OBS_STUB_HEAD_CHARS:
            head += "…"
        compacted.append(
            {
                **message,
                "content": TOOL_OBS_STUB_TEMPLATE.format(
                    tool=tool_name, size=len(content), head=head
                ),
            }
        )
    return compacted


_HISTORY_OBSERVATION_PATTERN = re.compile(r"Observation:.*?(?=\nStep |\Z)", re.DOTALL)


def compact_history_lines(
    history_lines: Sequence[str],
    *,
    keep_recent: int = TOOL_OBS_KEEP_RECENT,
) -> List[str]:
    """文本 ReAct 协议（``react_history_lines``）的历史步压缩。

    与 :func:`compact_tool_observations` 同口径：最近 ``keep_recent`` 步保留原文，
    更早的步骤把 ``Observation:`` 段落换成短桩，避免每轮用 ``history_lines``
    重建 user prompt 时把全部历史观察再背一遍。
    """
    if keep_recent < 0:
        keep_recent = 0
    if len(history_lines) <= keep_recent:
        return list(history_lines)

    head_count: int = len(history_lines) - keep_recent
    compacted: List[str] = []
    for index, line in enumerate(history_lines):
        if index >= head_count:
            compacted.append(line)
            continue
        compacted.append(
            _HISTORY_OBSERVATION_PATTERN.sub(
                lambda match: (
                    f"Observation: 【历史观察已压缩】原文 {len(match.group(0))} 字符，"
                    "如需完整原文请重新调用该工具。"
                ),
                line,
                count=1,
            )
        )
    return compacted
