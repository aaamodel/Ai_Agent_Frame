# -*- coding: utf-8 -*-
"""执行期就地纠偏：把"发明方向"降级为"在候选里选"。

为什么需要这一层（2026-09 实测教训）：

    一条 trace 里两次重规划共占 20,041 输入字符（53.5%），但它们要解决的其实是
    **步级**问题——t1 读了技能文档、摘要丢掉了里面的路径，重规划只能盲猜目录，
    连续两次扫 `/data` 均为空。根因不是模型不聪明，而是：

      1. 重规划的输入里**没有"可用的替代方向"**，模型只能发明方向。它发明了
         `/data`——一个已被同一份输入记录证否的路径。模型面对的是"我该往哪纠偏"
         这个开放式问题，而它缺的正是回答这个问题所需的信息基础。
      2. 执行期**完全封闭**：控制指令只能收缩（跳过 / 提前收尾），单个子任务取不到
         数据时只记账、不处理，必须等整轮跑完才由重规划从头修订。

    本模块把分工掰正：
      · 系统侧 —— 用**代码**算出确定性的候选方向集（已提取事实减去已试路径、
        工具白名单减去已用工具），并把已证否的方向**结构性剔除**；
      · 模型侧 —— 只做一件事：从这个封闭列表里选一项，或者选"不选"。
        它不需要构造工具名、参数或路径。

设计约束（均为硬性）：
    - 纯字符串 / 集合 / 本地分词运算，**不产生任何模型调用**；
    - 候选必须与当前工具白名单求交——白名单里没有能消费某资产的工具时，该资产
      不能进候选，否则等于让模型选中一个它执行不了的方向；
    - 任何异常都不得中断主链路（返回空候选即可）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from loguru import logger

from app.core.agent.planner import result_is_ineffective

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

#: 候选列表的渲染长度上限（含标题）。只注入"索引"级信息，不注入参数 schema。
CANDIDATE_MAX_CHARS: int = 300

#: 证据缺口判据：问题侧核心词在本步返回内容中的字面出现率低于该值即置位。
#: 初始取 0.3（宁可多触发以收集数据）；稳定后按 trace 统计上调。
EVIDENCE_COVERAGE_THRESHOLD: float = 0.3

#: 纠偏配额 = 剩余调用预算 × 该比例（向下取整）。
QUOTA_BUDGET_RATIO: float = 0.5

#: 未设总预算上限（`budget_remaining` 返回 None）时的保守配额。
QUOTA_WHEN_UNLIMITED: int = 2

_CANDIDATE_KIND_ASSET: str = "asset"
_CANDIDATE_KIND_TOOL: str = "tool"

#: 候选列表的提示词标题（渲染与测试共用，避免两处写死）
CANDIDATE_BLOCK_HEADER: str = "## 可选的替代方向（如需换源/补步，请从中选择）"

#: 高频、无区分度的词。留在核心词集合里只会稀释覆盖率——例如"我们/哪些/是否"
#: 在几乎任何返回文本里都出现，会让"跑题"被算成"覆盖正常"。
_STOPWORDS: frozenset = frozenset({
    "我们", "你们", "他们", "它们", "咱们", "自己", "这个", "那个", "这些", "那些",
    "哪些", "哪个", "什么", "怎么", "怎样", "如何", "为何", "为什么", "是否", "能否",
    "可以", "可以吗", "需要", "应该", "可能", "以及", "并且", "但是", "然后", "如果",
    "因为", "所以", "由于", "为了", "关于", "对于", "根据", "通过", "还是", "或者",
    "一个", "一些", "一下", "一次", "一直", "已经", "正在", "可以", "没有", "不是",
    "就是", "还有", "另外", "同时", "目前", "现在", "之前", "之后", "以上", "以下",
    "进行", "给出", "查看", "看看", "帮忙", "帮我", "麻烦", "请", "你", "我", "他",
    "她", "它", "的", "了", "和", "与", "或", "在", "是", "有", "无", "不", "也",
    "都", "就", "而", "及", "等", "把", "被", "让", "给", "对", "从", "到", "向",
    "个", "些", "这", "那", "上", "下", "中", "里", "外", "前", "后",
})

#: 纯数字 / 纯标点 / 纯 ASCII 单字母之类的无信息 token
_MEANINGFUL_RE: re.Pattern = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9_]+$")

#: 归一化路径用：把反斜杠与连续分隔符统一，去掉多余空白
_SEP_RE: re.Pattern = re.compile(r"[\\/\s]+")

#: 渲染候选列表时，从形如 "asset:客户线索台账.xlsx" 的 id 里取前缀
_ID_SPLIT_RE: re.Pattern = re.compile(r"^([a-zA-Z_]+)\s*[:：]\s*(.+)$")


# ─────────────────────────────────────────────────────────────────────────────
# §1.2 分词与核心词
# ─────────────────────────────────────────────────────────────────────────────

_jieba_ready: bool = False


def _jieba() -> Any:
    """惰性导入并静音 jieba（首次调用会构建前缀词典，约 1 秒，一次性）。"""
    global _jieba_ready
    import jieba  # noqa: PLC0415 - 惰性导入：避免进程启动就付词典构建成本

    if not _jieba_ready:
        # jieba 默认把 "Building prefix dict..." 打到 stderr，属于噪音
        jieba.setLogLevel(logging.WARNING)
        _jieba_ready = True
    return jieba


def _is_meaningful(token: str) -> bool:
    """token 是否值得计入核心词：长度 >= 2、非停用词、非纯符号。"""
    text = str(token or "").strip()
    if len(text) < 2:
        return False
    if text in _STOPWORDS:
        return False
    return bool(_MEANINGFUL_RE.match(text))


def extract_keywords(text: Any) -> Set[str]:
    """把一段文本切成核心词集合（本地分词 + 停用词过滤，零外部调用）。"""
    raw = str(text or "").strip()
    if not raw:
        return set()
    try:
        tokens = _jieba().cut(raw)
    except Exception as exc:  # noqa: BLE001 - 分词失败不得中断主链路
        logger.warning("分词失败（按无核心词处理）: {}", exc)
        return set()
    return {token.strip() for token in tokens if _is_meaningful(token)}


def build_question_keywords(state: Dict[str, Any]) -> Set[str]:
    """按优先级汇总"问题侧核心词"（数据集 A）。

    来源优先级：本轮目标 → 子问题 → 意图树命中路径 → 已提取的结构化事实；
    均为空时回退到改写后的问题文本。全部缺失时返回空集合（调用方据此不置位）。
    """
    intent: Dict[str, Any] = state.get("intent") or {}
    slots: Dict[str, Any] = intent.get("slots") or {}

    sources: List[str] = []

    goal: str = str(slots.get("agent_goal") or "").strip()
    if goal:
        sources.append(goal)

    for sub_question in slots.get("per_sub_questions") or []:
        text = str(sub_question or "").strip()
        if text:
            sources.append(text)

    for node_key in ("top_kb_node", "top_mcp_node", "top_system_node"):
        node = slots.get(node_key)
        if isinstance(node, dict):
            full_path = str(node.get("full_path") or "").strip()
            if full_path:
                sources.append(full_path)

    for fact in state.get("extracted_facts") or []:
        if isinstance(fact, dict):
            name = str(fact.get("name") or "").strip()
            if name:
                sources.append(name)

    if not sources:
        fallback = str(state.get("user_input") or "").strip()
        if fallback:
            sources.append(fallback)

    keywords: Set[str] = set()
    for source in sources:
        keywords |= extract_keywords(source)
    return keywords


def detect_evidence_gap(
    question_keywords: Iterable[str],
    observation: Any,
    threshold: float = EVIDENCE_COVERAGE_THRESHOLD,
) -> Tuple[bool, float]:
    """确定性的证据缺口检测：比对数据集 A 与数据集 B。

    Args:
        question_keywords: 数据集 A —— 问题侧核心词。
        observation: 数据集 B —— **本步**工具返回的原始文本。
            调用方 MUST 只传本步的内容：掺入历史结论会把覆盖率抬高，
            导致"本步返回跑题"被漏检。
        threshold: 覆盖率阈值。

    Returns:
        ``(是否置位, 实际覆盖率)``。

    两条不置位条件（宁可漏触发，不得误伤正常步骤）：
      · 数据集 A 为空 —— 没有可比对的基准；
      · 数据集 B 为空 —— 纯推理步骤无返回内容，覆盖率无意义。
    """
    keys: Set[str] = {k for k in (question_keywords or []) if k}
    text: str = str(observation or "")
    if not keys or not text.strip():
        return False, 0.0

    hits: int = sum(1 for key in keys if key in text)
    coverage: float = hits / len(keys)
    return coverage < threshold, coverage


# ─────────────────────────────────────────────────────────────────────────────
# §2 候选方向列表（全部由代码算）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Candidate:
    """一条候选方向：短标识 + 一句话描述。"""

    id: str
    description: str


@dataclass
class CandidateSet:
    """组装结果：候选本身 + 被排除的资产（供留痕）。"""

    candidates: List[Candidate]
    excluded: List[Dict[str, str]]

    def __bool__(self) -> bool:
        return bool(self.candidates)


def _normalize_path(text: Any) -> str:
    """路径归一化：反斜杠与连续分隔符统一、小写、去首尾分隔符。"""
    normalized = _SEP_RE.sub("/", str(text or "")).strip().lower()
    return normalized.strip("/")


def _attempted_blobs(results: Any) -> List[str]:
    """把每条执行记录的 `action_input` 拼成文本，供"反查"使用。

    反查而不是"抽取路径"：各工具的路径参数名互不相同（`file_path` / `path` /
    `dir` …），猜参数名既脆弱又要维护清单。这里只问一个问题——"这个资产的位置或
    文件名，是否出现在本次调用的真实入参里"。
    """
    blobs: List[str] = []
    for record in results or []:
        if not isinstance(record, dict):
            continue
        args = record.get("action_input")
        if isinstance(args, dict):
            blobs.append(" ".join(str(value) for value in args.values()))
        elif isinstance(args, str):
            blobs.append(args)
    return [_normalize_path(blob) for blob in blobs]


def attempted_asset_locations(facts: Any, results: Any) -> Set[str]:
    """本轮已尝试过的资产位置集合。

    完整位置与**文件名**同时匹配——后者能覆盖"模型自己拼了目录但文件名写对"的
    情形（本 trace 里两次扫 /data 都属于这种情况）。
    """
    blobs = _attempted_blobs(results)
    if not blobs:
        return set()

    hit: Set[str] = set()
    for fact in facts or []:
        if not isinstance(fact, dict):
            continue
        location = str(fact.get("location") or "").strip()
        if not location:
            continue
        targets: Set[str] = {_normalize_path(location)}
        basename = _normalize_path(location).rstrip("/").split("/")[-1]
        if basename:
            targets.add(basename)
        targets.discard("")
        if any(
            target and any(target in blob for blob in blobs)
            for target in targets
        ):
            hit.add(location)
    return hit


def build_candidates(state: Dict[str, Any]) -> CandidateSet:
    """确定性组装候选方向列表。

    两个差集：
      ① 工具侧 ＝ ``state["active_tool_names"]`` − 本轮已调用工具；
      ② 资产侧 ＝ 已提取的结构化事实 − 本轮已尝试过的资产。

    资产侧还会与白名单求交：需要白名单外工具才能消费的资产不进候选，并记入
    ``excluded``（这同时是"白名单过窄"的观测点——本 trace 的白名单就不含任何
    Excel 工具，而技能文档声明的资产全是 Excel）。
    """
    allowed: Set[str] = {
        str(name).strip()
        for name in (state.get("active_tool_names") or [])
        if str(name).strip()
    }
    results = state.get("subtask_results") or []
    facts = [
        fact for fact in (state.get("extracted_facts") or []) if isinstance(fact, dict)
    ]

    candidates: List[Candidate] = []
    excluded: List[Dict[str, str]] = []

    # ── 差集②：资产侧 ──────────────────────────────────────────────────
    attempted = attempted_asset_locations(facts, results)
    for fact in facts:
        name = str(fact.get("name") or "").strip()
        location = str(fact.get("location") or "").strip()
        if not name or not location:
            continue
        if location in attempted:
            continue  # 已尝试过（不论成败）→ 不再作为候选
        tool = str(fact.get("tool") or "").strip()
        if not tool or tool not in allowed:
            excluded.append({
                "asset": name,
                "location": location,
                "reason": f"当前白名单不含可消费该资产的工具（{tool or '文档未声明'}）",
            })
            continue
        # 描述只放位置：资产名已在标识里，重复一遍会白占长度上限。
        candidates.append(
            Candidate(id=f"{_CANDIDATE_KIND_ASSET}:{name}", description=location)
        )

    # ── 差集①：工具侧 ──────────────────────────────────────────────────
    used_tools: Set[str] = {
        str(record.get("tool_name"))
        for record in results
        if isinstance(record, dict) and record.get("tool_name")
    }
    for tool_name in sorted(allowed - used_tools):
        candidates.append(
            Candidate(id=f"{_CANDIDATE_KIND_TOOL}:{tool_name}", description="尚未尝试的工具")
        )

    return CandidateSet(candidates=candidates, excluded=excluded)


def render_candidates(
    candidates: Sequence[Candidate],
    max_chars: int = CANDIDATE_MAX_CHARS,
) -> str:
    """把候选列表渲染成提示词片段；为空或超限时按**整条**取舍。

    只渲染"索引"级信息（标识 + 一句话）——参数 schema 仍在真正调用工具的那一刻
    由 Function Calling 注入。把 schema 塞进每次注入会让成本反超重规划。
    """
    if not candidates:
        return ""

    # 标识直接作为每行的标签——不再另起一行列"可选标识"。
    # 实测：3 个资产 + 3 个工具的完整位置已经接近 300 字符，再列一遍标识会被
    # 长度上限截掉正文（位置是模型真正需要照抄的东西），等于把预算花在重复上。
    lines: List[str] = [f"[{item.id}] {item.description}" for item in candidates]

    budget: int = max_chars - len(CANDIDATE_BLOCK_HEADER) - 1
    if budget <= 0:
        return ""

    kept: List[str] = []
    used: int = 0
    for line in lines:
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    if not kept:
        return ""
    return f"{CANDIDATE_BLOCK_HEADER}\n" + "\n".join(kept) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# §3 注入时机
# ─────────────────────────────────────────────────────────────────────────────

def _last_result(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    results = state.get("subtask_results") or []
    for record in reversed(results):
        if isinstance(record, dict):
            return record
    return None


def injection_reason(
    state: Dict[str, Any],
    *,
    evidence_gap: bool = False,
) -> Optional[str]:
    """判定是否注入候选列表；返回触发原因，不注入则返回 None。

    三个条件任一成立即注入：
      1. 上一步未取得有效数据（错误 / 空数据，确定性判据）；
      2. 上一步虽有产出但自评为「未解决」或「部分解决」；
      3. 本步触发了证据缺口检测（返回内容与问题核心词不匹配）。

    正常成功且上一步自评为已解决时**不注入**——常驻注入会让模型分不清
    "我现在应该换方向"与"我只是需要知道有哪些工具"。
    """
    last = _last_result(state)
    if last is not None:
        if result_is_ineffective(last):
            return "取数失败"
        solved = str(last.get("solved") or "").strip().lower()
        if solved in {"no", "partial"}:
            return "上一步未解决"
    if evidence_gap:
        return "证据缺口"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# §4 模型只选不造：标识 → 具体动作
# ─────────────────────────────────────────────────────────────────────────────

def parse_candidate_id(raw: Any) -> Optional[str]:
    """从模型的输出里取出候选项标识；取不到或不合法时返回 None。

    容错：模型常把渲染出来的方括号一起抄回来（`[tool:web_search]`），
    这里把中英文方括号一并剥掉再解析。
    """
    text = str(raw or "").strip().strip("[]【】").strip()
    if not text:
        return None
    matched = _ID_SPLIT_RE.match(text)
    if not matched:
        return None
    return f"{matched.group(1).strip().lower()}:{matched.group(2).strip()}"


def translate_candidate(
    candidate_id: Any,
    *,
    candidates: Sequence[Candidate],
    allowed_tools: Iterable[str],
    facts: Any,
) -> Optional[Dict[str, Any]]:
    """把候选项标识翻译为具体动作（含白名单复核）。

    模型**只输出标识**，工具名 / 参数 / 路径全部由这里构造。标识不在本次注入的
    候选列表中、或翻译出的工具未过白名单校验时返回 ``None``——调用方据此降级为
    「继续执行下一个子任务」，并且**不得丢弃本步已产出的结论**。
    """
    parsed = parse_candidate_id(candidate_id)
    if not parsed:
        return None

    valid_ids = {item.id for item in candidates}
    if parsed not in valid_ids:
        return None

    kind, _, name = parsed.partition(":")
    allowed = {str(item) for item in (allowed_tools or []) if str(item).strip()}

    if kind == _CANDIDATE_KIND_TOOL:
        if name not in allowed:
            return None
        return {"tool_name": name, "action_input": {}, "title": f"调用 {name}"}

    if kind == _CANDIDATE_KIND_ASSET:
        for fact in facts or []:
            if not isinstance(fact, dict):
                continue
            if str(fact.get("name") or "").strip() != name:
                continue
            tool = str(fact.get("tool") or "").strip()
            if not tool or tool not in allowed:
                return None
            location = str(fact.get("location") or "").strip()
            if not location:
                return None
            return {
                "tool_name": tool,
                "action_input": {"file_path": location},
                "title": f"读取 {name}",
            }
        return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# §6 闸门与留痕
# ─────────────────────────────────────────────────────────────────────────────

def correction_quota(state: Dict[str, Any]) -> int:
    """单轮纠偏配额：剩余调用预算的一半（未设上限时取保守值）。"""
    from app.core.agent.graph.state import budget_remaining  # noqa: PLC0415

    remaining = budget_remaining(state.get("budget") or {})
    if remaining is None:
        return QUOTA_WHEN_UNLIMITED
    return max(0, int(remaining * QUOTA_BUDGET_RATIO))


def remaining_quota(state: Dict[str, Any]) -> int:
    """剩余纠偏配额；耗尽即关闭该能力。"""
    used = len(state.get("step_corrections") or [])
    return max(0, correction_quota(state) - used)
