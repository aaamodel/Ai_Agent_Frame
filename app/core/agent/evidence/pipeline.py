# -*- coding: utf-8 -*-
"""确定性证据管道：查询表示 → 打分 → 跨轮去重 → 按轮预算选择。

零 LLM、零 embedding、零新第三方依赖（jieba 已在项目依赖中）。
入管（ingest）是唯一写入口；证据板每轮从全量 Unit 重算。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jieba

from app.core.agent.evidence.chunkers import MAX_UNIT_CHARS, chunk_observation
from app.core.agent.evidence.models import (
    KIND_CONTENT,
    KIND_ERROR,
    KIND_STATUS,
    KIND_TABLE,
    EvidenceUnit,
    RoundReport,
)

# ── 预算 / 阈值常量（spec 锁定值）─────────────────────────────────────────
SCORE_MIN: float = 35.0
BOARD_BASE_CHARS: int = 1200
BOARD_STEP_CHARS: int = 400
BOARD_CAP_CHARS: int = 2400
MMR_LAMBDA: float = 0.7
SIMHASH_BITS: int = 64
SIMHASH_HAMMING_MAX: int = 3
SHORT_TEXT_CHARS: int = 50
JACCARD_DUP_MIN: float = 0.85
LEN_DIFF_MAX_RATIO: float = 0.2
FETCH_BONUS: float = 10.0
# 短观测豁免：≤300 字符整建一个豁免单元，ReAct 视图原文保留 3 个轮次，
# 超期收敛为一行桩（工具名+字符数+编号+首句预览）。
EXEMPT_OBS_CHARS: int = 300
EXEMPT_KEEP_ROUNDS: int = 3
EXEMPT_STUB_HEAD_CHARS: int = 60

_BM25_K1 = 1.5
_BM25_B = 0.75
_WEIGHT_CURRENT_QUERY = 1.0
_WEIGHT_BACKGROUND = 0.6

_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "及", "或", "都", "也", "就",
    "我", "你", "您", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "什么", "怎么", "如何", "为什么", "哪", "哪里", "哪些",
    "请问", "请", "帮", "帮忙", "帮我", "一下", "谢谢", "吗", "呢", "吧", "啊",
    "需要", "想要", "查", "查询", "查看", "看看", "知道", "了解", "一下",
    "可以", "可能", "应该", "已经", "还有", "没有", "不是", "一个", "一些",
    "对于", "关于", "根据", "按照", "通过", "进行", "以及", "并且", "但是",
    "to", "the", "a", "an", "of", "in", "on", "and", "or", "is", "are",
    "for", "please", "show", "me", "what", "how", "why",
}

_ENTITY_PATTERNS = [
    re.compile(r"\d+(?:\.\d+)?%"),                       # 百分比
    re.compile(r"\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?"),  # 日期
    re.compile(r"\d{1,2}月\d{1,2}日?"),
    re.compile(r"\d{1,2}月"),
    re.compile(r"\d+(?:\.\d+)?"),                        # 数字（含金额/计数）
    re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}"),           # 英文代号/型号
]
_ASCII_KEEP = re.compile(r"[A-Za-z0-9]")


# ── 1. 查询表示 ────────────────────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    """jieba 分词：保留 ≥2 字词 与 单字英文/数字 token，去停用词。"""
    tokens: List[str] = []
    for raw in jieba.cut(str(text or "")):
        token = raw.strip().lower()
        if not token or token in _STOPWORDS:
            continue
        if len(token) >= 2:
            tokens.append(token)
        elif _ASCII_KEEP.fullmatch(token):
            tokens.append(token)
    return tokens


def query_terms_weighted(
    current_query: str,
    background_terms: Sequence[str],
) -> Dict[str, float]:
    """加权查询词项：本轮 query 权重 1.0；背景词（用户原始问题/历史轮 query）0.6。

    同一词在两处出现时取较高权重。
    """
    weighted: Dict[str, float] = {}
    for term in background_terms:
        weighted[term] = max(weighted.get(term, 0.0), _WEIGHT_BACKGROUND)
    for term in tokenize(current_query):
        weighted[term] = max(weighted.get(term, 0.0), _WEIGHT_CURRENT_QUERY)
    return weighted


def _char_bigrams(text: str) -> List[str]:
    compact = re.sub(r"[\s\W_]+", "", str(text or ""), flags=re.UNICODE)
    return [compact[i : i + 2] for i in range(max(0, len(compact) - 1))]


def _entities(text: str) -> List[str]:
    found: List[str] = []
    for pattern in _ENTITY_PATTERNS:
        found.extend(pattern.findall(str(text or "")))
    # 去重保序
    seen: set = set()
    uniq: List[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def extract_query_text(action_input: Optional[Dict[str, Any]], tool_name: str) -> str:
    """从工具参数里取"本轮查询"的最佳文本（不同工具字段名不同）。"""
    if not isinstance(action_input, dict):
        return ""
    for key in ("query", "question", "pattern", "sql", "keyword", "keywords"):
        value = action_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # 兜底：参数整体序列化（文件路径等本身就是强信号）
    return " ".join(str(v) for v in action_input.values() if isinstance(v, (str, int, float)))


# ── 2. BM25 + 固定权重打分 ─────────────────────────────────────────────────
def _bm25_scores(
    units: List[EvidenceUnit],
    query_weights: Dict[str, float],
) -> Dict[int, float]:
    scorable = [i for i, u in enumerate(units) if u.kind in (KIND_CONTENT, KIND_TABLE)]
    docs_tokens = [tokenize(units[i].text) for i in scorable]
    n_docs = len(docs_tokens)
    if n_docs == 0 or not query_weights:
        return {}

    df: Counter = Counter()
    token_counters: List[Counter] = []
    doc_lens: List[int] = []
    for tokens in docs_tokens:
        counter = Counter(tokens)
        token_counters.append(counter)
        doc_lens.append(sum(counter.values()))
        df.update(counter.keys())

    avgdl = max(1.0, sum(doc_lens) / n_docs)
    idf = {
        term: math.log(1 + (n_docs - dfi + 0.5) / (dfi + 0.5))
        for term, dfi in df.items()
    }
    raw: Dict[int, float] = {}
    for slot, (i, counter, dl) in enumerate(zip(scorable, token_counters, doc_lens)):
        score = 0.0
        for term, weight in query_weights.items():
            tf = counter.get(term, 0)
            if tf <= 0:
                continue
            idf_t = idf.get(term, 0.0)
            score += (
                weight
                * idf_t
                * (tf * (_BM25_K1 + 1))
                / (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl))
            )
        raw[i] = score
    # 归一口径：以"命中单个最强加权词项（tf=1、文档长度=平均长度）的贡献"为满分
    # 参考——而不是以当轮最高词项数文档为分母。后者会让跨轮累积、但只命中任务
    # 核心词的旧证据被"本轮多词项文档"稀释到阈值以下，证据板无法跨轮留证。
    # 多词项命中直接截断 40；语料中不存在的词项不参与分母。
    saturation = 0.0
    for term, weight in query_weights.items():
        dfi = df.get(term, 0)
        if dfi > 0:
            saturation = max(saturation, weight * idf[term])
    if saturation <= 0:
        return raw
    return {i: min(40.0, score / saturation * 40.0) for i, score in raw.items()}


def _structure_score(text: str, kind: str) -> float:
    if kind == KIND_TABLE:
        return 10.0
    score = 0.0
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        first = lines[0].strip()
        if first.startswith(("#", "【", "* ")) or re.match(r"^\d+[.、)]", first):
            score += 5.0
    if lines:
        path_like = sum(
            1 for line in lines
            if ("/" in line or "\\" in line) and len(line.strip()) < 40
        )
        if len(lines) >= 3 and path_like / len(lines) > 0.5:
            score -= 15.0
    return max(-15.0, min(10.0, score))


def score_units(
    units: List[EvidenceUnit],
    *,
    query_weights: Dict[str, float],
    query_text: str,
    current_round: int,
) -> None:
    """就地填充 content/table 单元的 score（0–100）。error/status 保持 0。"""
    bm25 = _bm25_scores(units, query_weights)
    bigrams = set(_char_bigrams(query_text))
    entities = _entities(query_text)
    max_round = max((u.round_idx for u in units), default=0)

    for i, unit in enumerate(units):
        if unit.kind not in (KIND_CONTENT, KIND_TABLE):
            unit.score = 0.0
            continue
        if unit.exempt:
            # 豁免单元不打分，按 D5a 走限期原文通道
            unit.score = 0.0
            continue
        text = unit.text

        substring = 0.0
        if bigrams:
            hits = sum(1 for bg in bigrams if bg in text)
            if hits >= 2:
                substring = min(20.0, hits * 3.0)

        entity_score = 0.0
        if entities:
            overlap = sum(1 for ent in entities if ent in text)
            entity_score = 15.0 * overlap / len(entities)

        structure = _structure_score(text, unit.kind)

        if max_round > 0:
            freshness = 15.0 * (1 - unit.round_idx / max_round)
        else:
            freshness = 15.0

        base = bm25.get(i, 0.0) + substring + entity_score + structure + freshness
        # 取回加分仅在取回发生后的下一次入管/选择（同一轮号）生效一轮
        if unit.ref.get("fetched_round") == current_round:
            base += FETCH_BONUS
        unit.score = max(0.0, min(100.0, base))


# ── 3. 去重：sha1 → SimHash → 短文本 Jaccard ──────────────────────────────
def _hash64(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:16], 16)


def simhash(text: str) -> int:
    """64 位 SimHash：jieba 词项（tf 加权）+ 相邻词 bigram。"""
    words = tokenize(text)
    features: Counter = Counter(words)
    for left, right in zip(words, words[1:]):
        features[f"{left} {right}"] += 1
    if not features:
        # 无语义词项时退回字符 bigram，保证纯数字/代号块也有指纹
        features = Counter(_char_bigrams(text))

    vector = [0] * SIMHASH_BITS
    for feature, weight in features.items():
        digest = _hash64(feature)
        for bit in range(SIMHASH_BITS):
            vector[bit] += weight if (digest >> bit) & 1 else -weight
    fingerprint = 0
    for bit, value in enumerate(vector):
        if value > 0:
            fingerprint |= 1 << bit
    return fingerprint


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _char_ngrams(text: str, n: int = 3) -> set:
    compact = re.sub(r"\s+", "", str(text or ""))
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def _jaccard(left: str, right: str) -> float:
    a, b = _char_ngrams(left), _char_ngrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_duplicate(unit: EvidenceUnit, other: EvidenceUnit) -> bool:
    """判定两个单元是否重复（长度差 <20% 为前提，防止长短互吞）。"""
    if unit.kind != other.kind:
        return False
    long_len, short_len = max(len(unit.text), 1), max(len(other.text), 1)
    if min(long_len, short_len) / max(long_len, short_len) < 1 - LEN_DIFF_MAX_RATIO:
        return False

    short_text = min(len(unit.text), len(other.text)) < SHORT_TEXT_CHARS
    if short_text:
        # SimHash 对短中文不稳定：以 3-gram Jaccard 为准
        return _jaccard(unit.text, other.text) >= JACCARD_DUP_MIN
    return _hamming(unit.simhash, other.simhash) <= SIMHASH_HAMMING_MAX


def _exact_key(text: str) -> str:
    """逐字精确判重键：空白归一后的 sha1（短观测豁免通道专用）。"""
    compact = re.sub(r"\s+", "", str(text or ""))
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()


def is_exact_duplicate(unit: EvidenceUnit, other: EvidenceUnit) -> bool:
    """豁免单元的精确短路判重：同 kind 且归一化后逐字相同。"""
    if unit.kind != other.kind:
        return False
    return _exact_key(unit.text) == _exact_key(other.text)


# ── 4. 按轮收紧预算 + MMR 选择 ─────────────────────────────────────────────
def board_budget_chars(round_idx: int) -> int:
    return min(BOARD_BASE_CHARS + BOARD_STEP_CHARS * round_idx, BOARD_CAP_CHARS)


def board_top_k(round_idx: int) -> int:
    return min(4 + (round_idx + 1), 10)


def _mmr_select(
    candidates: List[EvidenceUnit],
    top_k: int,
    char_budget: int,
) -> List[EvidenceUnit]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda u: u.score, reverse=True)
    selected: List[EvidenceUnit] = []
    pool = list(ordered)
    used_chars = 0

    def pair_sim(unit: EvidenceUnit, other: EvidenceUnit) -> float:
        return _jaccard(unit.text, other.text)

    while pool and len(selected) < top_k and used_chars < char_budget:
        if not selected:
            choice = pool.pop(0)
        else:
            best_idx, best_mmr = 0, float("-inf")
            for idx, unit in enumerate(pool):
                max_sim = max(pair_sim(unit, picked) for picked in selected)
                mmr = MMR_LAMBDA * (unit.score / 100.0) - (1 - MMR_LAMBDA) * max_sim
                if mmr > best_mmr:
                    best_mmr, best_idx = mmr, idx
            choice = pool.pop(best_idx)
        if used_chars + len(choice.text) > char_budget and selected:
            # 预算装满即止（fetched 单元由调用方在预算外强制保留）
            break
        selected.append(choice)
        used_chars += len(choice.text)
    return selected


def select_board(units: List[EvidenceUnit], round_idx: int) -> List[EvidenceUnit]:
    """每轮从全量单元重选证据板。被 fetch 回取的单元只在取回轮强制保留一轮，
    同一 uid 不会因重复回取长期占用预算（fetched_round 旧于当前轮即失效）。
    """
    def _fresh_fetched(u: EvidenceUnit) -> bool:
        return u.ref.get("fetched_round") == round_idx

    eligible = [
        u for u in units
        if not u.dupe_of and u.kind in (KIND_CONTENT, KIND_TABLE)
        and (_fresh_fetched(u) or u.score >= SCORE_MIN)
    ]
    forced = [u for u in eligible if _fresh_fetched(u)]
    ranked = [u for u in eligible if not _fresh_fetched(u)]

    budget = board_budget_chars(round_idx)
    forced_chars = sum(len(u.text) for u in forced)
    picked = _mmr_select(
        ranked,
        top_k=board_top_k(round_idx),
        char_budget=max(0, budget - forced_chars),
    )
    return forced + picked


# ── 5. 入管编排 ────────────────────────────────────────────────────────────
def _board_chars(units: List[EvidenceUnit]) -> int:
    return sum(len(u.text) for u in units)


def ingest_observation(
    *,
    existing_units: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
    tool_name: str,
    round_idx: int,
    observation: str,
    current_query: str,
    user_question: str,
    action_input: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], RoundReport]:
    """把一次工具观测入管：切分 → 编号 → 全局重打分 → 去重 → 重选证据板。

    Returns:
        (all_unit_dicts, new_meta, round_report)
        all_unit_dicts 为**全量**单元（含 selected 标记），由节点整体替换写回
        state（evidence_units 非 add-reducer）；本函数不依赖调用方做合并。
    """
    meta = dict(meta or {})
    next_seq = int(meta.get("next_seq") or 1)
    history_queries = list(meta.get("queries") or [])

    blocks = chunk_observation(
        tool_name=tool_name,
        observation=observation,
        action_input=action_input,
        call_id=call_id,
    )

    # 短观测豁免（D5a）：非 error/status 且 strip 后 ≤300 字符 → 折叠为 1 个
    # 豁免单元（沿用首块 ref 作回取坐标），不切分、不打分。
    exempt_observation = False
    obs_stripped = str(observation or "").strip()
    if (
        obs_stripped
        and len(obs_stripped) <= EXEMPT_OBS_CHARS
        and blocks
        and all(
            (b.get("kind") or KIND_CONTENT) not in (KIND_ERROR, KIND_STATUS)
            for b in blocks
        )
    ):
        first = blocks[0]
        all_tables = all(
            (b.get("kind") or KIND_CONTENT) == KIND_TABLE for b in blocks
        )
        blocks = [{
            "text": obs_stripped,
            "source": first.get("source") or "",
            "ref": first.get("ref") or {},
            "kind": KIND_TABLE if all_tables else KIND_CONTENT,
            "truncated": False,
        }]
        exempt_observation = True

    new_units: List[EvidenceUnit] = []
    for block_idx, block in enumerate(blocks):
        unit = EvidenceUnit(
            uid=f"e{next_seq}",
            tool_name=tool_name,
            round_idx=int(round_idx),
            block_idx=block_idx,
            text=block["text"][:MAX_UNIT_CHARS],
            source=block.get("source") or "",
            ref=block.get("ref") or {},
            kind=block.get("kind") or KIND_CONTENT,
            truncated=bool(block.get("truncated")),
            exempt=exempt_observation,
        )
        next_seq += 1
        new_units.append(unit)

    all_units = [EvidenceUnit.from_dict(d) for d in existing_units] + new_units
    new_uids = {u.uid for u in new_units}

    # 指纹（旧单元已带指纹）
    for unit in all_units:
        if not unit.simhash:
            unit.simhash = simhash(unit.text)

    # 背景词 = 用户原始问题 + 历史轮 query；本轮 query 权重最高
    background = tokenize(user_question) + history_queries
    query_weights = query_terms_weighted(current_query, background)
    query_text = " ".join([user_question or "", current_query or ""])
    score_units(
        all_units,
        query_weights=query_weights,
        query_text=query_text,
        current_round=int(round_idx),
    )

    # 去重：本轮新单元逐个与"已确立单元"（存量 + 本轮已接纳的新单元）比对
    report = RoundReport(round_idx=int(round_idx), new=len(new_units))
    established: List[EvidenceUnit] = [
        u for u in all_units if u.uid not in new_uids
    ]
    accepted_new: List[EvidenceUnit] = []
    for unit in new_units:
        if unit.kind not in (KIND_CONTENT, KIND_TABLE):
            accepted_new.append(unit)
            continue
        winner: Optional[EvidenceUnit] = None
        if unit.exempt:
            # 豁免单元只做逐字精确判重（不做 SimHash/Jaccard 模糊判重）
            for other in established + accepted_new:
                if other.dupe_of:
                    continue
                if is_exact_duplicate(unit, other):
                    winner = other
                    break
        else:
            for other in established + accepted_new:
                if other.kind not in (KIND_CONTENT, KIND_TABLE) or other.dupe_of:
                    continue
                if is_duplicate(unit, other):
                    winner = other
                    break
        if winner is None:
            accepted_new.append(unit)
            if unit.exempt:
                report.exempt += 1
            continue
        report.duplicated += 1
        if unit.score > winner.score:
            # 新王登基：旧王降为重复，来源互证信息迁给新王
            winner.dupe_of = unit.uid
            unit.also_from = list(dict.fromkeys(
                ([winner.source] if winner.source else [])
                + list(winner.also_from)
            ))
            accepted_new.append(unit)
        else:
            unit.dupe_of = winner.uid
            if unit.source and unit.source != winner.source:
                winner.also_from = list(dict.fromkeys(
                    winner.also_from + [unit.source]
                ))

    # 低分计数（本轮、非 dup、非豁免、内容型；豁免单元不打分故不计低分）
    for unit in new_units:
        if (
            not unit.dupe_of
            and not unit.exempt
            and unit.kind in (KIND_CONTENT, KIND_TABLE)
            and unit.score < SCORE_MIN
        ):
            report.low_score += 1

    # 每轮全局重选（selected 仅作记录；视图渲染时以同口径重算，不依赖持久标记）
    board = select_board(all_units, int(round_idx))
    board_uids = {u.uid for u in board}
    for unit in all_units:
        unit.selected = unit.uid in board_uids
    report.on_board = len(board)
    report.board_chars = _board_chars(board)

    has_error = any(u.kind in (KIND_ERROR, KIND_STATUS) for u in new_units)
    content_new = [u for u in new_units if u.kind in (KIND_CONTENT, KIND_TABLE)]
    # 接纳的豁免原文本身就是新证据，不计入"全部低分"判定
    report.no_new_evidence = (
        bool(content_new)
        and all(
            u.dupe_of or (not u.exempt and u.score < SCORE_MIN)
            for u in content_new
        )
        and not has_error
    )

    new_meta: Dict[str, Any] = {
        "next_seq": next_seq,
        "queries": list(dict.fromkeys(history_queries + tokenize(current_query))),
        "rounds": list(meta.get("rounds") or []) + [report.to_dict()],
    }

    return [u.to_dict() for u in all_units], new_meta, report
