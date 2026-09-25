# -*- coding: utf-8 -*-
"""按工具类型把原始观测确定性切分为证据块（RawBlock）。

约定：
- 只切分、不改写；每个块是一个可独立上板/回取的原子；
- 块大小上限 MAX_UNIT_CHARS，超长按句边界滑窗（重叠 1 句，防边界丢证据）；
- 错误 / 无结果状态识别为 error / status 块（不参与打分，恒保留一行）；
- 新增工具只需在 ``CHUNKERS`` 注册一个函数，未注册走通用兜底。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from app.core.agent.evidence.models import (
    KIND_CONTENT,
    KIND_ERROR,
    KIND_STATUS,
    KIND_TABLE,
)

# ── 切分常量（集中管理，测试可调）─────────────────────────────────────────
MAX_UNIT_CHARS: int = 600
"""单块正文字符上限。"""

SHORT_OBS_CHARS: int = 200
"""不超过该长度的观测整体成块，不再切分。"""

TABLE_MAX_ROWS: int = 30
"""表格单元保留的最大数据行数（不含表头/分隔行）。"""

WEB_BLOCK_MAX_CHARS: int = 500
"""联网搜索 / 知识图谱 / 飞书等低频工具的单块正文字符上限。"""

# ── 错误 / 状态识别（只看首部，避免正文里出现"错误"二字被误判）────────────
_ERROR_PREFIXES = (
    "error:", "error：", "错误:", "错误：", "错误 ",
)
_ERROR_MARKERS = (
    "执行期间发生异常错误",
    "操作失败",
    "调用失败",
    "期间崩溃",
    "operation failed",
    "failed to ",
)
_STATUS_MARKERS = (
    "未匹配到任何",
    "no matches found",
    "is empty",
    "empty or offset out of bounds",
    "未找到",
    "无数据",
    "没有查询到",
    "0 条",
    "no data",
)

# RAG：--- 知识库检索结果 (查询: xxx) ---  /  [n] 来源文献: xxx\n内容片段: yyy
_RAG_HEADER = re.compile(r"^---\s*知识库检索结果\s*\(查询[:：]\s*(.*?)\)\s*---", re.S)
_RAG_BLOCK = re.compile(
    r"\[(\d+)\]\s*来源文献[:：]\s*(.*?)\s*\n\s*内容片段[:：]\s*(.*?)"
    r"(?=\n\s*\[\d+\]\s*来源文献[:：]|\Z)",
    re.S,
)

_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_GREP_LINE = re.compile(r"^(.+?):(\d+):\s?(.*)$")
_SENTENCE = re.compile(r"[^。！？；\n.!?]+(?:[。！？；\n.!?]+|$)")
_SOFT_BREAK = re.compile(r"[，、；,;]")


def _head(text: str, chars: int = 240) -> str:
    return text.strip()[:chars].lower()


def classify_kind(text: str) -> str:
    """识别 error / status；其余一律 content。"""
    stripped = text.strip()
    head = _head(stripped)
    if head.startswith(_ERROR_PREFIXES) or any(m in head for m in _ERROR_MARKERS):
        return KIND_ERROR
    if any(m in head for m in _STATUS_MARKERS):
        return KIND_STATUS
    return KIND_CONTENT


def split_sentences(text: str) -> List[str]:
    """中英文句边界切分（含换行作为硬边界）。"""
    return [part.strip() for part in _SENTENCE.findall(text) if part.strip()]


def _hard_split_long(text: str, max_chars: int) -> List[str]:
    """单句超过上限：在逗号/分号等软边界处断，找不到则硬切。"""
    pieces: List[str] = []
    rest = text
    while len(rest) > max_chars:
        window = rest[:max_chars]
        breaks = [m.start() for m in _SOFT_BREAK.finditer(window)]
        cut = breaks[-1] + 1 if breaks else max_chars
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest.strip())
    return [p for p in pieces if p]


def pack_sentences(sentences: List[str], max_chars: int = MAX_UNIT_CHARS,
                   overlap: int = 1) -> List[str]:
    """句子列表滑窗装填为块；超长单句软边界硬切。"""
    blocks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        if current:
            blocks.append("".join(current).strip())

    for sentence in sentences:
        if len(sentence) > max_chars:
            flush()
            current, current_len = [], 0
            for piece in _hard_split_long(sentence, max_chars):
                blocks.append(piece)
            continue
        if current and current_len + len(sentence) > max_chars:
            flush()
            carry = current[-overlap:] if overlap else []
            current = list(carry)
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
    flush()
    return [b for b in blocks if b]


def _make_block(text: str, *, kind: str, source: str = "",
                ref: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "text": text.strip(),
        "kind": kind,
        "source": source.strip() if source else "",
        "ref": dict(ref or {}),
    }


# ── 通用兜底切分器 ─────────────────────────────────────────────────────────
def default_chunker(
    observation: str,
    *,
    action_input: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    """双换行分段 → 句子滑窗；短观测整块；错误/状态直接成块。"""
    text = observation or ""
    kind = classify_kind(text)
    ref: Dict[str, Any] = {"call_id": call_id} if call_id else {}
    if kind in (KIND_ERROR, KIND_STATUS):
        return [_make_block(text, kind=kind, ref=ref)]
    if len(text.strip()) <= SHORT_OBS_CHARS:
        return [_make_block(text, kind=KIND_CONTENT, ref=ref)]

    blocks: List[Dict[str, Any]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= MAX_UNIT_CHARS:
            blocks.append(_make_block(paragraph, kind=KIND_CONTENT, ref=ref))
        else:
            for piece in pack_sentences(split_sentences(paragraph)):
                blocks.append(_make_block(piece, kind=KIND_CONTENT, ref=ref))
    return blocks


# ── RAG ────────────────────────────────────────────────────────────────────
def rag_chunker(
    observation: str,
    *,
    action_input: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    """解析 `--- 知识库检索结果 ---` + `[n] 来源文献/内容片段` 块。"""
    text = observation or ""
    header = _RAG_HEADER.match(text.strip())
    query = header.group(1).strip() if header else ""
    matches = list(_RAG_BLOCK.finditer(text))
    if not matches:
        # 空检索 / 异常文案：交给通用分类（status / error）
        return default_chunker(text, action_input=action_input, call_id=call_id)

    blocks: List[Dict[str, Any]] = []
    for match in matches:
        index = int(match.group(1))
        source = match.group(2).strip()
        content = match.group(3).strip()
        ref = {"doc": source, "rag_index": index, "query": query}
        if call_id:
            ref["call_id"] = call_id
        if len(content) <= MAX_UNIT_CHARS:
            pieces = [content]
        else:
            pieces = pack_sentences(split_sentences(content))
        for piece in pieces:
            blocks.append(_make_block(piece, kind=KIND_CONTENT, source=source, ref=ref))
    return blocks


# ── markdown 表格（SQL / 目录列举等）───────────────────────────────────────
def _extract_table_runs(lines: List[str]) -> List[tuple]:
    """返回 [(start, end), ...]：连续的 markdown 表格行区间（end 不含）。"""
    runs: List[tuple] = []
    start: Optional[int] = None
    for i, line in enumerate(lines):
        if _TABLE_LINE.match(line):
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(lines)))
    return runs


def table_aware_chunker(
    observation: str,
    *,
    action_input: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
    source: str = "",
    **_: Any,
) -> List[Dict[str, Any]]:
    """表格整表成块（超 TABLE_MAX_ROWS 数据行截断并保留行数信息）；
    表外文本走通用切分。"""
    text = observation or ""
    kind = classify_kind(text)
    ref_base: Dict[str, Any] = {"call_id": call_id} if call_id else {}
    if kind in (KIND_ERROR, KIND_STATUS):
        return [_make_block(text, kind=kind, ref=ref_base)]
    if len(text.strip()) <= SHORT_OBS_CHARS:
        return [_make_block(text, kind=KIND_CONTENT, source=source, ref=ref_base)]

    lines = text.splitlines()
    runs = _extract_table_runs(lines)
    if not runs:
        return default_chunker(text, action_input=action_input, call_id=call_id)

    blocks: List[Dict[str, Any]] = []
    cursor = 0
    for start, end in runs:
        # 表前散文
        if start > cursor:
            prose = "\n".join(lines[cursor:start]).strip()
            if prose:
                blocks.extend(default_chunker(prose, call_id=call_id))
        table_lines = lines[start:end]
        # 表头(0)/分隔(1) 之后才是数据行
        data_rows = max(0, len(table_lines) - 2)
        kept = table_lines
        truncated = False
        if data_rows > TABLE_MAX_ROWS:
            kept = table_lines[: 2 + TABLE_MAX_ROWS]
            kept.append(f"| …（共 {data_rows} 行，已显示前 {TABLE_MAX_ROWS} 行，可回取全文） |")
            truncated = True
        table_text = "\n".join(kept)
        if len(table_text) > MAX_UNIT_CHARS and not truncated:
            # 单行超宽的极端情况：退回通用切分，避免撑爆预算
            blocks.extend(default_chunker(table_text, call_id=call_id))
        else:
            block = _make_block(table_text, kind=KIND_TABLE, source=source, ref=ref_base)
            block["truncated"] = truncated
            blocks.append(block)
        cursor = end
    # 表尾散文
    if cursor < len(lines):
        tail = "\n".join(lines[cursor:]).strip()
        if tail:
            blocks.extend(default_chunker(tail, call_id=call_id))
    return blocks


def sql_chunker(observation: str, **kwargs: Any) -> List[Dict[str, Any]]:
    return table_aware_chunker(observation, **kwargs)


# ── 文件读取 ───────────────────────────────────────────────────────────────
def file_read_chunker(
    observation: str,
    *,
    action_input: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    action_input = action_input or {}
    path = str(action_input.get("file_path") or action_input.get("path") or "")
    blocks = default_chunker(observation, action_input=action_input, call_id=call_id)
    ref_extra = {"path": path}
    if action_input.get("offset") is not None:
        ref_extra["offset"] = action_input.get("offset")
    if action_input.get("limit") is not None:
        ref_extra["limit"] = action_input.get("limit")
    for block in blocks:
        block["source"] = path
        block["ref"].update({k: v for k, v in ref_extra.items() if v not in (None, "")})
    return blocks


# ── grep：按来源文件归组（path:line: content）──────────────────────────────
def grep_chunker(
    observation: str,
    *,
    action_input: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    text = observation or ""
    kind = classify_kind(text)
    if kind in (KIND_ERROR, KIND_STATUS):
        return [_make_block(text, kind=kind, ref={"call_id": call_id} if call_id else {})]

    groups: Dict[str, List[str]] = {}
    order: List[str] = []
    for line in text.splitlines():
        match = _GREP_LINE.match(line.strip())
        if not match or line.lstrip().startswith("#"):
            continue
        path, lineno, content = match.group(1), match.group(2), match.group(3)
        if path not in groups:
            groups[path] = []
            order.append(path)
        groups[path].append(f"{path}:{lineno}: {content}")

    if not groups:
        return default_chunker(text, action_input=action_input, call_id=call_id)

    blocks: List[Dict[str, Any]] = []
    for path in order:
        ref = {"path": path}
        if call_id:
            ref["call_id"] = call_id
        for piece in pack_sentences(split_sentences("\n".join(groups[path]))):
            blocks.append(_make_block(piece, kind=KIND_CONTENT, source=path, ref=ref))
    return blocks


def file_list_chunker(observation: str, **kwargs: Any) -> List[Dict[str, Any]]:
    action_input = kwargs.get("action_input") or {}
    return table_aware_chunker(
        observation, source=str(action_input.get("path") or ""), **kwargs
    )


# ── 联网搜索（web_search 豆包主通道 / tavily_search_internal 降级通道）──────
# 共同外层：以下是关于「q」的最新联网搜索结果：
# 豆包：【搜索综合摘要】…  + [n] 标题/链接/来源/摘要/发布时间
# Tavily：[n] 标题/内容
_WEB_WRAPPER = re.compile(r"^以下是关于[「『].*?[」』].*?：\s*")
_WEB_ENTRY = re.compile(
    r"\[(\d+)\]\s*标题[:：]\s*(.*?)"
    r"(?=\n\s*\[\d+\]\s*标题[:：]|\Z)",
    re.S,
)
_WEB_SOURCE_LINE = re.compile(r"^\s*(?:链接|来源)[:：]\s*(\S+)", re.M)
_SUMMARY_SEGMENT = re.compile(r"【搜索综合摘要】\s*(.*?)(?=\n\s*\[\d+\]\s*标题[:：]|\Z)", re.S)


def web_search_chunker(
    observation: str,
    *,
    action_input: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
    **_: Any,
) -> List[Dict[str, Any]]:
    """搜索结果按条目成块（摘要块 + 每篇结果一块），单块 ≤500 字。

    `【系统提示】` 开头的降级/失败文案成单条 status 块；不匹配任何结构时
    退回 500 字上限的通用切分。
    """
    text = (observation or "").strip()
    ref_base: Dict[str, Any] = {"call_id": call_id} if call_id else {}
    if text.startswith("【系统提示】"):
        return [_make_block(text, kind=KIND_STATUS, ref=ref_base)]
    kind = classify_kind(text)
    if kind in (KIND_ERROR, KIND_STATUS):
        return [_make_block(text, kind=kind, ref=ref_base)]

    body = _WEB_WRAPPER.sub("", text, count=1)
    blocks: List[Dict[str, Any]] = []

    summary_match = _SUMMARY_SEGMENT.search(body)
    if summary_match and summary_match.group(1).strip():
        summary = summary_match.group(1).strip()
        for piece in pack_sentences(split_sentences(summary), WEB_BLOCK_MAX_CHARS):
            blocks.append(_make_block(
                piece, kind=KIND_CONTENT, source="搜索综合摘要", ref=ref_base
            ))

    found = False
    for match in _WEB_ENTRY.finditer(body):
        found = True
        entry = match.group(0).strip()
        source_match = _WEB_SOURCE_LINE.search(entry)
        source = source_match.group(1).strip() if source_match else f"搜索结果{match.group(1)}"
        for piece in pack_sentences(split_sentences(entry), WEB_BLOCK_MAX_CHARS):
            blocks.append(_make_block(
                piece, kind=KIND_CONTENT, source=source,
                ref={**ref_base, "web_index": int(match.group(1))},
            ))

    if not blocks and not found:
        return _capped_default_chunker(
            text, call_id=call_id, max_chars=WEB_BLOCK_MAX_CHARS
        )
    return blocks


# ── 知识图谱 / 飞书多维表：500 字上限通用切分 ─────────────────────────────
def _capped_default_chunker(
    observation: str,
    *,
    call_id: Optional[str] = None,
    max_chars: int = WEB_BLOCK_MAX_CHARS,
    source: str = "",
) -> List[Dict[str, Any]]:
    text = (observation or "").strip()
    ref_base: Dict[str, Any] = {"call_id": call_id} if call_id else {}
    if text.startswith("【系统提示】"):
        return [_make_block(text, kind=KIND_STATUS, source=source, ref=ref_base)]
    kind = classify_kind(text)
    if kind in (KIND_ERROR, KIND_STATUS):
        return [_make_block(text, kind=kind, source=source, ref=ref_base)]
    if len(text) <= max_chars:
        return [_make_block(text, kind=KIND_CONTENT, source=source, ref=ref_base)]

    blocks: List[Dict[str, Any]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            blocks.append(_make_block(paragraph, kind=KIND_CONTENT,
                                      source=source, ref=ref_base))
        else:
            for piece in pack_sentences(split_sentences(paragraph), max_chars):
                blocks.append(_make_block(piece, kind=KIND_CONTENT,
                                          source=source, ref=ref_base))
    return blocks


def knowledge_graph_chunker(observation: str, **kwargs: Any) -> List[Dict[str, Any]]:
    return _capped_default_chunker(
        observation, call_id=kwargs.get("call_id"),
        max_chars=WEB_BLOCK_MAX_CHARS, source="私有知识图谱",
    )


def feishu_bitable_chunker(observation: str, **kwargs: Any) -> List[Dict[str, Any]]:
    text = (observation or "").strip()
    if text.startswith("成功"):
        # 写入确认：状态行恒显即可，不参与相关性打分
        return [_make_block(
            text, kind=KIND_STATUS, source="飞书多维表",
            ref={"call_id": kwargs.get("call_id")} if kwargs.get("call_id") else {},
        )]
    return _capped_default_chunker(
        observation, call_id=kwargs.get("call_id"),
        max_chars=WEB_BLOCK_MAX_CHARS, source="飞书多维表",
    )


# ── 注册表与分派 ───────────────────────────────────────────────────────────
CHUNKERS: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "rag_knowledge_search": rag_chunker,
    "sales_sql_query": sql_chunker,
    "sales_sql_write": sql_chunker,
    "database_query": sql_chunker,
    "describe_table": sql_chunker,
    "list_tables": sql_chunker,
    "file_read_tool": file_read_chunker,
    "file_grep_tool": grep_chunker,
    "file_list_tool": file_list_chunker,
    "web_search": web_search_chunker,
    "tavily_search_internal": web_search_chunker,
    "knowledge_graph_search": knowledge_graph_chunker,
    "feishu_bitable_tool": feishu_bitable_chunker,
}


def chunk_observation(
    *,
    tool_name: str,
    observation: str,
    action_input: Optional[Dict[str, Any]] = None,
    call_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """把一次工具观测切分为 RawBlock 列表（dict: text/kind/source/ref）。

    任何 chunker 自身异常都向上抛——由调用方的降级边界统一兜底
    （退回既有压缩策略），chunkers 内部不吞异常。
    """
    chunker = CHUNKERS.get(tool_name, default_chunker)
    blocks = chunker(
        observation,
        action_input=action_input,
        call_id=call_id,
        tool_name=tool_name,
    )
    return [block for block in blocks if block.get("text")]
