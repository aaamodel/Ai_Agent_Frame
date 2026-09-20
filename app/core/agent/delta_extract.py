# -*- coding: utf-8 -*-
"""显示层：把模型原始输出转换成"该显示给用户的文本"。

⚠️ 本模块**只服务于显示**。业务侧的解析路径（schema 校验、容错、回退）
一律不经过这里，也绝不修改——这是 spec §2 的硬约束。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

__all__ = ["IncrementalJsonFieldExtractor", "DisplayRouter"]


# JSON 字符串里的简单转义
_SIMPLE_ESCAPES: Dict[str, str] = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class IncrementalJsonFieldExtractor:
    """在累积的 JSON 文本里，增量抽取某个字符串字段当前已确定可见的值。

    用法：
        ex = IncrementalJsonFieldExtractor("rewritten_question")
        for chunk in stream:
            display = ex.feed(chunk)      # 只返回"本次新增"的可见文本
            if display:
                push_delta(display)

    ⚠️ 兜底方向是"宁可晚出现，也不返回半截错误内容"：
    字段还没出现、值还没闭合时一律返回空串。
    """

    def __init__(self, field: str) -> None:
        self._key: str = f'"{field}"'
        self._buffer: str = ""          # 尚未消费的原始文本
        self._search_from: int = 0      # 定位键时的搜索起点（避免重复扫描）
        self._state: str = "seek_key"   # seek_key → seek_colon → seek_quote → in_value → done
        self._escaped: bool = False
        self._value: List[str] = []

    # ------------------------------------------------------------------
    # 对外
    # ------------------------------------------------------------------
    def visible_value(self) -> str:
        """当前已确定可见的完整值。"""
        return "".join(self._value)

    def feed(self, chunk: str) -> str:
        """喂入一段新增文本，返回本次新增的可见文本（可能为空串）。"""
        if self._state == "done":
            return ""
        self._buffer += chunk

        if self._state == "seek_key" and not self._consume_until_key():
            return ""
        if self._state == "seek_colon" and not self._consume_colon():
            return ""
        if self._state == "seek_quote" and not self._consume_quote():
            return ""

        if self._state == "in_value":
            return "".join(self._consume_value())
        return ""

    # ------------------------------------------------------------------
    # 内部：逐阶段消费
    # ------------------------------------------------------------------
    def _consume_until_key(self) -> bool:
        idx = self._buffer.find(self._key, self._search_from)
        if idx == -1:
            # 键可能跨 chunk 被切开：保留最后 len(key) 个字符继续等
            self._search_from = max(0, len(self._buffer) - len(self._key))
            return False
        self._buffer = self._buffer[idx + len(self._key):]
        self._search_from = 0
        self._state = "seek_colon"
        return True

    def _consume_colon(self) -> bool:
        idx = self._buffer.find(":")
        if idx == -1:
            return False
        self._buffer = self._buffer[idx + 1:]
        self._state = "seek_quote"
        return True

    def _consume_quote(self) -> bool:
        idx = self._buffer.find('"')
        if idx == -1:
            return False
        self._buffer = self._buffer[idx + 1:]
        self._state = "in_value"
        return True

    def _consume_value(self) -> List[str]:
        produced: List[str] = []
        buf = self._buffer
        i = 0
        while i < len(buf):
            ch = buf[i]
            if self._escaped:
                text, consumed = self._unescape(buf, i)
                if consumed == 0:
                    # \u 还没收全：保住从反斜杠起的部分，等下一片再解
                    self._buffer = buf[i - 1:]
                    self._escaped = False
                    self._value.extend(produced)
                    return produced
                produced.append(text)
                i += consumed
                self._escaped = False
                continue
            if ch == "\\":
                self._escaped = True
                i += 1
                continue
            if ch == '"':
                # 值闭合
                self._state = "done"
                self._buffer = buf[i + 1:]
                self._value.extend(produced)
                return produced
            produced.append(ch)
            i += 1

        # 未闭合：悬空的转义引导符留在缓冲区等下一片。
        # ⚠️ `_escaped` 必须在这里复位：缓冲区的开头就是那个 `\`，
        #    下一片会重新扫到它并重新置位。不复位的话，下一片会把这个 `\`
        #    本身当成"被转义的字符"，吐出一个字面反斜杠（实测踩过：
        #    `abc\` + `nd"` 变成了 `abc\\nd` 而不是 `abc\nd`）。
        if self._escaped:
            self._buffer = "\\"
            self._escaped = False
        else:
            self._buffer = ""
        self._value.extend(produced)
        return produced

    @staticmethod
    def _unescape(buf: str, backslash_at: int) -> Tuple[str, int]:
        r"""解出反斜杠后的转义。

        Returns:
            ``(文本, 消费的字符数)``；``\uXXXX`` 没收全时返回 ``("", 0)``，
            表示"等下一片"。

        ⚠️ 本 docstring 必须是原始字符串：里面出现 ``\uXXXX``，普通字符串会把它
        当成 Unicode 转义并在解析期抛 SyntaxError（实测踩过）。
        """
        nxt = buf[backslash_at] if backslash_at < len(buf) else ""
        if nxt == "u":
            hex_part = buf[backslash_at + 1:backslash_at + 5]
            if len(hex_part) < 4:
                return "", 0
            try:
                return chr(int(hex_part, 16)), 5
            except ValueError:
                return "u", 1
        return _SIMPLE_ESCAPES.get(nxt, nxt), 1


# ---------------------------------------------------------------------------
# 文本分流
# ---------------------------------------------------------------------------

# ReAct 文本协议的草稿标记（行首）
_REACT_DRAFT_RE = re.compile(r"(?m)^\s*(Thought|Action|Action Input)\s*:")
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*", re.IGNORECASE)

# 草稿标记的关键词：用于"前缀歧义"等待（"Tho" 还不能断定是不是 "Thought:"）
_DRAFT_KEYWORDS = ("Thought", "Action", "Action Input")

# `answer` 阶段判定为"汇总结构化输出"的探针：SummaryVerdictSchema 的首个字段
# （summarize_node.py:46-72）。用它把"用户自己要的 JSON 答案"排除掉。
_SUMMARY_PROBE = '"sufficient"'
_SUMMARY_PROBE_WINDOW = 200


class DisplayRouter:
    """把某一阶段的模型原始输出，转成"该追加显示的文本"。

    判定规则**严格按此顺序**（spec §4.5），命中即锁定：

    1. `phase == "rewrite"`：缓冲以 `{` 开头 → json 模式（抽 `rewritten_question`）
    2. `phase == "answer"`：以 `{` 开头**且**前 200 字符含 `"sufficient"`
       → json 模式（抽 `answer`）
    3. 含行首 `Thought:` / `Action:` / `Action Input:` → suppress（内部草稿）
    4. 含 `Final Answer:` → final（只显示其后的内容）
    5. 其余 → plain（原样显示）

    ⚠️ 两处相对朴素实现必须加固的点（否则会漏草稿或吞文本）：

    - **suppress 不是终态**：文本协议的同一轮里可能是
      `Thought: …\\nFinal Answer: xxx`，所以处于 suppress 时仍要持续找
      `Final Answer:`，一旦出现就切到 final 并输出其后内容。
    - **前缀歧义要等待**：`"Tho"` 还看不出是不是 `"Thought:"`，
      此时若草率判成 plain，就会把草稿开头的几个字推给用户。

    ⚠️ 判定未定期间**不输出但不丢文本**：一旦判定为 plain，
    之前攒下的文本会在同一次 feed 里补出来。
    """

    def __init__(self, phase: str) -> None:
        if phase not in ("rewrite", "answer"):
            raise ValueError(f"未知 phase: {phase!r}（只支持 rewrite / answer）")
        self._phase = phase
        self._raw: str = ""
        self._mode: str = "undecided"
        self._emitted_len: int = 0  # plain/final 模式已输出到 _raw 的哪个位置
        self._emitted_any: bool = False  # 是否已经向外输出过任何可见文本
        self._extractor: Optional[IncrementalJsonFieldExtractor] = None
        self._extractor_primed: bool = False

    @property
    def mode(self) -> str:
        return self._mode

    def visible_any(self) -> bool:
        """是否已输出过可见文本。

        供 SSE 层判断"换候选时要不要插入上段废弃标注"——从没输出过就别插。
        """
        return self._emitted_any

    def feed(self, chunk: str) -> str:
        """喂入一段新增文本，返回本次应追加显示的文本。"""
        self._raw += chunk
        visible = self._compute_visible(chunk)
        if visible:
            self._emitted_any = True
        return visible

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _compute_visible(self, chunk: str) -> str:
        if self._mode == "undecided":
            self._decide()

        if self._mode == "json":
            assert self._extractor is not None
            if not self._extractor_primed:
                # ⚠️ json 模式可能是"读到一半才判定"的（例如 answer 阶段要等
                #    200 字符窗口），判定前累积的文本必须补喂一次，
                #    否则目标字段若已在那段里，就永远抽不出来（静默吞字）。
                self._extractor_primed = True
                return self._extractor.feed(self._raw)
            return self._extractor.feed(chunk)

        # ⚠️ 无条件优先检查 Final Answer：它在 suppress 之后才出现，
        #    所以不能等到"判定完成"才看——那样草稿轮永远切不到输出态。
        marker = _FINAL_ANSWER_RE.search(self._raw)
        if marker is not None:
            self._mode = "final"
            if self._emitted_len < marker.end():
                self._emitted_len = marker.end()
            return self._take_tail(marker.end())

        # ⚠️ "undecided" 也必须拦在这里：判定未定时若落到 _take_tail，
        #    缓冲内容会被直接吐给用户（实测：汇总 JSON 漏出 `{"sufficient`、
        #    草稿轮漏出 `Thought`）。
        if self._mode in ("undecided", "suppress"):
            return ""
        return self._take_tail(0)

    def _take_tail(self, start: int) -> str:
        """取出 `_raw` 中从 `_emitted_len` 起的新增部分。"""
        if self._emitted_len >= len(self._raw):
            return ""
        out = self._raw[self._emitted_len:]
        self._emitted_len = len(self._raw)
        return out

    def _decide(self) -> None:
        stripped = self._raw.lstrip()
        if self._phase == "rewrite":
            if stripped.startswith("{"):
                self._mode = "json"
                self._extractor = IncrementalJsonFieldExtractor("rewritten_question")
            return

        # ---- answer 阶段 ----
        if stripped.startswith("{"):
            if _SUMMARY_PROBE in self._raw[:_SUMMARY_PROBE_WINDOW]:
                self._mode = "json"
                self._extractor = IncrementalJsonFieldExtractor("answer")
            elif len(self._raw) >= _SUMMARY_PROBE_WINDOW:
                # 以 { 开头但排除了汇总 schema → 用户自己要的 JSON，当普通文本
                self._mode = "plain"
            return

        if not stripped:
            return

        if _REACT_DRAFT_RE.search(self._raw):
            self._mode = "suppress"
            return

        # 前缀歧义：当前头还可能是草稿标记的开头，先等更多字再判
        head = stripped.split("\n", 1)[0]
        for keyword in _DRAFT_KEYWORDS:
            if keyword.startswith(head):
                return  # 如 "Tho" —— 可能正在打 "Thought:"
            if head.startswith(keyword) and ":" not in head:
                return  # 如 "Action" —— 打完词还没到冒号
        self._mode = "plain"
