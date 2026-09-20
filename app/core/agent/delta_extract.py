# -*- coding: utf-8 -*-
"""显示层：把模型原始输出转换成"该显示给用户的文本"。

⚠️ 本模块**只服务于显示**。业务侧的解析路径（schema 校验、容错、回退）
一律不经过这里，也绝不修改——这是 spec §2 的硬约束。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

__all__ = ["IncrementalJsonFieldExtractor"]


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
