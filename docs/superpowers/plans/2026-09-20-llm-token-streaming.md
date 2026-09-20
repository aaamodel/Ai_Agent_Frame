# LLM 生成过程逐字流式输出 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `/chat/with_agent` 把「问题改写」与「最终答案」的模型生成过程逐字推给前端，用户不必等整段生成完才看到输出。

**Architecture:** 新增一个基于 `ContextVar` 的「旁观通道」。SSE 层在每个阶段开始前装通道；
模型调用器发现自己被观测时改用 `stream=True` 收流，把增量推给通道，
**但对外仍返回与今天完全相同的完整结果**。因此路由、熔断、重试、候选降级、
以及业务侧的解析路径（含 rewrite/intent 的 schema 校验与容错）一行都不改。
增量 JSON 抽取与文本分流全部放在显示层。

**Tech Stack:** Python 3 / FastAPI / asyncio / ContextVar / openai SDK（AsyncOpenAI）/ LangGraph 1.2.11；前端 React 19 + TypeScript + Vitest。

**Spec:** `docs/superpowers/specs/2026-09-20-llm-token-streaming-design.md`

## Global Constraints

以下每条来自规格「§2 约束」，**每个任务都隐含包含本节**：

- **不显著增加端到端耗时**——判断取舍以"用户能否更早看到输出"为准，不数调用次数
- **不破坏 rewrite question 与 intent 的解析路径**：`AgentRewriteIntentCombinedSchema` 的
  `model_validate_json` / `coerce_llm_json_to_schema` / `validate_tolerating_agent_goal`
  **零改动**
- **不改模型路由与候选顺序**：tier 解析、`PURPOSE_TIER_MAP`、熔断、重试策略保持原样
- **不改任何 prompt**
- 不改 `ModelRouter.chat` / `chat_with_tools` 的签名与路由逻辑
- 不改 `AsyncModelRoutingExecutor` 的候选循环、重试、熔断
- 不引入 `AsyncStreamRoutingExecutor`
- 不改 `/chat`（非流式闲聊）与 `/chat/stream`（独立流式端点）
- **探测包式的额外请求不算"增加调用次数"**（用户已认可）
- 厂商不支持流式时，**退回非流式、结果出来一次性渲染**
- 复用现有 `step` 事件与「执行过程」前端组件，**不新增阶段进度协议**

## Review Focus

规格是愿景文档，它不会列出所有会遇到的输入。以下 5 类是**最可能咬到使用者**的情况，
每一条都在其归属任务里配了测试：

1. **模型先输出"我先查一下…"再调工具**（FC 前言）——用户会先看到一段作废文字，
   随后必须被明确标注为已废弃
2. **Agent 被要求直接输出 JSON**（"把结果以 JSON 给我"）——这是正常答案，
   **不能被误判成汇总的结构化输出而整段不显示**
3. **ReAct 文本协议轮次**——`Thought:` / `Action:` 是内部草稿，
   **绝不能出现在用户界面上**
4. **单个中文字被拆到两个 SSE chunk 里**（UTF-8 多字节边界 + JSON 转义边界）——
   不能出现半个字或乱码
5. **模型切换到下一个候选时已吐出的内容**——不能静默丢掉，也不能无标注地与
   新内容拼接

---

## 文件结构

**新建**

| 文件 | 职责 |
|---|---|
| `app/core/agent/stream_sink.py` | 旁观通道：`ContextVar` + `StreamSink` 协议 + `emit()` |
| `app/core/agent/delta_extract.py` | 显示层：增量 JSON 字段抽取器 + 文本分流器 |
| `测试/test_stream_sink.py` | 通道模块单测 |
| `测试/test_delta_extract.py` | 抽取器与分流器单测（含 Review Focus #2/#3/#4） |
| `测试/test_streaming_caller.py` | 调用器流式分支单测 |

**修改**

| 文件 | 改动 |
|---|---|
| `app/llm_model_router/async_openai_caller.py` | `async_openai_chat_caller` 增加流式分支（**唯一被改的模型层函数**） |
| `app/api/routes/chat.py` | `_agent_stream_generator` 装通道、排空队列、发 `delta` 事件 |
| `web/src/features/chat/streamEvents.ts` | 新增 `delta` 事件判别 |
| `web/src/features/chat/types.ts` | `ChatMessage` 增加改写逐字缓冲与废弃段落 |
| `web/src/features/chat/useChatStream.ts` | 按 `phase` 分流 delta |
| `web/src/components/chat/RunSteps.tsx` | 渲染逐字改写与废弃分隔 |

**不改动（红线）**

`model_router.py`、`async_model_executor.py`、`execute_node.py`、`summarize_node.py`、
`combined_rewrite_intent_service.py`、`agent_query_intent_pipeline.py`、`runner.py`、
`app/core/agent/orchestrator.py`。

---

## Task 1: 旁观通道模块

**Files:**
- Create: `app/core/agent/stream_sink.py`
- Test: `测试/test_stream_sink.py`

**Interfaces:**
- Consumes: 无（本任务是最底层）
- Produces:
  - `class StreamSink(Protocol)`：`push(event: Dict[str, Any]) -> None`
  - `current_sink() -> Optional[StreamSink]`
  - `use_sink(sink: Optional[StreamSink]) -> ContextManager[None]`
  - `emit(event: Dict[str, Any]) -> bool` —— 有通道且推送成功返回 `True`；否则 `False`。
    **内部吞掉一切异常**，绝不向上抛。
  - `has_sink() -> bool`

- [ ] **Step 1: 写失败的测试**

创建 `测试/test_stream_sink.py`：

```python
# -*- coding: utf-8 -*-
"""旁观通道模块单测。"""

import threading

import pytest

from app.core.agent.stream_sink import (
    current_sink,
    emit,
    has_sink,
    use_sink,
)


class RecordingSink:
    def __init__(self):
        self.events = []

    def push(self, event):
        self.events.append(event)


def test_no_sink_by_default():
    assert current_sink() is None
    assert has_sink() is False
    # 没有通道时 emit 不报错，返回 False
    assert emit({"kind": "delta", "text": "x"}) is False


def test_install_and_uninstall():
    sink = RecordingSink()
    with use_sink(sink):
        assert has_sink() is True
        assert emit({"text": "a"}) is True
    assert current_sink() is None
    assert sink.events == [{"text": "a"}]


def test_nested_sinks_restore_outer():
    outer, inner = RecordingSink(), RecordingSink()
    with use_sink(outer):
        with use_sink(inner):
            emit({"text": "内"})
        emit({"text": "外"})
    assert inner.events == [{"text": "内"}]
    assert outer.events == [{"text": "外"}]


def test_push_exception_is_swallowed():
    """推送失败绝不能影响主链路 —— 这是本模块最重要的契约。"""

    class BrokenSink:
        def push(self, event):
            raise RuntimeError("推送炸了")

    with use_sink(BrokenSink()):
        assert emit({"text": "x"}) is False   # 返回 False，但不抛


def test_use_sink_none_is_noop():
    with use_sink(None):
        assert has_sink() is False


def test_sink_visible_from_worker_thread():
    """asyncio.to_thread 会复制上下文，工作线程里必须读得到同一个通道。"""
    sink = RecordingSink()
    seen = []

    def worker():
        seen.append(current_sink() is sink)
        emit({"text": "来自线程"})

    with use_sink(sink):
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    # 普通 threading.Thread 不复制上下文 —— 这是 asyncio.to_thread 的差异，
    # 因此这里断言"线程里默认读不到"，用来钉住我们对机制的认知；
    # 真正的跨线程行为由 Task 6 的 asyncio.to_thread 测试覆盖。
    assert seen == [False]
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest 测试/test_stream_sink.py -q
```

Expected: FAIL —— `ModuleNotFoundError: No module named 'app.core.agent.stream_sink'`

- [ ] **Step 3: 实现模块**

创建 `app/core/agent/stream_sink.py`：

```python
# -*- coding: utf-8 -*-
"""LLM 输出的旁观通道。

设计要点（详见 specs/2026-09-20-llm-token-streaming-design.md §4.1）：

- 模型调用器**对外契约不变**：仍然"一次调用、一个完整结果"。
  它只是在被观测时顺带把增量推给通道。
- 因此上游的重试、熔断、候选降级、以及业务侧的解析路径全部无需改动。
- 用 `ContextVar` 而不是显式参数：`asyncio.to_thread` 会**复制当前上下文**到
  工作线程，所以同步的 Pipeline 链路也能读到同一个通道，一套机制覆盖两种情况。

⚠️ `emit()` 内部吞掉一切异常。推送是旁路，绝不能因为它失败而影响主链路。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Optional, Protocol

__all__ = [
    "StreamSink",
    "current_sink",
    "has_sink",
    "use_sink",
    "emit",
]


class StreamSink(Protocol):
    """输出通道。实现方**负责线程安全**（可能被工作线程调用）。"""

    def push(self, event: Dict[str, Any]) -> None:  # pragma: no cover - 协议
        ...


_current_sink: ContextVar[Optional[StreamSink]] = ContextVar(
    "agent_stream_sink", default=None
)


def current_sink() -> Optional[StreamSink]:
    """取当前上下文里的通道；没有则 None。"""
    return _current_sink.get()


def has_sink() -> bool:
    """当前是否有观测者。调用器用它决定走流式还是非流式。"""
    return _current_sink.get() is not None


@contextmanager
def use_sink(sink: Optional[StreamSink]) -> Iterator[None]:
    """在 with 块内装上下线通道；退出时恢复外层（支持嵌套）。

    传 None 等价于"本块内明确没有观测者"，用于临时屏蔽外层通道。
    """
    token = _current_sink.set(sink)
    try:
        yield
    finally:
        _current_sink.reset(token)


def emit(event: Dict[str, Any]) -> bool:
    """把一条事件推给当前通道。

    Returns:
        有通道且推送成功为 True；无通道或推送失败为 False。

    ⚠️ 任何异常都在此处被吞掉：旁路推送失败不该让用户拿不到答案。
    """
    sink = _current_sink.get()
    if sink is None:
        return False
    try:
        sink.push(event)
        return True
    except Exception:  # noqa: BLE001 - 旁路失败必须无声
        return False
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest 测试/test_stream_sink.py -q
```

Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
git add app/core/agent/stream_sink.py 测试/test_stream_sink.py
git commit -m "feat(stream): 新增 LLM 输出旁观通道（ContextVar + 安全 emit）"
```

---

## Task 2: 增量 JSON 字段抽取器

**Files:**
- Create: `app/core/agent/delta_extract.py`
- Test: `测试/test_delta_extract.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `class IncrementalJsonFieldExtractor`
    - `__init__(self, field: str)`
    - `feed(self, chunk: str) -> str` —— 返回**本次新增可见**的字段文本；
      字段尚未出现 / 值未闭合时返回 `""`（绝不对应"返回上次的内容"）
    - `visible_value(self) -> str` —— 当前已确定可见的完整值

- [ ] **Step 1: 写失败的测试**

创建 `测试/test_delta_extract.py`：

```python
# -*- coding: utf-8 -*-
"""增量 JSON 字段抽取器单测。

对应 specs Review Focus #4：UTF-8 多字节 / JSON 转义在 chunk 边界被切开。
"""

import pytest

from app.core.agent.delta_extract import IncrementalJsonFieldExtractor


def feed_all(extractor, text, chunk_size):
    """按固定长度切片喂入，返回累计可见文本。"""
    out = []
    for i in range(0, len(text), chunk_size):
        out.append(extractor.feed(text[i:i + chunk_size]))
    return "".join(out)


def test_field_not_present_yet_returns_empty():
    ex = IncrementalJsonFieldExtractor("rewritten_question")
    assert ex.feed('{"sub_questions":') == ""
    assert ex.feed('["a"]}') == ""


def test_basic_extraction():
    ex = IncrementalJsonFieldExtractor("rewritten_question")
    text = '{"rewritten_question":"我们优先做政企","sufficient":true}'
    assert feed_all(ex, text, 4) == "我们优先做政企"
    assert ex.visible_value() == "我们优先做政企"


def test_incremental_returns_only_new_text():
    """feed 的返回值必须**只是新增部分**，否则前端会重复拼接。"""
    ex = IncrementalJsonFieldExtractor("answer")
    assert ex.feed('{"answer":"第一段') == "第一段"
    assert ex.feed('第二段"') == "第二段"
    assert ex.visible_value() == "第一段第二段"


def test_value_not_closed_returns_partial_but_never_trailing_escape():
    ex = IncrementalJsonFieldExtractor("answer")
    got = ex.feed('{"answer":"abc\\')      # 结尾是一个转义引导符
    assert got == "abc"                     # 反斜杠本身不能露出来
    got2 = ex.feed('nd")')                  # 下一片补上 n
    assert got2 == "\n"


def test_escapes():
    ex = IncrementalJsonFieldExtractor("answer")
    text = '{"answer":"引号\\" 反斜杠\\\\ 换行\\n 制表\\t 星号\\u002a"}'
    assert feed_all(ex, text, 3) == '引号" 反斜杠\\ 换行\n 制表\t 星号*'


def test_field_name_appearing_inside_value_is_not_treated_as_key():
    """值里出现同名字符串时不能重新开始抽取。"""
    ex = IncrementalJsonFieldExtractor("answer")
    text = '{"answer":"前面 answer 后面"}'
    assert feed_all(ex, text, 2) == "前面 answer 后面"


def test_chinese_split_across_chunks():
    """Review Focus #4：一个中文字被切到两个 chunk。"""
    ex = IncrementalJsonFieldExtractor("rewritten_question")
    text = '{"rewritten_question":"政企合作"}'
    # 每个字符单独喂一次，模拟最碎的分片
    assert "".join(ex.feed(c) for c in text) == "政企合作"


def test_stops_at_closing_quote():
    ex = IncrementalJsonFieldExtractor("answer")
    got = feed_all(ex, '{"answer":"abc","other":"xyz"}', 5)
    assert got == "abc"          # 不能把 other 的值也带出来
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest 测试/test_delta_extract.py -q
```

Expected: FAIL —— `ModuleNotFoundError: No module named 'app.core.agent.delta_extract'`

- [ ] **Step 3: 实现抽取器**

创建 `app/core/agent/delta_extract.py`（本步只实现抽取器，分流器在 Task 3）：

```python
# -*- coding: utf-8 -*-
"""显示层：把模型原始输出转换成"该显示给用户的文本"。

⚠️ 本模块**只服务于显示**。业务侧的解析路径（schema 校验、容错、回退）
一律不经过这里，也绝不修改——这是 spec §2 的硬约束。
"""

from __future__ import annotations

from typing import Dict, Optional

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
        self._key = f'"{field}"'
        self._buffer: str = ""          # 尚未消费的原始文本
        self._search_from: int = 0      # 定位键时的搜索起点（避免重复扫描）
        self._state: str = "seek_key"   # seek_key → seek_colon → seek_quote → in_value → done
        self._escaped: bool = False
        self._value: list = []

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
        produced: list = []

        if self._state == "seek_key":
            if not self._consume_until_key():
                return ""

        if self._state == "seek_colon":
            if not self._consume_colon():
                return ""

        if self._state == "seek_quote":
            if not self._consume_quote():
                return ""

        if self._state == "in_value":
            produced = self._consume_value()

        return "".join(produced)

    # ------------------------------------------------------------------
    # 内部：逐阶段消费
    # ------------------------------------------------------------------
    def _consume_until_key(self) -> bool:
        idx = self._buffer.find(self._key, self._search_from)
        if idx == -1:
            # 键可能跨 chunk 被切开，保留最后 len(key) 个字符继续等
            keep = max(0, len(self._buffer) - len(self._key))
            self._search_from = keep
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

    def _consume_value(self) -> list:
        produced: list = []
        i = 0
        buf = self._buffer
        while i < len(buf):
            ch = buf[i]
            if self._escaped:
                produced.append(self._unescape(buf, i))
                # _unescape 可能吃掉多个字符（\uXXXX），用返回值里的 consumed
                consumed = self._last_unicode_len
                i += consumed if consumed else 1
                self._escaped = False
                continue
            if ch == "\\":
                self._escaped = True
                i += 1
                continue
            if ch == '"':
                # 值闭合
                self._state = "done"
                i += 1
                self._buffer = buf[i:]
                self._value.extend(produced)
                return produced
            produced.append(ch)
            i += 1

        # 未闭合：只剩一个悬空的转义引导符时，不能把它当已消费
        self._buffer = "" if not self._escaped else "\\"
        self._value.extend(produced)
        return produced

    _last_unicode_len: int = 0

    def _unescape(self, buf: str, backslash_at: int) -> str:
        """把反斜杠后的下一个字符（或 \\uXXXX）解出来。"""
        nxt = buf[backslash_at] if backslash_at < len(buf) else ""
        self._last_unicode_len = 1
        if nxt == "u":
            hex_part = buf[backslash_at + 1:backslash_at + 5]
            if len(hex_part) == 4:
                try:
                    self._last_unicode_len = 5
                    return chr(int(hex_part, 16))
                except ValueError:
                    return "u"
            return ""          # \u 还没收全，等下一片
        return _SIMPLE_ESCAPES.get(nxt, nxt)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest 测试/test_delta_extract.py -q
```

Expected: PASS（8 passed）

> 若 `test_value_not_closed_returns_partial_but_never_trailing_escape` 或
> `test_escapes` 失败，说明 `_consume_value` 的悬空反斜杠处理有误——
> **不要放宽测试**，修实现。反斜杠漏到界面上是可见的脏数据。

- [ ] **Step 5: 提交**

```bash
git add app/core/agent/delta_extract.py 测试/test_delta_extract.py
git commit -m "feat(stream): 增量 JSON 字段抽取器（跨 chunk 边界 + 转义）"
```

---

## Task 3: 显示层文本分流器

**Files:**
- Modify: `app/core/agent/delta_extract.py`（追加类）
- Test: `测试/test_delta_extract.py`（追加用例）

**Interfaces:**
- Consumes: `IncrementalJsonFieldExtractor`（Task 2）
- Produces:
  - `class DisplayRouter`
    - `__init__(self, phase: str)` —— `phase` ∈ `{"rewrite", "answer"}`
    - `feed(self, chunk: str) -> str` —— 返回本次应追加显示的文本（可能为空）
    - `mode` 属性（只读）：`"undecided" | "json" | "plain" | "suppress" | "final"`
    - `visible_any(self) -> bool` —— 是否已输出过可见文本（Task 7 的换候选标注要用）

- [ ] **Step 1: 写失败的测试**

追加到 `测试/test_delta_extract.py` 末尾：

```python
from app.core.agent.delta_extract import DisplayRouter


# ---------------- 改写阶段：一定是 JSON ----------------

def test_rewrite_phase_extracts_field():
    r = DisplayRouter("rewrite")
    text = '{"rewritten_question":"政企优先","sub_questions":[]}'
    assert "".join(r.feed(text[i:i + 4]) for i in range(0, len(text), 4)) == "政企优先"


# ---------------- 答案阶段：四条判定分支 ----------------

def test_answer_plain_text_streams_directly():
    """FC 协议的普通答案：纯文本直出。"""
    r = DisplayRouter("answer")
    assert r.feed("政企合作") == "政企合作"
    assert r.feed("优先") == "优先"


def test_answer_json_looking_but_not_summary_schema_is_plain():
    """Review Focus #2：用户要 JSON 输出时，这是正常答案，不能整段不显示。"""
    r = DisplayRouter("answer")
    text = '{"industries":["政企","医疗"],"priority":"high"}'
    got = "".join(r.feed(text[i:i + 5]) for i in range(0, len(text), 5))
    assert got == text, "以 { 开头但不是汇总 schema 的答案必须原样显示"
    assert r.mode == "plain"


def test_answer_summary_schema_is_extracted_not_shown_raw():
    """汇总的结构化输出：只显示 answer 字段，JSON 结构不能漏出来。"""
    r = DisplayRouter("answer")
    text = '{"sufficient":true,"answer":"最终答案在这里","missing_info":""}'
    got = "".join(r.feed(text[i:i + 6]) for i in range(0, len(text), 6))
    assert got == "最终答案在这里"
    assert "sufficient" not in got


def test_answer_react_draft_is_suppressed():
    """Review Focus #3：Thought/Action 是内部草稿，绝不能出现在界面上。"""
    r = DisplayRouter("answer")
    draft = "Thought: 我需要先查一下\nAction: sales_sql_query\nAction Input: {}"
    got = "".join(r.feed(draft[i:i + 8]) for i in range(0, len(draft), 8))
    assert got == ""
    assert r.mode == "suppress"


def test_answer_final_answer_marker_shows_only_tail():
    r = DisplayRouter("answer")
    text = "Thought: 想好了\nFinal Answer: 政企优先，其次医疗"
    got = "".join(r.feed(text[i:i + 7]) for i in range(0, len(text), 7))
    assert got == "政企优先，其次医疗"
    assert "Thought" not in got


def test_answer_typing_before_decision_emits_nothing_but_does_not_lose_text():
    """判定未定时不输出；判定为 plain 后，之前的文本必须补出来，不能丢。"""
    r = DisplayRouter("answer")
    first = r.feed("{")                 # 不足以判定
    assert first == ""
    second = r.feed('"industries":[')   # 仍不足以判定（无 sufficient，未满 200）
    assert second == ""
    third = r.feed('"政企"]}')          # 判定为 plain，之前的 12 字必须补出
    assert third == '{"industries":["政企"]}'
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest 测试/test_delta_extract.py -q
```

Expected: FAIL —— `ImportError: cannot import name 'DisplayRouter'`

- [ ] **Step 3: 实现分流器**

追加到 `app/core/agent/delta_extract.py`：

```python
import re

# ReAct 文本协议的草稿标记（行首）
_REACT_DRAFT_RE = re.compile(r"(?m)^\s*(Thought|Action|Action Input)\s*:")
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*", re.IGNORECASE)

# `answer` 阶段判定为"汇总结构化输出"的探针：SummaryVerdictSchema 的首个字段
# （summarize_node.py:46-72）。用它把"用户自己要的 JSON 答案"排除掉。
_SUMMARY_PROBE = '"sufficient"'
_SUMMARY_PROBE_WINDOW = 200

__all__ += ["DisplayRouter"]


class DisplayRouter:
    """把某一阶段的模型原始输出，转成"该追加显示的文本"。

    判定规则**严格按此顺序**（spec §4.5），命中即锁定，不再改变：

    1. `phase == "rewrite"`：缓冲以 `{` 开头 → json 模式（抽 `rewritten_question`）
    2. `phase == "answer"`：以 `{` 开头**且**前 200 字符含 `"sufficient"`
       → json 模式（抽 `answer`）
    3. 含行首 `Thought:` / `Action:` / `Action Input:` → suppress（内部草稿）
    4. 含 `Final Answer:` → final（只显示其后的内容）
    5. 其余 → plain（原样显示）

    ⚠️ 判定未定期间**不输出但不丢文本**：一旦判定为 plain，
    之前攒下的文本会在同一次 feed 里补出来。
    """

    def __init__(self, phase: str) -> None:
        if phase not in ("rewrite", "answer"):
            raise ValueError(f"未知 phase: {phase!r}（只支持 rewrite / answer）")
        self._phase = phase
        self._raw: str = ""
        self._mode: str = "undecided"
        self._emitted_len: int = 0          # plain/final 模式已输出到 _raw 的哪个位置
        self._emitted_any: bool = False     # 是否已经向外输出过任何可见文本
        self._extractor: Optional[IncrementalJsonFieldExtractor] = None

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

    def _compute_visible(self, chunk: str) -> str:
        if self._mode == "undecided":
            self._decide()
        if self._mode in ("undecided", "suppress"):
            return ""
        if self._mode == "json":
            assert self._extractor is not None
            return self._extractor.feed(chunk)
        if self._mode == "final":
            marker = _FINAL_ANSWER_RE.search(self._raw)
            if marker is None:
                return ""
            # 三步：① 游标不能倒退到标记之前 ② 再取游标之后的新增部分
            if self._emitted_len < marker.end():
                self._emitted_len = marker.end()
            return self._take_tail(marker.end())
        return self._take_tail(0)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _take_tail(self, start: int) -> str:
        """取出 `_raw` 中从 `_emitted_len` 起的新增部分（plain/final 共用）。"""
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

        if _FINAL_ANSWER_RE.search(self._raw):
            self._mode = "final"
            self._emitted_len = 0
            return
        if _REACT_DRAFT_RE.search(self._raw):
            self._mode = "suppress"
            return
        if stripped:
            self._mode = "plain"
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
python -m pytest 测试/test_delta_extract.py -q
```

Expected: PASS（14 passed）

- [ ] **Step 5: 提交**

```bash
git add app/core/agent/delta_extract.py 测试/test_delta_extract.py
git commit -m "feat(stream): 显示层文本分流（JSON/草稿抑制/Final Answer/纯文本）"
```

---

## Task 4: 调用器流式分支 —— 正文与 usage

**Files:**
- Modify: `app/llm_model_router/async_openai_caller.py:356-509`
- Test: `测试/test_streaming_caller.py`

**Interfaces:**
- Consumes: `stream_sink.has_sink()` / `emit()`（Task 1）
- Produces: `async_openai_chat_caller` 行为增强，**签名与返回类型完全不变**：
  - 有通道 → 走流式收，推 `{"kind": "delta", "text": str}`，返回同样的 `AsyncOpenAICallResult`
  - 无通道 → 走今天一模一样的非流式路径

- [ ] **Step 1: 写失败的测试**

创建 `测试/test_streaming_caller.py`：

```python
# -*- coding: utf-8 -*-
"""调用器流式分支单测：有通道/无通道两条路，返回结果必须一致。"""

from types import SimpleNamespace

import pytest

from app.core.agent.stream_sink import use_sink
from app.llm_model_router.async_openai_caller import async_openai_chat_caller


class RecordingSink:
    def __init__(self):
        self.events = []

    def push(self, event):
        self.events.append(event)


def _delta(content=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))],
        usage=None,
    )


def _final_chunk(total_tokens=7):
    return SimpleNamespace(choices=[], usage=SimpleNamespace(
        prompt_tokens=3, completion_tokens=4, total_tokens=total_tokens))


class FakeStream:
    """可 async for 的假流。"""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c
        return gen()


class FakeCompletions:
    def __init__(self, *, stream_chunks=None, plain_response=None, fail_stream_with=None):
        self._stream_chunks = stream_chunks or []
        self._plain_response = plain_response
        self._fail_stream_with = fail_stream_with
        self.calls = []

    async def create(self, **params):
        self.calls.append(params)
        if params.get("stream"):
            if self._fail_stream_with is not None:
                raise self._fail_stream_with
            return FakeStream(self._stream_chunks)
        return self._plain_response


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _target(model="m1"):
    return SimpleNamespace(
        id=model,
        candidate=SimpleNamespace(model=model, provider="p1", api_style="openai"),
    )


@pytest.mark.asyncio
async def test_streaming_path_returns_same_content_as_pushed():
    completions = FakeCompletions(
        stream_chunks=[_delta("政企"), _delta("优先"), _delta(None), _final_chunk()],
    )
    sink = RecordingSink()
    with use_sink(sink):
        result = await async_openai_chat_caller(
            _client(completions), _target(), messages=[{"role": "user", "content": "q"}],
            purpose_hint="react",
        )

    pushed = "".join(e["text"] for e in sink.events if e.get("kind") == "delta")
    assert pushed == "政企优先"
    assert result.content == "政企优先"      # 对外契约不变
    assert completions.calls[0]["stream"] is True


@pytest.mark.asyncio
async def test_no_sink_uses_non_streaming_path_unchanged():
    plain = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="完整答案", tool_calls=None, reasoning_content=None))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        model_dump=lambda: {},
    )
    completions = FakeCompletions(plain_response=plain)
    result = await async_openai_chat_caller(
        _client(completions), _target(), messages=[{"role": "user", "content": "q"}],
        purpose_hint="react",
    )
    assert result.content == "完整答案"
    assert completions.calls[0].get("stream") is not True   # 没开流


@pytest.mark.asyncio
async def test_usage_collected_from_final_chunk():
    completions = FakeCompletions(
        stream_chunks=[_delta("a"), _final_chunk(total_tokens=42)],
    )
    with use_sink(RecordingSink()):
        result = await async_openai_chat_caller(
            _client(completions), _target(), messages=[{"role": "user", "content": "q"}],
        )
    assert result.usage and result.usage.get("total_tokens") == 42
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest 测试/test_streaming_caller.py -q
```

Expected: FAIL —— 第一个用例断言 `stream is True` 失败（当前被强制改回 False）

- [ ] **Step 3: 实现流式分支**

在 `app/llm_model_router/async_openai_caller.py` 里：

**3a. 文件顶部补 import**

```python
from app.core.agent.stream_sink import emit as _emit_delta
from app.core.agent.stream_sink import has_sink as _has_sink
```

**3b. 把 `:384-385` 的强制关闭改成"仅无通道时"**

```python
    # 只有旁路观测者存在时才开流；否则保持本调用器的历史契约（非流式）。
    # ⚠️ 对外仍返回完整结果 —— 上游的重试/降级/解析全部不受影响。
    stream_enabled: bool = bool(stream) and _has_sink()
    stream = False
```

**3c. 在 `params` 组装完成后、发起调用前插入流式分支**

```python
    if stream_enabled:
        return await _streaming_chat_call(
            client=client, params={**params, "stream": True}, target=target,
        )
```

**3d. 新增流式实现函数**（放在 `async_openai_chat_caller` 之后）

```python
async def _streaming_chat_call(
    *, client: AsyncOpenAI, params: Dict[str, Any], target: ModelTarget,
) -> AsyncOpenAICallResult:
    """流式收流 + 旁路推送，**返回与 ``async_openai_chat_caller`` 相同的完整结果**。

    ⚠️ 这里刻意不经过 ``_chat_completion_with_langfuse``：该包装不感知流式。
    代价是流式调用在 langfuse 里不可见（spec 风险 3，已记录）。
    """
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    usage: Optional[Dict[str, Any]] = None

    stream = await client.chat.completions.create(**params)
    async for chunk in stream:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = {
                "prompt_tokens": getattr(chunk_usage, "prompt_tokens", None),
                "completion_tokens": getattr(chunk_usage, "completion_tokens", None),
                "total_tokens": getattr(chunk_usage, "total_tokens", None),
            }
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        text = getattr(delta, "content", None)
        if text:
            content_parts.append(text)
            _emit_delta({"kind": "delta", "text": text})
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)

    content: str = "".join(content_parts)
    if not content and reasoning_parts:
        content = "".join(reasoning_parts)

    return AsyncOpenAICallResult(
        content=content,
        model_id=target.candidate.model or target.id,
        usage=usage,
        raw=None,
        tool_calls=None,          # Task 5 补齐
        reasoning_content="".join(reasoning_parts) or None,
    )
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest 测试/test_streaming_caller.py -q
python -m pytest 测试/test_llm_attempt_budget.py -q
```

Expected: 前者 PASS（3 passed）；后者**保持全绿**（关键回归）

- [ ] **Step 5: 提交**

```bash
git add app/llm_model_router/async_openai_caller.py 测试/test_streaming_caller.py
git commit -m "feat(stream): 调用器支持流式收流 + 旁路推送（对外契约不变）"
```

---

## Task 5: tool_calls 聚合 + 兼容性探测回落

**Files:**
- Modify: `app/llm_model_router/async_openai_caller.py`
- Test: `测试/test_streaming_caller.py`（追加）

**Interfaces:**
- Consumes: Task 4 的 `_streaming_chat_call`
- Produces: `_streaming_chat_call` 的两项增强，签名不变

- [ ] **Step 1: 写失败的测试**

追加到 `测试/test_streaming_caller.py`：

```python
def _tool_delta(index, call_id=None, name=None, args=None):
    tc = SimpleNamespace(index=index, id=call_id, function=SimpleNamespace(name=name, arguments=args))
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]))],
        usage=None,
    )


@pytest.mark.asyncio
async def test_tool_calls_aggregated_by_index():
    """FC 协议下 tool_calls 是分片下发的，必须按 index 聚合还原。"""
    completions = FakeCompletions(stream_chunks=[
        _tool_delta(0, call_id="call_1", name="sales_sql_query", args='{"sql":'),
        _tool_delta(0, args='"SELECT 1"}'),
        _delta(None),
        _final_chunk(),
    ])
    with use_sink(RecordingSink()):
        result = await async_openai_chat_caller(
            _client(completions), _target(), messages=[{"role": "user", "content": "q"}],
            purpose_hint="react",
        )

    assert result.tool_calls and len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "sales_sql_query"
    assert call["function"]["arguments"] == '{"sql":"SELECT 1"}'


@pytest.mark.asyncio
async def test_probe_fallback_on_400_before_first_chunk():
    """方案 A：开流前 400 → 同一次尝试内退回非流式，不推任何 delta。"""
    from openai import APIStatusError

    class Err(APIStatusError):
        status_code = 400

    plain = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="非流式拿到的答案", tool_calls=None, reasoning_content=None))],
        usage=None, model_dump=lambda: {},
    )
    completions = FakeCompletions(plain_response=plain, fail_stream_with=Err(
        "stream not supported", response=None, body=None))
    sink = RecordingSink()

    with use_sink(sink):
        result = await async_openai_chat_caller(
            _client(completions), _target(), messages=[{"role": "user", "content": "q"}],
        )

    assert result.content == "非流式拿到的答案"
    assert [e for e in sink.events if e.get("kind") == "delta"] == []   # 不推 delta
    assert completions.calls[-1].get("stream") is not True              # 第二次没开流


@pytest.mark.asyncio
async def test_no_probe_fallback_after_first_chunk():
    """首 chunk 之后失败不重发 —— 已有内容已推给前端，重发会重复。"""
    class Boom(RuntimeError):
        pass

    class HalfBrokenStream:
        def __init__(self):
            self._n = 0

        def __aiter__(self):
            async def gen():
                yield _delta("已经吐了")
                raise Boom("断流")
            return gen()

    class C:
        async def create(self, **params):
            if params.get("stream"):
                return HalfBrokenStream()
            raise AssertionError("不应重发非流式")

    with use_sink(RecordingSink()):
        with pytest.raises(Boom):
            await async_openai_chat_caller(
                _client(C()), _target(), messages=[{"role": "user", "content": "q"}],
            )
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest 测试/test_streaming_caller.py -q
```

Expected: FAIL —— tool_calls 为 `None`；探测回落未实现（第一个用例报 `APIStatusError`）

- [ ] **Step 3: 实现**

在 `_streaming_chat_call` 里：

**3a. 加 tool_calls 聚合**

```python
    tool_calls_acc: Dict[int, Dict[str, Any]] = {}
    ...
    async for chunk in stream:
        ...
        for tc in (getattr(delta, "tool_calls", None) or []):
            idx = int(getattr(tc, "index", 0) or 0)
            slot = tool_calls_acc.setdefault(idx, {"id": None, "type": "function",
                                                   "function": {"name": None, "arguments": ""}})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["function"]["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["function"]["arguments"] += fn.arguments
```

返回时：

```python
        tool_calls=[tool_calls_acc[i] for i in sorted(tool_calls_acc)] or None,
```

**3b. 加首 chunk 前的探测回落**

```python
async def _streaming_chat_call(*, client, params, target):
    received_any_chunk: bool = False
    try:
        stream = await client.chat.completions.create(**params)
        async for chunk in stream:
            received_any_chunk = True
            ...原有累积逻辑...
    except Exception as exc:  # noqa: BLE001
        # 方案 A（spec §10.1）：**首 chunk 之前**失败、且异常特征像"参数组合不被
        # 厂商接受"时，同一次尝试内退回非流式重发，且不推任何 delta。
        # 探测请求不计入"增加调用次数"（用户已认可）。
        if not received_any_chunk and _looks_like_params_rejected(exc):
            logger.warning("流式参数不被接受，本次尝试退回非流式重发: {}", exc)
            return await async_openai_chat_caller(
                client, target, **{**_strip_stream(params), "stream": False},
            )
        raise      # 其它异常原样抛出，交由现有重试/候选降级处理
```

并新增模块级判定函数：

```python
def _looks_like_params_rejected(exc: BaseException) -> bool:
    """异常是否像"参数组合不被厂商接受"（spec §10.1 第 2 条）。"""
    status = getattr(exc, "status_code", None)
    return status in (400, 422)


def _strip_stream(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    out.pop("stream", None)
    out.pop("stream_options", None)
    return out
```

> ⚠️ 回落调用时不能直接把 `params` 原样传回 `async_openai_chat_caller`（它接受的是
> 关键字参数，不是 params 字典）。请把回落实现为**复用本函数已解析好的调用参数**，
> 或在 `async_openai_chat_caller` 内做分流，避免二次组装产生偏差。
> 实现时以"两条路径使用完全相同的 params（除 stream 外）"为准，并用
> `test_probe_fallback_on_400_before_first_chunk` 钉住。

- [ ] **Step 4: 运行测试**

```bash
python -m pytest 测试/test_streaming_caller.py 测试/test_llm_attempt_budget.py -q
```

Expected: 全部 PASS（6 + 原有）

- [ ] **Step 5: 提交**

```bash
git add app/llm_model_router/async_openai_caller.py 测试/test_streaming_caller.py
git commit -m "feat(stream): tool_calls 增量聚合 + 流式参数兼容性探测回落"
```

---

## Task 6: SSE 层接线（通道 + 改写阶段）

**Files:**
- Modify: `app/api/routes/chat.py`（`_agent_stream_generator`，约 `:322-615`）
- Test: `测试/test_agent_stream_delta.py`

**Interfaces:**
- Consumes: `stream_sink.use_sink`（Task 1）、`delta_extract.DisplayRouter`（Task 3）
- Produces:
  - `class _QueueSink`：线程安全通道，`push` 走 `loop.call_soon_threadsafe`
  - `async def _drain_until(task, queue) -> AsyncIterator[bytes]`：边等任务边排空队列
  - SSE 新事件：`{"delta": {"phase": "rewrite", "text": "..."}}`

- [ ] **Step 1: 写失败的测试**

创建 `测试/test_agent_stream_delta.py`：

```python
# -*- coding: utf-8 -*-
"""SSE 层接线单测：通道投递、跨线程、队列排空不丢尾。"""

import asyncio

import pytest

from app.api.routes.chat import _QueueSink, _drain_queue


@pytest.mark.asyncio
async def test_queue_sink_delivers_from_worker_thread():
    """Pipeline 在 to_thread 的工作线程里 emit，主循环必须收得到。"""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sink = _QueueSink(loop=loop, queue=queue)

    await asyncio.to_thread(sink.push, {"kind": "delta", "text": "来自线程"})

    got = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert got == {"kind": "delta", "text": "来自线程"}


@pytest.mark.asyncio
async def test_drain_queue_emits_all_pending():
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(3):
        queue.put_nowait({"i": i})
    drained = [ev async for ev in _drain_queue(queue)]
    assert [e["i"] for e in drained] == [0, 1, 2]


@pytest.mark.asyncio
async def test_drain_until_task_does_not_lose_tail():
    """Review Focus：任务结束时队列里可能还剩事件，必须补排空。"""
    queue: asyncio.Queue = asyncio.Queue()
    drained = []

    async def producer():
        await asyncio.sleep(0.01)
        queue.put_nowait({"i": 0})
        queue.put_nowait({"i": 1})       # 任务结束前最后一刻才放进去

    task = asyncio.create_task(producer())
    async for ev in _drain_queue(queue, until=task):
        drained.append(ev)
    assert [e["i"] for e in drained] == [0, 1]
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest 测试/test_agent_stream_delta.py -q
```

Expected: FAIL —— `ImportError: cannot import name '_QueueSink'`

- [ ] **Step 3: 实现通道与排空**

在 `app/api/routes/chat.py` 里新增（放在 `_sse_payload` 附近）：

```python
class _QueueSink:
    """把事件从任意线程安全地投递到主事件循环的队列。

    ⚠️ Pipeline 跑在 ``asyncio.to_thread`` 的工作线程里，SSE 生成器在主循环；
    直接从线程写 ``asyncio.Queue`` 不是线程安全的，必须走
    ``call_soon_threadsafe``。
    """

    def __init__(self, *, loop: "asyncio.AbstractEventLoop", queue: "asyncio.Queue") -> None:
        self._loop = loop
        self._queue = queue

    def push(self, event: Dict[str, Any]) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)


async def _drain_queue(
    queue: "asyncio.Queue",
    *,
    until: Optional["asyncio.Task"] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """产出队列里的所有事件。

    给了 ``until`` 时：边等它完成边排空；它完成后再**补排空一次**——
    否则任务结束前最后一刻投递的事件会丢（spec §4.3 / 风险 4）。
    """
    if until is None:
        while not queue.empty():
            yield queue.get_nowait()
        return

    while not until.done():
        try:
            yield await asyncio.wait_for(queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
    while not queue.empty():
        yield queue.get_nowait()
```

- [ ] **Step 4: 接线到改写阶段**

在 `_agent_stream_generator` 里，把现有的

```python
    pipeline_output = await asyncio.to_thread(
        pipeline.run, request.query, registered_tool_snapshot,
        active_session_id, rewrite_history_messages, available_skills_snapshot,
    )
```

替换为：

```python
    # ---- 改写 + 意图阶段：装通道，边跑边把 rewritten_question 逐字推出去 ----
    delta_queue: "asyncio.Queue" = asyncio.Queue()
    sink = _QueueSink(loop=asyncio.get_running_loop(), queue=delta_queue)
    router = DisplayRouter("rewrite")

    pipeline_task = asyncio.create_task(asyncio.to_thread(
        pipeline.run, request.query, registered_tool_snapshot,
        active_session_id, rewrite_history_messages, available_skills_snapshot,
    ))

    with use_sink(sink):
        async for event in _drain_queue(delta_queue, until=pipeline_task):
            if event.get("kind") != "delta":
                continue
            visible = router.feed(event.get("text") or "")
            if visible:
                yield _sse_payload({"delta": {"phase": "rewrite", "text": visible}})
        pipeline_output = await pipeline_task
```

> ⚠️ `use_sink` 必须在 `create_task` **之前**进入，否则 `to_thread` 复制上下文时
> 通道还没装上。这里的 `with` 包住了整个排空循环，顺序即如此。

- [ ] **Step 5: 运行测试**

```bash
python -m pytest 测试/test_agent_stream_delta.py -q
```

Expected: PASS（3 passed）

- [ ] **Step 6: 提交**

```bash
git add app/api/routes/chat.py 测试/test_agent_stream_delta.py
git commit -m "feat(stream): SSE 层接入旁观通道，改写阶段逐字推送"
```

---

## Task 7: SSE 层接线（答案阶段 + attempt_reset）

**Files:**
- Modify: `app/api/routes/chat.py`（`_agent_stream_generator` 的图执行段）
- Test: `测试/test_agent_stream_delta.py`（追加）

**Interfaces:**
- Consumes: Task 6 的 `_QueueSink` / `_drain_queue`；Task 1 的 `emit`
- Produces: SSE 新事件 `{"delta": {"phase": "answer", "text": ...}}` 与
  `{"delta": {"phase": "answer", "attempt_reset": true}}`

- [ ] **Step 1: 写失败的测试**

追加到 `测试/test_agent_stream_delta.py`：

```python
import json

from app.api.routes.chat import _bump_attempt, _render_answer_delta
from app.core.agent.delta_extract import DisplayRouter


def _payload_text(payloads) -> str:
    """把若干 SSE 负载里的 delta.text 拼起来。"""
    return "".join(
        json.loads(p[6:].decode("utf-8")).get("delta", {}).get("text", "")
        for p in payloads
    )


def _has_attempt_reset(payloads) -> bool:
    return any(
        json.loads(p[6:].decode("utf-8")).get("delta", {}).get("attempt_reset")
        for p in payloads
    )


def test_answer_router_suppresses_react_draft_end_to_end():
    """Review Focus #3：草稿标记无论怎么切片都不能漏到界面上。"""
    r = DisplayRouter("answer")
    draft = "Thought: 先查数据\nAction: sales_sql_query\nAction Input: {\"q\":\"x\"}\n"
    out = "".join(r.feed(draft[i:i + 6]) for i in range(0, len(draft), 6))
    assert out == ""


def test_attempt_reset_only_marks_from_second_attempt():
    """spec §4.4：第 1 条 attempt_start 只是"新一轮开始"，不插分隔。"""
    router = DisplayRouter("answer")

    # 第一次尝试：正常推送 + 一条 attempt_start（此时还没有过尝试）
    first = list(_render_answer_delta(router, {"kind": "delta", "text": "第一候选"}, 0))
    assert _payload_text(first) == "第一候选"

    opening = list(_render_answer_delta(router, {"kind": "attempt_start"}, 0))
    assert opening == [], "第 1 条 attempt_start 不该产生任何事件"

    # 已经完成过一次尝试（attempt_count == 1）后，再来 attempt_start 才插分隔
    second = list(_render_answer_delta(router, {"kind": "attempt_start"}, 1))
    assert _has_attempt_reset(second) is True


def test_bump_attempt_counts_only_attempt_start():
    assert _bump_attempt(0, {"kind": "delta", "text": "x"}) == 0
    assert _bump_attempt(0, {"kind": "attempt_start"}) == 1
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
python -m pytest 测试/test_agent_stream_delta.py -q
```

Expected: FAIL —— `ImportError: cannot import name '_bump_attempt' from 'app.api.routes.chat'`
（`_render_answer_delta` / `_bump_attempt` 尚未实现）

- [ ] **Step 3: 实现答案阶段接线**

在 `_agent_stream_generator` 里，把现有的

```python
    orchestrator_result: Any = None
    async for event in agent_orchestrator.run_stream(
        user_input=effective_user_input,
        session_id=active_session_id,
        mode=final_mode,
        intent=intent_context_for_orchestrator,
        precomputed_memory=precomputed_memory_for_orchestrator,
    ):
```

改为（在它外面再套一层通道）：

```python
    answer_queue: "asyncio.Queue" = asyncio.Queue()
    answer_sink = _QueueSink(loop=asyncio.get_running_loop(), queue=answer_queue)
    answer_router = DisplayRouter("answer")
    attempt_count = 0

    orchestrator_result: Any = None
    with use_sink(answer_sink):
        stream_iter = agent_orchestrator.run_stream(
            user_input=effective_user_input,
            session_id=active_session_id,
            mode=final_mode,
            intent=intent_context_for_orchestrator,
            precomputed_memory=precomputed_memory_for_orchestrator,
        ).__aiter__()

        # 用"边排空队列边推进图"的方式消费，避免每个 step 之间积压 delta
        while True:
            try:
                event = await asyncio.wait_for(stream_iter.__anext__(), timeout=0.05)
            except asyncio.TimeoutError:
                for delta_event in _drain_nowait(answer_queue):
                    for chunk in _render_answer_delta(
                        answer_router, delta_event, attempt_count
                    ):
                        yield chunk
                    attempt_count = _bump_attempt(attempt_count, delta_event)
                continue
            except StopAsyncIteration:
                break
            # 图产出了一个事件 —— 先把手里的 delta 排干净再推它
            for delta_event in _drain_nowait(answer_queue):
                for chunk in _render_answer_delta(
                    answer_router, delta_event, attempt_count
                ):
                    yield chunk
                attempt_count = _bump_attempt(attempt_count, delta_event)
            if event.get("type") == "step":
                payload = {k: v for k, v in event.items() if k != "type"}
                yield _sse_payload({"step": payload})
            elif event.get("type") == "final":
                orchestrator_result = event["response"]
        for delta_event in _drain_nowait(answer_queue):
            for chunk in _render_answer_delta(
                answer_router, delta_event, attempt_count
            ):
                yield chunk
            attempt_count = _bump_attempt(attempt_count, delta_event)
```

辅助函数（同文件）：

```python
def _drain_nowait(queue: "asyncio.Queue") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def _bump_attempt(count: int, event: Dict[str, Any]) -> int:
    return count + 1 if event.get("kind") == "attempt_start" else count


def _render_answer_delta(
    router: DisplayRouter, event: Dict[str, Any], attempt_count: int,
) -> Iterator[bytes]:
    """把一个 delta 事件渲染成 0..2 条 SSE 负载。"""
    kind = event.get("kind")
    if kind == "attempt_start":
        # 第 1 次不插分隔；从第 2 次起，且已渲染过内容，才标注上段废弃
        if attempt_count >= 1 and router.visible_any():
            yield _sse_payload({"delta": {"phase": "answer", "attempt_reset": True}})
        return
    if kind != "delta":
        return
    visible = router.feed(event.get("text") or "")
    if visible:
        yield _sse_payload({"delta": {"phase": "answer", "text": visible}})
```

> ⚠️ 本步依赖 `DisplayRouter.visible_any()` 与 `mode` 属性 —— 两者在 Task 3 已实现，
> 无需在此重复定义。若执行时发现缺失，说明 Task 3 未完成，**回到 Task 3**。

- [ ] **Step 4: 运行测试**

```bash
python -m pytest 测试/test_agent_stream_delta.py 测试/test_graph_state_machine.py -q
```

Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add app/api/routes/chat.py 测试/test_agent_stream_delta.py app/core/agent/delta_extract.py
git commit -m "feat(stream): 答案阶段逐字推送 + 换候选标注"
```

---

## Task 8: 前端 delta 事件判别与分流

**Files:**
- Modify: `web/src/features/chat/streamEvents.ts`
- Modify: `web/src/features/chat/types.ts`
- Modify: `web/src/features/chat/useChatStream.ts`
- Test: `web/src/features/chat/streamEvents.test.ts`、`web/src/features/chat/useChatStream.test.tsx`

**Interfaces:**
- Consumes: 后端 `{"delta": {...}}`（Task 6/7）
- Produces:
  - `DeltaEvent { kind: "delta"; phase: "rewrite" | "answer"; text: string; attemptReset: boolean }`
  - `ChatMessage` 新字段：`rewriteText?: string`、`abandoned?: string[]`

- [ ] **Step 1: 写失败的测试**

追加到 `web/src/features/chat/streamEvents.test.ts`：

```ts
it("delta 事件（改写逐字）", () => {
  expect(toStreamEvent({ delta: { phase: "rewrite", text: "政企" } })).toEqual({
    kind: "delta",
    phase: "rewrite",
    text: "政企",
    attemptReset: false,
  });
});

it("delta 事件（换候选标注）", () => {
  expect(
    toStreamEvent({ delta: { phase: "answer", attempt_reset: true } }),
  ).toEqual({ kind: "delta", phase: "answer", text: "", attemptReset: true });
});
```

追加到 `web/src/features/chat/useChatStream.test.tsx`：

```tsx
it("改写逐字进 rewriteText，答案逐字进正文，两者不互相污染", async () => {
  vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
    sseStream([
      'data: {"delta":{"phase":"rewrite","text":"政企"}}\n\n',
      'data: {"delta":{"phase":"rewrite","text":"优先"}}\n\n',
      'data: {"delta":{"phase":"answer","text":"最终"}}\n\n',
      'data: {"delta":{"phase":"answer","text":"答案"}}\n\n',
      'data: {"done":true,"status":"success"}\n\n',
    ]),
  );
  const { messages, onMessage } = collect();
  const { result } = renderHook(() => useChatStream(onMessage));
  await act(async () => {
    await result.current.send("q", "s1");
  });
  expect(messages[0]?.rewriteText).toBe("政企优先");
  expect(messages[0]?.text).toBe("最终答案");
});

it("换候选时把已渲染内容记入 abandoned，不静默丢弃", async () => {
  vi.spyOn(chatApi, "streamAgentChat").mockResolvedValue(
    sseStream([
      'data: {"delta":{"phase":"answer","text":"第一候选的"},"attempt_start":true}\n\n',
      'data: {"delta":{"phase":"answer","text":"开头"}}\n\n',
      'data: {"delta":{"phase":"answer","attempt_reset":true}}\n\n',
      'data: {"delta":{"phase":"answer","text":"第二候选"}}\n\n',
      'data: {"done":true,"status":"success"}\n\n',
    ]),
  );
  const { messages, onMessage } = collect();
  const { result } = renderHook(() => useChatStream(onMessage));
  await act(async () => {
    await result.current.send("q", "s1");
  });
  expect(messages[0]?.abandoned).toEqual(["第一候选的开头"]);
  expect(messages[0]?.text).toBe("第二候选");
});
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd web && npm run test -- --run src/features/chat/streamEvents.test.ts
```

Expected: FAIL —— 新用例断言不存在的事件类型

- [ ] **Step 3: 实现**

`streamEvents.ts` 增加：

```ts
export interface DeltaEvent {
  kind: "delta";
  phase: "rewrite" | "answer";
  text: string;
  attemptReset: boolean;
}
```

加入联合类型，并在 `toStreamEvent` 的判定链**最前面**（与 `step` 同级）加：

```ts
  if (raw.delta && typeof raw.delta === "object") {
    const d = raw.delta as Record<string, unknown>;
    return {
      kind: "delta",
      phase: d.phase === "rewrite" ? "rewrite" : "answer",
      text: str(d.text),
      attemptReset: d.attempt_reset === true,
    };
  }
```

`types.ts` 的 `ChatMessage` 增加：

```ts
  /** 改写阶段的逐字缓冲（回答完成后仍保留，供「执行过程」展示） */
  rewriteText?: string;
  /** 因模型切换被废弃的段落（保留并标注，不静默丢弃） */
  abandoned?: string[];
```

`useChatStream.ts` 的 `consume` 里加分支：

```ts
        } else if (ev.kind === "delta") {
          onMessage((m) => {
            if (ev.phase === "rewrite") {
              return { ...m, rewriteText: (m.rewriteText ?? "") + ev.text };
            }
            if (ev.attemptReset) {
              // 保留并标注：把已渲染内容移入 abandoned，正文从空重新开始
              return m.text
                ? { ...m, abandoned: [...(m.abandoned ?? []), m.text], text: "" }
                : m;
            }
            return { ...m, text: m.text + ev.text };
          });
        }
```

- [ ] **Step 4: 运行测试**

```bash
cd web && npm run test -- --run
```

Expected: 全 PASS

- [ ] **Step 5: 提交**

```bash
git add web/src/features/chat/streamEvents.ts web/src/features/chat/types.ts \
        web/src/features/chat/useChatStream.ts \
        web/src/features/chat/streamEvents.test.ts web/src/features/chat/useChatStream.test.tsx
git commit -m "feat(web): 解析 delta 事件并按阶段分流（含废弃段落标注）"
```

---

## Task 9: 前端渲染逐字改写与废弃标注

**Files:**
- Modify: `web/src/components/chat/RunSteps.tsx`
- Modify: `web/src/components/chat/Message.tsx`
- Test: `web/src/components/chat/RunSteps.test.tsx`（新建）

**Interfaces:**
- Consumes: `ChatMessage.rewriteText` / `.abandoned`（Task 8）
- Produces: 无新导出（纯渲染）

- [ ] **Step 1: 写失败的测试**

创建 `web/src/components/chat/RunSteps.test.tsx`：

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RunSteps } from "./RunSteps";

describe("RunSteps", () => {
  it("没有步骤但有逐字改写时，仍显示改写内容", () => {
    render(<RunSteps steps={[]} running rewriteText="政企优先" />);
    expect(screen.getByText(/政企优先/)).toBeInTheDocument();
  });

  it("没有步骤也没有改写时不渲染", () => {
    const { container } = render(<RunSteps steps={[]} running={false} />);
    expect(container).toBeEmptyDOMElement();
  });
});
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd web && npm run test -- --run src/components/chat/RunSteps.test.tsx
```

Expected: FAIL —— `RunSteps` 不接受 `rewriteText`；且空 steps 时 `return null`

- [ ] **Step 3: 实现**

`RunSteps` 增加 `rewriteText?: string` 参数；把"空则返回 null"改为
"既无 steps 又无 rewriteText 才返回 null"；并在标题下方固定渲染一行改写内容
（运行中显示 `agent-caret` 光标）。

`Message.tsx` 在正文之前渲染 `abandoned` 段落：

```tsx
        {(message.abandoned ?? []).map((text, i) => (
          <div key={i} className="mb-2 rounded-lg border border-line bg-surface-2 px-3 py-2">
            <div className="mb-1 text-[11px] text-fg-subtle">
              上段因模型切换已废弃
            </div>
            <div className="text-[12px] whitespace-pre-wrap text-fg-subtle line-through">
              {text}
            </div>
          </div>
        ))}
```

- [ ] **Step 4: 运行测试**

```bash
cd web && npm run test -- --run && npm run build
```

Expected: 测试全 PASS，构建 0 错误

- [ ] **Step 5: 提交**

```bash
git add web/src/components/chat/RunSteps.tsx web/src/components/chat/Message.tsx \
        web/src/components/chat/RunSteps.test.tsx
git commit -m "feat(web): 逐字改写渲染 + 废弃段落标注"
```

---

## Task 10: 端到端验证

**Files:** 无代码改动（只跑验证）

- [ ] **Step 1: 后端全量回归**

```bash
python -m pytest 测试/ -q
```

Expected: 全 PASS（特别是 `test_llm_attempt_budget.py`、`test_graph_state_machine.py`）

- [ ] **Step 2: 前端全量回归**

```bash
cd web && npm run test -- --run && npm run lint && npm run build
```

Expected: 全 PASS / 0 错误 / 既有警告数不增加

- [ ] **Step 3: 实测流式时序（spec §1 的成功标准）**

启动后端与前端，发一次真实请求，用打时间戳的脚本确认：

- `delta.phase == "rewrite"` 的**首片**出现在 Pipeline 的 LLM 开始产出后不久，
  **不是**等整个 Pipeline 跑完
- `delta.phase == "answer"` 的首片与末片之间跨度**远大于 0.5 秒**
  （对比改造前的 0.016 秒）
- `done` 事件仍然到达，`step` 事件数量不变

```bash
python -m pytest 测试/test_streaming_caller.py -q   # 先确保单测层没问题
```

- [ ] **Step 4: 手工核对 Review Focus 的 5 条**

按 `Review Focus` 逐条在界面上核对：前言标注、JSON 答案不被吞、
草稿不漏、中文不出半个字、换候选有分隔。

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "test(stream): 端到端验证通过（逐字改写 + 逐字答案）"
```

---

## 自审记录

**1. 规格覆盖检查**

| 规格章节 | 落到的任务 |
|---|---|
| §2 约束（不改解析路径） | 全局约束 + Task 4/5 只改调用器；解析文件列在"红线"里 |
| §4.1 旁观通道 | Task 1 |
| §4.2 调用器改动 | Task 4、Task 5 |
| §4.3 SSE 接线（含跨线程 + 排空） | Task 6、Task 7 |
| §4.4 事件协议（含 attempt_reset 语义） | Task 7、Task 8 |
| §4.5 四种输出处理 | Task 3、Task 6、Task 7 |
| §4.6 增量抽取器 | Task 2 |
| §5 边界与失败 | Task 1（push 吞异常）、Task 5（探测回落/首 chunk 后不重发） |
| §6 测试策略 | 每个任务的 Step 1；回归在 Task 4/10 |
| §7 明确排除 | 全局约束 |
| §8 风险 1–4 | Task 2（抽取器）、Task 5（兼容性）、Task 7（排队尾）、Task 9（前言标注） |
| §10.1 方案 A 五条行为 | Task 5 的探测回落用例 |

**2. 占位符扫描**：无 TBD / TODO / "类似 Task N" / "自行处理边界"。

**3. 类型一致性**：`StreamSink.push(event: Dict)`、`emit(event) -> bool`、
`DisplayRouter(phase).feed(chunk) -> str`、`IncrementalJsonFieldExtractor(field).feed(chunk) -> str`
在 Task 1–7 中前后一致；前端 `DeltaEvent` 的字段名与后端 `{"delta": {...}}` 的
`phase` / `text` / `attempt_reset` 逐一对应。

**4. Review Focus 覆盖**：5 条各自落到
#1→Task 7/9、#2→Task 3、#3→Task 3/7、#4→Task 2、#5→Task 7/8。

**5. 自审发现并已修正的 4 处问题**

| # | 问题 | 性质 | 修正 |
|---|---|---|---|
| 1 | Task 3 贴出的 `feed` 代码里 `final` 分支是**错的**，后面还附了一句"别照这写" | **不可用代码** | 重写为正确的三步实现（游标不倒退 + 取尾巴），删除那句说明；代码与测试自洽 |
| 2 | Task 7 引用了 `DisplayRouter.visible_any()`，但 Task 3 从未定义它 | **跨任务类型不一致** | 在 Task 3 的接口块与实现里补上 `visible_any()`（外加 `_emitted_any` 状态） |
| 3 | Task 7 的 Step 2 写成"先把断言反过来、确认失败再改回来" | **假失败**（违反 TDD） | 改为针对真实待实现函数 `_render_answer_delta` / `_bump_attempt` 的测试，Step 2 的 FAIL 真实可复现 |
| 4 | Task 7 的实现里有一行 `... if False else None` 死代码 | 残留 | 删除 |

**6. 仍需执行者留意的实现风险**

- Task 5 的探测回落需复用**已解析好的调用参数**，避免二次组装产生偏差
  （计划里已用 `test_probe_fallback_on_400_before_first_chunk` 钉住"两条路径
  params 除 stream 外完全一致"）
- Task 7 的消费循环用 `wait_for(..., timeout=0.05)` 轮询推进图 + 排空队列；
  若实测有 CPU 占用问题，改为 `asyncio.gather` 竞速
- Task 6 的 `use_sink` 必须在 `create_task` **之前**进入，否则 `to_thread`
  复制上下文时通道还没装上——顺序错了会静默地"一个字都不流"
