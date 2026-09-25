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
        choices=[
            SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))
        ],
        usage=None,
    )


def _final_chunk(total_tokens=7):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=3, completion_tokens=4, total_tokens=total_tokens
        ),
    )


class FakeStream:
    """可 async for 的假流。"""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

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
        candidate=SimpleNamespace(
            id=model,
            model=model,
            provider="p1",
            api_style="openai",
            json_schema_supported=lambda: True,
            json_object_supported=lambda: True,
        ),
    )


@pytest.mark.asyncio
async def test_streaming_path_returns_same_content_as_pushed():
    completions = FakeCompletions(
        stream_chunks=[_delta("政企"), _delta("优先"), _delta(None), _final_chunk()],
    )
    sink = RecordingSink()
    with use_sink(sink):
        result = await async_openai_chat_caller(
            _client(completions),
            _target(),
            messages=[{"role": "user", "content": "q"}],
            purpose_hint="react",
        )

    pushed = "".join(e["text"] for e in sink.events if e.get("kind") == "delta")
    assert pushed == "政企优先"
    assert result.content == "政企优先"  # 对外契约不变
    assert completions.calls[0]["stream"] is True


@pytest.mark.asyncio
async def test_emits_attempt_start_before_first_delta():
    """每次流式尝试开流前先发一条 attempt_start（且恰好一条、排在首个 delta 前）。

    没有它，SSE 层的 attempt_count 恒为 0，换候选/重试时"上段废弃"分隔永不出现，
    DisplayRouter 也不会复位（旧尝试的半截文本污染新尝试）。
    """
    completions = FakeCompletions(
        stream_chunks=[_delta("政企"), _delta("优先"), _final_chunk()],
    )
    sink = RecordingSink()
    with use_sink(sink):
        await async_openai_chat_caller(
            _client(completions),
            _target(),
            messages=[{"role": "user", "content": "q"}],
            purpose_hint="react",
        )
    kinds = [e.get("kind") for e in sink.events]
    assert kinds[0] == "attempt_start", kinds
    assert kinds.count("attempt_start") == 1
    assert "".join(e.get("text", "") for e in sink.events if e.get("kind") == "delta") == "政企优先"


@pytest.mark.asyncio
async def test_no_sink_uses_non_streaming_path_unchanged():
    plain = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="完整答案", tool_calls=None, reasoning_content=None
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        model_dump=lambda: {},
    )
    completions = FakeCompletions(plain_response=plain)
    result = await async_openai_chat_caller(
        _client(completions),
        _target(),
        messages=[{"role": "user", "content": "q"}],
        purpose_hint="react",
    )
    assert result.content == "完整答案"
    assert completions.calls[0].get("stream") is not True  # 没开流


@pytest.mark.asyncio
async def test_usage_collected_from_final_chunk():
    completions = FakeCompletions(
        stream_chunks=[_delta("a"), _final_chunk(total_tokens=42)],
    )
    with use_sink(RecordingSink()):
        result = await async_openai_chat_caller(
            _client(completions),
            _target(),
            messages=[{"role": "user", "content": "q"}],
        )
    assert result.usage and result.usage.get("total_tokens") == 42


# ---------------------------------------------------------------------------
# Task 5：tool_calls 聚合 + 兼容性探测回落
# ---------------------------------------------------------------------------


def _tool_delta(index, call_id=None, name=None, args=None):
    tc = SimpleNamespace(
        index=index, id=call_id, function=SimpleNamespace(name=name, arguments=args)
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[tc]))],
        usage=None,
    )


class _ParamsRejected(Exception):
    """模拟"参数组合不被厂商接受"（400/422）。"""

    status_code = 400


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
            _client(completions),
            _target(),
            messages=[{"role": "user", "content": "q"}],
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
    plain = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="非流式拿到的答案", tool_calls=None, reasoning_content=None
                )
            )
        ],
        usage=None,
        model_dump=lambda: {},
    )
    completions = FakeCompletions(
        plain_response=plain, fail_stream_with=_ParamsRejected("stream not supported")
    )
    sink = RecordingSink()

    with use_sink(sink):
        result = await async_openai_chat_caller(
            _client(completions),
            _target(),
            messages=[{"role": "user", "content": "q"}],
        )

    assert result.content == "非流式拿到的答案"
    assert [e for e in sink.events if e.get("kind") == "delta"] == []  # 不推 delta
    assert completions.calls[-1].get("stream") is not True  # 第二次没开流


# ---------------------------------------------------------------------------
# show_structural 策略：答案阶段内部结构化（response_format）调用不进通道
# ---------------------------------------------------------------------------
def _plain(content="结构化JSON"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, tool_calls=None, reasoning_content=None
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        model_dump=lambda: {},
    )


@pytest.mark.asyncio
async def test_structural_call_non_streaming_when_policy_hides_it():
    """答案阶段（show_structural=False）：planner/distill 的 JSON 调用既不推
    delta 也不发 attempt_start——这是"成功计划被误划废弃 + 原始 JSON 上屏"
    事故（2026-09-23）的回归钉。"""
    completions = FakeCompletions(
        stream_chunks=[_delta('{"subtasks": []}'), _final_chunk()],
        plain_response=_plain(),
    )
    sink = RecordingSink()
    with use_sink(sink, show_structural=False):
        result = await async_openai_chat_caller(
            _client(completions),
            _target(),
            messages=[{"role": "user", "content": "q"}],
            response_format={"type": "json_object"},
        )

    assert sink.events == []  # 一个事件都没有：不流、不计数尝试
    assert completions.calls[0].get("stream") is not True  # 走的非流式
    assert result.content == "结构化JSON"  # 对外契约不变


@pytest.mark.asyncio
async def test_structural_call_still_streams_with_default_policy():
    """改写阶段默认策略：结构化 JSON 照常流（增量抽取 rewrite 字段依赖它）。"""
    completions = FakeCompletions(
        stream_chunks=[_delta('{"rewrite":'), _delta('"改写后"}'), _final_chunk()],
    )
    sink = RecordingSink()
    with use_sink(sink):  # 默认 show_structural=True
        await async_openai_chat_caller(
            _client(completions),
            _target(),
            messages=[{"role": "user", "content": "q"}],
            response_format={"type": "json_object"},
        )
    kinds = [e.get("kind") for e in sink.events]
    assert kinds[0] == "attempt_start"
    assert "".join(e.get("text", "") for e in sink.events if e["kind"] == "delta") == (
        '{"rewrite":"改写后"}'
    )


@pytest.mark.asyncio
async def test_free_text_call_still_streams_under_hide_structural_policy():
    """策略只屏蔽结构化调用：自由文本（react 推理/最终答复）在答案阶段照常流。"""
    completions = FakeCompletions(
        stream_chunks=[_delta("政企"), _delta("优先"), _final_chunk()],
    )
    sink = RecordingSink()
    with use_sink(sink, show_structural=False):
        result = await async_openai_chat_caller(
            _client(completions),
            _target(),
            messages=[{"role": "user", "content": "q"}],
        )
    assert "".join(e.get("text", "") for e in sink.events if e["kind"] == "delta") == "政企优先"
    assert result.content == "政企优先"
    assert completions.calls[0]["stream"] is True


@pytest.mark.asyncio
async def test_no_probe_fallback_after_first_chunk():
    """首 chunk 之后失败不重发 —— 已有内容已推给前端，重发会重复。"""

    class Boom(RuntimeError):
        pass

    class HalfBrokenStream:
        def __aiter__(self):
            async def gen():
                yield _delta("已经吐了")
                raise Boom("断流")

            return gen()

    class C:
        def __init__(self):
            self.calls = []

        async def create(self, **params):
            self.calls.append(params)
            if params.get("stream"):
                return HalfBrokenStream()
            raise AssertionError("不应重发非流式")

    client = C()
    with use_sink(RecordingSink()):
        with pytest.raises(Boom):
            await async_openai_chat_caller(
                _client(client),
                _target(),
                messages=[{"role": "user", "content": "q"}],
            )
    assert len(client.calls) == 1  # 只尝试了一次
