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
