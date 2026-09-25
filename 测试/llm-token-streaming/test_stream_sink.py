# -*- coding: utf-8 -*-
"""旁观通道模块单测。"""

import threading

from app.core.agent.stream_sink import (
    current_sink,
    emit,
    has_sink,
    structural_visible,
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
        assert emit({"text": "x"}) is False  # 返回 False，但不抛


def test_use_sink_none_is_noop():
    with use_sink(None):
        assert has_sink() is False


def test_plain_thread_does_not_inherit_context():
    """钉住机制认知：threading.Thread 不复制上下文（asyncio.to_thread 才复制）。

    真正的跨线程行为由 test_agent_stream_delta.py 的 asyncio.to_thread 用例覆盖；
    这里断言"普通线程读不到"，是为了避免有人误以为 ContextVar 会自动跨线程。
    """
    sink = RecordingSink()
    seen = []

    def worker():
        seen.append(current_sink() is sink)
        emit({"text": "来自线程"})

    with use_sink(sink):
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    assert seen == [False]
    assert sink.events == []


# ---------------------------------------------------------------------------
# show_structural 策略：答案阶段屏蔽内部结构化 JSON，改写阶段保持放行
# ---------------------------------------------------------------------------
def test_structural_visible_defaults_to_true():
    assert structural_visible() is True
    sink = RecordingSink()
    with use_sink(sink):  # 不传 = 历史行为（改写阶段依赖结构化流）
        assert structural_visible() is True


def test_show_structural_false_scopes_and_restores():
    sink = RecordingSink()
    with use_sink(sink, show_structural=False):
        assert has_sink() is True
        assert structural_visible() is False
    # 退出后恢复默认
    assert structural_visible() is True
    assert has_sink() is False


def test_nested_structural_policy_restores_outer():
    outer = RecordingSink()
    with use_sink(outer, show_structural=False):
        with use_sink(RecordingSink()):  # 内层默认 True（如改写阶段嵌套）
            assert structural_visible() is True
        # 内层退出后恢复外层的 False，而不是全局默认 True
        assert structural_visible() is False
    assert structural_visible() is True
