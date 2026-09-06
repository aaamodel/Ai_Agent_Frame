# -*- coding: utf-8 -*-
"""全链路追踪：内存存储 Span 树、事件日志与查询接口。"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class TraceSpan:
    """单次操作 Span（具体运输环节）。"""

    span_id: str
    trace_id: str
    operation: str
    parent_span_id: str | None
    start_time: float
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)  # 新增：支持编排层传入的属性
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class TraceRecord:
    """一次 Trace 的完整记录（单笔物流提单）。"""

    trace_id: str
    spans: list[TraceSpan] = field(default_factory=list)
    # 👇 把这里的 default_factory 改回标准的 list 即可！
    events: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class Tracer:
    """
    全链路追踪器：记录 Agent 执行的每个步骤。
    完美实现 app/core/agent/orchestrator.py 中的 Tracer(Protocol) 契约。
    """

    def __init__(self, max_trace_record: int = 5000) -> None:
        self._max_trace_record = max_trace_record
        self._trace_record: dict[str, TraceRecord] = {}
        self._lock = threading.RLock()
        self._current_parent: dict[str, str | None] = {}

    def start_trace(self, trace_id: str, operation: str, attributes: Optional[Dict[str, Any]] = None) -> TraceSpan:
        """开始新的根 Span。"""
        with self._lock:
            rec = self._trace_record.get(trace_id)
            if rec is None:
                rec = TraceRecord(trace_id=trace_id)
                self._trace_record[trace_id] = rec
                self._trim_locked()

            span = TraceSpan(
                span_id=str(uuid.uuid4()),
                trace_id=trace_id,
                operation=operation,
                parent_span_id=None,
                start_time=time.perf_counter(),
                attributes=attributes or {},
            )
            rec.spans.append(span)
            self._current_parent[trace_id] = span.span_id
            logger.debug("trace={} span={} op={} 根环节开始", trace_id, span.span_id, operation)
            return span

    def start_child_span(
            self,
            trace_id: str,
            operation: str,
            parent_span_id: str | None = None,
            attributes: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        """在已有 Trace 下创建子 Span。"""
        with self._lock:
            rec = self._trace_record.get(trace_id)
            if rec is None:
                rec = TraceRecord(trace_id=trace_id)
                self._trace_record[trace_id] = rec
                self._trim_locked()

            parent = parent_span_id or self._current_parent.get(trace_id)
            span = TraceSpan(
                span_id=str(uuid.uuid4()),
                trace_id=trace_id,
                operation=operation,
                parent_span_id=parent,
                start_time=time.perf_counter(),
                attributes=attributes or {},
            )
            rec.spans.append(span)
            self._current_parent[trace_id] = span.span_id
            logger.debug("trace={} span={} op={} 子环节开始 -> 父级={}", trace_id, span.span_id, operation, parent)
            return span

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        """按 trace_id 获取完整追踪记录。"""
        with self._lock:
            rec = self._trace_record.get(trace_id)
            if rec is None:
                return None
            return TraceRecord(
                trace_id=rec.trace_id,
                spans=list(rec.spans),
                events=list(rec.events),  # 确保浅拷贝返回事件
                created_at=rec.created_at,
            )

    def _trim_locked(self) -> None:
        """限制内存中 trace 数量。"""
        if len(self._trace_record) <= self._max_trace_record:
            return
        excess = len(self._trace_record) - self._max_trace_record
        for key in list(self._trace_record.keys())[:excess]:
            del self._trace_record[key]

    # ---------------------------------------------------------------------------
    # 精准实现 Orchestrator 的 Protocol 契约方法
    # ---------------------------------------------------------------------------

    def new_trace_id(self) -> str:
        """生成全新的全链路唯一追踪物流单号。"""
        return str(uuid.uuid4())

    def start_span(self, name: str, trace_id: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
        """
        给编排器调用的统一开工接口。
        智能判断：如果是当前 trace_id 的第一个环节，自动做根 Span；否则自动降级为子 Span。
        """
        with self._lock:
            # 判断当前物流单号是否已经存在且含有根 Span
            if trace_id in self._trace_record and self._trace_record[trace_id].spans:
                return self.start_child_span(trace_id, operation=name, attributes=attributes)
            else:
                return self.start_trace(trace_id, operation=name, attributes=attributes)

    def end_span(self, span: Any, error: Optional[BaseException] = None,
                 result: Optional[Dict[str, Any]] = None) -> None:
        if span is None:
            return
        with self._lock:
            rec = self._trace_record.get(span.trace_id)
            if rec is None:
                logger.warning("end_span 找不到 trace_id={}", span.trace_id)
                return
            span.end_time = time.perf_counter()
            if error is not None:
                span.error = f"{type(error).__name__}: {str(error)}"
            if result is not None:
                span.result = result
            logger.debug(
                "trace={} span={} op={} 完工，耗时={:.4f}s，错误={}，结果={}",
                span.trace_id,
                span.span_id,
                span.operation,
                span.end_time - span.start_time,
                span.error,
                span.result,
            )


    def log_event(self, trace_id: str, name: str, payload: Dict[str, Any]) -> None:
        """
        在当前trace上，随时盖章记录关键突发事件。
        """
        with self._lock:
            rec = self._trace_record.get(trace_id)
            if rec is None:
                # 健壮性：如果事件先于第一个 span 到达，先开辟空间
                rec = TraceRecord(trace_id=trace_id)
                self._trace_record[trace_id] = rec
                self._trim_locked()

            event_entry = {
                "timestamp": time.time(),
                "event_name": name,
                "payload": payload
            }
            rec.events.append(event_entry)
            #logger.info("【Trace事件日志】trace={} 事件={} 详情={}", trace_id, name, payload)