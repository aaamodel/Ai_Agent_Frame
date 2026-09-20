# -*- coding: utf-8 -*-
"""集合白名单守卫：越界取值的过滤、回落与留痕。

为什么需要它：

    知识库检索的 ``collection_names`` 此前是一个无约束的自由字符串数组。一次实测中
    Planner 传入了 ``product_docs`` / ``sales_policies`` / ``pricing_guide`` 三个
    **并不存在**的集合名，两个子任务全部空召回、工具被硬熔断、整轮降级收尾。

    修复分两层：取值域用**动态枚举**约束（模型看不到别的名字），以及**这一层**——
    即使枚举被支持不完整的网关忽略、模型仍然编了名字，也要在进入检索前拦下。

为什么是"静默回落"而不是报错：

    枚举的强制力取决于模型与网关是否支持严格 schema。一次幻觉不应让整轮请求失败，
    "还能检索到东西"优先于"严厉报错"。因此越界取值被过滤；过滤后若为空，回落为
    不限定集合的检索。

    代价是**幻觉被静默吸收**，事后无法从错误里发现。所以留痕不是可选项，而是这条
    决策的必要配套：只有留下可查询的记录，才谈得上统计幻觉发生率、评估枚举是否真的有效。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Iterable, List, Sequence, Tuple

from loguru import logger

#: 进程内保留的越界记录条数上限（仅用于观测，不参与业务判断）
MAX_RECORDS: int = 200


@dataclass(frozen=True)
class CollectionFallbackRecord:
    """一次越界过滤 / 回落的记录。"""

    requested: Tuple[str, ...]  #: 调用方原始请求的集合名
    kept: Tuple[str, ...]       #: 过滤后保留的合法集合名
    dropped: Tuple[str, ...]    #: 被判为越界而丢弃的取值
    fell_back_to_all: bool      #: 是否回落为"不限定集合"检索
    at: float                   #: 发生时刻（time.time()）


_records: Deque[CollectionFallbackRecord] = deque(maxlen=MAX_RECORDS)
_lock = threading.Lock()


def partition_collection_names(
    requested: Iterable[Any],
    is_valid: Callable[[str], bool],
) -> Tuple[List[str], List[str]]:
    """把请求的集合名分成 ``(合法, 越界)`` 两组，保持传入顺序并去重。

    ``is_valid`` 由调用方注入（通常是注册表查询），本模块不依赖具体注册表实现——
    这样既能单测，也避免把"哪些集合存在"的知识复制到这里。

    ⚠️ 合法性口径应当是"注册表内**全部**真实集合"，而不是"工具枚举里**可显式选择**的
    集合"。两者不同：枚举只决定模型能不能选中，过滤决定名字是不是真实资产。混用会把
    ``_untagged`` 这类"存在但无描述、故不可显式指定"的集合误判为幻觉并丢弃。
    """
    kept: List[str] = []
    dropped: List[str] = []
    seen: set[str] = set()
    for raw in requested or []:
        name = str(raw or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        if is_valid(name):
            kept.append(name)
        else:
            dropped.append(name)
    return kept, dropped


def record_fallback(
    requested: Sequence[str],
    kept: Sequence[str],
    dropped: Sequence[str],
    *,
    fell_back_to_all: bool,
) -> CollectionFallbackRecord:
    """登记一次越界过滤；返回该记录（同时落一条 warning 日志）。"""
    record = CollectionFallbackRecord(
        requested=tuple(str(item) for item in requested),
        kept=tuple(str(item) for item in kept),
        dropped=tuple(str(item) for item in dropped),
        fell_back_to_all=bool(fell_back_to_all),
        at=time.time(),
    )
    with _lock:
        _records.append(record)
    logger.warning(
        "RAG 集合白名单含越界取值，已静默过滤: dropped={} kept={} 回落全库={}",
        list(record.dropped), list(record.kept), record.fell_back_to_all,
    )
    return record


def recent_fallbacks(limit: int = 20) -> List[CollectionFallbackRecord]:
    """最近的越界记录（新→旧）。"""
    size = max(1, int(limit))
    with _lock:
        return list(_records)[-size:][::-1]


def clear_fallbacks() -> None:
    """清空记录（测试夹具用）。"""
    with _lock:
        _records.clear()
