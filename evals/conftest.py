# -*- coding: utf-8 -*-
"""pytest 共享 fixture。

**刻意保持轻量**：本文件在 pytest 收集阶段就会被 import，任何重型依赖
（llama_index / pymilvus / fastapi / redis）都不应出现在模块顶层，否则
``pytest evals/test_metrics.py`` 在干净环境里会直接 collection error。
真正需要连真实链路的用例放在 ``evals/runners/`` 里，由 runner 自己 import。
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from evals.loaders import (
    load_intent_cases,
    load_rag_cases,
    load_results,
    load_thresholds,
    load_tool_cases,
)


@pytest.fixture(scope="session")
def thresholds() -> Dict[str, Any]:
    """质量门阈值（evals/thresholds.yaml）。"""
    return load_thresholds()


@pytest.fixture(scope="session")
def intent_cases() -> List[Dict[str, Any]]:
    return load_intent_cases()


@pytest.fixture(scope="session")
def rag_cases() -> List[Dict[str, Any]]:
    return load_rag_cases()


@pytest.fixture(scope="session")
def tool_cases() -> List[Dict[str, Any]]:
    return load_tool_cases()


@pytest.fixture(scope="session")
def eval_results() -> Dict[str, Any]:
    """最近一次评测结果（evals/_results/latest.json）。

    文件不存在时 **skip 而不是 fail**：CI 上"只跑了指标单测、还没跑真实链路"
    是合法状态（真实链路需要 Milvus/Redis/模型 API）。质量门一旦有结果就会真正生效。
    """
    results = load_results()
    if results is None:
        pytest.skip(
            "尚无评测结果（evals/_results/latest.json 不存在）。"
            "请先运行：python -m evals.report --run-all"
        )
    return results
