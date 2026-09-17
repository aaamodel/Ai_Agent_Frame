# -*- coding: utf-8 -*-
"""mrr 最小样本门单测（spec: `evals/rag-quality-gate`）。

核心对照：**同一份数据**（期望文档都排第 3）在样本不足时不判定、在样本达标时判定。
这正是本次要消除的假阳性——`--limit 2` 只抽到 1 条用例时 mrr=0.333 被判红，
而全量 27 条判分样本下 mrr=0.699 是达标的。
"""

from __future__ import annotations

import pytest

from evals.metrics import (
    MIN_SAMPLES_RETRIEVAL,
    aggregate_rag_results,
    evaluate_gate,
)
from evals.report import flatten_metrics

_K = 5


def _record(case_id: str, expected_rank: int) -> dict:
    """构造一条检索明细：期望文档出现在第 ``expected_rank`` 位。"""
    names = [f"doc_{i}.md" for i in range(1, _K + 1)]
    names[expected_rank - 1] = "target.md"
    return {
        "id": case_id,
        "retrieved_ids": [],
        "expected_ids": [],
        "retrieved_doc_names": names,
        "expected_doc_names": ["target.md"],
        "latency_ms": 10.0,
    }


def _runs(metrics: dict) -> dict:
    return {"rag": {"ok": True, "result": {"metrics": metrics}}}


# ---------------------------------------------------------------------------
# 4.1 小样本时只标记 mrr，且数值照常展示
# ---------------------------------------------------------------------------
def test_single_sample_marks_mrr_insufficient_only():
    metrics = aggregate_rag_results([_record("R24", 3)])
    insufficient = metrics["sample_insufficient"]

    assert "mrr" in insufficient, "单条样本不足以判定 mrr"
    assert insufficient["mrr"] == {"n": 1, "required": MIN_SAMPLES_RETRIEVAL}
    # 同分母的 recall@k / hit@k 对位置不敏感，保持既有行为（见 design D6）
    assert "recall@5" not in insufficient
    assert "hit@5" not in insufficient
    # 数值照常展示：排除只影响判定，不影响观测
    assert metrics["mrr"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# 4.2 判定与展示口径一致：被标记的指标不进质量门
# ---------------------------------------------------------------------------
def test_insufficient_mrr_is_excluded_from_gate():
    metrics = aggregate_rag_results([_record("R24", 3)])
    flat = flatten_metrics(_runs(metrics))

    assert "mrr" not in flat, "被标记样本不足的指标不得进入质量门"
    assert "recall@5" in flat, "同分母的其他指标不受影响，仍照常参与判定"

    violations = evaluate_gate(flat, {"rag": {"mrr_min": 0.55}})
    assert not any("mrr" in item for item in violations), "不得出现 mrr 违规"


# ---------------------------------------------------------------------------
# 4.3 样本达标时照常判定（与 4.1/4.2 用同样的排名数据作对照）
# ---------------------------------------------------------------------------
def test_enough_samples_keeps_mrr_in_gate():
    records = [_record(f"R{i:02d}", 3) for i in range(MIN_SAMPLES_RETRIEVAL)]
    metrics = aggregate_rag_results(records)

    assert metrics["sample_insufficient"] == {}, "样本达标时不应有任何样本不足标记"
    assert metrics["mrr"] == pytest.approx(1 / 3)

    flat = flatten_metrics(_runs(metrics))
    assert flat["mrr"] == pytest.approx(1 / 3), "样本达标后 mrr 必须参与判定"

    violations = evaluate_gate(flat, {"rag": {"mrr_min": 0.55}})
    assert len(violations) == 1 and "mrr" in violations[0], (
        "样本达标后 0.333 < 0.55 应当判为违规"
    )


def test_unanswerable_samples_do_not_count_toward_the_gate():
    """应拒答样本既进分母以外，也不占样本量名额。"""
    records = [_record("R01", 1) for _ in range(2)]
    records.append(
        {
            "id": "R28",
            "unanswerable": True,
            "retrieved_ids": [],
            "expected_ids": [],
            "retrieved_doc_names": ["x.md"],
            "expected_doc_names": [],
            "latency_ms": 5.0,
        }
    )
    metrics = aggregate_rag_results(records)
    assert metrics["scored"] == 2
    assert metrics["skipped_unanswerable"] == 1
    assert metrics["sample_insufficient"]["mrr"]["n"] == 2
