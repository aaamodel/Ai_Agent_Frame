# -*- coding: utf-8 -*-
"""``evals/tools/grade_answers.py`` 的单测。

重点验证三件事：
1. 脏输入（重复 id / 坏 JSON / 缺字段）不会静默算成分数；
2. 缺答案的用例被标记 missing_answer，**不判 0 分**（跑挂了 ≠ 质量差）；
3. 拒答样本与可判分样本分开统计，不互相污染。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.tools.grade_answers import (
    aggregate_answer_records,
    grade_case,
    grade_records,
    load_answers,
)


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
def _case(cid: str, *, must_abstain: bool = False, facts=None):
    judge = {"must_abstain": True} if must_abstain else {"min_facts": 2}
    return {
        "id": cid,
        "query": f"问题 {cid}",
        "expected_facts": facts if facts is not None else ["要点甲", "要点乙"],
        "judge": judge,
    }


def test_load_answers_tolerates_bad_lines(tmp_path: Path):
    p = tmp_path / "a.jsonl"
    p.write_text(
        "\n".join([
            '{"id": "R01", "answer": "甲"}',
            "not a json line",
            '{"id": "R02"}',                      # 缺 answer
            "# 注释行应被忽略",
            '{"id": "R03", "pred": "兼容 pred 字段"}',
        ]),
        encoding="utf-8",
    )
    got = load_answers(p)
    assert got == {"R01": "甲", "R03": "兼容 pred 字段"}


def test_load_answers_duplicate_id_keeps_last(tmp_path: Path):
    p = tmp_path / "a.jsonl"
    p.write_text(
        '{"id": "R01", "answer": "旧"}\n{"id": "R01", "answer": "新"}\n',
        encoding="utf-8",
    )
    assert load_answers(p)["R01"] == "新"


def test_load_answers_missing_file_exits(tmp_path: Path):
    with pytest.raises(SystemExit):
        load_answers(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# 判分
# ---------------------------------------------------------------------------
def test_grade_case_marks_missing_answer_without_zero_score():
    record = grade_case(_case("R01"), None)
    assert record["missing_answer"] is True
    assert record["passed"] is None          # 不是 False —— 跑挂了不等于答错
    assert record["hit_rate"] is None


def test_grade_case_abstain_uses_abstain_path():
    case = _case("R25", must_abstain=True, facts=[])
    ok = grade_case(case, "抱歉，知识库里没有这个信息。")
    bad = grade_case(case, "我们公司 2027 年营收目标是 3.5 亿。")
    assert ok["passed"] is True and ok["abstained"] is True
    assert bad["passed"] is False and bad["abstained"] is False
    assert "拒答" in (bad["reason"] or "")


def test_grade_records_preserves_golden_order_and_missing():
    cases = [_case("R01"), _case("R02"), _case("R03")]
    answers = {"R01": "要点甲 要点乙", "R03": "要点甲 要点乙"}
    records = grade_records(cases, answers)
    assert [r["id"] for r in records] == ["R01", "R02", "R03"]
    assert records[1]["missing_answer"] is True


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def test_aggregate_separates_abstain_from_scored():
    records = [
        {"id": "R01", "missing_answer": False, "must_abstain": False,
         "passed": True, "hit_rate": 1.0, "distractors_hit": []},
        {"id": "R02", "missing_answer": False, "must_abstain": False,
         "passed": False, "hit_rate": 0.5, "distractors_hit": ["15 万"]},
        {"id": "R25", "missing_answer": False, "must_abstain": True,
         "passed": True, "abstained": True, "hit_rate": None},
        {"id": "R03", "missing_answer": True, "passed": None},
    ]
    m = aggregate_answer_records(records)
    assert m["total"] == 4
    assert m["scored"] == 2
    assert m["abstain_total"] == 1
    assert m["answer_pass_rate"] == 0.5
    assert m["abstain_correct_rate"] == 1.0
    assert m["distractor_hit_rate"] == 0.5
    assert m["coverage"] == pytest.approx(0.75)   # 3/4 提供了答案
    assert m["failed_cases"] == ["R02"]


def test_aggregate_returns_none_when_nothing_gradeable():
    m = aggregate_answer_records([{"id": "R01", "missing_answer": True}])
    assert m["scored"] == 0
    assert m["answer_pass_rate"] is None
    assert m["coverage"] == 0.0


def test_end_to_end_grade_file(tmp_path: Path):
    """走一遍真实文件 IO：写答案文件 → 判分 → 汇总。"""
    cases = [_case("R01"), _case("R25", must_abstain=True, facts=[])]
    ans = tmp_path / "answers.jsonl"
    ans.write_text(
        "\n".join([
            json.dumps({"id": "R01", "answer": "要点甲和要点乙都提到了"},
                       ensure_ascii=False),
            json.dumps({"id": "R25", "answer": "抱歉，我没有找到相关信息"},
                       ensure_ascii=False),
        ]),
        encoding="utf-8",
    )
    loaded = load_answers(ans)
    records = grade_records(cases, loaded)
    m = aggregate_answer_records(records)
    assert m["scored"] == 1 and m["abstain_total"] == 1
    assert m["answer_pass_rate"] == 1.0
    assert m["abstain_correct_rate"] == 1.0
