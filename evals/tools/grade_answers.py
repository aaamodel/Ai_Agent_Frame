#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成层（答案质量）离线判分器。

为什么单独成一个入口
--------------------
``evals/runners/run_rag.py`` 只测**检索层**（Recall@5 / MRR）。
但「召回到了正确的文档」不等于「说出来的答案是对的」——
召回满分而答案把「专业版 45 万」说成「旗舰版 90 万」，检索指标依然全绿。

``rag_cases.jsonl`` 里每条都有 ``expected_facts``（标准答案要点）与
``judge``（评分标准），但**需要一个「模型实际说了什么」作为输入**才能判分。
本脚本就是那个入口：把预测答案喂进来，产出生成层指标。

设计上刻意做成**离线**：预测答案从文件读，不调 LLM。好处是——
1. 预测答案可以来自任何地方：真实跑批、Langfuse 导出、人工填写都行；
2. 判分逻辑与取数解耦，换模型/换 prompt 不用改判分器；
3. 无中间件、无 API Key 也能跑，CI 里永远绿。

用法：
    # 1) 先准备预测答案文件（每行一条 {"id": "R01", "answer": "..."}）
    # 2) 判分
    python evals/tools/grade_answers.py --answers evals/_results/answers.jsonl

    # 只看看，不写文件
    python evals/tools/grade_answers.py --answers evals/_results/answers.jsonl --dry-run

    # 顺手生成一份 Markdown（不依赖 evals/report.py）
    python evals/tools/grade_answers.py --answers evals/_results/answers.jsonl --out-md evals/answer_report.md

预测答案文件格式（JSONL，每行一个 JSON 对象）：
    {"id": "R01", "answer": "专业版 45 万/年，支持 ≤200 席……"}
    {"id": "R25", "answer": "抱歉，知识库中没有相关信息。"}
``id`` 也接受 ``case_id``；``answer`` 也接受 ``pred`` / ``response`` / ``text``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

if __package__ in (None, ""):  # 允许 `python evals/tools/grade_answers.py` 直接跑
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.answer_quality import evaluate_answer  # noqa: E402
from evals.loaders import (  # noqa: E402
    GOLDEN_DIR,
    RESULTS_DIR,
)
from evals.metrics import mean  # noqa: E402

DEFAULT_ANSWERS: Path = RESULTS_DIR / "answers.jsonl"
DEFAULT_OUT: Path = RESULTS_DIR / "answer.json"
DEFAULT_MD: Path = Path("evals") / "answer_report.md"
RAG_CASES_PATH: Path = GOLDEN_DIR / "rag_cases.jsonl"

# 预测答案里「答案文本」字段的兼容别名（按优先级）
_ANSWER_KEYS: tuple = ("answer", "pred", "response", "text", "content", "output")
# 「用例 id」字段的兼容别名
_ID_KEYS: tuple = ("id", "case_id", "caseId", "qid")


# ---------------------------------------------------------------------------
# 载入
# ---------------------------------------------------------------------------
def load_answers(path: Path) -> Dict[str, str]:
    """读预测答案文件，返回 ``{case_id: answer_text}``。

    容忍脏数据：非 JSON 行跳过，缺字段跳过，重复 id 以**后者覆盖**并在
    stderr 提示（覆盖比静默丢一半更安全，但要让人看见）。
    """
    if not path.exists():
        raise SystemExit(
            f"❌ 预测答案文件不存在：{path}\n"
            f"   格式：每行 {{\"id\": \"R01\", \"answer\": \"...\"}}"
        )
    out: Dict[str, str] = {}
    dup: List[str] = []
    bad = 0
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(obj, Mapping):
                bad += 1
                continue
            case_id = _pick(obj, _ID_KEYS)
            text = _pick(obj, _ANSWER_KEYS)
            if not case_id or text is None:
                bad += 1
                continue
            if case_id in out:
                dup.append(case_id)
            out[str(case_id)] = str(text)
    if dup:
        print(f"[grade] ⚠️ 重复 id（后者覆盖前者）：{sorted(set(dup))}", file=sys.stderr)
    if bad:
        print(f"[grade] ⚠️ 跳过 {bad} 行无法解析/缺字段的数据", file=sys.stderr)
    return out


def _pick(obj: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        value = obj.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


# ---------------------------------------------------------------------------
# 判分（纯函数，可单测）
# ---------------------------------------------------------------------------
def grade_case(case: Mapping[str, Any], answer: Optional[str]) -> Dict[str, Any]:
    """对单条用例判分。``answer`` 为 None 表示「本次没跑出答案」。"""
    case_id = str(case.get("id"))
    judge: Mapping[str, Any] = case.get("judge") or {}
    facts: List[Any] = list(case.get("expected_facts") or [])
    must_abstain = bool(judge.get("must_abstain"))

    if answer is None:
        # 没跑出答案：不判 0 分，而是标注缺失——避免"跑挂了"被当成"质量差"
        return {
            "id": case_id,
            "query": case.get("query"),
            "missing_answer": True,
            "must_abstain": must_abstain,
            "passed": None,
            "hit_rate": None,
            "reason": "本次未提供预测答案",
        }

    result = evaluate_answer(answer, facts, dict(judge))
    record: Dict[str, Any] = {
        "id": case_id,
        "query": case.get("query"),
        "missing_answer": False,
        "must_abstain": must_abstain,
        "unanswerable": bool(case.get("unanswerable")),
    }
    record.update(result)
    return record


def grade_records(
    cases: Sequence[Mapping[str, Any]],
    answers: Mapping[str, str],
) -> List[Dict[str, Any]]:
    """按黄金集顺序逐条判分（缺答案的用例也会保留，标记为 missing_answer）。"""
    return [grade_case(case, answers.get(str(case.get("id")))) for case in cases]


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
def aggregate_answer_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """汇总生成层指标。

    口径（重要，报告里要照抄，否则数字无法复现）：
    - ``answer_pass_rate``：可判分样本中 ``passed`` 的比例（不含拒答样本）
    - ``avg_fact_hit_rate``：可判分且非拒答样本的要点命中率均值
    - ``distractor_hit_rate``：命中过干扰项的样本占比（**越低越好**）
    - ``abstain_correct_rate``：应拒答样本中**确实拒答**的比例（抗幻觉能力）
    - ``coverage``：提供了预测答案的样本占比（防止"只跑一半就说达标"）
    """
    graded = [r for r in records if not r.get("missing_answer")]
    scored = [r for r in graded if not r.get("must_abstain")]
    abstain = [r for r in graded if r.get("must_abstain")]

    total = len(records)
    pass_rate: Optional[float] = None
    if scored:
        pass_rate = sum(1 for r in scored if r.get("passed")) / len(scored)

    rates = [
        float(r["hit_rate"])
        for r in scored
        if r.get("hit_rate") is not None
    ]
    with_dis = [r for r in scored if "distractors_hit" in r]
    dis_rate: Optional[float] = None
    if with_dis:
        dis_rate = sum(1 for r in with_dis if r.get("distractors_hit")) / len(with_dis)

    abstain_rate: Optional[float] = None
    if abstain:
        abstain_rate = sum(1 for r in abstain if r.get("abstained")) / len(abstain)

    return {
        "total": total,
        "coverage": (len(graded) / total) if total else 0.0,
        "scored": len(scored),
        "abstain_total": len(abstain),
        "answer_pass_rate": pass_rate,
        "avg_fact_hit_rate": mean(rates) if rates else None,
        "distractor_hit_rate": dis_rate,
        "abstain_correct_rate": abstain_rate,
        "failed_cases": [r.get("id") for r in graded if r.get("passed") is False],
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def render_markdown(
    metrics: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    answers_path: Path,
) -> str:
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "未采集"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines: List[str] = []
    add = lines.append
    add("# 生成层（答案质量）评测报告")
    add("")
    add(f"- 预测答案来源：`{answers_path}`")
    add(f"- 覆盖：{metrics.get('scored', 0)} 条可判分 + "
        f"{metrics.get('abstain_total', 0)} 条应拒答，"
        f"覆盖率 {fmt(metrics.get('coverage'), 2)}")
    add("")
    add("| 指标 | 值 | 说明 |")
    add("| --- | --- | --- |")
    add(f"| 答案合格率 | {fmt(metrics.get('answer_pass_rate'))} | "
        "要点命中数达 min_facts 且无干扰项 |")
    add(f"| 平均要点命中率 | {fmt(metrics.get('avg_fact_hit_rate'))} | "
        "标准答案要点被说中的比例 |")
    add(f"| 干扰项命中率 | {fmt(metrics.get('distractor_hit_rate'))} | "
        "**越低越好**，命中即判不通过 |")
    add(f"| 应拒答正确率 | {fmt(metrics.get('abstain_correct_rate'))} | "
        "知识库没有时是否老实说不知道 |")
    add("")

    failed = [r for r in records if r.get("passed") is False]
    if failed:
        add("## 未通过样本")
        add("")
        add("| id | 问题 | 原因 |")
        add("| --- | --- | --- |")
        for item in failed[:20]:
            add(f"| {item.get('id')} | {_esc(item.get('query'))} | "
                f"{_esc(item.get('reason') or item.get('missing_facts'))} |")
        add("")
    return "\n".join(lines) + "\n"


def _esc(text: Any) -> str:
    s = "" if text is None else str(text)
    return s.replace("|", "\\|").replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="生成层（答案质量）离线判分器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--answers", type=Path, default=DEFAULT_ANSWERS,
        help="预测答案 JSONL（每行 {\"id\", \"answer\"}）",
    )
    parser.add_argument(
        "--cases", type=Path, default=RAG_CASES_PATH,
        help="黄金集（默认 evals/golden/rag_cases.jsonl）",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="结果 JSON 输出路径")
    parser.add_argument("--out-md", type=Path, default=None, help="额外写一份 Markdown")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = parser.parse_args()

    if not args.cases.exists():
        raise SystemExit(f"❌ 黄金集不存在：{args.cases}")
    cases: List[Dict[str, Any]] = []
    with args.cases.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    answers = load_answers(args.answers)
    missing_rubric = [str(c.get("id")) for c in cases if not c.get("judge")]
    if missing_rubric:
        print(f"❌ 以下 case 缺 judge 字段：{missing_rubric}")
        print("   请先运行: python evals/tools/add_rubric.py")
        return 1

    records = grade_records(cases, answers)
    metrics = aggregate_answer_records(records)

    print(f"[grade] 黄金集 {len(cases)} 条，提供预测答案 {len(answers)} 条")
    print(f"[grade] 可判分 {metrics['scored']} 条 / 应拒答 {metrics['abstain_total']} 条 "
          f"/ 覆盖率 {metrics['coverage']:.2%}")
    for key in ("answer_pass_rate", "avg_fact_hit_rate",
                "distractor_hit_rate", "abstain_correct_rate"):
        value = metrics.get(key)
        print(f"[grade] {key:<22} = {'未采集' if value is None else f'{value:.4f}'}")
    if metrics["failed_cases"]:
        print(f"[grade] 未通过：{metrics['failed_cases']}")

    if args.dry_run:
        print("\n--dry-run：未写入任何文件")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "result": {"records": records, "metrics": metrics},
        "path": str(args.out),
    }
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"\n[grade] 已写入 {args.out}")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            render_markdown(metrics, records, answers_path=args.answers),
            encoding="utf-8",
        )
        print(f"[grade] 已写入 {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
