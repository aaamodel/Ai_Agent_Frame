# -*- coding: utf-8 -*-
"""rubric 自检：拿标准答案自己当输入，验证判分器不会误杀。

这是数据集上线前必须做的一步，逻辑很简单：

> 如果连 ``answer_anchor``（人工写的标准答案）都判不过，
> 那这套 rubric 就是坏的 —— 它测的不是模型质量，是标注写法。

同时也跑三个"应该失败"的反例，确认判分器**不是无脑全过**。

用法::

    python evals/tools/selftest_rubric.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.answer_quality import evaluate_answer  # noqa: E402
from evals.loaders import load_rag_cases  # noqa: E402


def main() -> int:
    cases = load_rag_cases()
    if not cases:
        print("❌ 未加载到 rag_cases")
        return 1

    # 只要 judge 存在即可：不可答样本的 expected_facts 本就应该为空数组
    missing = [c["id"] for c in cases if not c.get("judge")]
    if missing:
        print(f"❌ 以下 case 缺 rubric 字段：{missing}")
        print("   请先运行: python evals/tools/add_rubric.py")
        return 1

    print(f"自检 {len(cases)} 条：把标准答案喂给判分器，验证不会误杀\n")

    failed = []
    for case in cases:
        must_abstain = bool((case.get("judge") or {}).get("must_abstain"))
        if must_abstain:
            # 不可答样本：标准行为是明确拒答，用一句标准拒答语喂进去
            predicted = "抱歉，知识库中没有找到相关信息，无法回答。"
        else:
            predicted = f"{case.get('answer_anchor', '')}"
        res = evaluate_answer(predicted, case.get("expected_facts") or [], case["judge"])
        status = "✅" if res["passed"] else "❌"
        if not res["passed"]:
            failed.append((case["id"], res))
        print(
            f"  {status} {case['id']}: {res['hit_count']}/{res['total_facts']} "
            f"(min={res['min_facts']}) {res['reason']}"
        )

    print()
    if failed:
        print(f"⚠️  {len(failed)}/{len(cases)} 条标准答案未通过 —— rubric 需要调整：")
        for cid, res in failed:
            print(f"   {cid}: 未命中要点 = {res['missing_facts']}")
        return 1

    print(f"✅ 全部 {len(cases)} 条标准答案均通过判分")
    return 0


if __name__ == "__main__":
    sys.exit(main())
