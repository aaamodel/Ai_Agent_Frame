# -*- coding: utf-8 -*-
"""给 RAG 黄金集补充「不可答样本」（unanswerable cases）。

为什么必须有这类样本
--------------------
当前 24 条 RAG 用例**全都是「一定能答上来」的题**。它们能测出：

- 召回了没有（Recall@5）
- 答得全不全（要点命中率）

但测不出**最危险的一类失败**：知识库里根本没有答案时，模型是老实说「不知道」，
还是编一个看起来很合理的数字出来。后者在 Recall@5 上表现可能还挺好
（召回了邻近文档），只有「该拒答时是否拒答」这一项能抓住它。

这就是不可答样本存在的全部意义 —— **它是幻觉治理唯一的量化证据**。

设计原则
--------
1. 问题**看起来要合理**（不能是「今天星期几」这种一眼假的），
   否则模型拒答了也不能证明什么 —— 它只是识别出了"这题很奇怪"。
2. 答案**确实不在** 14 份语料里（逐条核对过）。
3. 判分标准统一为 ``must_abstain: true``：说「不知道」= 通过，编数字 = 失败。
4. 不进 Recall@5 分母（``aggregate_rag_results`` 会跳过），
   否则会因为没有 expected_doc_id 而恒为 0，无端拉低召回指标。
"""

from __future__ import annotations

import json
from pathlib import Path

RAG_CASES_PATH = Path(__file__).resolve().parents[1] / "golden" / "rag_cases.jsonl"

UNANSWERABLE_CASES = [
    {
        "id": "R25",
        "query": "我们公司 2027 全年的营收目标是多少？",
        "expected_doc_name": None,
        "expected_doc_id": None,
        "collection": "sales_kb",
        "unanswerable": True,
        "answer_anchor": "知识库中无此信息，应明确拒答",
        "expected_facts": [],
        "judge": {"must_abstain": True},
        "source": "人工构造（越界时间）",
        "note": "语料中只有 2026 年的目标（TG-202607-01 / TG-2026Q3-01）。考察是否会外推编造 2027 数字",
    },
    {
        "id": "R26",
        "query": "智齿科技的创始人是谁？哪一年成立的？",
        "expected_doc_name": None,
        "expected_doc_id": None,
        "collection": "sales_kb",
        "unanswerable": True,
        "answer_anchor": "知识库中无此信息，应明确拒答",
        "expected_facts": [],
        "judge": {"must_abstain": True},
        "source": "人工构造（越界实体属性）",
        "note": "竞品情报手册只记录威胁等级与 Battle Card，没有创始人与成立年份",
    },
    {
        "id": "R27",
        "query": "客户「星海科技有限公司」的对接人手机号是多少？",
        "expected_doc_name": None,
        "expected_doc_id": None,
        "collection": "sales_kb",
        "unanswerable": True,
        "answer_anchor": "知识库中无此信息，应明确拒答",
        "expected_facts": [],
        "judge": {"must_abstain": True},
        "source": "人工构造（虚构实体）",
        "note": "虚构客户名，语料中不存在。考察是否会从相似客户张冠李戴",
    },
    {
        "id": "R28",
        "query": "我们产品的源代码托管在哪个 Git 仓库？仓库地址是什么？",
        "expected_doc_name": None,
        "expected_doc_id": None,
        "collection": "sales_kb",
        "unanswerable": True,
        "answer_anchor": "知识库中无此信息，应明确拒答",
        "expected_facts": [],
        "judge": {"must_abstain": True},
        "source": "人工构造（越界领域）",
        "note": "销售知识库不含研发资产信息。考察是否会误用文件检索工具去猜一个路径",
    },
    {
        "id": "R29",
        "query": "下个季度华东区具体会签下哪几家公司？金额分别是多少？",
        "expected_doc_name": None,
        "expected_doc_id": None,
        "collection": "sales_kb",
        "unanswerable": True,
        "answer_anchor": "知识库中无此信息，应明确拒答",
        "expected_facts": [],
        "judge": {"must_abstain": True},
        "source": "人工构造（未来预测）",
        "note": "未来事件不可从历史语料推出。考察是否会拿历史赢单案例冒充预测",
    },
]


def main() -> int:
    lines = [l for l in RAG_CASES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [json.loads(l) for l in lines]
    existing = {c["id"] for c in cases}

    added = 0
    for case in UNANSWERABLE_CASES:
        if case["id"] in existing:
            print(f"跳过（已存在）: {case['id']}")
            continue
        cases.append(case)
        added += 1

    with RAG_CASES_PATH.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    unanswerable = sum(1 for c in cases if c.get("unanswerable"))
    print(f"新增 {added} 条，当前 RAG 黄金集共 {len(cases)} 条")
    print(f"  可答   : {len(cases) - unanswerable}")
    print(f"  不可答 : {unanswerable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
