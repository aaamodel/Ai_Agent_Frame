# -*- coding: utf-8 -*-
"""给 RAG 黄金集补充「生成质量评分标准（rubric）」。

背景
----
原 rag_cases.jsonl 里有 ``answer_anchor``（标准答案要点），但它是**自由文本**，
代码里没有任何地方消费它 —— 也就是说我们只测了「检索召回」，没测「生成答案对不对」。

本脚本为每条 case 补两个字段：

- ``expected_facts``：把 answer_anchor 拆成**结构化要点数组**，供自动判分
- ``judge``：评分标准（rubric），声明这条题「怎么算答对」

    - ``min_facts``       至少命中几个要点才算通过（int）
    - ``number_strict``   数字是否必须精确出现（bool）
    - ``order_sensitive`` 要点顺序是否敏感（bool，如阶段流转）
    - ``distractors``     干扰项：命中即视为「混淆了邻近文档/条目」的信号（list[str]）

设计原则
--------
1. **只补字段，不改原字段** —— 已有的 query/expected_doc_id/answer_anchor 全部保留，
   脚本可重复执行（幂等）。
2. **min_facts 逐条人工定，不统一取一个值** —— 因为 24 条的信息密度差异极大
   （R09 只有 2 个要点，R19 有 6 条反馈），统一阈值会让简单题虚高、难题虚低。
3. **distractors 是本数据集最有价值的部分之一** —— 它让「召回对了但答错数字」
   这类最容易糊弄过去的错误暴露出来（如 R01 的标准版 15 万 / 旗舰版 90 万）。

用法::

    python evals/tools/add_rubric.py            # 写入
    python evals/tools/add_rubric.py --dry-run  # 只看不写
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

RAG_CASES_PATH = Path(__file__).resolve().parents[1] / "golden" / "rag_cases.jsonl"


# ---------------------------------------------------------------------------
# 逐条人工拆分的评分标准
# key = case id；value = (expected_facts, judge)
# ---------------------------------------------------------------------------
RUBRIC: Dict[str, Dict[str, Any]] = {
    "R01": {
        # 「含质检」/「与质检」/「支持质检」写法差异大 → 用别名数组，任一命中即算
        "expected_facts": [
            "专业版 45 万/年",
            ["≤200 席", "200 席", "不超过200席"],
            ["RAG 知识库", "知识库"],
            ["质检"],
        ],
        "judge": {
            "min_facts": 2,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": ["15 万", "90 万"],
        },
    },
    "R02": {
        "expected_facts": [
            "销售代表 9.5 折",
            "经理 9 折",
            "总监 8.5 折",
            "低于 8.5 折逐级审批",
        ],
        "judge": {
            "min_facts": 3,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R03": {
        "expected_facts": [
            "客服平台+知识库 9 折",
            "工单+知识库 9.5 折",
            "三线全签 8.5 折",
        ],
        "judge": {
            "min_facts": 3,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R04": {
        "expected_facts": ["满分 12 分", "六要素", "A 级 10~12 分"],
        "judge": {
            "min_facts": 2,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R05": {
        "expected_facts": ["线索", "MQL", "SQL", "商机", "方案验证", "商务谈判", "赢单"],
        "judge": {
            "min_facts": 6,
            "number_strict": False,
            "order_sensitive": True,
            "distractors": [],
        },
    },
    "R06": {
        "expected_facts": ["价格", "功能", "关系", "时机", "需求消失", "自建", "无明显原因"],
        "judge": {
            "min_facts": 5,
            "number_strict": False,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R07": {
        "expected_facts": ["P0 24 小时内首触", "P1 3 个工作日", "P2 月度培育"],
        "judge": {
            "min_facts": 2,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R08": {
        "expected_facts": [
            ["纯免费工具倾向", "免费工具"],
            ["数据必须境外", "数据出境", "境外存储"],
            ["员工<100 人且无增长", "员工不足 100 人", "少于 100 人", "100人以下"],
            ["集团已绑定其他供应商", "已绑定其他供应商", "集团绑定"],
        ],
        "judge": {
            "min_facts": 3,
            "number_strict": False,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R09": {
        "expected_facts": ["赢单数 ÷（赢单数+输单数）", "健康区间 20%~30%"],
        "judge": {
            "min_facts": 2,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R10": {
        "expected_facts": ["3.0 倍以上健康", "低于 2.5 倍有风险"],
        "judge": {
            "min_facts": 2,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R11": {
        "expected_facts": [
            "总体表现",
            "指标明细表",
            "瓶颈诊断",
            "分维度拆解",
            ["行动建议", "建议"],
            ["顺序不可变", "顺序固定", "不能调整顺序", "顺序不能变"],
        ],
        "judge": {
            "min_facts": 4,
            "number_strict": False,
            "order_sensitive": True,
            "distractors": [],
        },
    },
    "R12": {
        "expected_facts": [
            "A 官方发布可直接引用",
            "B 权威媒体需注明来源日期",
            "C 待验证",
        ],
        "judge": {
            "min_facts": 2,
            "number_strict": False,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R13": {
        "expected_facts": ["智齿", "美洽", "网易七鱼", "易维帮助台", "语雀企业版"],
        "judge": {
            "min_facts": 4,
            "number_strict": False,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R14": {
        "expected_facts": [
            ["拆成本结构", "成本结构", "拆解成本"],
            ["SLA"],
            ["100% 质检", "全量质检"],
            ["私有化差异化", "私有化"],
            ["三年总持有成本", "总持有成本", "TCO"],
            "智齿降价 30%",
        ],
        "judge": {
            "min_facts": 3,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R15": {
        "expected_facts": [
            "七鱼 AI 问答 2.0",
            ["幻觉控制", "减少幻觉", "幻觉率"],
            ["RAG 答案带引用", "带引用", "引用来源", "可溯源"],
            "各答 20 个真实历史问题",
        ],
        "judge": {
            "min_facts": 2,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R16": {
        "expected_facts": [
            "SC-202606-01",
            "中恒建设集团",
            "智能客服平台专业版",
            "45 万",
            "华北",
            "私有化",
        ],
        "judge": {
            "min_facts": 4,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R17": {
        "expected_facts": [
            "咨询量 -30%",
            "质检通过率 +25%",
            "检索效率 +50%",
            "60% 重复问题",
        ],
        "judge": {
            "min_facts": 3,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R18": {
        "expected_facts": [
            "智齿 汇智软件园 30 万",
            "七鱼 华东电商 25 万",
            "易维 华南物流 15 万",
            "语雀 西南教育 12 万",
        ],
        "judge": {
            "min_facts": 3,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R19": {
        "expected_facts": [
            "中恒 工单客服联动",
            "雅膳 移动端审批",
            "华南精工 知识库 API",
            "西南优购 折扣",
            "华东电商 竞品对比",
            "汇智 续约顾虑",
        ],
        "judge": {
            "min_facts": 4,
            "number_strict": False,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R20": {
        "expected_facts": [
            "赢单复盘 5 步法",
            "竞品应对",
            "ICP 识别",
            "报价与折扣合规红线",
            "9.5 折/9 折权限",
            ["9 折以下走审批", "低于9折", "9折以下"],
            ["内部毛利成本不得对客户输出", "内部毛利", "不对外输出成本", "成本不得对客户"],
        ],
        "judge": {
            "min_facts": 4,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R21": {
        "expected_facts": [
            "目标 130 万",
            "实际 95 万",
            "达成率 73.1%",
            "智齿降价 30%",
            "华东客服环比 -25%",
        ],
        "judge": {
            "min_facts": 3,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R22": {
        "expected_facts": [
            "目标 25%",
            "实际 20.6%",
            "达成率 82.4%",
            "6 月 23.1%",
            "7 月 18.2%",
            "8 月 20.0%",
        ],
        "judge": {
            "min_facts": 3,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R23": {
        "expected_facts": [
            # 除号写法可能是 ÷ / / ，别名覆盖
            ["CAC = 总投入 ÷ 转化赢单数", "总投入÷转化赢单数", "总投入/转化赢单数", "总投入除以转化赢单数"],
            ["MQL 率 = MQL 数 / 获取线索数", "MQL数/获取线索数", "MQL数÷获取线索数"],
            ["赢单率 = 转化赢单数 / 转化商机数", "转化赢单数/转化商机数", "转化赢单数÷转化商机数"],
        ],
        "judge": {
            "min_facts": 2,
            "number_strict": False,
            "order_sensitive": False,
            "distractors": [],
        },
    },
    "R24": {
        "expected_facts": [
            "MA-202606-01",
            "华东反击战",
            "1 赢单",
            "28 万",
            "智齿促销影响",
        ],
        "judge": {
            "min_facts": 3,
            "number_strict": True,
            "order_sensitive": False,
            "distractors": [],
        },
    },
}


def apply_rubric(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把 rubric 合并进 cases，返回统计信息。"""
    stats = {"total": len(cases), "filled": 0, "missing_rubric": [], "unknown_id": []}

    for case in cases:
        cid = case.get("id")
        rubric = RUBRIC.get(cid)
        if rubric is None:
            stats["unknown_id"].append(cid)
            continue
        case["expected_facts"] = rubric["expected_facts"]
        case["judge"] = rubric["judge"]
        stats["filled"] += 1

    known = set(RUBRIC)
    seen = {c.get("id") for c in cases}
    stats["missing_rubric"] = sorted(known - seen)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="给 RAG 黄金集补充生成质量评分标准")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写回文件")
    args = parser.parse_args()

    lines = [l for l in RAG_CASES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = [json.loads(l) for l in lines]

    stats = apply_rubric(cases)

    print(f"总条数        : {stats['total']}")
    print(f"已补 rubric   : {stats['filled']}")
    print(f"未知 case id  : {stats['unknown_id'] or '无'}")
    print(f"rubric 未用上 : {stats['missing_rubric'] or '无'}")

    total_facts = sum(len(r["expected_facts"]) for r in RUBRIC.values())
    print(f"要点总数      : {total_facts}（平均 {total_facts / len(RUBRIC):.1f} 个/条）")

    if not args.dry_run:
        with RAG_CASES_PATH.open("w", encoding="utf-8") as f:
            for case in cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"\n已写入: {RAG_CASES_PATH}")
    else:
        print("\n--dry-run：未写入")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
