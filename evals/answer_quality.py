# -*- coding: utf-8 -*-
"""生成质量判分（answer quality）。

为什么单独一个文件
-------------------
``evals/metrics.py`` 里已有的指标测的是「**过程**」：

- ``recall_at_k`` / ``mrr`` → 有没有**召回到**正确的文档
- ``tool_call_success``    → 有没有**选对**工具

但它们都回答不了用户最关心的问题：**模型最终说出来的那句话，对不对？**

召回对了但答错数字（比如把「专业版 45 万」答成「旗舰版 90 万」）是最容易糊弄过去的
一类错误 —— 检索指标全绿，用户拿到的是错的。本模块就是为堵这个口子而存在。

判分思路（刻意不用 LLM-as-judge）
---------------------------------
第一版**不用模型打分**，原因是：

1. **可解释**：每一分都能把对应要点逐条指出来
2. **可单测**：纯函数，零依赖，离线也能跑
3. **不引入 evaluator 自身的方差**：LLM judge 换个模型分数就变，baseline 不可复现

代价是**只能判「事实覆盖率」，判不了「表述质量/语气/是否啰嗦」**。
这是有意的取舍：先要有能跑的数字，再谈升级。LLM-as-judge 应作为**对照**叠加，而非替代。

判定链条
--------
1. 归一化（NFKC + 去空白 + 万元→万）
2. 逐要点命中：先试**精确子串**，再试**数字兜底**
3. 顺序敏感题额外校验要点出现次序是否递增
4. 干扰项检测（distractors）：命中即标记为「混淆信号」
5. 综合：``hit_count >= judge.min_facts`` 且无混淆 → 通过

局限性（写在明面上）
--------------------
- 数字兜底匹配是**粗粒度**的：只保证数字出现过，不保证它修饰的是同一个对象。
  因此本模块只用于**回归对比**（同一套规则下前后比较），不用于**绝对质量断言**。
- ``fact_hit`` 对「同义词」无能为力（如答案写「打九五折」而要点写「9.5 折」）。
  缓解方式：在 ``expected_facts`` 里给关键要点补别名，见 ``RUBRIC``。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "normalize_answer",
    "extract_numbers",
    "fact_hit",
    "fact_hit_detail",
    "fact_hit_rate",
    "check_order",
    "detect_abstain",
    "detect_distractors",
    "evaluate_answer",
    "aggregate_answer_results",
]

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# 归一化时视为「分隔符」的标点。
# 注意：刻意**不含** . % ~ + < > ≤ ≥ —— 它们承载语义（小数、百分比、区间、阈值）。
# 这些标点会被替换成空格（而非删除），原因见 extract_numbers 的注释。
_PUNCT_RE = re.compile(r"[：:=、，。；,;'\"()（）\[\]【】{}「」『』/\\|·—_\-]")


def _punct_to_space(text: str) -> str:
    """把分隔符类标点替换成空格，并做 NFKC + 金额口径统一。"""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.replace("万元", "万").replace("亿元", "亿")
    return _PUNCT_RE.sub(" ", s)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
def normalize_answer(text: Any) -> str:
    """归一化文本，便于稳定匹配。

    做了五件事：
    1. NFKC：全角数字/字母/百分号 → 半角（中文不受影响）
    2. 小写化：``RAG`` / ``rag`` / ``Rag`` 视作同一事实
    3. 金额口径统一：万元 → 万（「45万元」和「45 万」是同一事实）
    4. **分隔符类标点 → 空格 → 再消除空白**
       （这一步让「中恒/工单客服联动」与「中恒 工单客服联动」对齐，
       也让「A=官方发布可直接引用」与「A 官方发布可直接引用」对齐）
    5. 消除空白：答案里的换行缩进不应影响匹配

    对不可归一化的输入（None / 非字符串）返回空串，不抛异常 —— 判分应容忍脏数据。
    """
    if text is None:
        return ""
    s = _punct_to_space(text)
    # 消除所有空白（含全角空格，NFKC 后已转半角）
    s = re.sub(r"\s+", "", s)
    return s.lower()


def extract_numbers(text: Any) -> List[str]:
    """抽取文本中的数字串，用于数字兜底匹配。返回原始字符串列表（保序去重）。

    **关键实现细节**：这里刻意只把标点换成空格、**不消除空格**。
    若在此处就消除空白，``6/7/8 月`` 会变成 ``678月``，数字被粘连成 ``678``，
    导致 ``6 月 23.1%`` 这类要点永远匹配不上 —— 反而制造新的误杀。
    """
    if text is None:
        return []
    s = _punct_to_space(text)
    seen: List[str] = []
    for n in _NUM_RE.findall(s):
        if n not in seen:
            seen.append(n)
    return seen


# ---------------------------------------------------------------------------
# 要点命中
# ---------------------------------------------------------------------------
def _iter_aliases(fact: Any):
    """展开要点的候选写法。

    要点可以写成两种形式：
    - ``"RAG 知识库"``                    单一写法
    - ``["RAG 知识库", "知识库 rag"]``     别名列表，**任一命中即算命中**

    为什么必须支持别名：中文同一事实的写法差异极大
    （「含质检」/「与质检」/「支持质检」），纯子串匹配会把正确答案判成未命中，
    导致生成质量被**系统性低估**。
    """
    if isinstance(fact, (list, tuple)):
        for alias in fact:
            yield alias
    else:
        yield fact


def fact_hit(predicted: Any, fact: Any) -> bool:
    """单个要点是否命中（支持别名列表，任一命中即算）。

    两级匹配：
    - **精确**：归一化后 fact 是 predicted 的子串
    - **数字兜底**：fact 中含有的所有数字都在 predicted 中出现
      （应对「45 万/年」vs「45万元/年」这类写法差异）
    """
    return any(_hit_single(predicted, alias) for alias in _iter_aliases(fact))


def _hit_single(predicted: Any, fact: Any) -> bool:
    pred = normalize_answer(predicted)
    if not pred:
        return False
    fac = normalize_answer(fact)
    if not fac:
        return False
    if fac in pred:
        return True
    # 数字兜底
    fac_nums = extract_numbers(fac)
    if fac_nums:
        pred_nums = set(extract_numbers(pred))
        return all(n in pred_nums for n in fac_nums)
    return False


def fact_hit_detail(predicted: Any, fact: Any) -> Tuple[bool, str]:
    """同 ``fact_hit``，但额外返回命中方式，便于报告里区分可信度。

    返回 ``(是否命中, 方式)``，方式为 ``exact`` / ``numeric`` / ``miss``。
    命中方式取**最好**的那一个（exact > numeric > miss）。
    """
    best_hit, best_mode = False, "miss"
    for alias in _iter_aliases(fact):
        pred = normalize_answer(predicted)
        fac = normalize_answer(alias)
        if not pred or not fac:
            continue
        if fac in pred:
            return True, "exact"
        fac_nums = extract_numbers(fac)
        if fac_nums:
            pred_nums = set(extract_numbers(pred))
            if all(n in pred_nums for n in fac_nums):
                best_hit, best_mode = True, "numeric"
    return best_hit, best_mode


def fact_hit_rate(predicted: Any, expected_facts: Sequence[Any]) -> Optional[float]:
    """要点命中率 = 命中数 / 要点总数。

    ``expected_facts`` 为空时返回 ``None``（而不是 0）——
    「没有标注」和「一个都没答对」是两回事，不能混。
    """
    if not expected_facts:
        return None
    hits = sum(1 for f in expected_facts if fact_hit(predicted, f))
    return hits / len(expected_facts)


# ---------------------------------------------------------------------------
# 顺序 / 干扰项
# ---------------------------------------------------------------------------
def check_order(predicted: Any, expected_facts: Sequence[Any]) -> bool:
    """顺序敏感题：各要点在答案中首次出现的位置是否非递减。

    用于「阶段流转」「报告结构」这类**顺序即语义**的题。
    只要有要点未命中就跳过它（不参与次序比较）—— 缺项由 min_facts 负责判罚。
    """
    pred = normalize_answer(predicted)
    positions: List[int] = []
    for fact in expected_facts:
        # 支持别名：取该要点在答案中最早出现的位置
        found = -1
        for alias in _iter_aliases(fact):
            fac = normalize_answer(alias)
            if not fac:
                continue
            idx = pred.find(fac)
            if idx >= 0 and (found < 0 or idx < found):
                found = idx
        if found >= 0:
            positions.append(found)
    return all(b >= a for a, b in zip(positions, positions[1:]))


# 拒答信号词：模型「知道自己不知道」的常见中文表述
_ABSTAIN_SIGNALS = (
    "不知道",
    "未找到",
    "没有找到",
    "未提及",
    "没有提",
    "无法回答",
    "无法从",
    "无法判断",
    "未提供",
    "未收录",
    "没有相关",
    "无相关",
    "没有找到相关",
    "不清楚",
    "无法确定",
    "超出我的知识",
    "不在知识库",
    "知识库中没有",
    "没有足够信息",
    "信息不足",
    # 下面这几个是自检时补的：模型真实的拒答说法比预想的多，
    # 漏掉会让「明明拒答了却判幻觉」，从而低估抗幻觉能力。
    "没有这个信息",
    "没有该信息",
    "无此信息",
    "知识库中未",
    "知识库里没有",
    "未提供该",
    "未包含",
    "没有包含",
    "无法获取",
    "查不到",
)


def detect_abstain(predicted: Any) -> bool:
    """判断答案是否在「拒答」。

    用于**不可答样本**（知识库里根本没有答案）：这类题的正确行为是
    明确说「不知道」，而不是编一个看起来合理的数字。
    能不能在该拒答时拒答，是幻觉治理最直接的证据。
    """
    pred = normalize_answer(predicted)
    if not pred:
        return False
    return any(sig in pred for sig in _ABSTAIN_SIGNALS)


def detect_distractors(predicted: Any, distractors: Sequence[Any]) -> List[str]:
    """返回答案中命中的干扰项列表。

    干扰项 = 邻近但**不属于本题**的事实（如 R01 的标准版 15 万 / 旗舰版 90 万）。
    命中说明模型**召回对了邻近内容、但没锁定正确条目** —— 这类错误 Recall@5 完全看不出来。
    """
    if not distractors:
        return []
    pred = normalize_answer(predicted)
    if not pred:
        return []
    return [d for d in distractors if fact_hit(pred, d)]


# ---------------------------------------------------------------------------
# 综合判定
# ---------------------------------------------------------------------------
def evaluate_answer(
    predicted: Any,
    expected_facts: Sequence[Any],
    judge: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """对单条生成答案评分。

    参数
    ----
    predicted:
        模型实际输出的答案文本
    expected_facts:
        结构化要点列表（来自 rag_cases.jsonl 的 ``expected_facts``）
    judge:
        评分标准，字段见 ``evals/tools/add_rubric.py``：
        ``min_facts`` / ``number_strict`` / ``order_sensitive`` / ``distractors``

    返回
    ----
    dict，关键字段：
    - ``hit_count`` / ``total_facts`` / ``hit_rate``
    - ``passed``  是否通过（要点数达标 且 顺序正确 且 无干扰项）
    - ``hit_modes``  每个要点的命中方式（exact / numeric / miss）
    - ``distractors_hit`` 命中的干扰项
    - ``missing_facts`` 未命中的要点（报告里直接列出来，方便定位）
    - ``reason``  未通过的原因（人类可读）
    """
    judge = judge or {}
    facts = list(expected_facts or [])
    min_facts = int(judge.get("min_facts", len(facts) or 1))
    order_sensitive = bool(judge.get("order_sensitive", False))
    distractors = list(judge.get("distractors") or [])
    must_abstain = bool(judge.get("must_abstain", False))

    # ---- 不可答样本：判「有没有拒答」，而不是「有没有答对」 ----
    if must_abstain:
        abstained = detect_abstain(predicted)
        return {
            "must_abstain": True,
            "abstained": abstained,
            "passed": abstained,
            "hit_count": 0,
            "total_facts": 0,
            "hit_rate": None,
            "min_facts": 0,
            "order_ok": True,
            "hit_modes": [],
            "distractors_hit": [],
            "missing_facts": [],
            "reason": "" if abstained else "应拒答但给出了具体答案（幻觉风险）",
        }

    hit_flags = [fact_hit(predicted, f) for f in facts]
    hit_modes = [fact_hit_detail(predicted, f)[1] for f in facts]
    hit_count = sum(1 for h in hit_flags if h)
    total = len(facts)

    hit_rate: Optional[float] = (hit_count / total) if total else None

    order_ok = True
    if order_sensitive and total > 1:
        order_ok = check_order(predicted, facts)

    dis_hit = detect_distractors(predicted, distractors)

    reasons: List[str] = []
    if total == 0:
        reasons.append("无标注要点，无法判分")
        passed = False
    else:
        if hit_count < min_facts:
            reasons.append(f"要点命中 {hit_count}/{total}，未达 min_facts={min_facts}")
        if not order_ok:
            reasons.append("要点顺序与标准答案不一致（order_sensitive）")
        if dis_hit:
            reasons.append(f"命中干扰项：{', '.join(str(d) for d in dis_hit)}")
        passed = not reasons

    return {
        "must_abstain": False,
        "abstained": detect_abstain(predicted),
        "hit_count": hit_count,
        "total_facts": total,
        "hit_rate": hit_rate,
        "min_facts": min_facts,
        "passed": passed,
        "order_ok": order_ok,
        "hit_modes": hit_modes,
        "distractors_hit": dis_hit,
        "missing_facts": [f for f, h in zip(facts, hit_flags) if not h],
        "reason": "；".join(reasons),
    }


def aggregate_answer_results(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """汇总一批生成质量结果。

    期望每条 record 至少含 ``passed`` 与 ``hit_rate``（即 ``evaluate_answer`` 的输出）。
    缺字段的 record 会被跳过 —— 宁可样本数变少，也不静默记 0 分。
    """
    graded = [r for r in records if isinstance(r, dict) and "passed" in r]
    total = len(graded)
    if total == 0:
        return {
            "sample_count": 0,
            "pass_rate": None,
            "avg_fact_hit_rate": None,
            "distractor_hit_rate": None,
            "note": "无可用判分结果",
        }

    passed = sum(1 for r in graded if r.get("passed"))
    # 可答样本才进命中率统计；不可答样本走 abstain 通道
    answerable = [r for r in graded if not r.get("must_abstain")]
    rates = [
        r["hit_rate"] for r in answerable if r.get("hit_rate") is not None
    ]
    has_dis = [r for r in graded if "distractors_hit" in r]
    dis_hit = sum(1 for r in has_dis if r.get("distractors_hit"))

    abstain_cases = [r for r in graded if r.get("must_abstain")]

    return {
        "sample_count": total,
        "answerable_count": len(answerable),
        "pass_rate": passed / total,
        "avg_fact_hit_rate": (sum(rates) / len(rates)) if rates else None,
        "distractor_hit_rate": (dis_hit / len(has_dis)) if has_dis else None,
        "abstain_count": len(abstain_cases),
        "abstain_success_rate": (
            sum(1 for r in abstain_cases if r.get("abstained")) / len(abstain_cases)
            if abstain_cases
            else None
        ),
        "failed_cases": [r.get("case_id") for r in graded if not r.get("passed")],
    }
