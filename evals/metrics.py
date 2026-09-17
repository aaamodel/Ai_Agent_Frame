# -*- coding: utf-8 -*-
"""评测指标计算（纯函数集合）。

设计约定（对齐《评测-压测-监控落地手册》第 3 节）：

1. **纯函数**：本模块只做「输入 -> 标量/字典」的计算，不读文件、不发网络请求、
   不依赖项目内部模块。这样才能被 ``test_metrics.py`` 完整单测，面试时也能
   逐行讲清每个数字怎么来的。
2. **只用标准库**：不引入 numpy / ragas，避免"分怎么来的答不上"。
3. **边界行为显式定义**：每个函数在 docstring 里写清空输入、k 越界等边界语义。

术语：
    retrieved_ids  检索返回的文档/切片标识列表，**按相关度降序**。
    expected_ids   标注的期望标识集合（单期望时为 1 个元素的集合）。
    gold / pred    标注值与预测值（用于分类指标）。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

# =====================================================================
# 零、最小样本量（"样本不足"判定）—— 全模块唯一真源
# =====================================================================
# 背景：小样本下的某些指标在统计上不成立，若照常算出来会被质量门当成真实违规。
# 最典型的是 P95：n=3 时插值位置 pos=(3-1)*0.95=1.9，结果几乎等于最大值，
# 一条偶发的慢请求就能把整组判红——它测的是"那一次的抖动"，不是"整体水位"。
#
# 处理方式（与「零分母 = None」同一套语义）：
#   样本量低于下限时，该指标置为 None，**不参与 evaluate_gate 判定**，
#   同时在结果里写 ``sample_insufficient`` 说明原因，由报告渲染成可读提示。
#   注意是置 None 而不是删除键——保留原始数值留给人工诊断（report 分组明细仍会打印）。
MIN_SAMPLES_PERCENTILE: int = 20
"""P95 等分位数所需的最小样本量。

20 是业界对 p95 的常用下限（低于此 n 时分位数插值退化为取极值）。
⚠️ 已知约束：工具黄金集只有 15 条，**全量跑也达不到 20**，
因此工具组的 P95 永远不会进质量门。若要启用该红线，需二选一：
把工具集扩到 ≥20 条，或调低本常量。
"""

MIN_SAMPLES_BOUNDARY: int = 5
"""边界准确率所需的最小样本量。

黄金集里设计就是 5 条边界样本，所以**全量跑恰好达标**（5 >= 5）；
任何切片/抽样导致边界样本少于 5 条时，该指标退化为未采集。
"""


def sample_gate(count: int, required: int) -> Optional[Dict[str, int]]:
    """样本量是否达到可判定的下限。

    Returns:
        不足时返回 ``{"n": count, "required": required}``（调用方据此把指标置 None
        并记入 ``sample_insufficient``）；达到下限返回 ``None``。
    """
    if int(count) < int(required):
        return {"n": int(count), "required": int(required)}
    return None


def format_metric(value: Any, spec: str = ".3f", na: str = "不可用") -> str:
    """把可能为 ``None``（未采集 / 零分母）的指标格式化成可打印文本。

    之所以需要它：指标现在允许是 ``None``，直接 ``f"{value:.3f}"`` 会在
    ``None`` 上抛 ``TypeError``。三个 runner 的收尾打印统一走这里，
    避免出现三份各自写的兜底逻辑。

    Args:
        value: 指标值，可能为 ``None``。
        spec: ``format()`` 用的格式串。
        na: ``None`` 时显示的文本（默认"不可用"）。
    """
    if value is None:
        return na
    if isinstance(value, bool):
        return str(value)
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "MIN_SAMPLES_PERCENTILE",
    "MIN_SAMPLES_BOUNDARY",
    "sample_gate",
    "format_metric",
    "recall_at_k",
    "mrr",
    "hit_at_k",
    "accuracy",
    "confusion_counts",
    "per_class_metrics",
    "mean",
    "percentile",
    "token_cost",
    "normalize_value",
    "key_arg_matches",
    "key_arg_recall",
    "tool_call_success",
    "tool_call_success_rate",
    "aggregate_intent_results",
    "aggregate_rag_results",
    "aggregate_tool_results",
    "evaluate_gate",
]


# =====================================================================
# 一、检索类指标（RAG）
# =====================================================================
def recall_at_k(
    retrieved_ids: Sequence[str],
    expected_ids: Iterable[str],
    k: int = 5,
) -> float:
    """前 k 个结果中命中期望文档的**比例**。

    定义：``|expected ∩ retrieved[:k]| / |expected|``

    边界行为（面试高频追问点）：
        - 单期望文档时退化为 0/1（命中即 1.0，未命中即 0.0）；
        - 期望为空集合时返回 0.0（无标注即无法判定召回，计为未命中，
          避免"空标注刷高分"）；
        - k <= 0 时返回 0.0（不检索 = 不召回）；
        - retrieved 为空时返回 0.0；
        - **重复命中不重复计数**（用集合求交，去重）。

    Args:
        retrieved_ids: 检索结果标识，按相关度降序。
        expected_ids: 期望命中的标识集合。
        k: 只看前 k 个（Recall@k）。

    Returns:
        0.0 ~ 1.0 的召回比例。
    """
    expected: Set[str] = {str(x) for x in expected_ids}
    if not expected or k <= 0:
        return 0.0
    top: List[str] = [str(x) for x in list(retrieved_ids)[:k]]
    if not top:
        return 0.0
    return len(expected & set(top)) / len(expected)


def hit_at_k(
    retrieved_ids: Sequence[str],
    expected_ids: Iterable[str],
    k: int = 5,
) -> float:
    """前 k 个结果是否**至少命中一个**期望文档（0.0 / 1.0）。

    与 ``recall_at_k`` 的区别：多期望标注时，hit-rate 只要"沾到一个"就算 1，
    而 Recall@k 按命中比例给分。RAG 场景两者都记，方便定位是"完全没召回"
    还是"召回不全"。
    """
    expected: Set[str] = {str(x) for x in expected_ids}
    if not expected or k <= 0:
        return 0.0
    top: Set[str] = {str(x) for x in list(retrieved_ids)[:k]}
    return 1.0 if (expected & top) else 0.0


def mrr(
    retrieved_ids: Sequence[str],
    expected_ids: Iterable[str],
) -> float:
    """首个命中结果的排名倒数（Mean Reciprocal Rank 的单项）。

    定义：找到第一个 ``doc_id ∈ expected`` 的位置 i（从 1 开始），返回 ``1/i``；
    全程未命中返回 0.0。

    边界行为：
        - 期望为空集合时返回 0.0；
        - retrieved 为空时返回 0.0。
    """
    expected: Set[str] = {str(x) for x in expected_ids}
    if not expected:
        return 0.0
    for index, doc_id in enumerate(retrieved_ids, start=1):
        if str(doc_id) in expected:
            return 1.0 / index
    return 0.0


# =====================================================================
# 二、分类类指标（意图识别）
# =====================================================================
def accuracy(golds: Sequence[Any], preds: Sequence[Any]) -> float:
    """逐样本精确匹配准确率。

    边界行为：
        - 两个序列长度不一致时抛 ``ValueError``（静默对齐会导致评测失真）；
        - 空输入返回 0.0。
    """
    if len(golds) != len(preds):
        raise ValueError(
            f"gold 与 pred 长度不一致：{len(golds)} vs {len(preds)}"
        )
    if not golds:
        return 0.0
    hit: int = sum(1 for g, p in zip(golds, preds) if str(g) == str(p))
    return hit / len(golds)


def confusion_counts(
    golds: Sequence[Any],
    preds: Sequence[Any],
) -> Dict[str, Dict[str, int]]:
    """混淆计数矩阵：``{gold_label: {pred_label: count}}``。

    用于报告里定位"分错到哪去了"（例如 boundary 样本被分到相邻意图）。
    """
    if len(golds) != len(preds):
        raise ValueError("gold 与 pred 长度不一致")
    matrix: Dict[str, Dict[str, int]] = {}
    for gold, pred in zip(golds, preds):
        gold_key, pred_key = str(gold), str(pred)
        matrix.setdefault(gold_key, {})
        matrix[gold_key][pred_key] = matrix[gold_key].get(pred_key, 0) + 1
    return matrix


def per_class_metrics(
    golds: Sequence[Any],
    preds: Sequence[Any],
    label: Any,
) -> Dict[str, float]:
    """单类别 P / R / F1（macro 口径由调用方对各类平均）。

    边界行为：分母为 0 时该指标返回 0.0（而非抛异常），便于报告直接渲染。
    """
    tp: int = sum(1 for g, p in zip(golds, preds) if str(g) == str(label) and str(p) == str(label))
    fp: int = sum(1 for g, p in zip(golds, preds) if str(g) != str(label) and str(p) == str(label))
    fn: int = sum(1 for g, p in zip(golds, preds) if str(g) == str(label) and str(p) != str(label))

    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1: float = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "support": float(tp + fn)}


# =====================================================================
# 三、通用统计
# =====================================================================
def mean(values: Iterable[float]) -> float:
    """算术平均；空输入返回 0.0（不抛异常，报告可直接渲染）。"""
    items: List[float] = [float(v) for v in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def percentile(values: Iterable[float], p: float) -> float:
    """线性插值分位数（与 numpy.percentile 默认 ``linear`` 口径一致）。

    实现（便于面试复述）：
        1. 升序排序得到 x[0..n-1]；
        2. 取虚拟下标 ``pos = (n - 1) * p / 100``；
        3. ``lo = floor(pos)``、``hi = ceil(pos)``，权重 ``w = pos - lo``；
        4. 返回 ``x[lo] * (1 - w) + x[hi] * w``。

    边界行为：
        - 空输入返回 0.0；
        - p 越界时被截断到 [0, 100]；
        - 单元素输入恒返回该元素。

    Args:
        values: 数值序列（如各请求延迟毫秒）。
        p: 百分位（0~100），P95 传 95。

    Returns:
        分位数值。
    """
    items: List[float] = sorted(float(v) for v in values)
    if not items:
        return 0.0
    pct: float = min(100.0, max(0.0, float(p)))
    if len(items) == 1:
        return items[0]
    pos: float = (len(items) - 1) * pct / 100.0
    lo: int = int(pos)
    hi: int = min(lo + 1, len(items) - 1)
    weight: float = pos - lo
    return items[lo] * (1.0 - weight) + items[hi] * weight


def token_cost(
    input_tokens: int,
    output_tokens: int,
    price_in_per_1k: float,
    price_out_per_1k: float,
) -> float:
    """按「输入/输出分开计价」估算单次调用金额。

    公式：``input_tokens/1000*price_in + output_tokens/1000*price_out``

    说明：这是**纯函数**，价格由调用方从 ``monitoring/model_pricing.yaml`` 传入；
    真实账单还受缓存命中、批量折扣影响，本函数给的是量级参考。
    """
    safe_in: float = max(0.0, float(input_tokens))
    safe_out: float = max(0.0, float(output_tokens))
    return (safe_in / 1000.0) * float(price_in_per_1k) + (
        safe_out / 1000.0
    ) * float(price_out_per_1k)


# =====================================================================
# 四、工具调用指标
# =====================================================================
def normalize_value(value: Any) -> str:
    """把参数值归一化为可比较字符串。

    规则：None -> ""；数字 -> 去掉多余 0 的可解析形式；字符串 -> strip + 小写。
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:g}"
    return str(value).strip().casefold()


def key_arg_matches(
    predicted_args: Optional[Mapping[str, Any]],
    expected_key_args: Optional[Mapping[str, Any]],
) -> Dict[str, bool]:
    """逐项判定「关键参数」是否命中。

    判定规则（宽松但可解释）：
        - 仅对 ``expected_key_args`` 中出现的键判定（未声明的参数不扣分）；
        - 键名必须存在（或用 ``aliases`` 声明替代键名时命中任一即算）；
        - 值比较：归一化后**相等**，或（字符串长度 >= 2 时）**包含**即算命中。
          包含判定是为了容忍模型把查询词写得更长（如 query="智能客服平台专业版报价"
          对 expected query="专业版报价"）。

    未声明的期望键会导致该项**不参与**统计（返回空 dict），由调用方决定是否
    视为"该用例无参数标注"。

    Args:
        predicted_args: 模型实际传入的工具参数字典。
        expected_key_args: 标注的关键参数；可用 ``{"param": {"value": ..., "aliases": [...]}}``
            的扩展形式声明别名。

    Returns:
        ``{参数名: 是否命中}``。
    """
    predicted: Dict[str, Any] = dict(predicted_args or {})
    expected: Dict[str, Any] = dict(expected_key_args or {})
    if not expected:
        return {}

    result: Dict[str, bool] = {}
    for key, spec in expected.items():
        aliases: List[str] = []
        expected_value: Any = spec
        if isinstance(spec, Mapping) and ("value" in spec or "aliases" in spec):
            expected_value = spec.get("value")
            aliases = [str(a) for a in (spec.get("aliases") or [])]

        candidate_keys: List[str] = [key, *aliases]
        actual_present: bool = False
        actual_ok: bool = False
        for candidate in candidate_keys:
            if candidate in predicted:
                actual_present = True
                actual_norm: str = normalize_value(predicted[candidate])
                expected_norm: str = normalize_value(expected_value)
                if actual_norm == expected_norm:
                    actual_ok = True
                    break
                if (
                    expected_norm
                    and len(expected_norm) >= 2
                    and expected_norm in actual_norm
                ):
                    actual_ok = True
                    break
        result[key] = bool(actual_present and actual_ok)
    return result


def key_arg_recall(
    predicted_args: Optional[Mapping[str, Any]],
    expected_key_args: Optional[Mapping[str, Any]],
) -> Optional[float]:
    """关键参数命中率；无参数标注时返回 ``None``（表示该用例不计入均值）。"""
    matches = key_arg_matches(predicted_args, expected_key_args)
    if not matches:
        return None
    return sum(1 for ok in matches.values() if ok) / len(matches)


def tool_call_success(
    called_tools: Sequence[str],
    expected_tool: str,
    acceptable_tools: Optional[Iterable[str]] = None,
    *,
    inspect_first_n: Optional[int] = None,
) -> bool:
    """单条用例的工具调用是否成功。

    判定：调用序列的前 ``inspect_first_n`` 个（None = 全部）中出现
    ``expected_tool`` **或** ``acceptable_tools`` 中任一工具，即算成功。

    为什么允许 ``acceptable_tools``：一个需求往往有多个同样正确的工具
    （如"查线索"可用 Excel 只读工具，也可先用 file_list 定位再读），
    只认唯一工具会把"合理但不同"的路径误判为失败。标注时把这类
    都写进 ``acceptable_tools``，评测才公平、可辩护。
    """
    allowed: Set[str] = {str(expected_tool)}
    allowed.update(str(t) for t in (acceptable_tools or []))
    calls: List[str] = [str(t) for t in called_tools]
    if inspect_first_n is not None:
        calls = calls[: max(0, int(inspect_first_n))]
    return any(call in allowed for call in calls)


def tool_call_success_rate(
    cases: Sequence[Mapping[str, Any]],
) -> float:
    """批量工具调用成功率。

    每个 case 需含：``called_tools``（列表）、``expected_tool``、
    可选 ``acceptable_tools``、可选 ``inspect_first_n``。

    边界行为：空输入返回 0.0。
    """
    if not cases:
        return 0.0
    ok_count: int = sum(
        1
        for case in cases
        if tool_call_success(
            called_tools=case.get("called_tools") or [],
            expected_tool=str(case.get("expected_tool") or ""),
            acceptable_tools=case.get("acceptable_tools"),
            inspect_first_n=case.get("inspect_first_n"),
        )
    )
    return ok_count / len(cases)


# =====================================================================
# 五、结果聚合
# =====================================================================
def aggregate_intent_results(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """聚合意图分类明细 -> 报告用指标字典。

    Args:
        records: 每项含 ``gold``、``pred``、``latency_ms``、可选 ``boundary``、
            ``input_tokens``、``output_tokens``。

    Returns:
        ``{intent_accuracy, intent_scored, boundary_accuracy, boundary_total,
        boundary_scored, mean_latency_ms, ...}``。

    零分母口径（对齐 RAG 侧 ``skipped_unanswerable``）：
        - 没有任何意图样本时 ``intent_accuracy=None``；
        - 没有边界样本（如 ``--limit`` 切片漏掉了 B 类样本）时
          ``boundary_accuracy=None``。
        ``None`` 表示"未采集/不适用"，**不参与 gate 判定**，报告渲染为
        "不可用"——避免把"没跑到样本"误判成准确率 0 的违规。

    最小样本量口径：
        指标照常算出来供人工诊断，但当样本数低于下限时会写入返回值的
        ``sample_insufficient`` 字典，由 ``report.flatten_metrics`` 把它们挡在
        质量门之外。判定的不是"分数低"，而是"这个分数还不足以被判定"。
    """
    golds: List[Any] = [r.get("gold") for r in records]
    preds: List[Any] = [r.get("pred") for r in records]
    boundary_records: List[Mapping[str, Any]] = [r for r in records if r.get("boundary")]
    latencies: List[float] = [
        float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None
    ]
    insufficient: Dict[str, Dict[str, int]] = {}

    # 边界准确率：n=0 是"未采集"（None）；0<n<5 是"样本不足"（保留原值但不进质量门）
    boundary_accuracy: Optional[float] = None
    if boundary_records:
        boundary_accuracy = accuracy(
            [r.get("gold") for r in boundary_records],
            [r.get("pred") for r in boundary_records],
        )
        boundary_flag: Optional[Dict[str, int]] = sample_gate(
            len(boundary_records), MIN_SAMPLES_BOUNDARY
        )
        if boundary_flag:
            insufficient["boundary_accuracy"] = boundary_flag

    # P95 延迟：样本数不足时插值会退化成取极值，只留给人看，不进质量门
    latency_flag: Optional[Dict[str, int]] = sample_gate(
        len(latencies), MIN_SAMPLES_PERCENTILE
    )
    if latency_flag:
        insufficient["p95_latency_ms"] = latency_flag

    return {
        "total": len(records),
        "intent_scored": len(records),
        "intent_accuracy": accuracy(golds, preds) if records else None,
        "boundary_total": len(boundary_records),
        "boundary_scored": len(boundary_records),
        "boundary_accuracy": boundary_accuracy,
        "mean_latency_ms": mean(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "input_tokens": int(sum(int(r.get("input_tokens") or 0) for r in records)),
        "output_tokens": int(sum(int(r.get("output_tokens") or 0) for r in records)),
        "confusion": confusion_counts(golds, preds),
        "sample_insufficient": insufficient,
    }


def aggregate_rag_results(
    records: Sequence[Mapping[str, Any]],
    *,
    k: int = 5,
) -> Dict[str, Any]:
    """聚合 RAG 检索明细 -> 报告用指标字典。

    Args:
        records: 每项含 ``retrieved_ids``（降序）、``expected_ids``、可选
            ``retrieved_doc_names`` / ``expected_doc_names``（当 doc_id 尚未回填时
            用文件名兜底匹配）、``latency_ms``。

    Returns:
        ``{total, recall@k, hit@k, mrr, mean_latency_ms, p95_latency_ms}``。
    """
    recalls: List[float] = []
    hits: List[float] = []
    mrrs: List[float] = []
    skipped_unanswerable: int = 0
    for record in records:
        # 「不可答」样本（知识库没有答案、模型应明确拒答）没有 expected_doc_id。
        # 它们**不参与检索指标的分母** —— 否则会因为"本来就没有正确答案"而恒为 0，
        # 无端拉低 Recall@5，让人误判成检索能力退化。
        # 这类样本由 answer_quality 的 must_abstain 单独判分（测的是拒答/幻觉）。
        if record.get("unanswerable"):
            skipped_unanswerable += 1
            continue

        expected_ids: List[str] = list(record.get("expected_ids") or [])
        retrieved_ids: List[str] = list(record.get("retrieved_ids") or [])
        expected_names: List[str] = list(record.get("expected_doc_names") or [])
        retrieved_names: List[str] = list(record.get("retrieved_doc_names") or [])

        # doc_id 未回填时用文件名兜底（两条通道取较优者，避免因 ID 缺失误判为 0）
        recall_id: float = recall_at_k(retrieved_ids, expected_ids, k)
        recall_name: float = recall_at_k(retrieved_names, expected_names, k)
        mrr_id: float = mrr(retrieved_ids, expected_ids)
        mrr_name: float = mrr(retrieved_names, expected_names)

        recalls.append(max(recall_id, recall_name))
        hits.append(
            max(
                hit_at_k(retrieved_ids, expected_ids, k),
                hit_at_k(retrieved_names, expected_names, k),
            )
        )
        mrrs.append(max(mrr_id, mrr_name))

    latencies: List[float] = [
        float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None
    ]
    insufficient: Dict[str, Dict[str, int]] = {}
    latency_flag: Optional[Dict[str, int]] = sample_gate(
        len(latencies), MIN_SAMPLES_PERCENTILE
    )
    if latency_flag:
        insufficient["p95_latency_ms"] = latency_flag

    return {
        "total": len(records),
        "scored": len(recalls),
        "skipped_unanswerable": skipped_unanswerable,
        f"recall@{k}": mean(recalls),
        f"hit@{k}": mean(hits),
        "mrr": mean(mrrs),
        "mean_latency_ms": mean(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "sample_insufficient": insufficient,
    }


def aggregate_tool_results(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """聚合工具调用明细 -> 报告用指标字典。

    Args:
        records: 每项含 ``expected_tool``、``called_tools``、可选
            ``acceptable_tools``、``arg_matches``（``key_arg_matches`` 的输出）、
            ``latency_ms``、``error``。

    Returns:
        ``{total, tool_success_rate, key_arg_recall, error_rate,
        mean_latency_ms, p95_latency_ms}``。
    """
    arg_ratios: List[float] = []
    for record in records:
        matches: Mapping[str, bool] = record.get("arg_matches") or {}
        if matches:
            arg_ratios.append(sum(1 for ok in matches.values() if ok) / len(matches))

    latencies: List[float] = [
        float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None
    ]
    error_count: int = sum(1 for r in records if r.get("error"))

    insufficient: Dict[str, Dict[str, int]] = {}
    latency_flag: Optional[Dict[str, int]] = sample_gate(
        len(latencies), MIN_SAMPLES_PERCENTILE
    )
    if latency_flag:
        insufficient["p95_latency_ms"] = latency_flag

    return {
        "total": len(records),
        "tool_success_rate": tool_call_success_rate(records),
        "key_arg_recall": mean(arg_ratios),
        "key_arg_cases": len(arg_ratios),
        "error_rate": (error_count / len(records)) if records else 0.0,
        "mean_latency_ms": mean(latencies),
        "p95_latency_ms": percentile(latencies, 95),
        "sample_insufficient": insufficient,
    }


# =====================================================================
# 六、质量门（阈值判定）
# =====================================================================
def evaluate_gate(
    results: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> List[str]:
    """把「结果 + 阈值配置」判成违规列表（空列表 = 通过）。

    配置形状（``evals/thresholds.yaml``）::

        intent:
          accuracy_min: 0.85
        rag:
          recall_at_5_min: 0.75
          mrr_min: 0.60
        tool:
          call_success_min: 0.80
        cost:
          avg_tokens_per_turn_max: 4000
          p95_latency_ms_max: 8000

    约定：``*_min`` 为下界（低于即违规），``*_max`` 为上界（高于即违规）；
    键不存在（结果或阈值任一侧缺）时**跳过**该条，不误报。
    这样「先建 eval、后发数据」的过渡期不会因为指标没跑出来就把 CI 搞红。

    Returns:
        人类可读的违规说明列表，例如
        ``["intent.accuracy_min 违规：intent_accuracy=0.800 < 0.850"]``。
    """
    violations: List[str] = []

    pairs = [
        ("intent", "accuracy_min", "intent_accuracy", "min"),
        ("intent", "boundary_accuracy_min", "boundary_accuracy", "min"),
        ("rag", "recall_at_5_min", "recall@5", "min"),
        ("rag", "mrr_min", "mrr", "min"),
        ("rag", "hit_at_5_min", "hit@5", "min"),
        ("tool", "call_success_min", "tool_success_rate", "min"),
        ("tool", "key_arg_recall_min", "key_arg_recall", "min"),
        ("cost", "avg_tokens_per_turn_max", "avg_tokens_per_turn", "max"),
        ("cost", "p95_latency_ms_max", "p95_latency_ms", "max"),
        # 生成层（答案质量）：需要先跑 evals/tools/grade_answers.py 才有数据，
        # 缺则跳过——不因为"还没跑生成层"就把 CI 搞红。
        ("answer", "pass_rate_min", "answer_pass_rate", "min"),
        ("answer", "abstain_correct_rate_min", "abstain_correct_rate", "min"),
    ]

    for section, threshold_key, result_key, direction in pairs:
        section_cfg: Mapping[str, Any] = thresholds.get(section) or {}
        if threshold_key not in section_cfg:
            continue
        if result_key not in results:
            continue
        # None = 零分母/未采集（N/A），不参与阈值判定，避免把"没跑到样本"
        # 误判成指标违规（与键缺失同等处理）。
        if results[result_key] is None:
            continue
        limit: float = float(section_cfg[threshold_key])
        actual: float = float(results[result_key])
        if direction == "min" and actual < limit:
            violations.append(
                f"{section}.{threshold_key} 违规：{result_key}={actual:.4f} < {limit:.4f}"
            )
        if direction == "max" and actual > limit:
            violations.append(
                f"{section}.{threshold_key} 违规：{result_key}={actual:.4f} > {limit:.4f}"
            )

    return violations
