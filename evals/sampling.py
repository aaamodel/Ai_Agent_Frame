# -*- coding: utf-8 -*-
"""评测用例抽样——**分层抽样**，替代原来的「取文件前 N 条」切片。

## 为什么要分层（这是本模块存在的全部理由）

三份黄金集都是**按 id 顺序排列**的，天然带有"类别聚集"：

    intent_cases.jsonl : I01..I17（常规）在前，B01..B05（边界）在末尾
    rag_cases.jsonl    : R01..R27（可答）在前，R28..R32（应拒答）在末尾
    tool_cases.jsonl   : 按 expected_tool 成簇排列

所以 ``cases[:limit]`` 这种简单切片做冒烟时：

    --limit 3 → 只跑 I01/I02/I03，5 条边界样本一条都进不来
              → boundary_total = 0 → 边界准确率没有分母
              → 报告上呈现为"什么都没测到"，而不是"测了但没考好"

分层抽样的目标是：**给多少预算，就尽量覆盖多少种评测维度**，
让小样本也能反映整体分布，而不是永远抽到文件开头那几条。

## 算法（确定性，不含随机数）

1. **保护位**：先给"必须被覆盖"的分层各留 1 个名额（意图的边界样本、
   RAG 的应拒答样本）。预算不够时按声明顺序优先——宁可先守住最容易漏的那类。
2. **层间轮转**：剩余名额在所有分层之间**轮转**，每层每轮取 1 条。
   轮转天然带来层内多样性（同一个意图不会被一口气抽干），
   也避免大层独占预算。
3. 最后按**文件原始顺序**返回，保证结果与全量跑的顺序一致。

全程不含随机数，因此同样的 ``--limit N`` 永远得到同一批用例——
这是它能被 CI 依赖的前提。

## 用法

    from evals.sampling import select_cases
    selected = select_cases("intent", cases, limit=3)   # 必含 1 条边界样本
    selected = select_cases("rag", cases, limit=5)      # 必含 1 条应拒答样本
    selected = select_cases("tool", cases, limit=3)     # 每个工具尽量轮转覆盖

## 用例顺序：``order`` / ``seed``（冒烟测试专用）

上面那条确定性算法有个**冒烟场景下的副作用**：同样的 ``--limit N`` 永远抽同一批用例，
连着跑几次冒烟，测的还是那几条。所以增加一个顺序开关：

    order="file"    （默认）按文件原始顺序 —— 确定性，CI / 回归口径，结果可比
    order="random"  先打乱再分层 —— 每次抽到的**样本本身**不同，覆盖面更广

⚠️ 关键点在「**先打乱、再分层**」这个顺序：

    - 若只打乱最终结果 → 抽样集合不变，只是执行顺序变了，冒烟还是那几条（没解决问题）；
    - 先打乱输入列表，分层算法里的 ``free[0]``（每层取第一条）就会落在不同样本上
      → 抽样集合随种子变化，同时保护位（边界/应拒答）依然每层必进 1 条。

``seed`` 语义：

    seed=None  每次运行都不同（默认，真·随机冒烟）
    seed=N     固定种子 → 同一批用例、同一顺序，可复现（排查失败用例时用它锁定现场）
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# 用例顺序口径
ORDER_FILE: str = "file"
ORDER_RANDOM: str = "random"
DEFAULT_ORDER: str = ORDER_FILE

__all__ = [
    "ORDER_FILE",
    "ORDER_RANDOM",
    "DEFAULT_ORDER",
    "intent_stratum",
    "rag_stratum",
    "tool_stratum",
    "select_stratified",
    "select_cases",
]

Pred = Callable[[str], bool]


# =====================================================================
# 一、分层键（每种评测各自关心的维度）
# =====================================================================
def intent_stratum(case: Mapping[str, Any]) -> str:
    """意图集：``边界标记 × 期望意图叶子``。

    用「边界在前」的前缀写法，是为了让下面的保护位能用 ``startswith`` 精确挑出边界层。
    """
    tag: str = "boundary" if case.get("boundary") else "regular"
    return f"{tag}|{case.get('expected_intent') or 'unknown'}"


def rag_stratum(case: Mapping[str, Any]) -> str:
    """RAG 集：应拒答自成一层，可答的按期望文档分层。"""
    if case.get("unanswerable"):
        return "unanswerable"
    return f"doc={case.get('expected_doc_name') or 'unknown'}"


def tool_stratum(case: Mapping[str, Any]) -> str:
    """工具集：按期望工具分层，保证每个工具都有机会被抽到。"""
    return f"tool={case.get('expected_tool') or 'unknown'}"


# 保护位谓词：命中这些层的样本，在预算允许时必须至少进 1 条
BOUNDARY_PROTECTED: Tuple[Pred, ...] = (lambda key: key.startswith("boundary|"),)
UNANSWERABLE_PROTECTED: Tuple[Pred, ...] = (lambda key: key == "unanswerable",)


# =====================================================================
# 二、核心算法
# =====================================================================
def _group_cases(
    cases: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], str],
) -> Tuple[Dict[str, List[int]], List[str]]:
    """按下标分组，返回 ``({层: [下标...]}, [层出现顺序])``。

    用 dict 保序（Python 3.7+），因此 ``order`` 就是层在文件里的首次出现顺序，
    全程不依赖 Sorting/random，结果可复现。
    """
    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for index, case in enumerate(cases):
        stratum: str = str(key(case))
        if stratum not in groups:
            groups[stratum] = []
            order.append(stratum)
        groups[stratum].append(index)
    return groups, order


def select_stratified(
    cases: Sequence[Mapping[str, Any]],
    limit: Optional[int] = None,
    *,
    key: Callable[[Mapping[str, Any]], str],
    protected: Iterable[Pred] = (),
) -> List[Mapping[str, Any]]:
    """分层抽样，返回不超过 ``limit`` 条的用例子集。

    边界行为：
        - ``limit`` 为 ``None`` / ``0`` 或 ``>= len(cases)`` 时返回**全部**（不抽样）；
        - 保护位谓词按声明顺序执行，预算用尽即止，其余谓词被跳过；
        - 某个层被抽干后自动退出轮转，不会死循环。

    Args:
        cases: 完整用例列表。
        limit: 目标条数上限。
        key: 用例 -> 分层键。
        protected: 保护位谓词序列，每个谓词保证「至少一个所属层的样本」入选。

    Returns:
        抽样后的用例列表，**已按文件原始顺序排好**。
    """
    items: List[Mapping[str, Any]] = list(cases)
    if not limit or limit >= len(items):
        return items

    groups, order = _group_cases(items, key)
    taken: set = set()
    budget: int = int(limit)

    # ---- 阶段 1：保护位（预算不足时按声明顺序优先）----
    for predicate in protected:
        if budget <= 0:
            break
        for stratum in order:
            if not predicate(stratum):
                continue
            free: List[int] = [i for i in groups[stratum] if i not in taken]
            if free:
                taken.add(free[0])
                budget -= 1
                break

    # ---- 阶段 2：层间轮转，每层每轮取 1 条 ----
    exhausted: set = set()
    while budget > 0:
        progressed: bool = False
        for stratum in order:
            if budget <= 0:
                break
            if stratum in exhausted:
                continue
            free = [i for i in groups[stratum] if i not in taken]
            if not free:
                exhausted.add(stratum)
                continue
            taken.add(free[0])
            budget -= 1
            progressed = True
        if not progressed:  # 所有层都抽干了（理论上不会发生，防死循环）
            break

    return [items[i] for i in sorted(taken)]


# =====================================================================
# 三、按评测类型分发（唯一入口）
# =====================================================================
def select_cases(
    kind: str,
    cases: Sequence[Mapping[str, Any]],
    limit: Optional[int] = None,
    *,
    order: str = DEFAULT_ORDER,
    seed: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    """按评测类型抽样（可选随机顺序）。**三个 runner 一律走这里**，避免各自实现漂移。

    Args:
        kind: ``"intent"`` / ``"rag"`` / ``"tool"``。
        cases: 完整用例列表。
        limit: 目标条数上限；``None`` 表示全量。
        order: ``"file"``（默认，按文件顺序，结果确定）/ ``"random"``（先打乱再分层，
            每次抽到的样本与顺序都不同）。
        seed: 仅 ``order="random"`` 生效。``None`` = 每次运行都不同；
            传整数 = 可复现（同一 seed → 同一批用例 + 同一顺序）。

    Raises:
        ValueError: ``kind`` 不是上述三者之一，或 ``order`` 取值非法。
    """
    if order not in (ORDER_FILE, ORDER_RANDOM):
        raise ValueError(
            f"未知的 order：{order!r}（可选 {ORDER_FILE!r} / {ORDER_RANDOM!r}）"
        )

    items: List[Mapping[str, Any]] = list(cases)
    # ⚠️ 必须在分层**之前**打乱：分层算法的"每层取第一条"要落在随机样本上，
    # 否则只是执行顺序变了、抽样集合没变（冒烟还是永远测那几条）。
    if order == ORDER_RANDOM:
        random.Random(seed).shuffle(items)

    if kind == "intent":
        return select_stratified(
            items, limit, key=intent_stratum, protected=BOUNDARY_PROTECTED
        )
    if kind == "rag":
        return select_stratified(
            items, limit, key=rag_stratum, protected=UNANSWERABLE_PROTECTED
        )
    if kind == "tool":
        return select_stratified(items, limit, key=tool_stratum)
    raise ValueError(f"未知的评测类型：{kind!r}（可选 intent / rag / tool）")


def describe_selection(
    cases: Sequence[Mapping[str, Any]],
    order: str,
    seed: Optional[int],
) -> str:
    """生成一行"本次选了哪些用例"的说明，供 runner 打印/落盘（便于复现与审计）。"""
    ids: List[str] = [str(c.get("id")) for c in cases]
    if order == ORDER_RANDOM:
        scope = f"随机顺序（seed={'随机' if seed is None else seed}）"
    else:
        scope = "文件顺序"
    return f"{scope}，共 {len(ids)} 条: {', '.join(ids)}"


def add_order_cli_args(parser: Any) -> None:
    """给 runner 的 argparse 挂上 ``--order`` / ``--seed``（三个 runner 共用）。

    放在本模块是为了让"顺序口径"只有一个真源：选项名、默认值、help 文案的语义
    与 ``select_cases(order=..., seed=...)`` 始终对齐，不会出现某个 runner 漏改。
    """
    from argparse import ArgumentParser

    assert isinstance(parser, ArgumentParser)  # 仅用于类型提示，运行时零成本
    parser.add_argument(
        "--order",
        choices=[ORDER_FILE, ORDER_RANDOM],
        default=DEFAULT_ORDER,
        help=(
            "用例顺序：file（默认，按文件顺序，结果确定、可被 CI 依赖）/ "
            "random（先打乱再做分层抽样 —— 每次抽到的样本与顺序都不同，"
            "适合反复冒烟；分层保护位仍然生效，边界/应拒答样本仍必进 1 条）"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "仅 --order random 生效：不传 = 每次运行都不同；传整数 = 可复现"
            "（同一 seed 得到同一批用例与同一顺序，用于锁定失败现场）"
        ),
    )


def run_config_of(order: str, seed: Optional[int], limit: Optional[int]) -> Dict[str, Any]:
    """把本次运行的抽样配置打包进结果 JSON，便于事后复现/审计。"""
    return {"order": order, "seed": seed, "limit": limit}
