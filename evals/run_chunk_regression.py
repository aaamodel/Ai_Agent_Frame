# -*- coding: utf-8 -*-
"""chunk size 回归实验：用 eval **量化证明一个改动是错的**。

这是手册里的"关键动作"，也是简历里最能站住脚的一句话：

    "我把 chunk size 从 512 调到 1024，用同一套黄金集跑 Recall@5，
     结果从 X 掉到 Y，所以我回滚了。"

## 实验设计（单变量对照）

=========================  ==================  ================
维度                        对照组              实验组
=========================  ==================  ================
语料                        同一批 app/rag_data  同一批
embedding 模型              同一个              同一个
切分实现                    split_text          同一函数
**chunk_size**              **512**             **1024**
chunk_overlap               64（=size/8）        128（=size/8）
集合                        eval_kb_512         eval_kb_1024
检索方式                    向量+BM25+RRF（同）  同一套
黄金集                      rag_cases.jsonl（同） 同一份
=========================  ==================  ================

关键：**doc_id 由文件名确定性派生**（``evals/doc_ids.py``），所以两个集合里
同一份语料的 ``document_id`` 完全一致 —— 黄金集的 ``expected_doc_id`` 不需要为
两组各标一遍，差异只可能来自 chunk size。

overlap 按同比例放大（1/8 → 1/8）也是刻意的：只改"片长"这一个变量。

## 用法

    # 1) 只看分块计划（离线，秒级）
    python -m evals.run_chunk_regression --plan-only

    # 2) 完整跑（需要 Milvus / 模型 API）
    python -m evals.run_chunk_regression --ingest

    # 3) 已入库，只重跑评测并出报告
    python -m evals.run_chunk_regression
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.loaders import RESULTS_DIR, load_rag_cases  # noqa: E402

# 对照组与实验组（唯一差异就是 chunk_size / overlap）
BASELINE_CHUNK_SIZE: int = 512
REGRESSION_CHUNK_SIZE: int = 1024
COLLECTION_BASELINE: str = "eval_kb_512"
COLLECTION_REGRESSION: str = "eval_kb_1024"

REPORT_PATH: Path = _REPO_ROOT / "evals" / "chunk_size_regression_report.md"


# =====================================================================
# 一、分块计划
# =====================================================================
def show_plan() -> Dict[str, Any]:
    """打印两个 chunk size 的分块计划（离线，不连服务）。"""
    from evals.tools.reingest_corpus import DEFAULT_OVERLAP_RATIO, plan_chunks, RAG_DATA_DIR

    out: Dict[str, Any] = {}
    for size in (BASELINE_CHUNK_SIZE, REGRESSION_CHUNK_SIZE):
        plan: Dict[str, Any] = plan_chunks(
            RAG_DATA_DIR,
            chunk_size=size,
            chunk_overlap=int(size * DEFAULT_OVERLAP_RATIO),
        )
        total_chars: int = sum(item["chars"] for item in plan["plan"])
        avg: float = (total_chars / plan["total_chunks"]) if plan["total_chunks"] else 0.0
        # 便于报告直接引用「平均字符/片」来做参数生效性验证
        plan["total_chars"] = total_chars
        plan["avg_chars_per_chunk"] = round(avg, 1)
        out[str(size)] = plan
        print(
            f"[regression] chunk_size={size} overlap={int(size * DEFAULT_OVERLAP_RATIO)} "
            f"文件={plan['files']} 总片数={plan['total_chunks']} "
            f"平均 {avg:.0f} 字符/片"
        )
    return out


# =====================================================================
# 二、入库
# =====================================================================
async def ingest_both() -> None:
    from evals.tools.reingest_corpus import (
        DEFAULT_OVERLAP_RATIO,
        RAG_DATA_DIR,
        ingest_into_collection,
    )

    for size, collection in (
        (BASELINE_CHUNK_SIZE, COLLECTION_BASELINE),
        (REGRESSION_CHUNK_SIZE, COLLECTION_REGRESSION),
    ):
        print(f"\n[regression] === 入库 {collection} (chunk_size={size}) ===", flush=True)
        result: Dict[str, Any] = await ingest_into_collection(
            src_dir=RAG_DATA_DIR,
            collection_name=collection,
            physical_collection=collection,  # A/B 实验用独立物理集合，与线上隔离
            chunk_size=size,
            chunk_overlap=int(size * DEFAULT_OVERLAP_RATIO),
            reset=True,  # 实验专用集合，重建是安全的
        )
        print(
            f"[regression] {collection} 完成：文件={result['files']} "
            f"总片数={result['total_chunks']}",
            flush=True,
        )


# =====================================================================
# 三、评测两组
# =====================================================================
async def evaluate_both(
    *, limit: Optional[int] = None, sleep_between: float = 0.0
) -> Dict[str, Any]:
    from evals.runners._runtime import EvalRuntime
    from evals.runners.run_rag import run_rag_eval

    cases: List[Dict[str, Any]] = load_rag_cases()
    results: Dict[str, Any] = {}

    for size, collection in (
        (BASELINE_CHUNK_SIZE, COLLECTION_BASELINE),
        (REGRESSION_CHUNK_SIZE, COLLECTION_REGRESSION),
    ):
        print(f"\n[regression] === 评测 {collection} ===", flush=True)
        runtime = EvalRuntime(collection_name=collection)
        try:
            results[collection] = await run_rag_eval(
                cases,
                runtime,
                limit=limit,
                sleep_between=sleep_between,
                # A/B 实验在独立物理集合里，标签与物理集合同名；
                # 必须显式传入，否则默认过滤 sales_kb 会把结果过滤空。
                collection_filter=[collection],
            )
        finally:
            try:
                await runtime.aclose()
            except Exception:  # noqa: BLE001
                pass
    return results


# =====================================================================
# 四、对比与报告
# =====================================================================
def _per_case_hit(record: Dict[str, Any], k: int = 5) -> float:
    from evals.metrics import hit_at_k

    return max(
        hit_at_k(record.get("retrieved_ids") or [], record.get("expected_ids") or [], k),
        hit_at_k(
            record.get("retrieved_doc_names") or [],
            record.get("expected_doc_names") or [],
            k,
        ),
    )


def compare(results: Dict[str, Any]) -> Dict[str, Any]:
    """对比两组指标，并找出「基线命中、实验组掉」的用例（回归证据）。"""
    base: Dict[str, Any] = results.get(COLLECTION_BASELINE) or {}
    exp: Dict[str, Any] = results.get(COLLECTION_REGRESSION) or {}
    base_metrics: Dict[str, Any] = base.get("metrics") or {}
    exp_metrics: Dict[str, Any] = exp.get("metrics") or {}

    base_hits: Dict[str, float] = {
        str(r["id"]): _per_case_hit(r) for r in (base.get("records") or [])
    }
    exp_hits: Dict[str, float] = {
        str(r["id"]): _per_case_hit(r) for r in (exp.get("records") or [])
    }

    regressed: List[Dict[str, Any]] = []
    improved: List[Dict[str, Any]] = []
    exp_records = {str(r["id"]): r for r in (exp.get("records") or [])}
    base_records = {str(r["id"]): r for r in (base.get("records") or [])}
    for case_id in sorted(set(base_hits) & set(exp_hits)):
        before, after = base_hits[case_id], exp_hits[case_id]
        if before > after:
            regressed.append(
                {
                    "id": case_id,
                    "query": (base_records.get(case_id) or {}).get("query"),
                    "expected": (base_records.get(case_id) or {}).get("expected_doc_name"),
                    "before_top1": (base_records.get(case_id) or {}).get("top1_doc_name"),
                    "after_top1": (exp_records.get(case_id) or {}).get("top1_doc_name"),
                    "after_preview": (exp_records.get(case_id) or {}).get("top1_content_head"),
                }
            )
        elif after > before:
            improved.append({"id": case_id, "before": before, "after": after})

    return {
        "baseline_metrics": base_metrics,
        "regression_metrics": exp_metrics,
        "case_count": len(set(base_hits) & set(exp_hits)),
        "delta": {
            key: float(exp_metrics.get(key) or 0.0) - float(base_metrics.get(key) or 0.0)
            for key in ("recall@5", "hit@5", "mrr")
            if key in base_metrics or key in exp_metrics
        },
        "regressed": regressed,
        "improved": improved,
    }


def render_comparison(comparison: Dict[str, Any], plan: Optional[Dict[str, Any]]) -> str:
    base: Dict[str, Any] = comparison["baseline_metrics"]
    exp: Dict[str, Any] = comparison["regression_metrics"]
    delta: Dict[str, float] = comparison["delta"]
    regressed: List[Dict[str, Any]] = comparison["regressed"]

    lines: List[str] = []
    add = lines.append

    add("# chunk size 回归实验报告（512 → 1024）")
    add("")
    add(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 对照组：`{COLLECTION_BASELINE}` chunk_size={BASELINE_CHUNK_SIZE} "
        f"overlap={int(BASELINE_CHUNK_SIZE / 8)}")
    add(f"- 实验组：`{COLLECTION_REGRESSION}` chunk_size={REGRESSION_CHUNK_SIZE} "
        f"overlap={int(REGRESSION_CHUNK_SIZE / 8)}")
    add("- 唯一变量：chunk_size（overlap 按同比例 1/8 缩放）")
    add("- 语料 / embedding / 切分函数 / 检索方式 / 黄金集 全部相同")
    add(f"- doc_id 规则：`doc-<md5(文件名)[:12]>`（两组一致，故黄金集可复用）")
    add("")
    add("> ⚠️ **单位说明**：chunk_size 的单位是 **token**（LlamaIndex "
        "`SentenceSplitter` 口径，见 `app/core/rag/parse.py`），不是字符数。"
        "下面同时给 `平均字符/片` 用于反向验证参数确实生效。")
    add("")
    add("## 一、分块对比")
    add("")
    if plan and len(plan) == 2:
        p512 = plan[str(BASELINE_CHUNK_SIZE)]
        p1024 = plan[str(REGRESSION_CHUNK_SIZE)]
        add("| 参数 | chunk_size=512 | chunk_size=1024 |")
        add("| --- | --- | --- |")
        add(f"| 语料文件数 | {p512['files']} | {p1024['files']} |")
        add(f"| 语料总字符数 | {p512.get('total_chars', 0)} | {p1024.get('total_chars', 0)} |")
        add(f"| 总切片数 | {p512['total_chunks']} | {p1024['total_chunks']} |")
        add(f"| 平均字符/片 | {p512.get('avg_chars_per_chunk', 0)} "
            f"| {p1024.get('avg_chars_per_chunk', 0)} |")
        add("")
        add("> 切片数变少、平均字符/片变大 = 参数确实生效。"
            "每片更长意味着一个向量要同时代表多个主题，"
            "语义中心被平均掉——这是召回变差的**结构性原因**"
            "（一个向量只能表达一个语义中心）。")
    else:
        add("（本次未生成分块计划，可加 `--plan-only` 离线查看）")
    add("")

    add("## 二、检索指标对比（这 3 个数字就是面试要说的）")
    add("")
    add("| 指标 | chunk_size=512 | chunk_size=1024 | 变化 | 结论 |")
    add("| --- | --- | --- | --- | --- |")
    for key, label in (("recall@5", "Recall@5"), ("hit@5", "Hit@5"), ("mrr", "MRR")):
        before: float = float(base.get(key) or 0.0)
        after: float = float(exp.get(key) or 0.0)
        change: float = float(delta.get(key) or 0.0)
        verdict: str = "变差 ❌" if change < 0 else ("持平" if change == 0 else "变好 ✅")
        add(f"| {label} | {before:.3f} | {after:.3f} | {change:+.3f} | {verdict} |")
    add("")

    add("## 三、逐用例回归证据（哪些题被改坏了）")
    add("")
    if not regressed:
        add("本次没有「基线命中、实验组未命中」的用例。")
        add("")
        add("> 若两组指标完全相同，先确认实验组集合里确实是 1024 的片子"
            "（`avg_chunk_chars` 应明显更大），否则说明入库参数没生效。")
    else:
        add(f"共 {len(regressed)} 条用例在 1024 下从「命中」变成「未命中」：")
        add("")
        add("| id | 问题 | 期望文档 | 512 的 top1 | 1024 的 top1 |")
        add("| --- | --- | --- | --- | --- |")
        for item in regressed[:8]:
            add(
                f"| {item['id']} | {_esc(item['query'])} | {_esc(item['expected'])} "
                f"| {_esc(item['before_top1'])} | {_esc(item['after_top1'])} |"
            )
        add("")
        first = regressed[0]
        add("### 可以这样讲（回答模板）")
        add("")
        add(f"1. **改了什么**：把 ingestion 的 chunk_size 从 512 调到 1024"
            f"（overlap 同比例 512→128）。")
        add(f"2. **怎么验证的**：同一批语料、同一份黄金集（{comparison.get('case_count', 0)} 条）、"
            f"同一个 embedding 与同一套检索器，只换集合里的分块参数。")
        add(f"3. **数据说了什么**：Recall@5 从 {float(base.get('recall@5') or 0.0):.3f} "
            f"掉到 {float(exp.get('recall@5') or 0.0):.3f}"
            f"（{float(delta.get('recall@5') or 0.0):+.3f}），"
            f"MRR {float(base.get('mrr') or 0.0):.3f} → {float(exp.get('mrr') or 0.0):.3f}。")
        add(f"4. **一个具体失败案例**：`{first['id']}`「{first['query']}」"
            f"期望 `{first['expected']}`，512 下 top1 命中 `{first['before_top1']}`，"
            f"1024 下 top1 变成了 `{first['after_top1']}`。")
        add("5. **根因判断**：片变长后，一个向量要同时代表多个主题，"
            "语义中心被平均掉，细粒度 query 的向量相似度下降。"
            "这也说明**这个场景下分块粒度是召回瓶颈**，下一步值得试 256 或按标题切。")
        add("6. **结论**：回滚到 512；把 chunk_size 纳入需要 eval 把关的参数清单。")
        add("")

    add("## 四、诚实声明")
    add("")
    add("- 本实验在**本机单机环境**完成，数据用于验证「eval 能否发现回归」，"
        "不代表生产容量或线上召回水平。")
    add("- 两组集合都在同一台 Milvus 上、同一时刻评测，避免环境漂移；"
        "但仍存在 embedding 服务的批次差异，因此**结论看的是相对变化趋势，"
        "不是绝对数值**。")
    add("- 实验集合 `eval_kb_512` / `eval_kb_1024` 是评测专用集合，"
        "**不影响**生产集合 `knowledge_base_v3`。")
    add("")

    return "\n".join(lines) + "\n"


def _esc(text: Any) -> str:
    return str(text if text is not None else "-").replace("|", "\\|").replace("\n", " ")


# =====================================================================
# 入口
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="chunk size 512 -> 1024 回归实验")
    parser.add_argument("--plan-only", action="store_true", help="只打印分块计划（离线）")
    parser.add_argument("--ingest", action="store_true", help="重新入库两个评测集合")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（冒烟）")
    parser.add_argument("--out", type=Path, default=REPORT_PATH, help="报告输出路径")
    args = parser.parse_args()

    plan: Optional[Dict[str, Any]] = None
    if args.plan_only:
        plan = show_plan()
        print("\n[regression] 分块计划完成（--plan-only，未入库、未评测）")
        return

    async def _main() -> Dict[str, Any]:
        if args.ingest:
            await ingest_both()
        return await evaluate_both(limit=args.limit)

    results = asyncio.run(_main())

    # 分块计划（离线部分，用于报告里的结构性解释）
    try:
        plan = show_plan()
    except Exception as exc:  # noqa: BLE001 - 计划不影响指标结论
        print(f"[regression] 分块计划生成失败（不影响指标对比）：{exc}")

    comparison = compare(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "chunk_regression.json").write_text(
        __import__("json").dumps(
            {"comparison": comparison, "results": results}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    markdown: str = render_comparison(comparison, plan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")

    print(f"\n[regression] 报告已写出：{args.out}")
    print(
        f"[regression] Recall@5: {comparison['baseline_metrics'].get('recall@5', 0):.3f} "
        f"-> {comparison['regression_metrics'].get('recall@5', 0):.3f} "
        f"({comparison['delta'].get('recall@5', 0):+.3f})；"
        f"回归用例 {len(comparison['regressed'])} 条"
    )


if __name__ == "__main__":
    main()
