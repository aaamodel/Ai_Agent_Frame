# -*- coding: utf-8 -*-
"""Runner：RAG 检索离线评测（Recall@5 / Hit@5 / MRR）。

## 一个必须讲清的坑（本文件最核心的设计点）

``RAGService.retrieve_contexts`` 返回的 ``RetrievalResult.id`` 是
**LlamaIndex 的切片级 node_id**（UUID），**不是文档 ID**。文档身份只存在于
``metadata`` 里：

    metadata = {"document_id": ..., "chunk_index": ..., "filename": ..., "collection": ...}

所以"算 Recall@5"的正确口径是：**看前 5 个切片分别属于哪篇文档**，
而不是拿 node_id 去比对 ``expected_doc_id``。如果直接拿 node_id 当文档 ID 用，
Recall 会恒等于 0——这是这类评测最常见的低级错误。

本 runner 因此把每条用例记录成**双通道**：

    retrieved_ids        <- [metadata["document_id"] ...]   （主通道，文档级）
    retrieved_doc_names  <- [归一化 metadata["filename"] ...]（兜底通道）
    retrieved_chunk_ids  <- [RetrievalResult.id ...]         （仅排查用，不参与判分）

``aggregate_rag_results`` 会对两条通道分别算 Recall@5/Hit@5/MRR 并**取较优者**：

    - 评测目标是 ``reingest_corpus.py`` 建的集合 → document_id 通道命中；
    - 评测目标是线上真实集合（uuid4 的 doc_id，黄金集对不上）→ 文件名通道命中。

这样同一份黄金集在两种场景下都不会被误判成"没召回"。

## 为什么默认**不**指定任何集合

``metadata['collection']`` 只是"这批数据是哪次上传登记的"这一标签，一个物理集合里
往往同时躺着 ``sales_kb`` / ``enterprise_kb`` / 各次上传表单填的 ``collection_name``
（见 ``GET /vector/collections``）。而黄金集判分用的是**文档身份**（document_id /
文件名），跟标签无关。所以：

    - 默认：**不过滤**，在配置的物理集合内全量检索（多标签共存也不会漏召回）；
    - 只有做 chunk-size A/B 这类"物理集合里混了别的东西"的实验时，才用
      ``--collection-filter`` 主动收窄。

反例（旧行为）：默认写死 ``sales_kb`` 过滤，一旦线上数据登记的是别的标签，
检索结果被整段过滤掉，表现为 **Recall 恒等于 0**，且日志里看不出任何异常。

用法::

    # 默认：不传任何集合参数 —— 物理集合取 settings.milvus_kb_collection_name
    python -m evals.runners.run_rag

    # 需要收窄范围时才显式给逻辑标签（可逗号分隔多个）
    python -m evals.runners.run_rag --collection-filter sales_kb

    # chunk-size A/B 实验：物理集合指向独立实验集合
    python -m evals.runners.run_rag --collection eval_kb_512 \
        --collection-filter eval_kb_512 --out evals/_results/rag_512.json

    python -m evals.runners.run_rag --limit 3      # 冒烟：分层抽样 3 条
    python -m evals.runners.run_rag --limit 3 --order random          # 每次抽不同 3 条
    python -m evals.runners.run_rag --limit 3 --order random --seed 7 # 可复现的随机

⚠️ ``--limit`` 走的是**分层抽样**（见 ``evals/sampling.py``），不是"取前 N 条"。
本集的「应拒答」样本 R28..R32 排在文件末尾，若用简单切片，小样本下会一条都抽不到，
``skipped_unanswerable`` 永远为 0，拒答能力等于没被测到。

⚠️ ``--order`` 默认 ``file``（确定性，CI 依赖它）；``random`` 是"**先打乱再分层**"，
因此抽到的样本本身也会变（应拒答样本仍必进 1 条），连着冒烟不会永远测那几条。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.doc_ids import normalize_doc_name  # noqa: E402
from evals.metrics import aggregate_rag_results, format_metric  # noqa: E402
from evals.runners._runtime import EvalRuntime  # noqa: E402
from evals.sampling import (  # noqa: E402
    DEFAULT_ORDER,
    add_order_cli_args,
    describe_selection,
    run_config_of,
    select_cases,
)

# 与 thresholds.yaml::rag.recall_at_5_min 的 k 对齐
EVAL_K: int = 5
# 报告中每条用例保留的正文预览长度（仅用于失败归因，不参与判分）
_PREVIEW_CHARS: int = 120


def _extract_doc_fields(hits: Sequence[Any]) -> Dict[str, List[Any]]:
    """把 RetrievalResult 列表拆成「文档级」与「切片级」两组标识。"""
    doc_ids: List[str] = []
    doc_names: List[str] = []
    chunk_ids: List[str] = []
    scores: List[float] = []
    previews: List[str] = []

    for hit in hits:
        meta: Dict[str, Any] = dict(getattr(hit, "metadata", None) or {})
        # 文档级：优先 document_id；缺失时退回空串（由文件名通道兜底）
        doc_ids.append(str(meta.get("document_id") or ""))
        doc_names.append(normalize_doc_name(str(meta.get("filename") or "")))
        # 切片级：仅排查用
        chunk_ids.append(str(getattr(hit, "id", "") or ""))
        scores.append(round(float(getattr(hit, "score", 0.0) or 0.0), 6))
        previews.append(str(getattr(hit, "content", "") or "")[:_PREVIEW_CHARS])

    return {
        "doc_ids": doc_ids,
        "doc_names": doc_names,
        "chunk_ids": chunk_ids,
        "scores": scores,
        "previews": previews,
    }


async def _describe_target(
    rag_service: Any, collection_filter: Optional[Sequence[str]]
) -> None:
    """打印本次检索目标：物理集合 + 各逻辑标签的切片数。

    RAG 评测最常见的"全 0"假故障是**物理集合里根本没有向量**（或者语料被登记在
    别的标签下），而这种信息在逐条 `FAIL top1=-` 的输出里完全看不出来。这里先摊开，
    把"数据问题"和"检索问题"在日志层就分开。
    """
    physical: str = str(getattr(rag_service, "physical_collection_name", "") or "?")
    scope: str = (
        f"逻辑标签白名单={list(collection_filter)}"
        if collection_filter
        else "逻辑标签=不过滤（物理集合内全量检索）"
    )
    try:
        groups: Dict[str, Dict[str, int]] = await rag_service.list_logical_files()
    except Exception as exc:  # noqa: BLE001 - 目标描述失败不影响评测本身
        print(
            f"[rag] 检索目标：物理集合={physical} {scope}"
            f"（标签枚举失败：{type(exc).__name__}: {exc}）",
            flush=True,
        )
        return

    if not groups:
        print(
            f"[rag] ⚠️ 物理集合 [{physical}] 内没有任何向量，召回必然为 0。"
            "先入库语料（--dry-run 可先空跑看分块）：\n"
            "      python -m evals.tools.reingest_corpus --chunk-size 512",
            flush=True,
        )
        return

    summary: str = "，".join(
        f"{tag}={sum(files.values())}片" for tag, files in sorted(groups.items())
    )
    print(f"[rag] 检索目标：物理集合={physical} {scope}；现有标签：{summary}", flush=True)


async def run_rag_eval(
    cases: List[Dict[str, Any]],
    runtime: Optional[EvalRuntime] = None,
    *,
    limit: Optional[int] = None,
    top_k: int = EVAL_K,
    collection_filter: Optional[List[str]] = None,
    seed_bm25: bool = True,
    sleep_between: float = 0.0,
    order: str = DEFAULT_ORDER,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """跑完整 RAG 检索评测，返回 ``{"records": [...], "metrics": {...}}``。

    ``collection_filter`` 口径（``metadata['collection']`` 白名单）：
        - ``None``（默认）/ ``[]``：**不过滤**，物理集合内全量检索。物理集合里
          同时存在 sales_kb / enterprise_kb / 各次上传登记的逻辑标签属正常现象，
          黄金集按文档身份判分，不需要按标签裁剪；
        - 显式传非空列表：只检索白名单内的标签（chunk-size A/B 实验用）。
    """
    runtime = runtime or EvalRuntime()
    rag_service = runtime.get_rag_service()

    # BM25 通道必须预热，否则混合检索退化成纯向量，Recall 被系统性低估
    if seed_bm25:
        from evals.runners._runtime import seed_bm25 as do_seed

        await do_seed(rag_service)

    await _describe_target(rag_service, collection_filter)

    # 分层抽样而非 cases[:limit]：黄金集里 5 条「应拒答」样本排在文件末尾，
    # 简单切片会把它们整体漏掉（同样的坑见意图集的边界样本）。
    selected = select_cases("rag", cases, limit, order=order, seed=seed)
    print(f"[rag] 本次用例：{describe_selection(selected, order, seed)}", flush=True)
    records: List[Dict[str, Any]] = []

    for index, case in enumerate(selected, start=1):
        query: str = str(case["query"])
        expected_doc_name: str = str(case.get("expected_doc_name") or "")
        expected_doc_id: Optional[str] = case.get("expected_doc_id")

        started = time.perf_counter()
        error: Optional[str] = None
        fields: Dict[str, List[Any]] = {
            "doc_ids": [],
            "doc_names": [],
            "chunk_ids": [],
            "scores": [],
            "previews": [],
        }
        try:
            hits = await rag_service.retrieve_contexts(
                query=query,
                top_k=top_k,
                collection_names=collection_filter,
            )
            fields = _extract_doc_fields(hits)
        except Exception as exc:  # noqa: BLE001 - 单条失败不中断整轮
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000.0

        record: Dict[str, Any] = {
            "id": case["id"],
            "query": query,
            # 不可答样本（应拒答）：没有 expected_doc_id，
            # aggregate_rag_results 会据此把它排除出 Recall@5 的分母，
            # 否则会因为"本来就没有正确答案"而恒为 0，无端拉低召回指标。
            "unanswerable": bool(case.get("unanswerable")),
            # ---- 判分通道（aggregate_rag_results 直接消费这些键）----
            "retrieved_ids": fields["doc_ids"],
            "expected_ids": [str(expected_doc_id)] if expected_doc_id else [],
            "retrieved_doc_names": fields["doc_names"],
            "expected_doc_names": [normalize_doc_name(expected_doc_name)]
            if expected_doc_name
            else [],
            # ---- 排查信息（不参与判分）----
            "expected_doc_name": expected_doc_name,
            "retrieved_chunk_ids": fields["chunk_ids"],
            "scores": fields["scores"],
            "top1_doc_name": fields["doc_names"][0] if fields["doc_names"] else None,
            "top1_score": fields["scores"][0] if fields["scores"] else None,
            "top1_content_head": fields["previews"][0] if fields["previews"] else None,
            "latency_ms": round(latency_ms, 2),
        }
        if error:
            record["error"] = error
        records.append(record)

        _print_progress(index, len(selected), record, k=top_k)
        if sleep_between:
            await asyncio.sleep(sleep_between)

    return {"records": records, "metrics": aggregate_rag_results(records, k=top_k)}


def _print_progress(index: int, total: int, record: Dict[str, Any], *, k: int) -> None:
    """逐条打印；命中判定复用 metrics 的口径，保证打印与汇总一致。"""
    from evals.metrics import hit_at_k

    top1: str = str(record.get("top1_doc_name") or "-")
    latency: str = f"{record['latency_ms']:.0f}ms"

    # ⚠️ 应拒答样本（unanswerable）**没有期望文档**，检索指标按设计不算它
    # （见 aggregate_rag_results 里 skipped_unanswerable 的分母口径）。
    # 旧实现照样按 hit 打印，而 hit_at_k(空, 空) 恒为 0 → 日志出现
    # "FAIL R30 expect=（空）"，看起来像"检索失败"，实际是设计内的跳过。
    # 这会把排查引到错误方向（真实踩过：误以为是召回退化），因此显式标 SKIP。
    if record.get("unanswerable"):
        print(
            f"[rag {index}/{total}] SKIP {record['id']} "
            f"（应拒答样本：不计检索指标，由生成层判是否拒答）top1={top1} {latency}",
            flush=True,
        )
        return

    hit: float = max(
        hit_at_k(record["retrieved_ids"], record["expected_ids"], k),
        hit_at_k(record["retrieved_doc_names"], record["expected_doc_names"], k),
    )
    mark: str = "PASS" if hit else "FAIL"
    print(
        f"[rag {index}/{total}] {mark} {record['id']} "
        f"expect={record['expected_doc_name']} top1={top1} {latency}",
        flush=True,
    )


def main() -> None:
    from evals.loaders import load_rag_cases, save_results

    parser = argparse.ArgumentParser(description="RAG 检索离线评测（Recall@5 / MRR）")
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "evals" / "_results" / "rag.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑 N 条（冒烟用）。按**分层抽样**选取：必含至少 1 条应拒答样本，"
             "其余在各期望文档间轮转分配，而不是简单取文件前 N 条",
    )
    parser.add_argument("--top-k", type=int, default=EVAL_K, help="召回截断 k（默认 5）")
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "覆盖物理 Milvus 集合名；不传则与线上同一个"
            "（settings.milvus_kb_collection_name）。常规评测不用传；"
            "chunk-size A/B 实验传 eval_kb_512 / eval_kb_1024。"
        ),
    )
    parser.add_argument(
        "--collection-filter",
        default=None,
        help=(
            "可选：按 metadata['collection'] 白名单收窄检索范围（逗号分隔多个）。"
            "**不传即不过滤**（推荐）——物理集合里通常同时有 sales_kb / enterprise_kb / "
            "各次上传的 collection_name，写死单一标签会在标签不一致时整段过滤成 0 召回。"
            "只有 chunk-size A/B 这类实验才需要传。"
        ),
    )
    parser.add_argument(
        "--no-seed-bm25",
        action="store_true",
        help="跳过 BM25 预热（不建议：会让 Recall 偏低）",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="用例间隔秒数（避免限流）")
    add_order_cli_args(parser)
    args = parser.parse_args()

    # 不传 --collection-filter 就是 None = 不过滤（物理集合内全量检索）
    collection_filter: Optional[List[str]] = (
        [c.strip() for c in args.collection_filter.split(",") if c.strip()]
        if args.collection_filter
        else None
    )

    async def _main() -> Dict[str, Any]:
        runtime = EvalRuntime(collection_name=args.collection)
        try:
            result = await run_rag_eval(
                load_rag_cases(),
                runtime,
                limit=args.limit,
                top_k=args.top_k,
                collection_filter=collection_filter,
                seed_bm25=not args.no_seed_bm25,
                sleep_between=args.sleep,
                order=args.order,
                seed=args.seed,
            )
        finally:
            await runtime.aclose()
        result["run_config"] = run_config_of(args.order, args.seed, args.limit)
        return result

    result = asyncio.run(_main())
    save_results(result, args.out)

    metrics = result["metrics"]
    k: int = args.top_k
    print(
        f"[rag] 完成：recall@{k}={format_metric(metrics[f'recall@{k}'])} "
        f"hit@{k}={format_metric(metrics[f'hit@{k}'])} mrr={format_metric(metrics['mrr'])} "
        f"p95={format_metric(metrics['p95_latency_ms'], '.0f')}ms -> {args.out}"
    )


if __name__ == "__main__":
    main()
