# -*- coding: utf-8 -*-
"""Runner：意图分类离线评测。

链路：复用应用的 ``AgentQueryIntentPipeline.run(...)``（改写 → 意图聚合 → 模式决策），
因此测的是**真实三阶段流水线**，不是单独调一次分类器。这样得到的
「意图准确率」才反映线上那条链路，而不是一次孤立的分类调用。

预测意图的提取口径（重要，容易被忽略）：
    Pipeline 输出 ``intents_result.primary_intent_text`` 是节点 ``full_path``
    （如「企业知识问答 > 人事制度」）。本 runner 用 full_path 的**首段域名**
    判断命中哪个通道（KB / MCP / SYSTEM），再从对应通道的 TOP1 节点
    （``raw_slots["top_kb_node"] / top_mcp_node / top_system_node``）取
    ``node_id`` 作为预测标签——与黄金集里的 ``expected_intent``（叶子 id）同域。

用法::

    python -m evals.runners.run_intent --out evals/_results/intent.json
    python -m evals.runners.run_intent --limit 3      # 冒烟：分层抽样 3 条
    python -m evals.runners.run_intent --limit 3 --order random           # 每次抽不同的 3 条
    python -m evals.runners.run_intent --limit 3 --order random --seed 7  # 可复现的随机

⚠️ ``--limit`` 走的是**分层抽样**（见 ``evals/sampling.py``），不是"取前 N 条"。
本集的边界样本 B01..B05 排在文件末尾，若用简单切片，``--limit 3`` 会一条边界
样本都抽不到，``boundary_accuracy`` 将因无样本而显示"不可用"。

⚠️ ``--order`` 默认 ``file``（确定性，CI 依赖它）；``random`` 是"**先打乱再分层**"，
因此不仅执行顺序变，**抽到的样本本身也会变**，连着冒烟不会永远测那几条。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.metrics import aggregate_intent_results, format_metric, percentile  # noqa: E402
from evals.runners._runtime import EvalRuntime  # noqa: E402
from evals.runners._usage import record_llm_usage  # noqa: E402
from evals.sampling import (  # noqa: E402
    DEFAULT_ORDER,
    add_order_cli_args,
    describe_selection,
    run_config_of,
    select_cases,
)

# full_path 首段域名 -> 对应通道的 TOP1 槽位键
DOMAIN_TO_SLOT: Dict[str, str] = {
    "企业知识问答": "top_kb_node",
    "外部公开信息检索": "top_mcp_node",
    "业务数据操作": "top_mcp_node",
    "本地文件操作": "top_mcp_node",
    "任务规划": "top_mcp_node",
    "系统交互": "top_system_node",
}
_SLOT_KEYS = ("top_kb_node", "top_mcp_node", "top_system_node")


def extract_predicted_intent(pipeline_output: Any) -> str:
    """从 Pipeline 输出提取预测意图（叶子 node_id）。"""
    intents = getattr(pipeline_output, "intents_result", None)
    if intents is None:
        return "general"
    slots: Dict[str, Any] = dict(getattr(intents, "raw_slots", None) or {})
    primary_text: str = str(getattr(intents, "primary_intent_text", "") or "")

    domain: str = primary_text.split(">")[0].strip()
    slot_key: Optional[str] = DOMAIN_TO_SLOT.get(domain)
    if slot_key:
        entry = slots.get(slot_key) or {}
        node_id = entry.get("node_id")
        if node_id:
            return str(node_id)

    # 兜底：在全通道 TOP1 里找 full_path 与主意图文本完全一致的那个
    for key in _SLOT_KEYS:
        entry = slots.get(key) or {}
        if entry.get("full_path") and str(entry["full_path"]) == primary_text and entry.get("node_id"):
            return str(entry["node_id"])
    return "general"


def predict_intent(pipeline: Any, query: str, session_id: str = "eval_intent") -> str:
    """同步执行一次 Pipeline 并返回预测意图 id。

    Pipeline 是同步实现，直接调用即可（线上由 ``asyncio.to_thread`` 包装）。
    """
    output = pipeline.run(
        query,
        [],            # available_tool_ids：不注入额外工具，走意图树自身的工具路由
        session_id,    # session_id
        [],            # conversation_history：单轮评测，无上下文
        [],            # available_skills：不注入技能，避免技能号令干扰意图判定
    )
    return extract_predicted_intent(output)


async def run_intent_eval(
    cases: List[Dict[str, Any]],
    runtime: Optional[EvalRuntime] = None,
    *,
    limit: Optional[int] = None,
    sleep_between: float = 0.0,
    order: str = DEFAULT_ORDER,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """跑完整意图评测，返回 ``{"records": [...], "metrics": {...}}``。

    ``order`` / ``seed``：用例选取顺序（``file`` 确定 / ``random`` 随机）。
    语义见 ``evals.sampling``——随机是"先打乱再分层"，不是只打乱结果。
    """
    runtime = runtime or EvalRuntime()
    pipeline = runtime.get_pipeline()
    # 分层抽样而非 cases[:limit]：黄金集按 id 排序，简单切片会把末尾的
    # 边界样本（B01..B05）整体漏掉，导致 boundary_accuracy 没有分母。
    selected = select_cases("intent", cases, limit, order=order, seed=seed)
    print(f"[intent] 本次用例：{describe_selection(selected, order, seed)}", flush=True)

    records: List[Dict[str, Any]] = []
    per_case_tokens: List[int] = []
    # 意图阶段也会真实调用 LLM（改写 + 意图聚合），这里旁路采集真实 usage，
    # 让报告能回答"意图阶段吃掉整轮 token 的多少"。
    with record_llm_usage() as usage:
        for index, case in enumerate(selected, start=1):
            query: str = str(case["query"])
            tokens_before: int = usage.total_tokens
            started = time.perf_counter()
            error: Optional[str] = None
            predicted = "general"
            try:
                predicted = await asyncio.to_thread(
                    predict_intent, pipeline, query, f"eval_intent_{case['id']}"
                )
            except Exception as exc:  # noqa: BLE001 - 单条失败不中断整轮
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - started) * 1000.0
            per_case_tokens.append(usage.total_tokens - tokens_before)

            gold: str = str(case["expected_intent"])
            record: Dict[str, Any] = {
                "id": case["id"],
                "query": query,
                "gold": gold,
                "pred": predicted,
                "correct": predicted == gold,
                "boundary": bool(case.get("boundary")),
                "latency_ms": round(latency_ms, 2),
            }
            if error:
                record["error"] = error
            records.append(record)

            _print_progress(index, len(selected), record)
            if sleep_between:
                await asyncio.sleep(sleep_between)

    metrics: Dict[str, Any] = aggregate_intent_results(records)
    per_turn = usage.per_turn(len(records))
    if per_turn is not None:
        metrics["avg_tokens_per_turn"] = per_turn
    # 中位数抗极值：个别长对话会把均值拉高，中位数才是"典型一轮"的稳健估计。
    metrics["median_tokens_per_turn"] = (
        percentile([float(t) for t in per_case_tokens], 50) if per_case_tokens else None
    )
    # 一轮问答折合多少次 LLM 调用——比总 token 更能解释"成本花在哪"。
    metrics["llm_calls_per_request"] = (
        usage.calls / len(records) if records else None
    )
    metrics["llm_usage"] = usage.as_dict()
    return {"records": records, "metrics": metrics}


def _print_progress(index: int, total: int, record: Dict[str, Any]) -> None:
    mark = "PASS" if record["correct"] else "FAIL"
    print(
        f"[intent {index}/{total}] {mark} {record['id']} "
        f"gold={record['gold']} pred={record['pred']} {record['latency_ms']:.0f}ms",
        flush=True,
    )


def main() -> None:
    from evals.loaders import load_intent_cases, save_results

    parser = argparse.ArgumentParser(description="意图分类离线评测")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "evals" / "_results" / "intent.json")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑 N 条（冒烟用）。按**分层抽样**选取：必含至少 1 条边界样本，"
             "其余在各意图间轮转分配，而不是简单取文件前 N 条",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="用例间隔秒数（避免限流）")
    add_order_cli_args(parser)
    args = parser.parse_args()

    async def _main() -> Dict[str, Any]:
        runtime = EvalRuntime()
        try:
            result = await run_intent_eval(
                load_intent_cases(), runtime, limit=args.limit, sleep_between=args.sleep,
                order=args.order, seed=args.seed,
            )
        finally:
            await runtime.aclose()
        result["run_config"] = run_config_of(args.order, args.seed, args.limit)
        return result

    result = asyncio.run(_main())
    save_results(result, args.out)
    metrics = result["metrics"]
    print(
        f"[intent] 完成：accuracy={format_metric(metrics['intent_accuracy'])} "
        f"boundary={format_metric(metrics['boundary_accuracy'])} "
        f"p95={format_metric(metrics['p95_latency_ms'], '.0f')}ms -> {args.out}"
    )


if __name__ == "__main__":
    main()
