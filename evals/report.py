# -*- coding: utf-8 -*-
"""评测报告生成器：一条命令跑完全部评测并产出 ``eval_report.md``。

## 一条命令

    python -m evals.report --run-all

它会依次跑「意图 / RAG / 工具」三组评测，把指标汇到一张表，过一遍
``thresholds.yaml`` 质量门，从失败样本里自动挑一个**最有代表性的**案例，
写出 ``evals/eval_report.md``。

## 设计要点

1. **失败不减配**：某个 runner 挂了（比如 Milvus 没起）不会让整条命令失败——
   该段标记为「未采集」并写明原因，其余指标照常出报告。理由是 CI 上
   "先建 eval、中间件还没好"是常见过渡态，报告必须能出来。
2. **不编数字**：拿不到的指标一律写「不可用」，绝不用估算值填充。
3. **单轮 token 成本**取自**工具评测那条端到端链路**（意图改写+分类+Agent 多步），
   这才对应「一轮对话」的真实成本；意图阶段单独采集的 usage 另列一行做归因。
4. **质量门结论与阈值版本绑定**，报告头写清"这次红线是哪版标准"。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from traceback import format_exc
from typing import Any, Dict, List, Mapping, Optional

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.loaders import (  # noqa: E402
    EVALS_DIR,
    RESULTS_DIR,
    load_rag_cases,
    load_intent_cases,
    load_thresholds,
    load_tool_cases,
    save_results,
)
from evals.metrics import evaluate_gate  # noqa: E402
from evals.sampling import DEFAULT_ORDER, add_order_cli_args, run_config_of  # noqa: E402

INTENT_RESULT_PATH: Path = RESULTS_DIR / "intent.json"
RAG_RESULT_PATH: Path = RESULTS_DIR / "rag.json"
TOOL_RESULT_PATH: Path = RESULTS_DIR / "tool.json"
# 生成层：不由 run_all 跑（需要外部提供预测答案），存在时会被读进报告
ANSWER_RESULT_PATH: Path = RESULTS_DIR / "answer.json"
REPORT_PATH: Path = EVALS_DIR / "eval_report.md"
MERGED_RESULT_PATH: Path = RESULTS_DIR / "latest.json"

# 需要按「最小」/「最大」判定的键，用于把各 runner 的指标摊平给 evaluate_gate
_METRIC_KEYS: tuple[str, ...] = (
    "intent_accuracy",
    "boundary_accuracy",
    "recall@5",
    "hit@5",
    "mrr",
    "tool_success_rate",
    "key_arg_recall",
    "avg_tokens_per_turn",
    "p95_latency_ms",
    "answer_pass_rate",
    "abstain_correct_rate",
)


# =====================================================================
# 一、执行
# =====================================================================
async def run_all(
    *,
    limit: Optional[int] = None,
    skip: Optional[List[str]] = None,
    order: str = DEFAULT_ORDER,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """跑三组评测，逐组隔离异常，返回 ``{runner: {"ok": bool, "result"|"error"}}``。

    ``order`` / ``seed`` 会透传给三个 runner 的抽样（语义见 ``evals.sampling``）：
    三组**共用同一个 seed 与同一套顺序口径**，避免出现"意图随机、工具按文件顺序"
    这种不同口径的混合结果。
    """
    skip = skip or []
    out: Dict[str, Any] = {}

    from evals.runners._runtime import EvalRuntime

    # 三组共用同一个运行时（同一套 ModelRouter / RAG / 意图树），
    # 避免重复装配导致"三份配置"与线上不一致。
    runtime = EvalRuntime()
    # 三组共用的抽样配置：随每组结果一起落盘（见 _guard 的说明）。
    run_config: Dict[str, Any] = run_config_of(order, seed, limit)
    try:
        if "intent" not in skip:
            out["intent"] = await _guard(
                "intent",
                lambda: _run_intent(runtime, limit, order, seed),
                INTENT_RESULT_PATH,
                run_config=run_config,
            )
        if "rag" not in skip:
            out["rag"] = await _guard(
                "rag", lambda: _run_rag(runtime, limit, order, seed), RAG_RESULT_PATH,
                run_config=run_config,
            )
        if "tool" not in skip:
            out["tool"] = await _guard(
                "tool", lambda: _run_tool(runtime, limit, order, seed), TOOL_RESULT_PATH,
                run_config=run_config,
            )
    finally:
        try:
            await runtime.aclose()
        except Exception:  # noqa: BLE001
            pass
    return out


async def _guard(
    name: str, factory: Any, save_path: Path, *, run_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """执行一个 runner；异常转成 ``{"ok": False, "error": ...}`` 而不上抛。

    ``run_config`` 会写进**分组结果文件**：否则 ``evals/_results/rag.json`` 只有一个
    孤零零的指标字典，事后无法判断这次是全量还是 ``--limit 2`` 冒烟、按文件顺序还是随机
    （真实踩过：拿一份旧的分组结果判断"当前水位"，结果那其实是 2 条冒烟的数据）。
    各 runner 自己 ``main()`` 时也会写这个键，所以用 ``setdefault`` 不覆盖。
    """
    try:
        result: Dict[str, Any] = await factory()
        if run_config is not None and isinstance(result, dict):
            result.setdefault("run_config", run_config)
        save_results(result, save_path)
        return {"ok": True, "result": result, "path": str(save_path)}
    except Exception as exc:  # noqa: BLE001 - 单组失败不影响出报告
        message: str = f"{type(exc).__name__}: {exc}"
        # ⚠️ 只打印一行摘要会让人无从下手（实测踩过：报告里只剩
        # "AttributeError: 'dict' object has no attribute 'abs'"，既不在本仓代码里、
        # 又定位不到是哪一层调用的）。报告里仍写摘要保持可读，
        # 完整 traceback 打到 stderr 供排查。
        detail: str = format_exc()
        print(f"[report] ⚠️ {name} 评测未完成：{message}", flush=True)
        print(detail, file=sys.stderr, flush=True)
        return {"ok": False, "error": message, "path": str(save_path), "traceback": detail}


async def _run_intent(
    runtime: Any, limit: Optional[int], order: str, seed: Optional[int]
) -> Dict[str, Any]:
    from evals.runners.run_intent import run_intent_eval

    return await run_intent_eval(load_intent_cases(), runtime, limit=limit, order=order, seed=seed)


async def _run_rag(
    runtime: Any, limit: Optional[int], order: str, seed: Optional[int]
) -> Dict[str, Any]:
    from evals.runners.run_rag import run_rag_eval

    return await run_rag_eval(load_rag_cases(), runtime, limit=limit, order=order, seed=seed)


async def _run_tool(
    runtime: Any, limit: Optional[int], order: str, seed: Optional[int]
) -> Dict[str, Any]:
    from evals.runners.run_tool import run_tool_eval

    return await run_tool_eval(load_tool_cases(), runtime, limit=limit, order=order, seed=seed)


def load_saved_results() -> Dict[str, Any]:
    """从 ``evals/_results/*.json`` 读回已有结果（不重新跑）。"""
    out: Dict[str, Any] = {}
    for name, path in (
        ("intent", INTENT_RESULT_PATH),
        ("rag", RAG_RESULT_PATH),
        ("tool", TOOL_RESULT_PATH),
        ("answer", ANSWER_RESULT_PATH),
    ):
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                out[name] = {"ok": True, "result": json.load(handle), "path": str(path)}
        else:
            # 生成层需要外部提供预测答案，缺失是预期内的常态，提示语要区分开
            if name == "answer":
                hint = ("（生成层未跑：需先准备预测答案再执行 "
                        "python evals/tools/grade_answers.py --answers "
                        "evals/_results/answers.jsonl）")
            else:
                hint = "（请先运行 python -m evals.report --run-all）"
            out[name] = {
                "ok": False,
                "error": f"缺少结果文件 {path.name}{hint}",
                "path": str(path),
            }
    return out


# =====================================================================
# 二、汇总
# =====================================================================
def flatten_metrics(runs: Mapping[str, Any]) -> Dict[str, float]:
    """把三组 metrics 摊平成 ``evaluate_gate`` 需要的扁平字典。

    同名键的优先级：工具(端到端) > RAG > 意图 —— 因为 ``avg_tokens_per_turn``
    与 ``p95_latency_ms`` 只有工具组的端到端口径才对应「一轮对话」。

    **样本量不足的键会被挡在这里**：每个 runner 的 ``metrics["sample_insufficient"]``
    记录了"样本数没达到可判定下限"的指标（如 n=3 时的 P95 延迟）。这类指标的数值
    照常在分组明细里给人看，但**不进入质量门**——
    它们判定的不是"分数低"，而是"这个分数还不足以被判定"。
    """
    flat: Dict[str, float] = {}
    for name in ("intent", "rag", "tool"):
        entry = runs.get(name) or {}
        if not entry.get("ok"):
            continue
        metrics: Mapping[str, Any] = (entry.get("result") or {}).get("metrics") or {}
        insufficient: Mapping[str, Any] = metrics.get("sample_insufficient") or {}
        for key in _METRIC_KEYS:
            if key in insufficient:
                continue
            value = metrics.get(key)
            if value is None:
                continue
            try:
                flat[key] = float(value)
            except (TypeError, ValueError):
                continue
    return flat


def pick_failure_case(runs: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """从所有失败样本里挑一个"最值得讲"的，附完整上下文。

    选择优先级（越靠前越有解释价值）：
        1. **RAG 完全没召回**（top1 是别的文档）——最能体现"混合检索/RRF 排序"的改进空间；
        2. 意图**边界样本**分错——体现 eval 的区分度；
        3. 工具调用选错工具；
        4. 其它失败。
    """
    candidates: List[Dict[str, Any]] = []

    rag = (runs.get("rag") or {})
    if rag.get("ok"):
        from evals.metrics import hit_at_k

        for record in (rag["result"] or {}).get("records") or []:
            # 应拒答样本没有期望文档，检索层判不了它（由生成层判是否拒答）——
            # 不排除的话它会被当成"RAG 完全没召回"，抢占"最值得讲的失败案例"
            # 这个位置，把真正有信息量的召回失败挤掉。
            if record.get("unanswerable"):
                continue
            hit: float = max(
                hit_at_k(record.get("retrieved_ids") or [], record.get("expected_ids") or [], 5),
                hit_at_k(
                    record.get("retrieved_doc_names") or [],
                    record.get("expected_doc_names") or [],
                    5,
                ),
            )
            if not hit:
                candidates.append(
                    {
                        "priority": 1,
                        "stage": "RAG 检索",
                        "id": record.get("id"),
                        "query": record.get("query"),
                        "expected": record.get("expected_doc_name"),
                        "observed": f"top1={record.get('top1_doc_name')}"
                        f"（score={record.get('top1_score')}）"
                        f" 召回列表={record.get('retrieved_doc_names')}",
                        "top1_content_head": record.get("top1_content_head"),
                    }
                )

    intent = (runs.get("intent") or {})
    if intent.get("ok"):
        for record in (intent["result"] or {}).get("records") or []:
            if record.get("correct"):
                continue
            candidates.append(
                {
                    "priority": 2 if record.get("boundary") else 4,
                    "stage": "意图分类"
                    + ("（边界样本）" if record.get("boundary") else ""),
                    "id": record.get("id"),
                    "query": record.get("query"),
                    "expected": record.get("gold"),
                    "observed": f"pred={record.get('pred')}",
                }
            )

    tool = (runs.get("tool") or {})
    if tool.get("ok"):
        from evals.metrics import tool_call_success

        for record in (tool["result"] or {}).get("records") or []:
            ok: bool = tool_call_success(
                record.get("called_tools") or [],
                str(record.get("expected_tool") or ""),
                record.get("acceptable_tools"),
                inspect_first_n=record.get("inspect_first_n"),
            )
            if ok:
                continue
            candidates.append(
                {
                    "priority": 3,
                    "stage": "工具调用",
                    "id": record.get("id"),
                    "query": record.get("query"),
                    "expected": record.get("expected_tool"),
                    "observed": f"called={record.get('called_tools') or '（无）'}"
                    f" mode={record.get('mode_used')}"
                    f" 白名单={record.get('allowed_tools')}"
                    f" 疑似未放行={record.get('tool_not_whitelisted') or '无'}",
                }
            )

    if not candidates:
        return None
    candidates.sort(key=lambda item: item["priority"])
    return candidates[0]


def render_report(
    runs: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    generated_at: Optional[str] = None,
) -> str:
    """渲染 Markdown 报告。"""
    flat: Dict[str, float] = flatten_metrics(runs)
    violations: List[str] = evaluate_gate(flat, thresholds)
    failure: Optional[Dict[str, Any]] = pick_failure_case(runs)

    lines: List[str] = []
    add = lines.append

    add("# Ai_Agent_Frame 评测报告（eval_report）")
    add("")
    add(f"- 生成时间：{generated_at or datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 阈值版本：`thresholds.version = {thresholds.get('version')}`")
    add(f"- 运行环境：{platform.system()} {platform.release()} / "
        f"Python {platform.python_version()}")
    add("- 复现命令：`python -m evals.report --run-all`")
    add("")

    # ---------------- 结论 ----------------
    add("## 一、结论（先看这里）")
    add("")
    if not flat:
        add("⚠️ **本次未采集到任何指标，质量门未生效。**")
        add("")
        add("常见原因：Milvus / Redis 未启动，或模型 API Key 不可用。"
            "先确认 `curl http://127.0.0.1:8000/api/v1/health` 有响应，"
            "再运行 `python -m evals.report --run-all`。")
    elif violations:
        add(f"❌ **质量门未通过**，共 {len(violations)} 项违规：")
        add("")
        for item in violations:
            add(f"- {item}")
    else:
        add(f"✅ **质量门通过**（本次采集到 {len(flat)} 项指标，均不低于阈值）。")
    add("")
    add("> 说明：阈值未配置的项、以及本次未采集到的指标会被**跳过**而不是判失败，"
        "所以「通过」的含义是「已测量的部分达标」。")
    add("")

    # ---------------- 指标表 ----------------
    add("## 二、四个必测指标")
    add("")
    add("| 指标 | 本次实测 | 阈值 | 阈值版本键 |")
    add("| --- | --- | --- | --- |")
    add(_row("意图准确率", flat.get("intent_accuracy"), thresholds, "intent", "accuracy_min"))
    add(_row("意图准确率（5 条边界样本）", flat.get("boundary_accuracy"), thresholds,
             "intent", "boundary_accuracy_min"))
    add(_row("RAG Recall@5", flat.get("recall@5"), thresholds, "rag", "recall_at_5_min"))
    add(_row("RAG Hit@5", flat.get("hit@5"), thresholds, "rag", "hit_at_5_min"))
    add(_row("RAG MRR", flat.get("mrr"), thresholds, "rag", "mrr_min"))
    add(_row("工具调用成功率", flat.get("tool_success_rate"), thresholds, "tool",
             "call_success_min"))
    add(_row("工具关键参数命中率", flat.get("key_arg_recall"), thresholds, "tool",
             "key_arg_recall_min"))
    add(_row("单轮 token 成本（端到端）", flat.get("avg_tokens_per_turn"), thresholds, "cost",
             "avg_tokens_per_turn_max"))
    add(_row("P95 延迟 (ms)", flat.get("p95_latency_ms"), thresholds, "cost",
             "p95_latency_ms_max"))
    add(_row("答案合格率（生成层）", flat.get("answer_pass_rate"), thresholds,
             "answer", "pass_rate_min"))
    add(_row("应拒答正确率（生成层）", flat.get("abstain_correct_rate"), thresholds,
             "answer", "abstain_correct_rate_min"))
    add("")

    # ---------------- 分组明细 ----------------
    add("## 三、分组明细")
    add("")
    sections = (("intent", "意图分类"), ("rag", "RAG 检索"), ("tool", "工具调用"))
    for order, (name, title) in enumerate(sections, start=1):
        entry = runs.get(name) or {}
        add(f"### 3.{order} {title}")
        add("")
        if not entry.get("ok"):
            add(f"⛔ **未采集**：{entry.get('error')}")
            add("")
            continue
        metrics: Mapping[str, Any] = (entry.get("result") or {}).get("metrics") or {}
        records: List[Mapping[str, Any]] = (entry["result"] or {}).get("records") or []
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (int, float)):
                add(f"- `{key}` = {value:.4f}" if isinstance(value, float) else f"- `{key}` = {value}")
        add(f"- 样本数 = {len(records)}")
        usage = metrics.get("llm_usage")
        if isinstance(usage, Mapping) and usage.get("calls"):
            add(f"- 真实 LLM 调用 {usage.get('calls')} 次，"
                f"input={usage.get('input_tokens')} output={usage.get('output_tokens')} "
                f"total={usage.get('total_tokens')} tokens")
        _render_insufficient(add, metrics)
        add("")
        misses: List[Mapping[str, Any]] = _collect_misses(name, records)
        if misses:
            add(f"失败样本（最多列 5 条，共 {len(misses)} 条）：")
            add("")
            add("| id | 问题 | 期望 | 实际 |")
            add("| --- | --- | --- | --- |")
            for item in misses[:5]:
                add(f"| {item['id']} | {_esc(item['query'])} | {_esc(item['expected'])} "
                    f"| {_esc(item['observed'])} |")
            add("")

    # ---------------- 失败案例 ----------------
    add("## 四、一个可讲的失败案例")
    add("")
    if failure is None:
        add("本次没有失败样本（或全部指标未采集）。")
    else:
        add(f"- **阶段**：{failure['stage']}")
        add(f"- **用例**：`{failure['id']}` —— {failure['query']}")
        add(f"- **期望**：{failure['expected']}")
        add(f"- **实际**：{failure['observed']}")
        if failure.get("top1_content_head"):
            add(f"- **实际召回的 top1 内容开头**：{_esc(failure['top1_content_head'])}")
        add("")
        add("> 讲法建议：先说「期望什么」，再说「实际发生了什么」，最后说"
            "「我认为根因是什么、下一步怎么验证」——这三点比结论本身更有信息量。")
    add("")

    # ---------------- 覆盖与局限 ----------------
    add("## 五、覆盖范围与已知局限")
    add("")
    add(f"- 黄金集规模：意图 {len(load_intent_cases())} 条 / "
        f"RAG {len(load_rag_cases())} 条 / 工具 {len(load_tool_cases())} 条。")
    add("- ``--limit`` 为**分层抽样**（见 ``evals/sampling.py``），不是'取前 N 条'："
        "必含边界样本 / 应拒答样本，其余在各层间轮转分配。"
        "因此**冒烟跑的数字不能代表全量**，只能验证链路通不通。"
        "``--order random`` 会在抽样前先打乱（连带抽样集合一起变化，适合反复冒烟）；"
        "``--seed N`` 可锁定同一批用例。")
    add("- **样本量不足的指标不判定、只展示**：如 P95 需要 ≥20 条、边界准确率需要 ≥5 条，"
        "未达下限时会在分组明细里标注原因并从质量门剔除。"
        "这是有意为之——小样本下的分位数测的是偶发抖动，不是整体水位。")
    add("- ⚠️ 已知约束：工具黄金集仅 15 条 < 20，**该组 P95 即使全量跑也不进质量门**。"
        "如需启用这条红线，请把工具集扩到 ≥20 条。")
    add("- `expected_doc_id` 若为 `null`，RAG 召回判定走**文件名通道**兜底"
        "（两条通道取较优者），这是设计内的行为。")
    add("- RAG 段（`rag`）测的是**检索层**（Recall@5 / MRR）；**生成层**另由 answer 段"
        "（`evals/tools/grade_answers.py`）判分，两者互补：召回对了不代表说对了。")
    add("- 生成层当前用**规则判分**（要点命中 + 干扰项检测 + 拒答检测），不是 LLM-as-judge。"
        "理由是先要有可复现的数字；LLM judge 应作为**对照**叠加。")
    add("- 单轮 token 成本来自旁路采集的真实 LLM `usage`；若采集失败该行显示"
        "「不可用」，不使用字符数估算。")
    add("- 全套评测依赖 Milvus / Redis / 模型 API，**均为本机单机环境**，"
        "不代表生产容量。")
    add("")

    return "\n".join(lines) + "\n"


def _render_insufficient(add: Any, metrics: Mapping[str, Any]) -> None:
    """渲染「样本量不足」提示。

    这些指标的**数值照常显示**给人看，但被 ``flatten_metrics`` 挡在质量门之外——
    否则小样本下一次偶发的慢请求就能把整条红线判红，而那测的是抖动不是水位。
    """
    insufficient: Any = metrics.get("sample_insufficient")
    if not isinstance(insufficient, Mapping) or not insufficient:
        return
    add("")
    add("- ⚠️ **样本量不足**：以下数值仅供参考，**不参与质量门判定**")
    for key, info in insufficient.items():
        add(
            f"  - `{key}`：本次 {info.get('n')} 条，需 ≥{info.get('required')} 条。"
            f"低于下限时该指标在统计上不成立（如 P95 在 n=3 时插值结果几乎等于最大值）"
        )


def _collect_misses(name: str, records: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """收集某组的失败样本（统一成 id/query/expected/observed 四列）。"""
    misses: List[Dict[str, Any]] = []
    if name == "intent":
        for record in records:
            if record.get("correct"):
                continue
            misses.append(
                {
                    "id": record.get("id"),
                    "query": record.get("query"),
                    "expected": record.get("gold"),
                    "observed": record.get("pred"),
                }
            )
    elif name == "rag":
        from evals.metrics import hit_at_k

        for record in records:
            # 应拒答样本不属于"检索失败"（它本来就没有期望文档），排除掉，
            # 否则失败样本表里会出现一行 expect 为空、实际=某个无关文档的假失败。
            if record.get("unanswerable"):
                continue
            hit: float = max(
                hit_at_k(record.get("retrieved_ids") or [], record.get("expected_ids") or [], 5),
                hit_at_k(
                    record.get("retrieved_doc_names") or [],
                    record.get("expected_doc_names") or [],
                    5,
                ),
            )
            if hit:
                continue
            misses.append(
                {
                    "id": record.get("id"),
                    "query": record.get("query"),
                    "expected": record.get("expected_doc_name"),
                    "observed": f"top1={record.get('top1_doc_name')}",
                }
            )
    elif name == "tool":
        from evals.metrics import tool_call_success

        for record in records:
            ok: bool = tool_call_success(
                record.get("called_tools") or [],
                str(record.get("expected_tool") or ""),
                record.get("acceptable_tools"),
                inspect_first_n=record.get("inspect_first_n"),
            )
            if ok:
                continue
            misses.append(
                {
                    "id": record.get("id"),
                    "query": record.get("query"),
                    "expected": record.get("expected_tool"),
                    "observed": "、".join(record.get("called_tools") or []) or "（无）",
                }
            )
    elif name == "answer":
        for record in records:
            if record.get("passed") is not False:
                continue
            observed = record.get("reason") or "未通过"
            if record.get("missing_facts"):
                observed = f"{observed}；漏掉要点：{'、'.join(record['missing_facts'])}"
            misses.append(
                {
                    "id": record.get("id"),
                    "query": record.get("query"),
                    "expected": "按 rubric 判分通过",
                    "observed": observed,
                }
            )
    return misses


def _row(
    label: str,
    value: Optional[float],
    thresholds: Mapping[str, Any],
    section: str,
    key: str,
) -> str:
    """渲染指标表一行；值缺失时写「不可用」，阈值缺失时写「未配置」。"""
    limit: Any = (thresholds.get(section) or {}).get(key)
    limit_text: str = f"{float(limit):g}" if isinstance(limit, (int, float)) else "未配置"
    value_text: str = "不可用" if value is None else f"{value:g}"
    if isinstance(limit, (int, float)) and value is not None:
        ok: bool = (value >= float(limit)) if key.endswith("_min") else (value <= float(limit))
        value_text = f"{value:g} {'✅' if ok else '❌'}"
    return f"| {label} | {value_text} | {limit_text} | `{section}.{key}` |"


def _esc(text: Any) -> str:
    """单元格转义：竖线与换行会破坏 Markdown 表格。"""
    return str(text if text is not None else "-").replace("|", "\\|").replace("\n", " ")


# =====================================================================
# 三、入口
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="跑评测并生成 eval_report.md")
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="重新跑三组评测（不加则只渲染已有的 evals/_results/*.json）",
    )
    parser.add_argument("--limit", type=int, default=None, help="每组只跑前 N 条（冒烟）")
    parser.add_argument(
        "--skip",
        default="",
        help="跳过某几组，逗号分隔，如 intent,rag（用于只跑一部分）",
    )
    parser.add_argument("--out", type=Path, default=REPORT_PATH, help="报告输出路径")
    add_order_cli_args(parser)
    args = parser.parse_args()

    skip: List[str] = [s.strip() for s in args.skip.split(",") if s.strip()]

    if args.run_all:
        runs: Dict[str, Any] = asyncio.run(
            run_all(limit=args.limit, skip=skip, order=args.order, seed=args.seed)
        )
        save_results(
            {
                "metrics": flatten_metrics(runs),
                "thresholds_version": load_thresholds().get("version"),
                "runs": {k: {"ok": v.get("ok"), "error": v.get("error")} for k, v in runs.items()},
                # 抽样配置也要落盘：否则只看 latest.json 无法判断"这次是全量还是冒烟、
                # 按文件顺序还是随机"，复现失败现场时第一步就卡住。
                "run_config": run_config_of(args.order, args.seed, args.limit),
            },
            MERGED_RESULT_PATH,
        )
    else:
        runs = load_saved_results()
        if skip:
            for name in skip:
                runs.pop(name, None)

    thresholds: Dict[str, Any] = load_thresholds()
    markdown: str = render_report(runs, thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")

    flat: Dict[str, float] = flatten_metrics(runs)
    violations: List[str] = evaluate_gate(flat, thresholds)
    print(f"[report] 报告已写出：{args.out}")
    print(f"[report] 已采集指标 {len(flat)} 项；质量门违规 {len(violations)} 项")
    if violations:
        for item in violations:
            print(f"[report]   - {item}")


if __name__ == "__main__":
    main()
