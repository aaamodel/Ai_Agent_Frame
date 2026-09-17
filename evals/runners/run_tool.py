# -*- coding: utf-8 -*-
"""Runner：工具调用离线评测（工具调用成功率 + 关键参数命中率）。

链路：**跑完整真实链路**——意图三阶段 Pipeline → ``build_orchestrator_input``
→ ``AgentOrchestrator.run``（ReAct / Plan-and-Execute）。因此测的是"给定这句话，
Agent 到底调了哪个工具、带了什么参数"，而不是把工具名硬塞给模型。

## 工具调用序列从哪里读（本文件的核心口径）

不自己包装 ``ToolRegistry.invoke``，而是读 **``AgentResponse.steps``**。
理由是 steps 是应用自己产出的权威账本（也通过 SSE 推给前端），并且它比
"invoke 包装"多覆盖两种关键情形：

===============================  ==============================
情形                              steps 里怎么体现
===============================  ==============================
FC 路径正常调用                   ``kind="fc_tool_call"`` + ``action``/``action_input``
文本协议路径                      ``action``/``action_input``（见 execute_node.py:744）
Plan 路径                         ``tool_name``（subtask 声明的工具）
**白名单拒绝**（没走到 invoke）   仍有 ``action`` 记账（invoke 包装会漏掉）
**危险工具 HITL 挂起**           步骤可能没落盘，但 ``approval_payloads[].tool_name`` 有
最终答复                          ``kind="fc_final"``，**不计入**工具调用
===============================  ==============================

因此提取规则 = 「steps 里所有带工具名的步骤」∪「approval_payloads 里的工具名」。
`ToolRegistry.invoke` 包装是**旁路一致性校验**（``executed_invocations``，连实际
入参一起记）：若 steps 与审批挂起都没记账、但 invoke 确实执行过（plan_execute
路径的已知缺口），则用旁路记录**回填判分序列**（工具名 + 真实参数），避免把
"账本没写"误判成"没调工具"；同时保留 ``steps_incomplete=True`` 标记暴露记账
缺口本身。回填来源记录在 ``tool_calls_source``（``steps`` / ``invoke_bypass``）。

## 关键参数命中率的判定对象

取「**第一个命中的工具调用**」的参数来判分（命中 = ``expected_tool`` 或
``acceptable_tools`` 中任一）。若一个都没命中，则取第一个调用的参数——这样
报告里能看到"它调错了工具、参数也不对"，而不是简单记 0。

用法::

    python -m evals.runners.run_tool --limit 3            # 冒烟
    python -m evals.runners.run_tool                      # 全量（慢：每条一次多步 Agent）
    python -m evals.runners.run_tool --strategy react     # 强制 react（参数记全）
    python -m evals.runners.run_tool --limit 3 --order random          # 每次抽不同 3 条
    python -m evals.runners.run_tool --limit 3 --order random --seed 7 # 可复现的随机
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.metrics import aggregate_tool_results, format_metric, key_arg_matches, percentile  # noqa: E402
from evals.runners._isolation import build_eval_session_id, new_run_tag, purge_session  # noqa: E402
from evals.runners._runtime import EvalRuntime  # noqa: E402
from evals.runners._usage import record_llm_usage  # noqa: E402
from evals.sampling import (  # noqa: E402
    DEFAULT_ORDER,
    add_order_cli_args,
    describe_selection,
    run_config_of,
    select_cases,
)


# ---------------------------------------------------------------------
# 1. 从 AgentResponse.steps 提取工具调用
# ---------------------------------------------------------------------
def extract_tool_calls(steps: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
    """从 ``AgentResponse.steps`` 提取工具调用序列。

    跳过最终答复步（``kind == "fc_final"`` 或 ``final is True``）——它没有工具名，
    若不过滤会被 ``step.get("tool_name")`` 之类的兜底逻辑误当成调用。

    Returns:
        每项 ``{"tool", "args", "args_raw", "status", "approval_denied",
        "budget_denied", "step"}``。
    """
    calls: List[Dict[str, Any]] = []
    for step in steps or []:
        if not isinstance(step, Mapping):
            continue
        if str(step.get("kind") or "") == "fc_final" or step.get("final") is True:
            continue

        # plan 路径的 steps 是嵌套形态 ``{"phase": "execute", "record": {...}}``
        # （见 `_common.emit_plan_step`），工具名/参数都在 `record` 里。
        # 不展开的话，plan_execute 用例的工具调用会被整体漏判（steps 看起来是空的，
        # 判分只能靠 invoke 旁路回填），key_arg_recall 也会因为拿不到入参而掉样本。
        record: Any = step.get("record")
        if isinstance(record, Mapping):
            step = {**record, "ts": step.get("ts"), "phase": step.get("phase")}

        name: Any = step.get("action") or step.get("tool_name")
        if not name:
            continue

        raw_args: Any = step.get("action_input")
        if raw_args is None:
            parsed: Any = step.get("parsed")
            if isinstance(parsed, Mapping):
                raw_args = parsed.get("action_input")

        args: Dict[str, Any] = (
            dict(raw_args) if isinstance(raw_args, Mapping) else {}
        )
        calls.append(
            {
                "tool": str(name),
                "args": args,
                "args_raw": raw_args,
                "status": step.get("status"),
                "approval_denied": bool(step.get("approval_denied")),
                "budget_denied": bool(step.get("budget_denied")),
                "step": step.get("step"),
            }
        )
    return calls


def _approval_tool_names(response: Any) -> List[str]:
    """从 HITL 挂起载荷里取被审批拦下的工具名（此时 steps 可能尚未落盘）。"""
    names: List[str] = []
    for payload in getattr(response, "approval_payloads", None) or []:
        if isinstance(payload, Mapping) and payload.get("tool_name"):
            names.append(str(payload["tool_name"]))
    return names


def reconcile_tool_calls(
    calls: Sequence[Mapping[str, Any]],
    approval_tools: Sequence[str],
    executed_invocations: Sequence[Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], List[str], bool, str]:
    """统一三路工具记账，产出判分序列。

    三路来源：
        - ``calls``：``AgentResponse.steps`` 提取的工具调用（权威账本，含被白名单/
          预算/审批拦下的调用，可能带参数）；
        - ``approval_tools``：HITL 挂起载荷里的工具名（steps 可能还没落盘）；
        - ``executed_invocations``：包装 ``ToolRegistry.invoke`` 的旁路记录
          （确实执行过，且带真实入参）。

    Returns:
        ``(judging_calls, called_tools, steps_incomplete, source)``：
        - ``judging_calls``：用于成功率/关键参数判分的调用序列；
        - ``called_tools``：保序去重的工具名序列；
        - ``steps_incomplete``：steps 与审批都没记账、但 invoke 确实执行过
          （plan_execute 路径的记账缺口，与是否回填无关，始终暴露）；
        - ``source``：``steps`` / ``steps+approval`` / ``approval`` /
          ``invoke_bypass`` / ``none``。
    """
    steps_names: List[str] = [str(c.get("tool")) for c in calls]
    steps_incomplete: bool = bool(
        not steps_names and not approval_tools and executed_invocations
    )

    judging_calls: List[Dict[str, Any]] = [dict(c) for c in calls]
    if calls:
        source = "steps+approval" if approval_tools else "steps"
    elif approval_tools:
        source = "approval"
    elif executed_invocations:
        # C2：steps 完全没记账时用旁路 invoke 记录回填（含真实入参，
        # key_arg_recall 也能判），避免把"账本缺口"误判成"没调工具"。
        judging_calls = [
            {
                "tool": str(inv.get("tool")),
                "args": dict(inv.get("args") or {}),
                "args_raw": inv.get("args"),
                "status": "executed",
                "approval_denied": False,
                "budget_denied": False,
                "step": None,
                "source": "invoke_bypass",
            }
            for inv in executed_invocations
        ]
        source = "invoke_bypass"
    else:
        source = "none"

    called_tools: List[str] = []
    for name in [str(c.get("tool")) for c in judging_calls] + list(approval_tools):
        if name and name not in called_tools:
            called_tools.append(name)
    return judging_calls, called_tools, steps_incomplete, source


# ---------------------------------------------------------------------
# 2. 单条用例执行
# ---------------------------------------------------------------------
def _build_intent_context(pipeline: Any, pipeline_output: Any, force_mode: Optional[str]) -> tuple:
    """复用应用自身的 ``build_orchestrator_input``，产出 (mode, IntentContext, input)。

    不自己拼 IntentContext：chat.py 线上走的就是这个方法，复用它才能保证
    「评测的 pipeline→orchestrator 交接」与「线上交接」完全一致
    （包括 mode 强覆盖时的 hint 清理、confidence 兜底等细节）。
    """
    from app.core.agent.orchestrator import IntentContext

    mode, payload, effective_input = pipeline.build_orchestrator_input(
        pipeline_output, force_mode
    )
    return mode, IntentContext(**payload), effective_input


def _pick_args_for_judging(
    calls: Sequence[Mapping[str, Any]],
    expected_tool: str,
    acceptable_tools: Optional[Sequence[str]],
) -> Dict[str, Any]:
    """选第一个命中工具调用的参数；全都没命中则退回第一个调用的参数。"""
    allowed = {str(expected_tool), *[str(t) for t in (acceptable_tools or [])]}
    for call in calls:
        if str(call.get("tool")) in allowed:
            return dict(call.get("args") or {})
    if calls:
        return dict(calls[0].get("args") or {})
    return {}


async def run_tool_eval(
    cases: List[Dict[str, Any]],
    runtime: Optional[EvalRuntime] = None,
    *,
    limit: Optional[int] = None,
    strategy: Optional[str] = None,
    sleep_between: float = 0.0,
    isolate: bool = True,
    order: str = DEFAULT_ORDER,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """跑完整工具调用评测，返回 ``{"records": [...], "metrics": {...}, "usage": {...}}``。

    Args:
        cases: 工具黄金集用例。
        runtime: 评测运行时（None 则新建）。
        limit: 分层抽样的条数上限。
        strategy: 强制编排模式（None = 用 Pipeline 自己的决策）。
        sleep_between: 用例间隔秒数。
        isolate: 是否启用**记忆隔离**（一次性 session_id + 跑完清理）。
            默认 True。关闭它会让指标随运行次数膨胀（见 ``_isolation`` 模块说明），
            仅在排查"记忆本身是否影响工具选择"时才该关。
    """
    runtime = runtime or EvalRuntime()
    pipeline = runtime.get_pipeline()
    orchestrator = runtime.get_orchestrator(strategy=strategy or "auto")
    registry = runtime.get_tool_registry()

    # ---- 记忆隔离：本次运行的唯一 tag，贯穿所有用例的 session_id ----
    run_tag: str = new_run_tag()
    cleanup_failures: int = 0

    # 分层抽样而非 cases[:limit]：黄金集按 expected_tool 成簇排列，
    # 简单切片会让抽样结果集中在头几个工具上，失去代表性。
    selected = select_cases("tool", cases, limit, order=order, seed=seed)
    print(f"[tool] 本次用例：{describe_selection(selected, order, seed)}", flush=True)
    records: List[Dict[str, Any]] = []
    per_case_tokens: List[int] = []

    # ---- 旁路一致性校验通道：包装 registry.invoke 记录"确实执行了"的工具与入参 ----
    executed_invocations: List[Dict[str, Any]] = []
    original_invoke: Any = registry.invoke

    async def _recording_invoke(name: str, arguments: Dict[str, Any]) -> Any:
        executed_invocations.append(
            {"tool": str(name), "args": dict(arguments or {})}
        )
        return await original_invoke(name, arguments)

    registry.invoke = _recording_invoke  # type: ignore[method-assign]

    try:
        with record_llm_usage() as usage:
            for index, case in enumerate(selected, start=1):
                tokens_before: int = usage.total_tokens
                record, session_id = await _run_single_case(
                    case, pipeline, orchestrator, executed_invocations, strategy, run_tag=run_tag
                )
                per_case_tokens.append(usage.total_tokens - tokens_before)
                records.append(record)
                _print_progress(index, len(selected), record)
                # 跑完即清：否则下一轮评测会召回本轮写下的记忆，成本逐次膨胀
                if isolate and not await purge_session(runtime, session_id):
                    cleanup_failures += 1
                if sleep_between:
                    await asyncio.sleep(sleep_between)
    finally:
        registry.invoke = original_invoke  # type: ignore[method-assign]

    metrics: Dict[str, Any] = aggregate_tool_results(records)
    per_turn = usage.per_turn(len(records))
    if per_turn is not None:
        metrics["avg_tokens_per_turn"] = per_turn
    # 中位数抗极值：工具组是"意图 Pipeline + 多轮 Agent"，个别用例会跑很久，
    # 均值被它们主导；中位数才是"典型一轮"的稳健估计。
    metrics["median_tokens_per_turn"] = (
        percentile([float(t) for t in per_case_tokens], 50) if per_case_tokens else None
    )
    # 一轮问答折合多少次 LLM 调用——比总 token 更能解释"多轮成本花在哪"。
    metrics["llm_calls_per_request"] = usage.calls / len(records) if records else None
    metrics["llm_usage"] = usage.as_dict()
    # 隔离审计：非 0 说明有会话没清干净，下一轮指标可能被污染（显式暴露，不静默）
    metrics["memory_isolated"] = bool(isolate)
    metrics["memory_cleanup_failures"] = cleanup_failures
    metrics["eval_run_tag"] = run_tag

    return {"records": records, "metrics": metrics}


async def _run_single_case(
    case: Dict[str, Any],
    pipeline: Any,
    orchestrator: Any,
    executed_invocations: List[Dict[str, Any]],
    strategy: Optional[str],
    *,
    run_tag: str,
) -> tuple[Dict[str, Any], str]:
    """执行单条工具用例，返回 ``(评测记录, 本用例 session_id)``（异常已隔离）。

    session_id 是**一次性**的：``eval::tool::<run_tag>::<case_id>``。
    固定 id 会让上一次运行写下的记忆被这一次召回，指标逐次膨胀。
    """
    query: str = str(case["query"])
    case_id: str = str(case["id"])
    session_id: str = build_eval_session_id("tool", case_id, run_tag)
    expected_tool: str = str(case["expected_tool"])
    acceptable: List[str] = [str(t) for t in (case.get("acceptable_tools") or [])]
    key_args: Mapping[str, Any] = case.get("key_args") or {}
    inspect_first_n: Optional[int] = case.get("inspect_first_n")

    executed_invocations.clear()
    started = time.perf_counter()
    error: Optional[str] = None
    calls: List[Dict[str, Any]] = []
    approval_tools: List[str] = []
    mode_used: Optional[str] = None
    intent_path: Optional[str] = None
    allowed_tools: List[str] = []
    answer_head: Optional[str] = None
    agent_success: Optional[bool] = None
    awaiting_approval: bool = False

    try:
        # ---- 步骤 1：真实意图三阶段 Pipeline（拿到 mode / allowed_tools / slots）----
        pipeline_output = await asyncio.to_thread(
            pipeline.run, query, [], session_id, [], []
        )
        mode, intent_context, effective_input = _build_intent_context(
            pipeline, pipeline_output, strategy
        )
        mode_used = str(mode)
        allowed_tools = [str(t) for t in (intent_context.allowed_tools or [])]
        intents_result = getattr(pipeline_output, "intents_result", None)
        intent_path = str(getattr(intents_result, "primary_intent_text", "") or "")

        # ---- 步骤 2：跑真实编排器（ReAct / Plan-Execute）----
        response = await orchestrator.run(
            user_input=effective_input,
            session_id=session_id,
            mode=mode,
            intent=intent_context,
        )
        calls = extract_tool_calls(getattr(response, "steps", None))
        approval_tools = _approval_tool_names(response)
        agent_success = bool(getattr(response, "success", False))
        awaiting_approval = bool(getattr(response, "awaiting_approval", False))
        answer_head = str(getattr(response, "answer", "") or "")[:200] or None
    except Exception as exc:  # noqa: BLE001 - 单条失败不中断整轮
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000.0

    # ---- 三路记账对账（steps / 审批挂起 / invoke 旁路，含 C2 缺口回填）----
    judging_calls, called_tools, steps_incomplete, tool_calls_source = (
        reconcile_tool_calls(calls, approval_tools, executed_invocations)
    )

    arg_matches: Dict[str, bool] = {}
    if key_args:
        judged_args = _pick_args_for_judging(judging_calls, expected_tool, acceptable)
        arg_matches = key_arg_matches(judged_args, key_args)

    record: Dict[str, Any] = {
        "id": case_id,
        "query": query,
        # ---- 判分键（aggregate_tool_results 直接消费）----
        "expected_tool": expected_tool,
        "acceptable_tools": acceptable,
        "called_tools": called_tools,
        "inspect_first_n": inspect_first_n,
        "arg_matches": arg_matches,
        "latency_ms": round(latency_ms, 2),
        # ---- 归因信息（不参与判分，但报告要用）----
        "calls": calls,
        "mode_used": mode_used,
        "intent_path": intent_path,
        "allowed_tools": allowed_tools,
        "tool_not_whitelisted": [
            t for t in called_tools if allowed_tools and t not in allowed_tools
        ],
        "approval_tool_names": approval_tools,
        "awaiting_approval": awaiting_approval,
        "agent_success": agent_success,
        "answer_head": answer_head,
        "executed_tools": list(
            dict.fromkeys(str(inv["tool"]) for inv in executed_invocations)
        ),
        # steps 记账缺口（按 steps/审批 本身是否为空判定，不受回填影响）
        "steps_incomplete": steps_incomplete,
        # 判分序列来源：steps / steps+approval / approval / invoke_bypass / none
        "tool_calls_source": tool_calls_source,
        # 本次用例的隔离 session_id（归因用：可据此复查记忆分区）
        "session_id": session_id,
        "denied": {
            "approval": [c["tool"] for c in calls if c["approval_denied"]],
            "budget": [c["tool"] for c in calls if c["budget_denied"]],
        },
    }
    if error:
        record["error"] = error
    return record, session_id


def _print_progress(index: int, total: int, record: Dict[str, Any]) -> None:
    from evals.metrics import tool_call_success

    ok: bool = tool_call_success(
        record["called_tools"],
        record["expected_tool"],
        record.get("acceptable_tools"),
        inspect_first_n=record.get("inspect_first_n"),
    )
    mark: str = "PASS" if ok else "FAIL"
    print(
        f"[tool {index}/{total}] {mark} {record['id']} "
        f"expect={record['expected_tool']} called={record['called_tools'] or '-'} "
        f"mode={record.get('mode_used')} {record['latency_ms']:.0f}ms",
        flush=True,
    )


def main() -> None:
    from evals.loaders import load_tool_cases, save_results

    parser = argparse.ArgumentParser(description="工具调用离线评测")
    parser.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "evals" / "_results" / "tool.json"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑 N 条（冒烟用）。按**分层抽样**选取：在各 expected_tool 之间"
             "轮转分配，而不是简单取文件前 N 条",
    )
    parser.add_argument(
        "--strategy",
        choices=["react", "plan_execute"],
        default=None,
        help=(
            "强制编排模式。默认 None = 用 Pipeline 自己的决策（最真实）。"
            "注意：plan_execute 路径的 steps 不记录工具参数，key_arg_recall 会少样本。"
        ),
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="用例间隔秒数（避免限流）")
    parser.add_argument(
        "--keep-sessions",
        action="store_true",
        help=(
            "关闭记忆隔离（不清 Redis/Milvus 里的评测记忆）。默认清理是有意为之："
            "不清的话下一轮会召回本轮写下的记忆，avg_tokens_per_turn 会随运行次数逐次膨胀。"
            "仅在排查「记忆是否影响工具选择」时才该加这个开关。"
        ),
    )
    add_order_cli_args(parser)
    args = parser.parse_args()

    async def _main() -> Dict[str, Any]:
        runtime = EvalRuntime()
        try:
            result = await run_tool_eval(
                load_tool_cases(),
                runtime,
                limit=args.limit,
                strategy=args.strategy,
                sleep_between=args.sleep,
                isolate=not args.keep_sessions,
                order=args.order,
                seed=args.seed,
            )
        finally:
            await runtime.aclose()

        # ⚠️ 落盘/打印必须在事件循环**内部**完成，不能放到 asyncio.run() 之后。
        # 原因：Windows 的 Proactor 事件循环在 Runner.__exit__ → loop.close() 里会等待
        # 所有未完成的 IOCP 操作（第三方 SDK——openai/httpx、pymilvus、redis——常留下
        # 未关闭的连接），一旦有残留就永久阻塞在 _poll()。此时 asyncio.run() 永远不返回，
        # 写在它后面的 save_results 就永远执行不到 —— 症状是「跑完 3 条用例后卡住、
        # 没有 tool.json、只能 Ctrl+C」，本函数就是为此把落盘挪进来的。
        # 该用例结果已完整采到，卡在收尾不应连累数据。
        result["run_config"] = run_config_of(args.order, args.seed, args.limit)
        save_results(result, args.out)

        metrics = result["metrics"]
        print(
            f"[tool] 完成：success_rate={format_metric(metrics['tool_success_rate'])} "
            f"key_arg_recall={format_metric(metrics['key_arg_recall'])}"
            f"（{metrics['key_arg_cases']} 条有参数标注）"
            f" p95={format_metric(metrics['p95_latency_ms'], '.0f')}ms "
            f"avg_tokens/turn={format_metric(metrics.get('avg_tokens_per_turn'), '.0f')} "
            f"median_tokens/turn={format_metric(metrics.get('median_tokens_per_turn'), '.0f')} "
            f"memory_isolated={metrics.get('memory_isolated')}"
            f"（清理失败 {metrics.get('memory_cleanup_failures')} 条）"
            f" -> {args.out}",
            flush=True,
        )
        print(
            "[tool] 指标与 tool.json 均已就绪；若进程在退出阶段静止不动，"
            "是事件循环收尾在等残留连接，可直接 Ctrl+C，不影响本次结果。",
            flush=True,
        )
        return result

    asyncio.run(_main())


if __name__ == "__main__":
    main()
