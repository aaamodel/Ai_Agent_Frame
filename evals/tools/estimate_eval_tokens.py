# -*- coding: utf-8 -*-
"""评测 token 成本预估工具（**完全离线**，不需要 Milvus / Redis / API Key）。

## 为什么需要它

``python -m evals.report --run-all`` 会真实计费（意图组每条一次大 prompt、
工具组每条还要多轮 Agent + observation 回灌）。跑之前先知道"大概多少钱"，
才能决定要不要跑、以及跑几遍。跑完之后再用 ``_results/*.json`` 里的
``llm_usage`` 反查真相——**本工具的价值是"跑之前有数"，不是"替代真实采集"**。

## 口径（重要，别把估算当实测）

1. **token 换算用字符比例**：中文 1.6 字符/token、ASCII 4 字符/token。
   这是 Qwen 系 BPE 的经验压缩率，**不是精确 tokenizer 计数**；
   真实值以 `_results/*.json` 的 `llm_usage.by_model` 为准。
2. **固定开销按实测的 prompt 体积算**，包括：
   - 意图阶段 system prompt 模板（``agent-rewrite-intent-combined.st``）
   - ``response_format`` 的 JSON Schema（这部分同样计入 input token）
   - 意图候选清单（``INTENT_VECTOR_TOP_K`` 个叶子节点，按 **实测值** 取）
   - 工具 schema 下发载荷（只保留真正发给模型的 type/function 两键）
3. **变量项靠参数估**：Agent 平均步数、单次 observation 体积无法离线得知
   （取决于模型行为与工具返回），用 ``--agent-calls`` / ``--obs-tokens`` 显式传入，
   并给出敏感性区间，而不是假装知道。
4. **生成层不计**：``--run-all`` 不跑 answer 段（需外部提供预测答案），
   因此 ``answer_pass_rate`` / ``abstain_correct_rate`` 的成本恒为 0。

用法::

    # 默认口径（中位估计）
    python -m evals.tools.estimate_eval_tokens

    # 悲观口径：Agent 步数更多、observation 更大
    python -m evals.tools.estimate_eval_tokens --agent-calls 4 --obs-tokens 4000

    # 只估某几组 / 导出 JSON
    python -m evals.tools.estimate_eval_tokens --skip intent --json out.json

    # 跑完真实 baseline 后，对照估算与实际
    python -m evals.tools.estimate_eval_tokens --compare
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EVALS_DIR: Path = _REPO_ROOT / "evals"
RESULTS_DIR: Path = EVALS_DIR / "_results"

# ---- token 换算比例（估算用，非精确 tokenizer）----
CJK_CHARS_PER_TOKEN: float = 1.6
ASCII_CHARS_PER_TOKEN: float = 4.0

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]")

# ---- 各阶段默认的 prompt 路径 ----
COMBINED_PROMPT_PATH: str = "app/query_intent/prompts/agent-rewrite-intent-combined.st"
TOOL_SCHEMA_JSON: Path = EVALS_DIR / "golden" / "_tool_schemas.json"
GOLDEN_FILES: Dict[str, str] = {
    "intent": "intent_cases.jsonl",
    "rag": "rag_cases.jsonl",
    "tool": "tool_cases.jsonl",
}


# =====================================================================
# 1. 计量原语
# =====================================================================
def est_tokens(text: str) -> int:
    """按字符比例估算 token 数（中文 1.6 字符/token，ASCII 4 字符/token）。"""
    cjk: int = len(_CJK.findall(text))
    other: int = len(text) - cjk
    return round(cjk / CJK_CHARS_PER_TOKEN + other / ASCII_CHARS_PER_TOKEN)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_cases(name: str) -> List[Dict[str, Any]]:
    path: Path = EVALS_DIR / "golden" / GOLDEN_FILES[name]
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# =====================================================================
# 2. 固定开销（全部实测自仓库内的真实 prompt / schema / 意图树）
# =====================================================================
def measure_rewrite_prompt() -> Dict[str, int]:
    """意图阶段 system prompt 模板的 token 量。"""
    path = _REPO_ROOT / COMBINED_PROMPT_PATH
    if not path.exists():
        return {"chars": 0, "tokens": 0}
    text = _read(path)
    return {"chars": len(text), "tokens": est_tokens(text)}


def measure_response_format_schema() -> Dict[str, int]:
    """``response_format`` 的 JSON Schema 体积（同样计入 input token）。

    导入失败（缺 pydantic 等）时返回 0 并置 ``available=False``——
    不要用猜的数字填，那会让整份预估看起来精确但实际错误。
    """
    try:
        from app.query_intent.llm_schemas import AgentRewriteIntentCombinedSchema

        schema = AgentRewriteIntentCombinedSchema.model_json_schema()
        compact = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        return {"chars": len(compact), "tokens": est_tokens(compact), "available": True}
    except Exception as exc:  # noqa: BLE001 - 离线环境可能装不上 pydantic
        print(f"[estimate] ⚠️ response_format schema 不可测（{type(exc).__name__}），按 0 计")
        return {"chars": 0, "tokens": 0, "available": False}


def measure_intent_candidates(top_k: int) -> Dict[str, Any]:
    """意图候选清单体积（``_render_intent_list`` 的渲染格式，按实测叶子节点算）。

    复刻 ``AgentCombinedRewriteIntentService._render_intent_list``：
    只渲染 id / path / description / type / tools / example（**最多 1 条**），
    **不含** tool_usage_hint（那一项不进 prompt，别多算）。

    ⚠️ 这里的渲染必须与 ``_render_intent_list`` 保持逐字一致，否则估算值会系统性
    偏离真实 prompt（例如示例从"全部渲染"改成"只留 1 条"后，本函数若不同步就会高估）。
    """
    try:
        from app.query_intent.intent_classify_resolver.intent_tree import IntentTreeFactory
    except Exception as exc:  # noqa: BLE001
        print(f"[estimate] ⚠️ 意图树不可读（{type(exc).__name__}），候选清单按 0 计")
        return {"leaves": 0, "per_node_chars": 0.0, "full_chars": 0,
                "full_tokens": 0, "topk_chars": 0, "topk_tokens": 0}

    nodes: List[Any] = []

    def walk(items: List[Any]) -> None:
        for node in items:
            nodes.append(node)
            walk(list(getattr(node, "children", None) or []))

    walk(list(IntentTreeFactory.build_intent_tree()))
    leaves = [n for n in nodes if not (getattr(n, "children", None) or [])]

    def render(items: List[Any]) -> str:
        parts: List[str] = []
        for node in items:
            parts.append(
                f"- id={node.id}\n  path={node.full_path}\n"
                f"  description={node.description}\n"
            )
            kind = "MCP" if node.is_mcp() else ("SYSTEM" if node.is_system() else "KB")
            parts.append(f"  type={kind}\n")
            tools = node.get_effective_agent_tool_names()
            if tools:
                parts.append(f"  tools={','.join(tools)}\n")
            examples = getattr(node, "examples", None) or []
            if examples:
                parts.append(f"  example={examples[0]}\n")
            parts.append("\n")
        return "".join(parts)

    full = render(leaves)
    n_leaves: int = max(1, len(leaves))
    # ⚠️ 不能按 chars/1.6 一刀切：清单是中英混排，纯中文比例会高估。
    # 先量出「每节点的字符数 + 每节点的中文字符数」，再按 TopK 缩放，
    # 最后交给 est_tokens 用统一口径换算。
    per_node_chars: float = len(full) / n_leaves
    per_node_cjk: float = len(_CJK.findall(full)) / n_leaves
    topk_cjk: int = round(per_node_cjk * top_k)
    topk_other: int = round((per_node_chars - per_node_cjk) * top_k)
    topk_tokens: int = round(topk_cjk / CJK_CHARS_PER_TOKEN + topk_other / ASCII_CHARS_PER_TOKEN)
    return {
        "leaves": len(leaves),
        "per_node_chars": round(per_node_chars, 1),
        "full_chars": len(full),
        "full_tokens": est_tokens(full),
        "topk_chars": round(per_node_chars * top_k),
        "topk_tokens": topk_tokens,
    }


def measure_tool_schemas() -> Dict[str, Any]:
    """工具 schema 下发载荷体积（只算真正发给模型的 ``type`` + ``function``）。"""
    if not TOOL_SCHEMA_JSON.exists():
        return {"tools": 0, "chars": 0, "tokens": 0}
    raw: Dict[str, Any] = json.loads(_read(TOOL_SCHEMA_JSON))
    payload = [
        {"type": t.get("type"), "function": t.get("function")}
        for t in raw.get("tools", [])
    ]
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return {"tools": len(payload), "chars": len(text), "tokens": est_tokens(text)}


# =====================================================================
# 3. 分组估算
# =====================================================================
def estimate(args: argparse.Namespace) -> Dict[str, Any]:
    prompt = measure_rewrite_prompt()
    schema = measure_response_format_schema()
    candidates = measure_intent_candidates(args.intent_topk)
    schemas = measure_tool_schemas()

    cases = {name: load_cases(name) for name in GOLDEN_FILES}
    if args.limit:
        cases = {k: v[: args.limit] for k, v in cases.items()}

    # 意图阶段单次调用的输入（工具组的 pipeline 前置调用完全同构）
    intent_in_per_call: int = (
        prompt["tokens"]
        + schema["tokens"]
        + candidates["topk_tokens"]
        + args.query_tokens
    )
    intent_out_per_call: int = args.rewrite_output_tokens

    groups: Dict[str, Any] = {}

    # ---- 意图组：每条 1 次调用（改写+分类合并；Stage2 走预计算零 LLM）----
    n = len(cases["intent"])
    groups["intent"] = {
        "cases": n,
        "calls": n,
        "calls_per_case": 1.0,
        "input_tokens": n * intent_in_per_call,
        "output_tokens": n * intent_out_per_call,
        "note": "改写+意图合并为单次调用；模式决策为纯规则零 LLM",
    }

    # ---- RAG 组：0 次 chat 调用，只有 query embedding ----
    n = len(cases["rag"])
    groups["rag"] = {
        "cases": n,
        "calls": 0,
        "calls_per_case": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "embedding_tokens": n * args.query_tokens,
        "note": "retrieve_contexts 只调 embedding，不经 chat 出口，故 llm_usage 记 0",
    }

    # ---- 工具组：pipeline 前置 1 次 + Agent 多轮（每轮重发全文）----
    n = len(cases["tool"])
    # Agent 单次调用的稳定部分：系统提示词 + 工具 schema 子集 + 用户/记忆上下文
    agent_system: int = args.agent_system_tokens
    agent_tools: int = round(
        schemas["tokens"] * args.tool_subset_ratio
    )
    agent_user: int = args.agent_user_tokens
    first_call_in: int = agent_system + agent_tools + agent_user

    # 后续轮：历史累积（近似取首轮 + 输出 + observation 的等比增长）
    steps: float = args.agent_calls
    later_calls: float = max(0.0, steps - 1.0)
    obs: int = args.obs_tokens
    # 第 k 轮输入 ≈ 首轮 + (k-1) 次「输出 + observation」
    later_input: float = later_calls * first_call_in + obs * later_calls * (later_calls + 1) / 2
    agent_in: float = (first_call_in + later_input) * n
    agent_out: float = n * (
        args.tool_call_output_tokens + later_calls * args.answer_output_tokens
    )

    groups["tool"] = {
        "cases": n,
        "calls": round(n * (1 + steps), 1),
        "calls_per_case": round(1 + steps, 2),
        "input_tokens": round(n * intent_in_per_call + agent_in),
        "output_tokens": round(n * intent_out_per_call + agent_out),
        "breakdown": {
            "pipeline_input": n * intent_in_per_call,
            "agent_input": round(agent_in),
            "agent_first_call_input": first_call_in,
            "agent_obs_tokens": obs,
        },
        "note": f"pipeline 前置 1 次 + Agent 约 {steps} 轮（每轮重发全文，输入线性增长）",
    }

    total_in: int = sum(g["input_tokens"] for g in groups.values())
    total_out: int = sum(g["output_tokens"] for g in groups.values())
    return {
        "assumptions": {
            "cjk_chars_per_token": CJK_CHARS_PER_TOKEN,
            "ascii_chars_per_token": ASCII_CHARS_PER_TOKEN,
            "intent_topk": args.intent_topk,
            "agent_calls_per_case": args.agent_calls,
            "obs_tokens_per_call": args.obs_tokens,
            "tool_subset_ratio": args.tool_subset_ratio,
        },
        "fixed_costs": {
            "rewrite_prompt": prompt,
            "response_format_schema": schema,
            "intent_candidates": candidates,
            "tool_schemas": schemas,
            "intent_input_per_call": intent_in_per_call,
            "intent_output_per_call": intent_out_per_call,
        },
        "groups": groups,
        "total": {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "llm_calls": round(sum(g["calls"] for g in groups.values()), 1),
        },
    }


def sensitivity(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """对最不确定的两个变量做敏感性分析（Agent 步数 / observation 体积）。"""
    rows: List[Dict[str, Any]] = []
    for calls, obs, label in (
        (2.0, args.obs_tokens * 0.5, "乐观：2 轮、observation 减半"),
        (args.agent_calls, args.obs_tokens, "中位：默认口径"),
        (3.5, args.obs_tokens * 1.5, "悲观：3.5 轮、observation ×1.5"),
        (4.0, args.obs_tokens * 2.0, "极端：4 轮、observation ×2（压力上限）"),
    ):
        clone = argparse.Namespace(**vars(args))
        clone.agent_calls = calls
        clone.obs_tokens = round(obs)
        totals = estimate(clone)["total"]
        rows.append({"label": label, **totals})
    return rows


# =====================================================================
# 4. 与真实采集对照
# =====================================================================
def compare_with_actual() -> None:
    """读 ``_results/*.json`` 里的 ``llm_usage``，与估算对照（跑过才有）。"""
    found = False
    for name in ("intent", "rag", "tool"):
        path = RESULTS_DIR / f"{name}.json"
        if not path.exists():
            continue
        found = True
        try:
            data = json.loads(_read(path))
        except Exception:  # noqa: BLE001
            print(f"[compare] {name}: 结果文件无法解析")
            continue
        usage = (data.get("metrics") or {}).get("llm_usage") or {}
        records = len(data.get("records") or [])
        calls = usage.get("calls") or 0
        print(
            f"[compare] {name:6s} 样本={records:3d} LLM调用={calls:4d} "
            f"input={usage.get('input_tokens', 0):>8d} "
            f"output={usage.get('output_tokens', 0):>7d} "
            f"total={usage.get('total_tokens', 0):>8d}"
            + (
                f"  平均 {usage.get('total_tokens', 0) / records:.0f} token/条"
                if records
                else ""
            )
        )
        by_model = usage.get("by_model") or {}
        for model, stats in by_model.items():
            print(
                f"           └─ {model}: calls={stats.get('calls')} "
                f"in={stats.get('input')} out={stats.get('output')}"
            )
    if not found:
        print("[compare] 还没有 evals/_results/*.json，先跑 baseline 再来对照")


# =====================================================================
# 5. 入口
# =====================================================================
def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.1f}"
    return f"{value:,}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评测 token 成本预估（离线，不连任何服务）"
    )
    parser.add_argument("--intent-topk", type=int, default=8,
                        help="意图候选清单条数（对齐 rag_constant.INTENT_VECTOR_TOP_K）")
    parser.add_argument("--agent-calls", type=float, default=2.5,
                        help="工具组每条用例的 Agent 平均轮数（不含 pipeline 前置调用）")
    parser.add_argument("--obs-tokens", type=int, default=2500,
                        help="每次工具 observation 回灌的 token（RAG top5 切片约 2500）")
    parser.add_argument("--tool-subset-ratio", type=float, default=0.35,
                        help="单次传递给模型的工具 schema 占全量 schema 的比例")
    parser.add_argument("--agent-system-tokens", type=int, default=170,
                        help="Agent 系统提示词 token（REACT_FC_SYSTEM_PROMPT 实测约 168）")
    parser.add_argument("--agent-user-tokens", type=int, default=500,
                        help="Agent 首轮 user 段（意图上下文/记忆/问题）token")
    parser.add_argument("--query-tokens", type=int, default=12, help="单条 query 的 token")
    parser.add_argument("--rewrite-output-tokens", type=int, default=300,
                        help="改写+意图打分单次输出的 token")
    parser.add_argument("--tool-call-output-tokens", type=int, default=60,
                        help="Agent 发起一次工具调用的输出 token")
    parser.add_argument("--answer-output-tokens", type=int, default=400,
                        help="Agent 最终答复的输出 token")
    parser.add_argument("--limit", type=int, default=None, help="只估前 N 条（对齐 --limit）")
    parser.add_argument("--json", type=Path, default=None, help="把结果写成 JSON")
    parser.add_argument("--compare", action="store_true",
                        help="读 _results/*.json 里的真实 llm_usage 做对照")
    args = parser.parse_args()

    if args.compare:
        compare_with_actual()
        return

    result = estimate(args)

    fc = result["fixed_costs"]
    print("=" * 78)
    print("固定开销（实测自仓库内真实 prompt / schema / 意图树）")
    print("=" * 78)
    print(f"  意图 system prompt 模板      {fc['rewrite_prompt']['chars']:>7,} chars  "
          f"{fc['rewrite_prompt']['tokens']:>6,} tokens")
    rf = fc["response_format_schema"]
    print(f"  response_format JSON Schema  {rf['chars']:>7,} chars  "
          f"{rf['tokens']:>6,} tokens" + ("" if rf.get("available") else "  （不可测，按 0 计）"))
    ic = fc["intent_candidates"]
    print(f"  意图候选清单 TopK={args.intent_topk:<3d}          {ic['topk_chars']:>7,} chars  "
          f"{ic['topk_tokens']:>6,} tokens  （叶子节点共 {ic['leaves']} 个）")
    ts = fc["tool_schemas"]
    print(f"  工具 schema 全量载荷         {ts['chars']:>7,} chars  "
          f"{ts['tokens']:>6,} tokens  （{ts['tools']} 个工具）")
    print(f"  ── 意图阶段单次调用输入合计    {fc['intent_input_per_call']:>21,} tokens")
    print()

    print("=" * 78)
    print("分组预估")
    print("=" * 78)
    print(f"{'组':6s} {'样本':>5s} {'LLM调用':>8s} {'输入token':>12s} {'输出token':>11s}")
    for name, g in result["groups"].items():
        print(f"{name:6s} {g['cases']:>5d} {_fmt(g['calls']):>8s} "
              f"{g['input_tokens']:>12,} {g['output_tokens']:>11,}")
        print(f"       └ {g['note']}")
    t = result["total"]
    print("-" * 78)
    print(f"{'合计':6s} {'':>5s} {_fmt(t['llm_calls']):>8s} "
          f"{t['input_tokens']:>12,} {t['output_tokens']:>11,}")
    print(f"{'总计':30s} {t['total_tokens']:>12,} tokens")
    print()

    print("=" * 78)
    print("敏感性（最不确定的两个变量：Agent 轮数 / observation 体积）")
    print("=" * 78)
    for row in sensitivity(args):
        print(f"  {row['label']:34s} 总计 {row['total_tokens']:>9,} tokens  "
              f"（输入 {row['input_tokens']:,} / 输出 {row['output_tokens']:,}）")
    print()
    print("⚠️ 本结果是**估算**：token 换算用字符比例（中文 1.6 字符/token），")
    print("   且 Agent 轮数、observation 体积是参数假设。真实值以 baseline 跑完后")
    print("   evals/_results/*.json 的 metrics.llm_usage 为准（--compare 可对照）。")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[estimate] 已写出 {args.json}")


if __name__ == "__main__":
    main()
