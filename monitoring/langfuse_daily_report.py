# -*- coding: utf-8 -*-
"""Langfuse 日报：把线上观测数据汇总成「调用量 / token / 成本 / 延迟 / 错误」日报。

## 为什么直接打 REST API 而不是用 langfuse SDK

项目依赖 ``langfuse>=4.0.0``（OTEL 架构）。SDK 的**读**接口在各版本间改过多次
（``fetch_traces`` / ``api.trace.list`` / ``api.trace.get`` 的签名与返回结构都有差异），
而 Langfuse 的**公开 REST API**（``/api/public/*``）是有版本承诺的稳定契约
（Basic Auth：public key 作用户名、secret key 作密码）。

对本脚本的用途（离线汇总日报）来说，直接用 REST 更稳、更容易解释、
也不需要在评测环境装 SDK。

## 取数口径

===========================  =========================================================
数据                          接口
===========================  =========================================================
调用量与错误率                 ``GET /api/public/traces``（时间窗）
LLM 调用 / token / 延迟        ``GET /api/public/observations?type=GENERATION``
成本                          本地 ``model_pricing.yaml`` + ``cost_calculator``
===========================  =========================================================

**关键点**：token 与模型名来自 **GENERATION 类型的 observation**，不是 trace。
因为一次对话可能包含多次 LLM 调用（意图改写 + 分类 + Agent 多步），
按 trace 统计会把它们合并成一个数，"哪一档在烧钱"就看不出来了。

## 用法

    # 今天的日报
    python monitoring/langfuse_daily_report.py

    # 指定日期 + 输出路径
    python monitoring/langfuse_daily_report.py --date 2026-09-12 \
        --out monitoring/langfuse_daily_2026-09-12.md

    # 先看看 Langfuse 里到底记录了什么模型名（填 price 表之前必跑）
    python monitoring/langfuse_daily_report.py --list-models

    # 输出 JSON 供其它脚本消费
    python monitoring/langfuse_daily_report.py --json

⚠️ 本脚本只读不写：不会修改任何 Langfuse 数据。
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_MONITORING_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _MONITORING_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from monitoring.cost_calculator import (  # noqa: E402
    ModelUsage,
    Pricing,
    compute_cost,
    load_pricing,
    normalize_usage,
)

PAGE_LIMIT: int = 100
MAX_PAGES: int = 200  # 硬上限：防止时间窗写错导致把整个项目拉下来
DEFAULT_HOST: str = "https://cloud.langfuse.com"


# =====================================================================
# 一、凭据与 HTTP
# =====================================================================
def resolve_credentials() -> Tuple[str, str, str]:
    """取 (public_key, secret_key, host)。优先环境变量，其次应用配置。"""
    import os

    public_key: str = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key: str = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    host: str = (
        os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or ""
    ).strip()

    if not (public_key and secret_key):
        try:
            from app.infrastructure.trace.langfuse import resolve_langfuse_keys

            cfg_pk, cfg_sk, cfg_host = resolve_langfuse_keys()
            public_key = public_key or cfg_pk
            secret_key = secret_key or cfg_sk
            host = host or cfg_host
        except Exception:  # noqa: BLE001 - 配置不可读时仍允许纯环境变量用法
            pass

    return public_key, secret_key, host or DEFAULT_HOST


class LangfuseReader:
    """只读的 Langfuse 公开 API 客户端（带分页）。"""

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        if not (public_key and secret_key):
            raise SystemExit(
                "缺少 Langfuse 凭据。请设置环境变量 LANGFUSE_PUBLIC_KEY / "
                "LANGFUSE_SECRET_KEY（可选 LANGFUSE_HOST），"
                "或在 .env 里配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY。"
            )
        self.host: str = host.rstrip("/")
        token: str = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self.headers: Dict[str, str] = {"Authorization": f"Basic {token}"}
        self._client: Any = self._build_client()

    @staticmethod
    def _build_client() -> Any:
        try:
            import httpx

            return httpx.Client(timeout=30.0)
        except ImportError:  # pragma: no cover - 兜底到 requests
            import requests

            return requests.Session()

    def get(self, path: str, params: Mapping[str, Any]) -> Dict[str, Any]:
        url: str = f"{self.host}{path}"
        response: Any = self._client.get(url, params=dict(params), headers=self.headers)
        status: int = int(getattr(response, "status_code", 0))
        if status == 401:
            raise SystemExit(
                "Langfuse 返回 401：凭据无效。请核对 LANGFUSE_PUBLIC_KEY / SECRET_KEY "
                f"与 host（当前 host={self.host}，注意区分 cloud.langfuse.com 与自建地址）。"
            )
        if status >= 400:
            body: str = str(getattr(response, "text", ""))[:400]
            raise SystemExit(f"Langfuse {path} 返回 HTTP {status}: {body}")
        data: Any = response.json()
        return data if isinstance(data, dict) else {}

    def paged(self, path: str, params: Mapping[str, Any], *, max_pages: int = MAX_PAGES) -> List[Dict[str, Any]]:
        """按页拉全量（Langfuse 用 page/limit 分页，meta.totalPages 给出总页数）。"""
        out: List[Dict[str, Any]] = []
        page: int = 1
        while page <= max_pages:
            query: Dict[str, Any] = dict(params)
            query["page"] = page
            query["limit"] = PAGE_LIMIT
            payload: Dict[str, Any] = self.get(path, query)
            rows: Any = payload.get("data")
            if not isinstance(rows, list) or not rows:
                break
            out.extend(row for row in rows if isinstance(row, dict))
            meta: Mapping[str, Any] = payload.get("meta") or {}
            total_pages: Any = meta.get("totalPages")
            if isinstance(total_pages, int) and page >= total_pages:
                break
            if len(rows) < PAGE_LIMIT:
                break
            page += 1
        return out


# =====================================================================
# 二、字段解析（版本兼容）
# =====================================================================
def extract_tokens(observation: Mapping[str, Any]) -> Tuple[int, int, int]:
    """从 generation observation 里取 (input, output, total)。

    兼容三种历史命名（Langfuse 改过口径，不兼容就会全部读成 0）：
        1. ``usage: {"input": n, "output": n, "total": n}``   ← v4 / OTEL 口径
        2. ``usage: {"promptTokens": n, "completionTokens": n, "totalTokens": n}``
        3. observation 顶层 ``promptTokens`` / ``completionTokens``
    另外 ``usageDetails`` / ``usage_details`` 也可能带明细，作为最后兜底。
    """
    usage: Any = observation.get("usage")
    if isinstance(usage, Mapping) and usage:
        in_tok: Any = _pick(usage, "input", "promptTokens", "prompt_tokens", "input_tokens")
        out_tok: Any = _pick(usage, "output", "completionTokens", "completion_tokens", "output_tokens")
        tot_tok: Any = _pick(usage, "total", "totalTokens", "total_tokens")
    else:
        in_tok = _pick(observation, "promptTokens", "prompt_tokens", "input")
        out_tok = _pick(observation, "completionTokens", "completion_tokens", "output")
        tot_tok = _pick(observation, "totalTokens", "total_tokens", "total")

    if in_tok is None and out_tok is None:
        for key in ("usageDetails", "usage_details"):
            details: Any = observation.get(key)
            if isinstance(details, Mapping) and details:
                in_tok = _pick(details, "input", "promptTokens")
                out_tok = _pick(details, "output", "completionTokens")
                break

    input_tokens: int = int(in_tok or 0)
    output_tokens: int = int(out_tok or 0)
    total_tokens: int = int(tot_tok or 0) or (input_tokens + output_tokens)
    return input_tokens, output_tokens, total_tokens


def _pick(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value: Any = mapping.get(key)
        if value is not None:
            return value
    return None


def extract_latency_ms(observation: Mapping[str, Any]) -> Optional[float]:
    """取单次调用延迟（毫秒）。``latency`` 单位是秒。"""
    latency: Any = observation.get("latency")
    if isinstance(latency, (int, float)):
        return float(latency) * 1000.0
    start: Any = _pick(observation, "startTime", "start_time")
    end: Any = _pick(observation, "endTime", "end_time")
    if isinstance(start, str) and isinstance(end, str):
        try:
            start_dt: datetime = datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt: datetime = datetime.fromisoformat(end.replace("Z", "+00:00"))
            return (end_dt - start_dt).total_seconds() * 1000.0
        except ValueError:
            return None
    return None


def extract_model(observation: Mapping[str, Any]) -> str:
    return str(_pick(observation, "model", "modelId", "model_id") or "unknown")


def percentile(values: List[float], pct: float) -> float:
    """线性插值分位数（与 ``evals/metrics.percentile`` 同口径）。"""
    items: List[float] = sorted(float(v) for v in values)
    if not items:
        return 0.0
    if len(items) == 1:
        return items[0]
    pos: float = (len(items) - 1) * min(100.0, max(0.0, pct)) / 100.0
    lo: int = int(pos)
    hi: int = min(lo + 1, len(items) - 1)
    weight: float = pos - lo
    return items[lo] * (1.0 - weight) + items[hi] * weight


# =====================================================================
# 三、汇总
# =====================================================================
def _window(day: date) -> Tuple[str, str]:
    """把日期换成 ISO8601 时间窗（UTC），Langfuse 接口用 ``Z`` 结尾。"""
    start: datetime = datetime(
        day.year, day.month, day.day, tzinfo=timezone.utc
    )
    end: datetime = start + timedelta(days=1)
    fmt = "%Y-%m-%dT%H:%M:%S.000Z"
    return start.strftime(fmt), end.strftime(fmt)


def collect(reader: LangfuseReader, day: date) -> Dict[str, Any]:
    """拉取一天的 traces 与 generations，汇总成结构化结果。"""
    from_ts, to_ts = _window(day)

    traces: List[Dict[str, Any]] = reader.paged(
        "/api/public/traces",
        {"fromTimestamp": from_ts, "toTimestamp": to_ts},
    )

    generations: List[Dict[str, Any]] = []
    generations_error: Optional[str] = None
    try:
        generations = reader.paged(
            "/api/public/observations",
            {"type": "GENERATION", "fromStartTime": from_ts, "toStartTime": to_ts},
        )
    except SystemExit as exc:
        # 观测接口不可用（权限/版本差异）时降级为仅 trace 维度，并显式记录
        generations_error = str(exc)

    usage_by_model: Dict[str, ModelUsage] = {}
    latencies: List[float] = []
    latencies_by_name: Dict[str, List[float]] = {}
    by_name_calls: Dict[str, int] = {}

    for generation in generations:
        model: str = extract_model(generation)
        in_tok, out_tok, _ = extract_tokens(generation)
        entry: ModelUsage = usage_by_model.setdefault(model, ModelUsage(model=model))
        entry.requests += 1
        entry.input_tokens += in_tok
        entry.output_tokens += out_tok

        latency_ms: Optional[float] = extract_latency_ms(generation)
        if latency_ms is not None:
            latencies.append(latency_ms)
            name: str = str(generation.get("name") or "unnamed")
            latencies_by_name.setdefault(name, []).append(latency_ms)
        name_key: str = str(generation.get("name") or "unnamed")
        by_name_calls[name_key] = by_name_calls.get(name_key, 0) + 1

    error_traces: int = 0
    for trace in traces:
        level: str = str(trace.get("level") or "").upper()
        if level in {"ERROR", "WARNING"} or trace.get("statusMessage"):
            error_traces += 1

    return {
        "day": day.isoformat(),
        "window_utc": {"from": from_ts, "to": to_ts},
        "traces_total": len(traces),
        "traces_error": error_traces,
        "generations_total": len(generations),
        "generations_error": generations_error,
        "usage_by_model": {k: _usage_dict(v) for k, v in usage_by_model.items()},
        "latency_ms": {
            "count": len(latencies),
            "mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "max": round(max(latencies), 1) if latencies else 0.0,
        },
        "latency_by_name": {
            name: {
                "calls": len(values),
                "p50": round(percentile(values, 50), 1),
                "p95": round(percentile(values, 95), 1),
            }
            for name, values in sorted(latencies_by_name.items())
        },
        "calls_by_name": dict(sorted(by_name_calls.items())),
    }


def _usage_dict(usage: ModelUsage) -> Dict[str, Any]:
    return {
        "requests": usage.requests,
        "input": usage.input_tokens,
        "output": usage.output_tokens,
        "total": usage.total_tokens,
    }


# =====================================================================
# 四、渲染
# =====================================================================
def render_markdown(result: Mapping[str, Any], report: Any, pricing: Pricing) -> str:
    lines: List[str] = []
    add = lines.append

    add(f"# Langfuse 日报 —— {result['day']}")
    add("")
    add(f"- 统计窗口（UTC）：`{result['window_utc']['from']}` ~ `{result['window_utc']['to']}`")
    add(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 币种：{report.currency}")
    add("")

    add("## 一、总览")
    add("")
    add("| 指标 | 值 |")
    add("| --- | --- |")
    add(f"| Trace 总数（一次对话 = 一个 trace） | {result['traces_total']} |")
    add(f"| 异常 Trace（ERROR/WARNING） | {result['traces_error']} |")
    error_rate: float = (
        result["traces_error"] / result["traces_total"] * 100.0
        if result["traces_total"]
        else 0.0
    )
    add(f"| 异常率 | {error_rate:.2f}% |")
    add(f"| LLM 调用次数（GENERATION 数） | {result['generations_total']} |")
    if result["traces_total"] and result["generations_total"]:
        add(
            f"| 每轮对话平均 LLM 调用次数 | "
            f"{result['generations_total'] / result['traces_total']:.2f} |"
        )
    add(f"| 总成本 | "
        + (f"{report.total:.4f} {report.currency}" if report.total is not None
           else "**不可计算**（价格表未填）")
        + " |")
    add("")

    add("## 二、按模型（token 与成本）")
    add("")
    if not result["usage_by_model"]:
        add("本日没有 GENERATION 记录"
            + (f"（观测接口报错：{result['generations_error']}）"
               if result.get("generations_error") else "")
            + "。")
    else:
        add("| 模型 | 调用次数 | 输入 token | 输出 token | 合计 | 成本 | 状态 |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for line in report.lines:
            cost: str = (
                f"{line['cost']:.4f}" if isinstance(line["cost"], (int, float)) else "-"
            )
            add(f"| `{line['model']}` | {line['requests']} | {line['input_tokens']} "
                f"| {line['output_tokens']} | {line['total_tokens']} | {cost} "
                f"| {line['status']} |")
    add("")

    add("## 三、延迟（按调用类型）")
    add("")
    overall: Mapping[str, Any] = result["latency_ms"]
    add(f"- 全部 LLM 调用：P50 = **{overall['p50']} ms**，"
        f"P95 = **{overall['p95']} ms**，最大 = {overall['max']} ms，"
        f"样本 {overall['count']} 次")
    add("")
    if result["latency_by_name"]:
        add("| 调用类型（observation name） | 次数 | P50 (ms) | P95 (ms) |")
        add("| --- | --- | --- | --- |")
        for name, item in result["latency_by_name"].items():
            add(f"| `{name}` | {item['calls']} | {item['p50']} | {item['p95']} |")
        add("")
    add("> 按 observation name 拆开看很有用：意图改写/分类是高频短调用，"
        "Agent 步进是低频长调用，混在一起看 P95 会被长尾主导。")
    add("")

    add("## 四、数据质量说明")
    add("")
    if not pricing.prices_filled:
        add("⚠️ `monitoring/model_pricing.yaml` 的 `prices_filled=false`："
            "**价格尚未核实，上表成本不可作为结论**。"
            "请按下述命令查看实际模型名并补齐单价：")
        add("")
        add("```bash")
        add("python monitoring/langfuse_daily_report.py --list-models")
        add("```")
        add("")
    if report.warnings:
        add("其它提示：")
        add("")
        for warning in report.warnings:
            add(f"- {warning}")
        add("")
    if result.get("generations_error"):
        add(f"- ⚠️ GENERATION 观测拉取失败，token/延迟部分不可用："
            f"`{result['generations_error']}`")
        add("")
    add("- 本日报基于 Langfuse 已上报数据；若应用未配置 LANGFUSE_PUBLIC_KEY/"
        "SECRET_KEY，`@observe` 会退化为 no-op，此时**不会有任何数据**——"
        "这不是「没流量」，是「没开启观测」。")
    add("")

    return "\n".join(lines) + "\n"


# =====================================================================
# 五、CLI
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Langfuse 日报（调用量/token/成本/延迟）")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="统计日期 YYYY-MM-DD（默认今天；窗口按 UTC 切）",
    )
    parser.add_argument("--out", type=Path, default=None, help="Markdown 输出路径")
    parser.add_argument("--json", action="store_true", help="输出 JSON（不写 Markdown）")
    parser.add_argument("--list-models", action="store_true",
                        help="只列出当日出现过的模型名与调用数（填价格表前必跑）")
    parser.add_argument("--pricing", type=Path, default=None, help="价格表路径")
    parser.add_argument("--host", default=None, help="覆盖 Langfuse host")
    args = parser.parse_args()

    try:
        day: date = date.fromisoformat(args.date)
    except ValueError:
        raise SystemExit(f"--date 格式应为 YYYY-MM-DD，收到：{args.date}")

    pricing: Pricing = load_pricing(args.pricing) if args.pricing else load_pricing()

    public_key, secret_key, host = resolve_credentials()
    if args.host:
        host = args.host
    reader: LangfuseReader = LangfuseReader(public_key, secret_key, host)

    if args.list_models:
        result_probe: Dict[str, Any] = collect(reader, day)
        usage_map: Dict[str, ModelUsage] = normalize_usage(
            {"by_model": result_probe["usage_by_model"]}
        )
        print(f"# {day.isoformat()} 出现过的模型（共 {len(usage_map)} 个）")
        print()
        print(f"{'模型名':<32} {'调用':>6} {'输入tok':>10} {'输出tok':>10}  价格表")
        print("-" * 78)
        for model, usage in sorted(usage_map.items()):
            price = pricing.get(model)
            state: str = "✅ 已定价" if (price and price.is_priced) else (
                "⚠️ 未定价" if price else "❌ 不在价格表"
            )
            print(f"{model:<32} {usage.requests:>6} {usage.input_tokens:>10} "
                  f"{usage.output_tokens:>10}  {state}")
        print()
        print("→ 把 ❌ 的模型补进 monitoring/model_pricing.yaml 的 models: 段，"
              "再把 ⚠️ 的单价填上。")
        return

    result: Dict[str, Any] = collect(reader, day)
    usage: Dict[str, ModelUsage] = normalize_usage({"by_model": result["usage_by_model"]})
    cost_report: Any = compute_cost(usage, pricing)

    if args.json:
        print(json.dumps(
            {**result, "cost": cost_report.as_dict()}, ensure_ascii=False, indent=2
        ))
        return

    markdown: str = render_markdown(result, cost_report, pricing)
    out_path: Path = args.out or (_MONITORING_DIR / f"langfuse_daily_{day.isoformat()}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    print(f"[langfuse] 日报已写出：{out_path}")
    print(f"[langfuse] trace={result['traces_total']} "
          f"异常={result['traces_error']} "
          f"LLM调用={result['generations_total']} "
          f"P95={result['latency_ms']['p95']}ms")
    print(f"[langfuse] 成本："
          + (f"{cost_report.total:.4f} {cost_report.currency}"
             if cost_report.total is not None else "不可计算（价格表未填）"))
    if cost_report.unpriced or cost_report.unknown:
        print(f"[langfuse] ⚠️ 未定价 {len(cost_report.unpriced)} 个 / "
              f"未知 {len(cost_report.unknown)} 个模型，成本不完整")


if __name__ == "__main__":
    main()
