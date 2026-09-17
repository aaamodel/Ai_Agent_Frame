# -*- coding: utf-8 -*-
"""把 eval 指标推到 Langfuse，作为 score 留痕（让"评测结果"与"线上观测"在同一个面板）。

## 为什么要做这件事

eval 报告是一个静态文件，跑完就躺在那儿；Langfuse 是**随时间变化**的观测面板。
把 eval 指标作为 score 推进去，就能回答两个静态报告回答不了的问题：

    - 上周 Recall@5 是 0.86，这周 0.79 —— 是哪次改动引入的？
    - 阈值 0.8 的这条线，最近 10 次评测有没有稳定在线上？

## 实现说明

同样直接用公开 REST API ``POST /api/public/scores``（不依赖 langfuse SDK 版本）。

⚠️ **Langfuse 的 score 是追加写入，不去重**。同一个 ``--run-name`` 重复推送
会产生多条同名 score。所以：

    - 默认先 ``--dry-run`` 看要推什么；
    - 真要重复跑同一批，请用 ``--run-name`` 区分（例如带上日期）。

## 用法

    # 1) 先看要推什么（不写任何数据）
    python monitoring/push_eval_scores.py --dry-run

    # 2) 从 eval 结果推（默认读 evals/_results/latest.json）
    python monitoring/push_eval_scores.py --run-name "ci-20260913"

    # 3) 手工推单条
    python monitoring/push_eval_scores.py --metric "recall@5=0.83" --run-name manual

    # 4) 全部挂到某个 trace 上（便于在 trace 详情页直接看到本次评测结论）
    python monitoring/push_eval_scores.py --trace-id <trace_id>
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

_MONITORING_DIR: Path = Path(__file__).resolve().parent
_REPO_ROOT: Path = _MONITORING_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from monitoring.langfuse_daily_report import resolve_credentials  # noqa: E402

DEFAULT_LATEST: Path = _REPO_ROOT / "evals" / "_results" / "latest.json"
THRESHOLDS_PATH: Path = _REPO_ROOT / "evals" / "thresholds.yaml"

# score 名 -> thresholds.yaml 里的 (section, key, 方向)
# 方向 'min' 表示越大越好（低于阈值即不达标），'max' 表示越小越好。
METRIC_TO_THRESHOLD: Dict[str, Tuple[str, str, str]] = {
    "intent_accuracy": ("intent", "accuracy_min", "min"),
    "boundary_accuracy": ("intent", "boundary_accuracy_min", "min"),
    "recall@5": ("rag", "recall_at_5_min", "min"),
    "hit@5": ("rag", "hit_at_5_min", "min"),
    "mrr": ("rag", "mrr_min", "min"),
    "tool_success_rate": ("tool", "call_success_min", "min"),
    "key_arg_recall": ("tool", "key_arg_recall_min", "min"),
    "avg_tokens_per_turn": ("cost", "avg_tokens_per_turn_max", "max"),
    "p95_latency_ms": ("cost", "p95_latency_ms_max", "max"),
}

# score 名前缀：加前缀是为了在 Langfuse 面板里能把 eval 指标与其它 score 区分开
SCORE_PREFIX: str = "eval."


# =====================================================================
# 一、取指标
# =====================================================================
def load_thresholds() -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return {}
    if not THRESHOLDS_PATH.exists():
        return {}
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def collect_metrics(
    latest_path: Path,
    explicit: Optional[List[str]] = None,
) -> Dict[str, float]:
    """汇总要推送的指标：显式 ``--metric`` 优先，否则读 eval 结果文件。"""
    metrics: Dict[str, float] = {}

    if explicit:
        for item in explicit:
            if "=" not in item:
                raise SystemExit(f"--metric 需要 name=value 形式，收到：{item}")
            name, _, raw = item.partition("=")
            try:
                metrics[name.strip()] = float(raw.strip())
            except ValueError:
                raise SystemExit(f"--metric 的值必须是数字：{item}")
        return metrics

    if not latest_path.exists():
        raise SystemExit(
            f"找不到 {latest_path}。请先跑 `python -m evals.report --run-all`，"
            "或用 --metric name=value 手工指定。"
        )
    with latest_path.open("r", encoding="utf-8") as handle:
        data: Any = json.load(handle)
    if not isinstance(data, Mapping):
        raise SystemExit(f"{latest_path} 内容不是对象")

    raw_metrics: Any = data.get("metrics")
    if not isinstance(raw_metrics, Mapping):
        raise SystemExit(
            f"{latest_path} 里没有 metrics 字段（可能只跑了部分评测）。"
            "请先跑完整的 `python -m evals.report --run-all`。"
        )
    for key, value in raw_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[str(key)] = float(value)
    return metrics


def build_scores(metrics: Mapping[str, float], thresholds: Mapping[str, Any],
                 run_name: str) -> List[Dict[str, Any]]:
    """把指标转成 Langfuse score 载荷。"""
    version: Any = thresholds.get("version")
    scores: List[Dict[str, Any]] = []
    for name, value in sorted(metrics.items()):
        comment_parts: List[str] = [f"run={run_name}"]
        spec = METRIC_TO_THRESHOLD.get(name)
        if spec:
            section, key, direction = spec
            limit: Any = (thresholds.get(section) or {}).get(key)
            if isinstance(limit, (int, float)):
                passed: bool = (
                    value >= float(limit) if direction == "min" else value <= float(limit)
                )
                comment_parts.append(f"threshold={float(limit):g}")
                comment_parts.append(f"pass={str(passed).lower()}")
        if version is not None:
            comment_parts.append(f"thresholds_version={version}")

        scores.append(
            {
                "name": f"{SCORE_PREFIX}{name}",
                "value": float(value),
                "dataType": "NUMERIC",
                "comment": " | ".join(comment_parts),
            }
        )
    return scores


# =====================================================================
# 二、推送
# =====================================================================
def _client() -> Any:
    try:
        import httpx

        return httpx.Client(timeout=30.0)
    except ImportError:  # pragma: no cover
        import requests

        return requests.Session()


def push_scores(
    scores: List[Dict[str, Any]],
    *,
    public_key: str,
    secret_key: str,
    host: str,
    trace_id: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """逐条 POST /api/public/scores。返回成功/失败统计。"""
    token: str = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    headers: Dict[str, str] = {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
    }
    url: str = f"{host.rstrip('/')}/api/public/scores"

    ok: int = 0
    failures: List[Dict[str, Any]] = []
    client: Any = _client()

    for payload in scores:
        body: Dict[str, Any] = dict(payload)
        if trace_id:
            body["traceId"] = trace_id
        if dry_run:
            print(f"[dry-run] POST {url} <- {json.dumps(body, ensure_ascii=False)}")
            ok += 1
            continue
        try:
            response: Any = client.post(url, json=body, headers=headers)
            status: int = int(getattr(response, "status_code", 0))
            if 200 <= status < 300:
                ok += 1
            else:
                failures.append(
                    {
                        "name": payload["name"],
                        "status": status,
                        "body": str(getattr(response, "text", ""))[:300],
                    }
                )
        except Exception as exc:  # noqa: BLE001 - 单条失败不中断整批
            failures.append({"name": payload["name"], "status": None, "body": str(exc)})

    return {"ok": ok, "failed": len(failures), "failures": failures}


# =====================================================================
# 三、CLI
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="把 eval 指标作为 score 推到 Langfuse"
    )
    parser.add_argument("--latest", type=Path, default=DEFAULT_LATEST,
                        help="eval 结果文件（默认 evals/_results/latest.json）")
    parser.add_argument("--metric", action="append", default=None,
                        help="手工指定 name=value（可重复；给了就不读结果文件）")
    parser.add_argument("--run-name", default=None,
                        help="本次运行标识，写进 score 的 comment（建议带日期）")
    parser.add_argument("--trace-id", default=None,
                        help="把 score 挂到指定 trace 上（可选）")
    parser.add_argument("--host", default=None, help="覆盖 Langfuse host")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印要推送的内容，不写任何数据（建议先跑这个）")
    args = parser.parse_args()

    run_name: str = args.run_name or "manual"
    metrics: Dict[str, float] = collect_metrics(args.latest, args.metric)
    if not metrics:
        raise SystemExit("没有任何指标可推送。")

    thresholds: Dict[str, Any] = load_thresholds()
    scores: List[Dict[str, Any]] = build_scores(metrics, thresholds, run_name)

    print(f"[push-eval] 待推送 {len(scores)} 条 score（run={run_name}）")
    for payload in scores:
        print(f"  - {payload['name']:<32} = {payload['value']:<12g}  {payload['comment']}")
    if not thresholds:
        print("  ⚠️ 读不到 thresholds.yaml，score 里不会带阈值与通过标记")

    public_key, secret_key, host = resolve_credentials()
    if args.host:
        host = args.host
    print(f"[push-eval] 目标：{host}")

    if not args.dry_run:
        print(
            "⚠️ Langfuse 的 score 是追加写入、不去重；重复推送会产生多条同名 score。"
        )

    result: Dict[str, Any] = push_scores(
        scores,
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        trace_id=args.trace_id,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print(f"[push-eval] dry-run 完成（未写入任何数据），共 {result['ok']} 条")
        return

    print(f"[push-eval] 成功 {result['ok']} 条，失败 {result['failed']} 条")
    for failure in result["failures"]:
        print(f"  ❌ {failure['name']} -> HTTP {failure['status']}: {failure['body']}")
    if result["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
