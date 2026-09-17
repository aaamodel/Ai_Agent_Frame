# -*- coding: utf-8 -*-
"""熔断切换实测：把"检测到故障 → 完成切换"这句话变成数字。

## 为什么不用 HTTP 压测来测这个

HTTP 层看到的只是"这个请求慢了"，看不出慢在等超时还是慢在切换。
而且真实压测里你没法让主模型"恰好挂掉"。

所以本脚本**在进程内直接驱动真实 ModelRouter**，用故障注入让指定候选一定失败，
再测三类时间：

=========================================  ==========================================
指标                                        含义 / 为什么重要
=========================================  ==========================================
``t_first_fallback_ms``                     第 1 次请求：发起 → 拿到降级模型的成功响应。
                                            这是**用户实际多等的时间**（含等待超时）。
``t_second_fallback_ms``                    第 2 次请求：同样降级，但熔断计数在本轮达阈值
                                            （``selection.failure_threshold=2``）。
``t_after_open_ms``                         第 3 次请求：熔断已打开，坏模型被直接跳过，
                                            应该**显著快于**前两次。这是"熔断真的生效了"
                                            的证据——否则它只是个装饰。
``circuit_opened``                          熔断后 ``is_unavailable(primary)`` 是否为 True。
=========================================  ==========================================

## 故障注入方式

patch ``app.llm_model_router.async_openai_caller._chat_completion_with_langfuse``：
该函数是所有真实 LLM 调用的唯一出口，且被同模块以**运行时名称查找**方式调用
（见 ``async_openai_chat_caller`` 里 ``await _chat_completion_with_langfuse(...)``），
所以替换模块属性即可精确注入。用完在 finally 里还原。

- ``--fault instant``：立刻抛 ``APITimeoutError`` → 模拟"连接被拒/快速失败"
- ``--fault hang``（默认）：挂起到 tier 超时 → 模拟"主模型不响应"，
  这正是简历里"手动让主模型超时"的做法

## 用法

    # 默认：hang 故障，跑 2 轮
    python benchmark/test_circuit_breaker.py

    # 快一点：只跑 1 轮
    python benchmark/test_circuit_breaker.py --repeat 1

    # 立刻失败（不等待超时）——用来对比"快速失败"和"等超时"的差距
    python benchmark/test_circuit_breaker.py --fault instant

    # 先看配置，不实际调用（不花钱）
    python benchmark/test_circuit_breaker.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

RESULTS_DIR: Path = Path(__file__).resolve().parent / "_results"
REPORT_PATH: Path = Path(__file__).resolve().parent / "熔断切换实测.md"

# 用 FAST tier（react 场景，15s 超时）：Agent 的主链路走这一档，
# 也是超时最短的一档，测出来的切换耗时最贴近用户真实体感。
TEST_PURPOSE: str = "react"


def _test_tier() -> Any:
    """按 PURPOSE_TIER_MAP 解析出被测 tier，避免与真实调用用错档位。

    ⚠️ 这里必须与 ``router.chat(purpose_hint=TEST_PURPOSE)`` 解析出的 tier 一致。
    否则会出现"选主候选用 STANDARD、实际调用走 FAST"的错位，
    导致注入的故障模型根本不是这次请求的主候选。
    """
    from app.llm_model_router.model_router import PURPOSE_TIER_MAP

    return PURPOSE_TIER_MAP[TEST_PURPOSE]


def _tier_name() -> str:
    tier: Any = _test_tier()
    return str(getattr(tier, "value", tier))


def _tier_timeout_ms(properties: Any) -> Optional[int]:
    """从配置里取目标 tier 的超时（毫秒），拿不到返回 None。"""
    try:
        tiers: Dict[str, Any] = dict(getattr(properties.chat, "tiers", None) or {})
        tier_cfg: Any = tiers.get(_tier_name())
        if tier_cfg is None:
            return None
        timeout_ms: Any = getattr(tier_cfg, "timeout_ms", None)
        return int(timeout_ms) if timeout_ms else None
    except Exception:  # noqa: BLE001
        return None


def _tier_candidate_ids(properties: Any) -> List[str]:
    try:
        tiers: Dict[str, Any] = dict(getattr(properties.chat, "tiers", None) or {})
        tier_cfg: Any = tiers.get(_tier_name())
        return [str(c) for c in (getattr(tier_cfg, "candidates", None) or [])]
    except Exception:  # noqa: BLE001
        return []


class FaultInjector:
    """让指定候选的 LLM 调用按指定模式失败。"""

    def __init__(self, bad_identifiers: set, mode: str, hang_seconds: float) -> None:
        self.bad = {str(x) for x in bad_identifiers if x}
        self.mode = mode
        self.hang_seconds = hang_seconds
        self.injected_calls: int = 0
        self._original: Any = None

    def install(self) -> None:
        from app.llm_model_router import async_openai_caller as caller_module

        self._original = caller_module._chat_completion_with_langfuse
        original = self._original

        async def _injected(  # noqa: ANN001
            client: Any, params: Dict[str, Any], *, model: str, provider: str
        ) -> Any:
            # 命中判定同时看「候选 id」（kwargs model）和「真实模型名」（params["model"]），
            # 两者任一命中即注入故障，避免因配置命名差异而注入失败。
            target_names = {str(model), str(params.get("model") or "")}
            if target_names & self.bad:
                self.injected_calls += 1
                if self.mode == "hang":
                    await asyncio.sleep(self.hang_seconds)
                import httpx
                from openai import APITimeoutError

                raise APITimeoutError(request=httpx.Request("POST", "http://injected.invalid/v1"))
            return await original(client, params, model=model, provider=provider)

        caller_module._chat_completion_with_langfuse = _injected

    def uninstall(self) -> None:
        if self._original is None:
            return
        from app.llm_model_router import async_openai_caller as caller_module

        caller_module._chat_completion_with_langfuse = self._original
        self._original = None


async def _one_request(router: Any) -> Tuple[float, str]:
    """发一次 chat 请求，返回 (耗时毫秒, 命中的 model_id)。"""
    started: float = time.perf_counter()
    response: Any = await router.chat(
        [{"role": "user", "content": "熔断切换测速：请回复 ok"}],
        purpose_hint=TEST_PURPOSE,
        temperature=0.0,
        max_tokens=8,
    )
    elapsed_ms: float = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, str(getattr(response, "model_id", "") or "")


async def run_cycle(
    *,
    fault_mode: str,
    fault_hang_seconds: float,
    health_wait_s: float = 0.0,
) -> Dict[str, Any]:
    """跑一轮「故障 → 降级 → 熔断开 → 跳过」三请求序列。"""
    from app.main import _build_global_router

    router: Any = _build_global_router()
    properties: Any = getattr(router, "properties", None)
    tier_timeout_ms: Optional[int] = _tier_timeout_ms(properties) if properties else None
    configured_ids: List[str] = _tier_candidate_ids(properties) if properties else []

    # 候选顺序由 Selector 决定（含熔断过滤）；必须用与本轮调用相同的 tier 取，
    # 否则会挑错主候选、注入到不该注入的模型上。
    targets: List[Any] = await router._selector.select_chat_candidates(
        thinking=False, override=_test_tier(), preferred_model_id=None
    )
    if len(targets) < 2:
        raise RuntimeError(
            f"当前 tier({_tier_name()}) 可用候选只有 {len(targets)} 个，无法测降级切换。"
            f"请在配置里为该档配置至少 2 个候选（配置里的 candidates={configured_ids}）。"
        )

    primary: Any = targets[0]
    secondary: Any = targets[1]
    primary_ids: set = {primary.id, getattr(primary.candidate, "model", None)}

    injector: FaultInjector = FaultInjector(
        bad_identifiers=primary_ids,
        mode=fault_mode,
        hang_seconds=fault_hang_seconds,
    )
    injector.install()
    try:
        t1_ms, m1 = await _one_request(router)
        t2_ms, m2 = await _one_request(router)
        t3_ms, m3 = await _one_request(router)
    finally:
        injector.uninstall()

    if health_wait_s:
        await asyncio.sleep(health_wait_s)
    circuit_opened: bool = bool(await router._health_store.is_unavailable(primary.id))

    return {
        "primary_id": primary.id,
        "primary_model": str(getattr(primary.candidate, "model", "") or ""),
        "secondary_id": secondary.id,
        "secondary_model": str(getattr(secondary.candidate, "model", "") or ""),
        "tier_timeout_ms": tier_timeout_ms,
        "fault_mode": fault_mode,
        "injected_calls": injector.injected_calls,
        "t_first_fallback_ms": round(t1_ms, 1),
        "t_second_fallback_ms": round(t2_ms, 1),
        "t_after_open_ms": round(t3_ms, 1),
        "model_first": m1,
        "model_second": m2,
        "model_third": m3,
        "switched": bool(m1 and m1 != primary.id),
        "skipped_after_open": bool(m3 and m3 != primary.id),
        "circuit_opened": circuit_opened,
    }


def _stat(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "min": round(min(values), 1),
        "median": round(statistics.median(values), 1),
        "max": round(max(values), 1),
    }


def render_report(cycles: List[Dict[str, Any]], *, fault_mode: str) -> str:
    first: List[float] = [c["t_first_fallback_ms"] for c in cycles]
    second: List[float] = [c["t_second_fallback_ms"] for c in cycles]
    after: List[float] = [c["t_after_open_ms"] for c in cycles]
    s1, s2, s3 = _stat(first), _stat(second), _stat(after)
    sample: Dict[str, Any] = cycles[0] if cycles else {}

    lines: List[str] = []
    add = lines.append
    add("# 熔断降级切换实测报告")
    add("")
    add(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"- 故障注入模式：`{fault_mode}`"
        + ("（挂起到 tier 超时，模拟主模型不响应）" if fault_mode == "hang"
           else "（立刻抛错，模拟连接被拒）"))
    add(f"- 被测 tier：`{_tier_name()}`（purpose=`{TEST_PURPOSE}`）"
        f"超时配置 {sample.get('tier_timeout_ms')} ms")
    add(f"- 主候选：`{sample.get('primary_id')}`（model={sample.get('primary_model')}）")
    add(f"- 降级候选：`{sample.get('secondary_id')}`"
        f"（model={sample.get('secondary_model')}）")
    add(f"- 轮次：{len(cycles)}（每轮 3 次请求：故障降级 ×2 → 熔断后跳过 ×1）")
    add("")
    add("## 一、核心数字（简历里那句「检测到故障 → 完成切换」）")
    add("")
    add("| 指标 | 最小 | 中位 | 最大 | 说明 |")
    add("| --- | --- | --- | --- | --- |")
    add(f"| 首次降级耗时 (ms) | {s1['min']} | {s1['median']} | {s1['max']} | "
        f"发起 → 拿到降级模型成功响应，含等待主模型超时 |")
    add(f"| 第二次降级耗时 (ms) | {s2['min']} | {s2['median']} | {s2['max']} | "
        f"熔断计数达阈值那一轮 |")
    add(f"| 熔断后耗时 (ms) | {s3['min']} | {s3['median']} | {s3['max']} | "
        f"坏模型被直接跳过，**应显著小于前两行** |")
    add("")
    if s1["median"] and s3["median"]:
        speedup: float = s1["median"] / s3["median"] if s3["median"] else 0.0
        add(f"→ 熔断打开后，单次请求耗时中位数从 **{s1['median']} ms** "
            f"降到 **{s3['median']} ms**（约 **{speedup:.1f}×**）。")
        add("")
        add("> 这个对比就是「熔断有意义」的证明：如果第三行和前两行一样慢，"
            "说明熔断状态没有真正参与候选筛选，降级每次都还在白等超时。")
    add("")
    add("## 二、行为正确性")
    add("")
    add("| 检查项 | 结果 |")
    add("| --- | --- |")
    add(f"| 第 1 次请求确实发生了降级 | {'✅' if all(c['switched'] for c in cycles) else '❌'} |")
    add(f"| 熔断后仍走非主候选 | "
        f"{'✅' if all(c['skipped_after_open'] for c in cycles) else '❌'} |")
    add(f"| 熔断状态已打开（is_unavailable） | "
        f"{'✅' if all(c['circuit_opened'] for c in cycles) else '❌'} |")
    add(f"| 故障注入命中次数 | {sample.get('injected_calls')} 次/轮 |")
    add("")
    add("## 三、逐轮明细")
    add("")
    add("| 轮次 | 首次降级(ms) | 第二次(ms) | 熔断后(ms) | 第1次命中模型 | 第3次命中模型 |")
    add("| --- | --- | --- | --- | --- | --- |")
    for index, cycle in enumerate(cycles, start=1):
        add(f"| {index} | {cycle['t_first_fallback_ms']} | {cycle['t_second_fallback_ms']} "
            f"| {cycle['t_after_open_ms']} | `{cycle['model_first']}` "
            f"| `{cycle['model_third']}` |")
    add("")
    add("## 四、诚实声明")
    add("")
    add("- 本测试在**本机单机环境**完成，故障是**注入**的（不是真实线上故障），"
        "因此测量的是「机制正确性 + 切换开销量级」，不是线上故障率。")
    add("- 降级候选走的是**真实模型 API**，所以 `t_*` 里含一次真实模型调用的耗时；"
        "切换本身的净开销约为 `t_after_open_ms` 减去一次正常调用的耗时。")
    add("- `--fault hang` 模式下，首次降级耗时天然会接近 tier 超时值"
        "（因为要等主模型超时才判定失败）——这是设计使然，不是缺陷。"
        "要测「快速失败」场景请用 `--fault instant` 对比。")
    add("- 熔断阈值来自配置 ``selection.failure_threshold``（当前为 2），"
        "所以第 1 次请求只是降级、第 2 次才把熔断打开。")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="熔断降级切换耗时实测")
    parser.add_argument("--repeat", type=int, default=2, help="轮次（每轮 3 次请求）")
    parser.add_argument(
        "--fault",
        choices=["hang", "instant"],
        default="hang",
        help="故障模式：hang=挂起到超时（默认，模拟主模型不响应）；instant=立刻报错",
    )
    parser.add_argument(
        "--hang-seconds",
        type=float,
        default=None,
        help="hang 模式挂起时长；默认为 tier 超时 × 3",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读配置并打印候选顺序，不实际调用（不花钱）",
    )
    parser.add_argument("--out", type=Path, default=REPORT_PATH, help="报告输出路径")
    args = parser.parse_args()

    async def _dry() -> None:
        from app.main import _build_global_router

        router: Any = _build_global_router()
        properties: Any = getattr(router, "properties", None)
        print(f"[circuit] tier={_tier_name()} "
              f"timeout={_tier_timeout_ms(properties)}ms "
              f"configured_candidates={_tier_candidate_ids(properties)}")
        targets: List[Any] = await router._selector.select_chat_candidates(
            thinking=False, override=_test_tier(), preferred_model_id=None
        )
        for index, target in enumerate(targets, start=1):
            print(f"[circuit]   #{index} id={target.id} "
                  f"model={getattr(target.candidate, 'model', None)} "
                  f"provider={getattr(target.candidate, 'provider', None)}")
        print(f"[circuit] 可用候选 {len(targets)} 个"
              + ("（✅ 可测降级）" if len(targets) >= 2 else "（❌ 不足 2 个，无法测降级）"))

    if args.dry_run:
        asyncio.run(_dry())
        return

    async def _get_timeout() -> Optional[int]:
        from app.main import _build_global_router

        properties: Any = getattr(_build_global_router(), "properties", None)
        return _tier_timeout_ms(properties) if properties else None

    tier_timeout_ms: Optional[int] = asyncio.run(_get_timeout())
    hang_seconds: float = (
        args.hang_seconds
        if args.hang_seconds is not None
        else (max(1.0, (tier_timeout_ms or 15000) / 1000.0 * 3.0))
    )

    print(
        f"[circuit] 模式={args.fault} tier={_tier_name()} "
        f"timeout={tier_timeout_ms}ms 轮次={args.repeat}"
        + (f" 挂起={hang_seconds:.0f}s（每轮约需 {hang_seconds * 2:.0f}s）"
           if args.fault == "hang" else "")
    )

    cycles: List[Dict[str, Any]] = []
    for index in range(1, args.repeat + 1):
        print(f"[circuit] 第 {index}/{args.repeat} 轮开始…", flush=True)
        try:
            cycle: Dict[str, Any] = asyncio.run(
                run_cycle(fault_mode=args.fault, fault_hang_seconds=hang_seconds)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[circuit] 第 {index} 轮失败：{type(exc).__name__}: {exc}")
            if not cycles:
                raise SystemExit(
                    "一轮都没跑通。请先确认模型 API Key 可用，"
                    "并用 --dry-run 检查候选数量。"
                ) from exc
            break
        cycles.append(cycle)
        print(
            f"[circuit]   首次降级={cycle['t_first_fallback_ms']}ms "
            f"第二次={cycle['t_second_fallback_ms']}ms "
            f"熔断后={cycle['t_after_open_ms']}ms "
            f"熔断状态={'OPEN' if cycle['circuit_opened'] else 'CLOSED'}",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "circuit_breaker.json").write_text(
        json.dumps({"cycles": cycles, "fault_mode": args.fault}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown: str = render_report(cycles, fault_mode=args.fault)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")

    first: List[float] = [c["t_first_fallback_ms"] for c in cycles]
    after: List[float] = [c["t_after_open_ms"] for c in cycles]
    print(f"\n[circuit] 报告已写出：{args.out}")
    print(
        f"[circuit] 首次降级中位数 {statistics.median(first):.0f}ms / "
        f"熔断后中位数 {statistics.median(after):.0f}ms"
    )


if __name__ == "__main__":
    main()
