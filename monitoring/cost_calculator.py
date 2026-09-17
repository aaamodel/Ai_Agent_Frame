# -*- coding: utf-8 -*-
"""成本核算（纯函数 + CLI）：把 token 用量按 ``model_pricing.yaml`` 换算成金额。

## 设计原则

1. **纯函数**：核心计算只做「用量 + 单价 → 金额」，可被单测覆盖、可逐行解释。
2. **不编数字**：价格没填（``null``）时**不返回 0**，而是把该模型计入
   ``unpriced`` 列表并报警告。0 会被误读成"免费"，是最危险的那种错。
3. **输入输出分开计价**：输出 token 通常贵 2~4 倍，合并计价会系统性低估。
4. **未知模型不猜价**：按配置的 ``fallback.on_unknown_model`` 处理，
   默认 ``warn_and_skip``（只列出来、不算进总额）。

## 用法

    # 用 benchmark 产出的 token 统计算成本
    python monitoring/cost_calculator.py --usage benchmark/_results/token_usage.json

    # 手工指定用量
    python monitoring/cost_calculator.py \
        --model qwen3.7-flash --input-tokens 120000 --output-tokens 40000

    # 输出 JSON（供脚本消费）
    python monitoring/cost_calculator.py --usage ... --json

    # 只看价格表状态（不计算）
    python monitoring/cost_calculator.py --check-pricing
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
PRICING_PATH: Path = Path(__file__).resolve().parent / "model_pricing.yaml"


# =====================================================================
# 一、价格表
# =====================================================================
@dataclass
class ModelPrice:
    """单个模型的价格（每 1000 token）。"""

    name: str
    input_per_1k: Optional[float] = None
    output_per_1k: Optional[float] = None
    tier: Optional[str] = None
    source_url: Optional[str] = None
    price_effective_date: Optional[str] = None

    @property
    def is_priced(self) -> bool:
        """是否两个单价都已填（缺任一即视为未定价）。"""
        return self.input_per_1k is not None and self.output_per_1k is not None


@dataclass
class Pricing:
    """整张价格表。"""

    currency: str = "CNY"
    unit: str = "per_1k_tokens"
    prices_filled: bool = False
    last_verified: Optional[str] = None
    models: Dict[str, ModelPrice] = field(default_factory=dict)
    on_unknown_model: str = "warn_and_skip"
    tier_defaults: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    def get(self, model: str) -> Optional[ModelPrice]:
        return self.models.get(model)


def load_pricing(path: Optional[Path] = None) -> Pricing:
    """读取 ``model_pricing.yaml``。

    缺 PyYAML 时给出明确的安装提示（不静默降级）。
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("读取价格表需要 PyYAML：pip install pyyaml") from exc

    target: Path = Path(path) if path else PRICING_PATH
    with target.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{target} 顶层必须是映射")

    models: Dict[str, ModelPrice] = {}
    for name, spec in (raw.get("models") or {}).items():
        if not isinstance(spec, dict):
            continue
        models[str(name)] = ModelPrice(
            name=str(name),
            input_per_1k=_opt_float(spec.get("input_per_1k")),
            output_per_1k=_opt_float(spec.get("output_per_1k")),
            tier=spec.get("tier"),
            source_url=spec.get("source_url"),
            price_effective_date=spec.get("price_effective_date"),
        )

    fallback: Any = raw.get("fallback") or {}
    return Pricing(
        currency=str(raw.get("currency") or "CNY"),
        unit=str(raw.get("unit") or "per_1k_tokens"),
        prices_filled=bool(raw.get("prices_filled")),
        last_verified=raw.get("last_verified"),
        models=models,
        on_unknown_model=str(fallback.get("on_unknown_model") or "warn_and_skip"),
        tier_defaults=dict(fallback.get("tier_defaults") or {}),
    )


def _opt_float(value: Any) -> Optional[float]:
    """把 YAML 值转成 float；``None``/空串保持 ``None``（区分"未填"与"0"）。"""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# =====================================================================
# 二、核心计算
# =====================================================================
def cost_of(
    input_tokens: int,
    output_tokens: int,
    price: ModelPrice,
) -> Optional[float]:
    """按输入/输出分开计价算金额。

    公式：``input/1000*price_in + output/1000*price_out``

    返回 ``None`` 表示**价格未填**（而不是 0）——调用方必须显式处理这种情况。
    """
    if not price.is_priced:
        return None
    safe_in: float = max(0.0, float(input_tokens or 0))
    safe_out: float = max(0.0, float(output_tokens or 0))
    return (safe_in / 1000.0) * float(price.input_per_1k or 0.0) + (
        safe_out / 1000.0
    ) * float(price.output_per_1k or 0.0)


@dataclass
class ModelUsage:
    """一个模型的用量。"""

    model: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def normalize_usage(raw: Any) -> Dict[str, ModelUsage]:
    """把多种输入形状归一化成 ``{model: ModelUsage}``。

    支持的形状（尽量宽松，避免"算不出成本"是因为格式对不上）：
        1. ``{"by_model": {"m": {"input": 1, "output": 2, "requests": 3}}}``
           —— ``benchmark/_results/token_usage.json`` 的形状
        2. ``{"m": {"input": 1, "output": 2}}``  —— 直接给 by_model 内容
        3. ``[{"model": "m", "input": 1, "output": 2}, ...]`` —— 列表形式
    字段别名：``input`` / ``input_tokens`` / ``prompt_tokens``；
              ``output`` / ``output_tokens`` / ``completion_tokens``。
    """
    rows: Any
    if isinstance(raw, Mapping) and "by_model" in raw:
        rows = raw.get("by_model")
    elif isinstance(raw, Mapping):
        rows = raw
    elif isinstance(raw, list):
        rows = raw
    else:
        return {}

    result: Dict[str, ModelUsage] = {}

    if isinstance(rows, Mapping):
        for model, spec in rows.items():
            if not isinstance(spec, Mapping):
                continue
            result[str(model)] = ModelUsage(
                model=str(model),
                requests=int(spec.get("requests") or spec.get("calls") or 0),
                input_tokens=int(_first(spec, "input", "input_tokens", "prompt_tokens") or 0),
                output_tokens=int(_first(spec, "output", "output_tokens", "completion_tokens") or 0),
            )
    elif isinstance(rows, list):
        for spec in rows:
            if not isinstance(spec, Mapping):
                continue
            model: str = str(spec.get("model") or spec.get("model_id") or "unknown")
            entry: ModelUsage = result.setdefault(model, ModelUsage(model=model))
            entry.requests += int(spec.get("requests") or 1)
            entry.input_tokens += int(
                _first(spec, "input", "input_tokens", "prompt_tokens") or 0
            )
            entry.output_tokens += int(
                _first(spec, "output", "output_tokens", "completion_tokens") or 0
            )
    return result


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


@dataclass
class CostReport:
    """成本核算结果。"""

    currency: str
    total: Optional[float]
    lines: List[Dict[str, Any]] = field(default_factory=list)
    unpriced: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    pricing_filled: bool = False
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "currency": self.currency,
            "total": self.total,
            "total_available": self.total is not None,
            "lines": self.lines,
            "unpriced_models": self.unpriced,
            "unknown_models": self.unknown,
            "pricing_filled": self.pricing_filled,
            "warnings": self.warnings,
        }


def compute_cost(usage: Mapping[str, ModelUsage], pricing: Pricing) -> CostReport:
    """按价格表核算 ``{model: ModelUsage}`` 的总成本。

    关键行为：
        - 某模型**在价格表里但价格未填** → 计入 ``unpriced``，**不计入总额**
          （总额仍给出已定价部分的合计，并附带 warning）；
        - 某模型**不在价格表里** → 计入 ``unknown``，**不计入总额**。
          刻意不猜价：既不知道它的 tier，也不知道它是不是贵一个量级的推理模型，
          猜一个价会让总额"看起来完整但实际是错的"，比缺数字危险。
        - ``pricing_filled=false`` → 顶部加一条 warning，明确"这份成本不完整"。
    """
    warnings: List[str] = []
    lines: List[Dict[str, Any]] = []
    unpriced: List[str] = []
    unknown: List[str] = []
    total: float = 0.0
    any_priced: bool = False

    if not pricing.prices_filled:
        warnings.append(
            "model_pricing.yaml 的 prices_filled=false：价格表尚未核实，"
            "下表金额只能视为**结构性示例**，不能作为成本结论。"
        )
    if pricing.on_unknown_model != "warn_and_skip":
        warnings.append(
            f"fallback.on_unknown_model={pricing.on_unknown_model!r} 不是受支持的值"
            "（当前实现只支持 warn_and_skip），已按 warn_and_skip 处理。"
        )

    for model, item in sorted(usage.items()):
        price: Optional[ModelPrice] = pricing.get(model)
        amount: Optional[float] = None

        if price is None:
            # 未知模型**刻意不猜价**：既不知道它的 tier，也不知道它是不是贵 10 倍的
            # 推理模型。猜一个价会让总额看起来完整但实际错误——比缺数字更危险。
            unknown.append(model)
            lines.append(
                {
                    "model": model,
                    "tier": None,
                    "requests": item.requests,
                    "input_tokens": item.input_tokens,
                    "output_tokens": item.output_tokens,
                    "total_tokens": item.total_tokens,
                    "cost": None,
                    "status": "unknown_model",
                }
            )
            continue

        amount = cost_of(item.input_tokens, item.output_tokens, price)
        status: str = "ok" if amount is not None else "unpriced"
        if amount is None:
            unpriced.append(model)
            warnings.append(
                f"模型 {model!r} 在价格表里，但 input/output 单价未填（null），"
                f"无法计价——请填 pricing.yaml（来源：{price.source_url or '未填写'}）。"
            )
        else:
            total += amount
            any_priced = True

        lines.append(
            {
                "model": model,
                "tier": price.tier,
                "requests": item.requests,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "total_tokens": item.total_tokens,
                "cost": amount,
                "status": status,
            }
        )

    return CostReport(
        currency=pricing.currency,
        total=total if any_priced else None,
        lines=lines,
        unpriced=unpriced,
        unknown=unknown,
        pricing_filled=pricing.prices_filled,
        warnings=warnings,
    )


# =====================================================================
# 三、CLI
# =====================================================================
def _print_report(report: CostReport) -> None:
    print(f"币种：{report.currency}")
    if report.total is None:
        print("总成本：**不可计算**（没有任何模型填了价格）")
    else:
        print(f"总成本：{report.total:.4f} {report.currency}")
        if report.unpriced or report.unknown:
            print(
                f"  ⚠️ 该总额**不包含** {len(report.unpriced)} 个未定价模型"
                f"和 {len(report.unknown)} 个未知模型"
            )
    print()
    header = f"{'模型':<26} {'请求':>8} {'输入tok':>10} {'输出tok':>10} {'成本':>12}  状态"
    print(header)
    print("-" * len(header))
    for line in report.lines:
        amount: str = (
            f"{line['cost']:.4f}" if isinstance(line["cost"], (int, float)) else "-"
        )
        print(
            f"{line['model']:<26} {line['requests']:>8} {line['input_tokens']:>10} "
            f"{line['output_tokens']:>10} {amount:>12}  {line['status']}"
        )
    if report.warnings:
        print()
        for warning in report.warnings:
            print(f"⚠️ {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="按 model_pricing.yaml 核算 token 成本")
    parser.add_argument("--usage", type=Path, default=None,
                        help="用量 JSON（支持 benchmark/_results/token_usage.json 形状）")
    parser.add_argument("--model", default=None, help="手工指定单个模型名")
    parser.add_argument("--input-tokens", type=int, default=0)
    parser.add_argument("--output-tokens", type=int, default=0)
    parser.add_argument("--pricing", type=Path, default=PRICING_PATH)
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--check-pricing", action="store_true",
                        help="只检查价格表填写状态")
    args = parser.parse_args()

    pricing: Pricing = load_pricing(args.pricing)

    if args.check_pricing:
        print(f"价格表：{args.pricing}")
        print(f"  currency      = {pricing.currency}")
        print(f"  prices_filled = {pricing.prices_filled}")
        print(f"  last_verified = {pricing.last_verified}")
        print(f"  覆盖模型 {len(pricing.models)} 个：")
        for name, price in sorted(pricing.models.items()):
            mark: str = "✅ 已定价" if price.is_priced else "❌ 未定价"
            print(f"    - {name:<26} tier={price.tier or '-':<10} {mark}")
        if not pricing.prices_filled:
            print()
            print("⚠️ 价格尚未核实。请按 YAML 顶部说明填写，"
                  "否则任何成本数字都不可辩护。")
        return

    # 组装用量
    if args.usage:
        with args.usage.open("r", encoding="utf-8") as handle:
            raw: Any = json.load(handle)
        usage: Dict[str, ModelUsage] = normalize_usage(raw)
        if not usage:
            raise SystemExit(
                f"从 {args.usage} 里没解析出任何模型用量。"
                "期望形状见 cost_calculator.normalize_usage 的 docstring。"
            )
        # benchmark 的形状里 by_model 不一定是唯一来源，若有顶层累计则补一条
        top_in: Any = raw.get("prompt_tokens") if isinstance(raw, Mapping) else None
        top_out: Any = raw.get("completion_tokens") if isinstance(raw, Mapping) else None
        if top_in or top_out:
            usage.setdefault(
                "<!--UNATTRIBUTED-->",
                ModelUsage(
                    model="<!--UNATTRIBUTED-->",
                    input_tokens=int(top_in or 0),
                    output_tokens=int(top_out or 0),
                ),
            )
    elif args.model:
        usage = {
            args.model: ModelUsage(
                model=args.model,
                requests=1,
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
            )
        }
    else:
        parser.error("需要 --usage 或 --model 之一（或 --check-pricing）")
        return

    report: CostReport = compute_cost(usage, pricing)
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        _print_report(report)

    # 未定价时用非零退出码，方便 CI / 脚本发现"成本算不出来"
    if report.unpriced or report.unknown:
        sys.exit(2)


if __name__ == "__main__":
    main()
