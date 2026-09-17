# -*- coding: utf-8 -*-
"""黄金集与结果的读写工具（薄封装，保持无副作用）。

放在独立模块而不是塞进 ``metrics.py``：metrics 必须保持**纯函数**（可单测、
可解释），任何文件 IO 都会破坏这个性质。runner / report / 质量门三方共用本模块，
避免各自实现一份路径拼接。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

EVALS_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = EVALS_DIR.parent
GOLDEN_DIR: Path = EVALS_DIR / "golden"
RESULTS_DIR: Path = EVALS_DIR / "_results"
LATEST_RESULTS_PATH: Path = RESULTS_DIR / "latest.json"
THRESHOLDS_PATH: Path = EVALS_DIR / "thresholds.yaml"


# ---------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """按行读取 JSONL；自动跳过空行与 ``#`` 注释行。"""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            text: str = raw_line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError as exc:  # pragma: no cover - 数据错误需显式暴露
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
    return records


def dump_jsonl(path: Path, records: Iterator[Dict[str, Any]] | List[Dict[str, Any]]) -> int:
    """写出 JSONL（UTF-8、不转义中文），返回写入行数。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count: int = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------
# 黄金集
# ---------------------------------------------------------------------
def load_intent_cases() -> List[Dict[str, Any]]:
    return load_jsonl(GOLDEN_DIR / "intent_cases.jsonl")


def load_rag_cases() -> List[Dict[str, Any]]:
    return load_jsonl(GOLDEN_DIR / "rag_cases.jsonl")


def load_tool_cases() -> List[Dict[str, Any]]:
    return load_jsonl(GOLDEN_DIR / "tool_cases.jsonl")


def load_all_cases() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "intent": load_intent_cases(),
        "rag": load_rag_cases(),
        "tool": load_tool_cases(),
    }


# ---------------------------------------------------------------------
# 阈值
# ---------------------------------------------------------------------
def load_thresholds(path: Optional[Path] = None) -> Dict[str, Any]:
    """读取 thresholds.yaml。缺 PyYAML 时给出明确安装提示。"""
    try:
        import yaml  # 局部导入：保证 metrics 单测不需要 yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "读取阈值需要 PyYAML：pip install pyyaml"
        ) from exc
    target: Path = Path(path) if path else THRESHOLDS_PATH
    with target.open("r", encoding="utf-8") as handle:
        data: Any = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{target} 顶层必须是映射（dict），实际是 {type(data)}")
    return data


# ---------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------
def save_results(results: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """保存评测结果 JSON；默认写 ``evals/_results/latest.json``。"""
    target: Path = Path(path) if path else LATEST_RESULTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    return target


def load_results(path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """读取评测结果；文件不存在返回 ``None``（质量门据此决定是否 skip）。"""
    target: Path = Path(path) if path else LATEST_RESULTS_PATH
    if not target.exists():
        return None
    with target.open("r", encoding="utf-8") as handle:
        data: Any = json.load(handle)
    return data if isinstance(data, dict) else None
