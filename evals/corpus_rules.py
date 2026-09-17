# -*- coding: utf-8 -*-
"""语料收录规则（单一真源，供 ``reingest_corpus.py`` 与 ``dump_corpus.py`` 共用）。

## 为什么需要"排除规则"

``app/rag_data`` 不只是"知识答案语料"的家，还可能混入**评测资产**。最典型的
反例是 ``app/rag_data/other/销售助手评测问题集.md``：它是一份带「期望要点」
列的问题集，一旦被递归灌进检索库，黄金集里的问题就会**直接命中这份问题集
本身** —— 检索返回的不是知识，而是"题目 + 答案要点"。

后果是 **Recall@5 虚高**，而且虚高得看不出来：分数很漂亮，但测的不是检索能力。
这与 D1（黄金集抄 examples）是同一类病：**评测集与被测对象之间出现了不该有的
信息通路**。

## 规则

1. ``EXCLUDED_DIRS``：整目录排除。``other/`` 在本项目里放的是评测参考资产
   （问题集、提示词集等），不是答案语料；
2. ``EXCLUDED_NAME_PATTERNS``：按文件名模糊排除，覆盖未来新增的评测集文件。

两条规则都只作用于**收录阶段**；原始文件保持原样不动，随时可用
``--include-excluded`` 让收录行为回到"什么都收"的旧口径做对照实验。
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Tuple

__all__ = [
    "EXCLUDED_DIRS",
    "EXCLUDED_NAME_PATTERNS",
    "is_excluded",
    "exclusion_reason",
    "resolve_rag_data_dir",
]

# 整目录排除（相对语料根目录的子目录名，大小写不敏感）
EXCLUDED_DIRS: frozenset[str] = frozenset({"other"})

# 文件名模糊排除（fnmatch 语法，大小写不敏感）
EXCLUDED_NAME_PATTERNS: Tuple[str, ...] = (
    "*评测问题集*",
    "*评测集*",
    "*golden*",
    "*_cases.jsonl",
)


def is_excluded(path: Path, src_dir: Path) -> bool:
    """判断 ``path`` 是否应被排除在语料收录之外。

    Args:
        path: 候选语料文件。
        src_dir: 语料根目录（用于把绝对路径折算成相对子目录判断）。

    Returns:
        True 表示应排除。
    """
    return exclusion_reason(path, src_dir) is not None


def resolve_rag_data_dir(repo_root: Path) -> Path:
    """解析销售语料根目录（单一真源，供入库/导出/体检脚本共用）。

    语料历史上在 ``app/rag_data``，后移到仓库根 ``rag_data``。路径写死成旧位置会
    **静默地收到 0 个文件**：``reingest_corpus`` 打印"文件=0 总片数=0"、Milvus 里
    一行向量都没有，最终表现为「RAG 评测 Recall 恒为 0」——排查成本极高。
    因此这里两个位置都探，优先新位置。

    Returns:
        存在的语料目录；都不存在时返回新位置（调用方自行决定报错还是空跑）。
    """
    for candidate in (repo_root / "rag_data", repo_root / "app" / "rag_data"):
        if candidate.is_dir():
            return candidate
    return repo_root / "rag_data"


def exclusion_reason(path: Path, src_dir: Path) -> str | None:
    """返回排除原因文案；不排除时返回 ``None``（便于打印"为什么没收这个文件"）。"""
    try:
        relative_parts = path.resolve().relative_to(src_dir.resolve()).parts
    except ValueError:
        relative_parts = path.parts

    # 规则 1：目录名命中
    for part in relative_parts[:-1]:
        if part.casefold() in {d.casefold() for d in EXCLUDED_DIRS}:
            return f"位于排除目录「{part}/」（评测参考资产，非答案语料）"

    # 规则 2：文件名模式命中
    name_cf = path.name.casefold()
    for pattern in EXCLUDED_NAME_PATTERNS:
        if fnmatch.fnmatch(name_cf, pattern.casefold()):
            return f"文件名命中排除模式「{pattern}」（评测集类文件会被黄金集直接命中）"

    return None
