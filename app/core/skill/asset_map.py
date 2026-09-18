# -*- coding: utf-8 -*-
"""从技能文档中**确定性**提取结构化事实（目前主要是「数据资产地图」）。

为什么要这一层（2026-09 实测教训）：

    技能文档（约 4,830 字符）被整篇交给子任务模型做自由摘要，摘要产出约 180 字，
    把文档里逐字写明的资产路径（`raw_data/sales_intel/客户线索台账.xlsx` 等）
    **丢得一个不剩**。后续重规划因此只能盲猜目录，连续两次扫 `/data` 均为空。

    根因不是模型不聪明，而是我们把"确定性事实的传递"交给了概率模型。提取器把
    表格里的路径原样搬进运行状态，模型摘要只负责语义，不再承担事实传递。

设计约束：
    - 纯字符串/正则运算，不产生任何模型调用；
    - 解析不出任何结果时返回空列表，由调用方回退既有行为，**绝不中断链路**。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# 表头里用于识别各列的关键词（命中任一即可）
_NAME_HINTS = ("资产", "名称", "文件", "表名", "数据集")
_LOCATION_HINTS = ("位置", "路径", "目录", "存放")
_USAGE_HINTS = ("用途", "工具", "调用", "说明")

# 用途列里出现的工具名形态：local_excel_read_tool / rag_knowledge_search 等
_TOOL_PATTERN = re.compile(r"[a-z][a-z0-9_]*(?:_tool|_search|_query|_export)")

# 单元格里的反引号与空白
_CELL_CLEAN = re.compile(r"[`\s]+")


@dataclass
class SkillAssetFact:
    """一条已提取的结构化事实：某个数据资产在哪、用什么工具消费。"""

    name: str
    location: str
    tool: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _split_row(line: str) -> List[str]:
    """拆一行 markdown 表格（`| a | b |` → ['a', 'b']）。"""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _clean_cell(cell: str) -> str:
    """去掉反引号与多余空白——位置列常写成 `raw_data/xxx.xlsx`。"""
    return _CELL_CLEAN.sub("", str(cell or ""))


def _pick_column(headers: List[str], hints: tuple) -> Optional[int]:
    for index, header in enumerate(headers):
        if any(hint in header for hint in hints):
            return index
    return None


def extract_asset_facts(markdown: str) -> List[SkillAssetFact]:
    """从 markdown 文本中提取数据资产事实。

    只认**规整的 markdown 表格**：表头必须能识别出「名称」与「位置」两列，
    否则该表被跳过。这样既能覆盖技能文档里的「数据资产地图」，又不会把
    任意表格误当成资产清单。

    Args:
        markdown: 技能文档（或其它工具返回）的原文。

    Returns:
        事实列表；解析失败或不含该结构时返回空列表。
    """
    if not markdown:
        return []

    lines: List[str] = str(markdown).splitlines()
    facts: List[SkillAssetFact] = []
    seen: set = set()

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("|"):
            index += 1
            continue

        # 收集一个连续的表格块
        block: List[str] = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            block.append(lines[index].strip())
            index += 1

        if len(block) < 3:
            continue

        headers = _split_row(block[0])
        separator = block[1]
        if not re.fullmatch(r"\|[\s:\-|]+\|", separator):
            continue

        name_col = _pick_column(headers, _NAME_HINTS)
        location_col = _pick_column(headers, _LOCATION_HINTS)
        usage_col = _pick_column(headers, _USAGE_HINTS)
        if name_col is None or location_col is None:
            continue

        for row_line in block[2:]:
            cells = _split_row(row_line)
            if len(cells) <= max(name_col, location_col):
                continue
            name = _clean_cell(cells[name_col])
            location = _clean_cell(cells[location_col])
            if not name or not location:
                continue
            # 只收"看起来真的像个位置"的：含路径分隔符或带扩展名
            if "/" not in location and "\\" not in location and "." not in location:
                continue
            tool = ""
            if usage_col is not None and len(cells) > usage_col:
                matched = _TOOL_PATTERN.search(cells[usage_col])
                if matched:
                    tool = matched.group(0)
            key = (name, location)
            if key in seen:
                continue
            seen.add(key)
            facts.append(SkillAssetFact(name=name, location=location, tool=tool))

    return facts


def merge_facts(
    existing: Any,
    new_facts: List[SkillAssetFact],
) -> List[Dict[str, Any]]:
    """把新提取的事实并入已有集合（按 name+location 去重）。

    Args:
        existing: 状态里已有的事实（dict 列表，可能缺失或类型异常）。
        new_facts: 本次新提取的事实。

    Returns:
        可序列化的 dict 列表。
    """
    merged: List[Dict[str, Any]] = []
    seen: set = set()

    def _add(fact: Any) -> None:
        if isinstance(fact, dict):
            name = str(fact.get("name") or "").strip()
            location = str(fact.get("location") or "").strip()
            tool = str(fact.get("tool") or "").strip()
        elif isinstance(fact, SkillAssetFact):
            name, location, tool = fact.name, fact.location, fact.tool
        else:
            return
        if not name or not location:
            return
        key = (name, location)
        if key in seen:
            return
        seen.add(key)
        merged.append({"name": name, "location": location, "tool": tool})

    if isinstance(existing, list):
        for item in existing:
            _add(item)
    for fact in new_facts:
        _add(fact)
    return merged


def render_facts_for_prompt(facts: Any, max_chars: int = 400) -> str:
    """把已提取事实渲染成给提示词用的极简文本（供后续环节"看得见"）。

    渲染成 `名称=位置` 的短行，便于模型直接照抄路径调用工具。超出长度上限时
    整段截断——宁可少给几条，也不要把提示词撑大。
    """
    if not isinstance(facts, list) or not facts:
        return ""
    lines: List[str] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        location = str(item.get("location") or "").strip()
        if not name or not location:
            continue
        lines.append(f"- {name}={location}")
    if not lines:
        return ""
    header = "已提取的数据资产（调用取数工具时 file_path 请逐字使用）：\n"
    # ⚠️ 上限是**整段**的上限（含标题），不能只对正文截断——否则实际输出会
    # 比 max_chars 多出标题那一截。按整行取舍，不切半行（半行路径反而有害）。
    budget = max_chars - len(header)
    kept: List[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > budget:
            break
        kept.append(line)
        used += len(line) + 1
    if not kept:
        return ""
    return header + "\n".join(kept)
