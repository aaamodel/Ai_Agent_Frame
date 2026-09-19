# -*- coding: utf-8 -*-
"""规则与口径的提取：从规则类 sheet 产出**两份不同用途**的产物。

为什么不是"8 张 sheet 全量进 RAG"（design D7）：

    `raw_data/sales_intel` 里的规则类 sheet 实际装着**三类**信息，混在一起会导致把冗余
    注入检索：

      1. 列名 / 类型 / 枚举取值 —— **DDL 里已经有了**（`CHECK` 约束），Vanna 从
         `sqlite_master` 自动学到；再进 RAG 是重复注入；
      2. 业务口径 / 判据（如"员工规模 ≥ 200 为 ICP 达标"）—— DDL 装不下，
         **必须预置进 SQL 生成阶段**（见 D7b），否则模型写不出 `WHERE 员工规模 >= 200`；
      3. 纯规则（阶段流转、折扣权限、组合策略）—— 回答"规则是什么"类问题用的，
         走 RAG 检索。

    因此本模块产出两份东西：
      - ``build_glossary()`` → 口径行，喂给 Vanna 的 ``train(documentation=...)``；
      - ``build_rule_documents()`` → 规则文档，导入 RAG 知识库。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

XLSX_DIR: Path = Path(__file__).resolve().parents[3] / "raw_data" / "sales_intel"

#: 字段字典 sheet（5 份，列名各不相同，故按语义抓列而不按固定名）
GLOSSARY_SHEETS: List[Tuple[str, str]] = [
    ("客户线索台账.xlsx", "字段字典"),
    ("市场活动效果.xlsx", "字段字典"),
    ("竞品追踪台账.xlsx", "字段字典"),
    ("输赢单分析表.xlsx", "字段字典"),
    ("销售业绩月度表.xlsx", "字段字典"),
]

#: 纯规则 sheet（3 份）→ 进 RAG
RULE_SHEETS: List[Tuple[str, str, str]] = [
    ("客户线索台账.xlsx", "阶段流转规则", "线索阶段流转规则"),
    ("产品与报价表.xlsx", "折扣权限", "折扣权限与审批规则"),
    ("产品与报价表.xlsx", "组合策略", "产品组合与折扣策略"),
]

#: 这些列只描述 schema，DDL 已覆盖 → 提取口径时丢弃
_SCHEMA_ONLY_COLUMNS = ("类型", "必填", "数据类型")


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def build_glossary(xlsx_dir: Path = XLSX_DIR) -> List[str]:
    """从 5 张字段字典中提取**口径与判据**，丢弃纯 schema 描述。

    丢弃规则（可验证）：
      - 列名为 类型/必填 的整列丢弃；
      - ``类型 == 枚举`` 的行，其"取值规则"就是枚举清单，已由 DDL 的 ``CHECK`` 承载，
        因此也丢弃——保留它等于把同一份信息注入两次。
    """
    lines: List[str] = []
    for file_name, sheet in GLOSSARY_SHEETS:
        frame = pd.read_excel(xlsx_dir / file_name, sheet_name=sheet)
        for record in frame.to_dict("records"):
            name = _text(record.get("字段"))
            if not name:
                continue
            kind = _text(record.get("类型"))

            parts: List[str] = []
            for column, value in record.items():
                if column in ("字段",) or column in _SCHEMA_ONLY_COLUMNS:
                    continue
                text = _text(value)
                if not text or text in ("—", "-"):
                    continue
                # 枚举取值清单已由 DDL 的 CHECK 承载，不再重复
                if column in ("取值规则", "取值") and kind == "枚举":
                    continue
                parts.append(f"{column}：{text}")
            if parts:
                lines.append(f"- {name} —— " + "；".join(parts))
    return lines


def build_rule_documents(xlsx_dir: Path = XLSX_DIR) -> Dict[str, str]:
    """把 3 张纯规则 sheet 渲染成可检索文档（表头 + 逐行判据）。"""
    documents: Dict[str, str] = {}
    for file_name, sheet, title in RULE_SHEETS:
        frame = pd.read_excel(xlsx_dir / file_name, sheet_name=sheet)
        lines: List[str] = [f"# {title}", ""]
        for record in frame.to_dict("records"):
            cells = [f"{k}：{_text(v)}" for k, v in record.items() if _text(v) not in ("", "—")]
            if cells:
                lines.append("- " + "；".join(cells))
        documents[title] = "\n".join(lines) + "\n"
    return documents


def glossary_as_documentation(glossary: List[str], header: str = "") -> str:
    """把口径行拼成一段 documentation（喂 ``vn.train(documentation=...)``）。"""
    body = "\n".join(glossary)
    if header:
        return f"{header}\n{body}"
    return body


#: 默认的 documentation 抬头——把"这些是口径、不是幻想"讲清楚
DEFAULT_GLOSSARY_HEADER: str = (
    "以下是本业务库的字段口径与判据，写 SQL 时**必须**按此执行，不得自行假设："
)
