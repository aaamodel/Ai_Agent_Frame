# -*- coding: utf-8 -*-
"""确定性文档 ID 与文件名归一化（离线可复现，不依赖 Milvus）。

## 为什么需要"确定性"ID

手册要求黄金集每条标注 ``expected_doc_id`` 以计算 Recall@5。但线上上传链路
``POST /documents/upload`` 用的是 ``uuid.uuid4()``——**每次入库都不一样**，
标注无法复用、黄金集无法离线填好、也没法在两个集合之间横向对比。

因此评测专用语料入库脚本 ``evals/tools/reingest_corpus.py`` 改用
**由文件名派生的确定性 ID**：

    document_id = "doc-" + md5(原始文件名 utf-8)[:12]

好处：
    1. 纯函数，**不需要 Milvus 在线**就能算出 ID → ``expected_doc_id`` 可离线回填；
    2. 同一份语料换集合（eval_kb_512 / eval_kb_1024）ID 不变 →
       chunk-size 回归实验是**同一批文档**的 A/B 对照，差异只来自分块参数；
    3. 十六进制 ASCII，避免 Milvus VARCHAR 主键的编码/长度风险。

注意：这与线上 ``/documents/upload`` 的 uuid4 是**两套 ID 空间**。因此
``evals/runners/run_rag.py`` 的召回判定同时保留**文件名通道**作为兜底——当评测
目标是线上真实集合（而非 reingest 的评测集合）时，用 ``metadata["filename"]``
匹配即可；两条通道取较优者，避免"ID 空间不同"被误判成"没召回"。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["DOC_ID_PREFIX", "make_doc_id", "doc_id_from_path", "normalize_doc_name"]


DOC_ID_PREFIX: str = "doc-"

# 会被剥离的文档扩展名（只剥离白名单，避免误伤 "V1.2" 这类名字）
_DOC_EXTENSIONS: frozenset[str] = frozenset(
    {"md", "markdown", "txt", "pdf", "docx", "doc", "csv", "tsv", "xlsx", "xls"}
)


def make_doc_id(filename: str) -> str:
    """由文件名派生确定性文档 ID。

    Args:
        filename: 文档文件名，如 ``公司产品知识库.md``（可含路径，会先取 basename）。

    Returns:
        ``doc-<12 位十六进制>``，例如 ``doc-3f1a9c0b2d4e``。
    """
    base: str = Path(str(filename)).name
    digest: str = hashlib.md5(base.encode("utf-8")).hexdigest()[:12]
    return f"{DOC_ID_PREFIX}{digest}"


def doc_id_from_path(path: str | Path) -> str:
    """便捷包装：从文件路径算 ID。"""
    return make_doc_id(Path(path).name)


def normalize_doc_name(name: str) -> str:
    """归一化文件名用于比对（去路径、去扩展名、去首尾空白、小写）。

    为什么去扩展名：Milvus 的 ``metadata["filename"]`` 是上传时的原始名（含
    ``.md``），而黄金集里有人可能写不带扩展名。归一化后 ``A.md`` 与 ``A``
    视为同一文档，避免因命名习惯差异产生**假阴性**。

    为什么只剥离白名单扩展名：防止 ``Sales_v1.2`` 被误剥成 ``Sales_v1``。
    """
    base: str = Path(str(name)).name.strip()
    if "." in base:
        stem, _, ext = base.rpartition(".")
        if ext.casefold() in _DOC_EXTENSIONS:
            base = stem
    return base.casefold()
