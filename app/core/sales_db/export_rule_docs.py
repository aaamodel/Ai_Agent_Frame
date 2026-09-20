# -*- coding: utf-8 -*-
"""把**纯规则**导出成待上传 RAG 知识库的文档。

用法：

    python -m app.core.sales_db.export_rule_docs      # 写出 raw_data/sales_kb_docs/*.md

⚠️ 本模块产出的是 **RAG 文档**（进项目的 Milvus 知识库 `sales_kb`，由
    `rag_knowledge_search` 消费），**不是** Vanna 的训练数据。
    Vanna 的口径训练走 `python -m app.core.sales_db.train_vanna`。
    两者目的地不同（Milvus 知识库 vs Vanna 的 ChromaDB 训练库）、消费者不同
    （自然语言问答 vs SQL 生成），因此是两个入口——这个模块曾经被塞在
    `train_vanna.py` 里，是个概念混淆。

导出后需要**人工上传**到 `sales_kb` 集合（上传接口 `POST /documents/upload`，
`collection_name=sales_kb`）。业务数据不在知识库里，无需上传。

只导"纯规则"这一类的理由见 design D7：
    字段字典里的列名/类型/枚举取值已被 DDL 覆盖（Vanna 从 `sqlite_master` 自动学到），
    再进 RAG 是重复注入；真正的口径走 Vanna 的 documentation。因此这里只剩
    阶段流转规则 / 折扣权限 / 组合策略这三张。
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from app.core.sales_db.knowledge import build_rule_documents

#: 规则文档输出目录（人工上传到 `sales_kb` 的来源）
RULES_OUT_DIR: Path = Path(__file__).resolve().parents[3] / "raw_data" / "sales_kb_docs"


def write_rule_documents(out_dir: Path = RULES_OUT_DIR) -> List[str]:
    """把 3 份规则文档写到磁盘（供导入 `sales_kb` 知识库）；返回写出的路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for title, content in build_rule_documents().items():
        target = out_dir / f"{title}.md"
        target.write_text(content, encoding="utf-8")
        written.append(str(target))
    return written


def _main() -> None:
    for path in write_rule_documents():
        print("写出:", path)
    print("→ 请人工上传到知识库集合 sales_kb（POST /documents/upload）")


if __name__ == "__main__":
    _main()
