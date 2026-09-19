# -*- coding: utf-8 -*-
"""从 Postgres 重跑 embedding，把向量补回 Milvus（不重新上传、不重填集合描述）。

## 为什么不需要重新上传

``POST /documents/upload`` 把同一次上传写进了**两处**，两者的生命周期是解耦的：

    1. Milvus 向量 —— metadata 里带 ``document_id`` / ``chunk_index`` /
       ``filename`` / ``collection``；
    2. Postgres —— ``documents``（文件与集合标签）、``document_chunks``（切片正文 +
       ``vector_id``）、``vector_collections``（集合的 description）。

Milvus 侧被清空（例如历史遗留的 ``MILVUS_KB_OVERWRITE=true`` 启动即 drop 重建集合）
时，Postgres 那半边完好无损。所以恢复检索能力只要拿 ``document_chunks.content``
**重跑一次 embedding** 写回 Milvus 即可：

    - 切片边界与原文完全一致（直接复用已存切片，不重新分块）；
    - ``document_id`` / ``filename`` / ``collection`` 全部沿用 Postgres 原值 →
      Milvus 与 Postgres 的文档身份继续对齐，``GET /vector/collections`` 立刻恢复；
    - 集合的 description 存在 ``vector_collections`` 表里，本脚本**完全不碰**，
      因此不需要再填一遍上传表单。

## 只补缺，不重复灌（重要）

默认只处理「该文件在 Milvus 里目前一片向量都没有」的文件。已经有向量的文件会跳过：
重复灌会产生**同内容不同 node_id 的重复切片**，检索结果出现重复、评测指标失真。
另外，若 Milvus 里的切片数与 Postgres 不一致（灌到一半就被打断），会作为
``切片数不一致`` 单独列出来**不动它**——正确做法是先用
``DELETE /vector/collections/{collection}/files/{filename}`` 删干净再补，避免重复。

## 用法::

    # 1) 先看要补哪些（只读 Postgres + Milvus，不调用 embedding API，不花钱）
    python -m evals.tools.reembed_from_db --dry-run

    # 2) 真正重跑 embedding 补回去
    python -m evals.tools.reembed_from_db

    # 3) 只补指定逻辑集合（逗号分隔多个）
    python -m evals.tools.reembed_from_db --collection enterprise_kb,sales_kb

⚠️ 本脚本只写 Milvus 向量与回写 ``document_chunks.vector_id``，**不删任何数据**。
   进程内的 BM25 内存索引在应用重启时会自动从 Milvus 全量重建（``seed_bm25``）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _log(message: str) -> None:
    print(f"[reembed] {message}", flush=True)


async def _fetch_documents(
    session: Any, tags: Optional[List[str]]
) -> List[Dict[str, Any]]:
    """从 Postgres 取「文档 → 切片」清单（含切片正文，即 embedding 的输入）。"""
    from sqlalchemy import select

    from app.infrastructure.database.models import Document, DocumentChunk

    rows = (
        await session.execute(
            select(
                Document.id,
                Document.filename,
                Document.meta["collection"].as_string(),
                Document.created_at,
            ).order_by(Document.created_at.asc())
        )
    ).all()

    documents: List[Dict[str, Any]] = []
    for doc_id, filename, tag, created_at in rows:
        # 与 document.py::_build_collection_infos 同口径：空标签记为 _untagged
        tag = str(tag or "").strip() or "_untagged"
        if tags and tag not in tags:
            continue
        chunk_rows = (
            await session.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.chunk_index,
                    DocumentChunk.content,
                )
                .where(DocumentChunk.document_id == doc_id)
                .order_by(DocumentChunk.chunk_index.asc())
            )
        ).all()
        pairs = [
            {"chunk_id": str(chunk_id), "chunk_index": int(index or 0), "text": str(content)}
            for chunk_id, index, content in chunk_rows
            if content and str(content).strip()
        ]
        documents.append(
            {
                "document_id": str(doc_id),
                "filename": str(filename),
                "collection": tag,
                "created_at": created_at.isoformat() if created_at else None,
                "pairs": pairs,
            }
        )
    return documents


def _split_plan(
    documents: List[Dict[str, Any]], milvus_groups: Dict[str, Dict[str, int]]
) -> tuple:
    """把 Postgres 文档分成「待补 / 已存在 / 切片数不一致 / 无切片」四类。"""
    pending: List[Dict[str, Any]] = []
    existed: List[tuple] = []
    partial: List[tuple] = []
    empty: List[tuple] = []

    for doc in documents:
        tag: str = doc["collection"]
        filename: str = doc["filename"]
        pg_count: int = len(doc["pairs"])
        mv_count: int = int(milvus_groups.get(tag, {}).get(filename, 0))

        if pg_count == 0:
            empty.append((tag, filename, "Postgres 里没有切片正文"))
        elif mv_count == 0:
            pending.append(doc)
        elif mv_count < pg_count:
            partial.append((tag, filename, mv_count, pg_count))
        else:
            existed.append((tag, filename, mv_count))

    return pending, existed, partial, empty


async def _reembed(
    *, tags: Optional[List[str]], dry_run: bool
) -> int:
    """执行重跑；返回实际写入的切片数。"""
    from app.infrastructure.database.session import configure_session, init_engine
    from evals.runners._runtime import build_rag_service, build_settings

    settings: Any = build_settings()
    engine: Any = init_engine(settings.database_url)
    factory: Any = configure_session(engine)

    from sqlalchemy import update

    from app.infrastructure.database.models import DocumentChunk

    written: int = 0
    try:
        async with factory() as session:
            documents = await _fetch_documents(session, tags)
            _log(
                f"Postgres 文档 {len(documents)} 份"
                f"（切片合计 {sum(len(d['pairs']) for d in documents)} 片）"
                + (f"，逻辑集合过滤={tags}" if tags else "")
            )

            # overwrite 恒为 False：本脚本只做"补写"，绝不删表重建
            rag_service: Any = build_rag_service(settings, overwrite=False)
            milvus_groups: Dict[str, Dict[str, int]] = await rag_service.list_logical_files()
            _log(
                "Milvus 当前分布："
                + (
                    "，".join(
                        f"{tag}={sum(files.values())}片"
                        for tag, files in sorted(milvus_groups.items())
                    )
                    or "（空）"
                )
            )

            pending, existed, partial, empty = _split_plan(documents, milvus_groups)

            for tag, filename, mv_count in existed:
                _log(f"  跳过（已有向量 {mv_count} 片）：[{tag}] {filename}")
            for tag, filename, mv_count, pg_count in partial:
                _log(
                    f"  ⚠️ 切片数不一致，未处理（Milvus={mv_count} / Postgres={pg_count}）："
                    f"[{tag}] {filename} —— 先用 "
                    f"DELETE /vector/v1/collections/{tag}/files/{filename} 删干净再补，"
                    "否则会重复"
                )
            for tag, filename, reason in empty:
                _log(f"  ⚠️ 跳过（{reason}）：[{tag}] {filename}")

            if not pending:
                _log("没有需要补的文件（Milvus 里向量已齐）。")
                return 0

            total_chunks: int = sum(len(doc["pairs"]) for doc in pending)
            if dry_run:
                _log(f"[dry-run] 待补 {len(pending)} 个文件 / {total_chunks} 片：")
                for doc in pending:
                    _log(
                        f"  - [{doc['collection']}] {doc['filename']} "
                        f"-> {len(doc['pairs'])} 片 (doc_id={doc['document_id']})"
                    )
                return 0

            _log(f"开始重跑 embedding：{len(pending)} 个文件 / {total_chunks} 片")
            for index, doc in enumerate(pending, start=1):
                texts: List[str] = [pair["text"] for pair in doc["pairs"]]
                metadatas: List[Dict[str, Any]] = [
                    {
                        # 沿用 Postgres 原值：Milvus 与 DB 的文档身份保持一致
                        "document_id": doc["document_id"],
                        "chunk_index": pair["chunk_index"],
                        "filename": doc["filename"],
                        "collection": doc["collection"],
                    }
                    for pair in doc["pairs"]
                ]
                node_ids: List[str] = await rag_service.ingest_texts(texts, metadatas)

                # 回写 vector_id，避免 document_chunks.vector_id 与 Milvus 实际 node_id 漂移
                if node_ids and len(node_ids) == len(doc["pairs"]):
                    await session.execute(
                        update(DocumentChunk),
                        [
                            {"id": pair["chunk_id"], "vector_id": node_id}
                            for pair, node_id in zip(doc["pairs"], node_ids)
                        ],
                    )
                    await session.commit()

                written += len(node_ids)
                _log(
                    f"  [{index}/{len(pending)}] [{doc['collection']}] {doc['filename']} "
                    f"-> {len(node_ids)} 片"
                )

            after: Dict[str, Dict[str, int]] = await rag_service.list_logical_files()
            _log(
                "完成：写入 "
                f"{written} 片；Milvus 现分布："
                + (
                    "，".join(
                        f"{tag}={sum(files.values())}片"
                        for tag, files in sorted(after.items())
                    )
                    or "（空）"
                )
            )
            return written
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 Postgres 的切片正文重跑 embedding，把向量补回 Milvus"
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="只补指定逻辑集合（逗号分隔多个，如 enterprise_kb,sales_kb）；"
        "不传则处理全部",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出会补哪些文件与片数，不写 Milvus、不调用 embedding API",
    )
    args = parser.parse_args()

    tags: Optional[List[str]] = (
        [t.strip() for t in args.collection.split(",") if t.strip()]
        if args.collection
        else None
    )
    written: int = asyncio.run(_reembed(tags=tags, dry_run=args.dry_run))
    if args.dry_run:
        _log("dry-run 结束（未写入任何数据）。确认无误后去掉 --dry-run 执行。")
    else:
        _log(f"结束：本次共写入 {written} 片向量。")


if __name__ == "__main__":
    main()
