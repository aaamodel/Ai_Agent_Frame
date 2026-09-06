# -*- coding: utf-8 -*-
"""文档管理 API：上传与列表（LlamaIndex 重构：解析→分块→入库→Pg 元数据）。"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.rag.parse import parse_bytes_to_text, split_text
from app.core.rag.rag_service import RAGService
from app.infrastructure.database.models import Document, DocumentChunk
from app.infrastructure.database.session import get_async_session
from app.api.depends.dependencies import get_rag_service
from app.models.agent_schemas import DocumentInfo, DocumentUploadResponse

router = APIRouter(tags=["documents"])

# 默认分块参数（与旧 LangChain chunker 对齐）
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
        file: UploadFile = File(..., description="上传的文件"),
        session: AsyncSession = Depends(get_async_session),
        rag_service: RAGService = Depends(get_rag_service),
        collection_name: str = "",  # 空串表示使用配置默认知识库集合
) -> DocumentUploadResponse:
    """上传文档：LlamaIndex 解析 + SentenceSplitter 分块，写入 Milvus 与 BM25，
    并保留 Postgres 文档元数据记录以支撑列表接口。"""
    settings = get_settings()
    kb_collection: str = (collection_name or "").strip() or settings.milvus_kb_collection_name
    upload_root = Path("uploads")
    upload_root.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    safe_name = file.filename or "unnamed"
    dest = upload_root / f"{doc_id}_{safe_name}"

    # 1. 落地保存原始文件
    try:
        raw = await file.read()
        await asyncio.to_thread(dest.write_bytes, raw)
    except Exception as exc:
        logger.exception("保存上传文件失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"保存文件失败: {exc!s}") from exc

    # 2. 解析 + 分块
    text: str = parse_bytes_to_text(raw, safe_name)
    chunks: List[str] = split_text(text, chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=422, detail="未能从文件中解析出任何可检索文本。")

    # 3. 入库向量库 + 刷新 BM25（LlamaIndex 内嵌 embedding，返回 node_id 列表）
    metadatas = [
        {
            "document_id": doc_id,
            "chunk_index": idx,
            "filename": safe_name,
            "collection": kb_collection,
        }
        for idx in range(len(chunks))
    ]
    try:
        node_ids: List[str] = await rag_service.ingest_texts(chunks, metadatas)
    except Exception as exc:
        logger.exception("向量库写入失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"向量库入库失败: {exc!s}") from exc

    # 4. Postgres 元数据记录（支撑 GET /documents 列表）
    try:
        doc = Document(
            id=doc_id,
            filename=safe_name,
            mime_type=file.content_type,
            storage_path=str(dest),
            status="ready",
            meta={"chunk_count": len(chunks), "collection": kb_collection},
        )
        session.add(doc)
        for i, chunk_text in enumerate(chunks, start=0):
            session.add(
                DocumentChunk(
                    id=node_ids[i] if i < len(node_ids) else str(uuid.uuid4()),
                    document_id=doc_id,
                    chunk_index=i,
                    content=chunk_text[:65000],
                    vector_id=node_ids[i] if i < len(node_ids) else "",
                    meta=None,
                )
            )
        await session.commit()
    except Exception as exc:
        await session.rollback()
        logger.exception("Postgres 文档元数据写入失败: {}", exc)
        # 元数据记录失败不阻断已经成功的向量入库，降级返回
        logger.warning("文档元数据写入失败，但向量已入库。doc_id={}", doc_id)

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=safe_name,
        status="ready",
        chunk_count=len(chunks),
        message="LlamaIndex 解析/分块/入库全链路成功",
    )


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    session: AsyncSession = Depends(get_async_session),
) -> list[DocumentInfo]:
    """列出已入库文档元数据。"""
    try:
        result = await session.execute(select(Document).order_by(Document.created_at.desc()))
        rows = result.scalars().all()
        out: list[DocumentInfo] = []
        for d in rows:
            out.append(
                DocumentInfo(
                    id=d.id,
                    filename=d.filename,
                    mime_type=d.mime_type,
                    status=d.status,
                    created_at=d.created_at.isoformat() if d.created_at else None,
                )
            )
        return out
    except Exception as exc:
        logger.exception("查询文档列表失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"查询失败: {exc!s}") from exc