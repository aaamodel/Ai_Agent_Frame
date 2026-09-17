# -*- coding: utf-8 -*-
"""文档管理 API：上传、列表与向量集合管理。

- 上传：LlamaIndex 解析→分块→入 Milvus/BM25→Postgres 元数据；可携带逻辑集合的
  「功能描述 / 检索时机描述」（用于意图识别匹配集合，最终透传给 Agent 编排层）。
- 集合管理：列出物理/逻辑集合、集合内文件、按集合+文件名删除向量。
"""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.rag.parse import parse_bytes_to_text, split_text
from app.core.rag.rag_service import RAGService
from app.infrastructure.database.collection_repo import (
    list_vector_collections,
    refresh_kb_registry,
    upsert_vector_collection,
)
from app.infrastructure.database.models import Document, DocumentChunk
from app.infrastructure.database.session import get_async_session
from app.api.depends.dependencies import get_rag_service
from app.models.agent_schemas import (
    CollectionFileDeleteResponse,
    CollectionFileInfo,
    DocumentInfo,
    DocumentUploadResponse,
    KbCollectionInfo,
    KbCollectionListResponse,
)

router = APIRouter(tags=["documents"])

# 默认分块参数（与旧 LangChain chunker 对齐）
_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64

# 逻辑集合名校验：中英文/数字/下划线/中划线，1-128 字符（要进 Milvus 表达式与意图树 id）
_COLLECTION_NAME_RE = re.compile(r"^[\w\u4e00-\u9fff][\w.\u4e00-\u9fff\-]{0,127}$", re.UNICODE)


def _validate_collection_name(name: str) -> str:
    name = (name or "").strip()
    if name and not _COLLECTION_NAME_RE.match(name):
        raise HTTPException(
            status_code=422,
            detail=(
                "collection_name 仅允许中英文、数字、下划线、中划线与点，长度 1-128；"
                f"收到：{name!r}"
            ),
        )
    return name


async def _reset_intent_vector_index(request: Request) -> None:
    """集合描述/成员变更后：reset 意图向量索引并后台重新预热。"""
    retriever = getattr(request.app.state, "intent_vector_retriever", None)
    if retriever is None:
        return
    try:
        retriever.reset()

        def _preheat() -> None:
            try:
                retriever._ensure_index()  # noqa: SLF001 - 复用既有预热入口
                request.app.state.intent_vector_retriever = retriever
                logger.info("集合变更后意图向量索引重新预热完成。")
            except Exception as preheat_error:  # noqa: BLE001
                logger.warning("集合变更后意图索引预热失败（下次请求懒加载兜底）：{}", preheat_error)

        asyncio.create_task(asyncio.to_thread(_preheat))
    except Exception as reset_error:  # noqa: BLE001 - 索引刷新失败不阻断主操作
        logger.warning("意图向量索引 reset 失败（不影响本次操作）：{}", reset_error)


async def _build_collection_infos(
    session: AsyncSession,
    rag_service: RAGService,
) -> List[KbCollectionInfo]:
    """合并 Milvus 实际向量、Postgres 文档行与集合描述，产出逻辑集合清单。"""
    milvus_groups: Dict[str, Dict[str, int]] = await rag_service.list_logical_files()
    descriptor_rows = {row.name: row for row in await list_vector_collections(session)}

    doc_result = await session.execute(
        select(
            Document.id,
            Document.filename,
            Document.created_at,
            Document.meta["collection"].as_string(),
        ).order_by(Document.created_at.desc())
    )
    # tag -> {filename -> {document_id, created_at}}
    pg_docs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for doc_id, filename, created_at, tag in doc_result.all():
        tag = str(tag or "").strip() or "_untagged"
        pg_docs.setdefault(tag, {})[str(filename)] = {
            "document_id": str(doc_id),
            "created_at": created_at.isoformat() if created_at else None,
        }

    all_tags = set(milvus_groups) | set(pg_docs) | set(descriptor_rows)
    infos: List[KbCollectionInfo] = []
    for tag in sorted(all_tags):
        milvus_files = milvus_groups.get(tag, {})
        pg_files = pg_docs.get(tag, {})
        file_names = sorted(set(milvus_files) | set(pg_files))
        files = [
            CollectionFileInfo(
                filename=fname,
                chunk_count=int(milvus_files.get(fname, 0)),
                document_id=pg_files.get(fname, {}).get("document_id"),
                created_at=pg_files.get(fname, {}).get("created_at"),
            )
            for fname in file_names
        ]
        descriptor = descriptor_rows.get(tag)
        infos.append(
            KbCollectionInfo(
                name=tag,
                description=descriptor.description if descriptor else None,
                retrieval_hint=descriptor.retrieval_hint if descriptor else None,
                document_count=len(file_names),
                vector_chunk_count=sum(milvus_files.values()),
                files=files,
            )
        )
    return infos


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
        request: Request,
        file: UploadFile = File(..., description="上传的文件"),
        session: AsyncSession = Depends(get_async_session),
        rag_service: RAGService = Depends(get_rag_service),
        collection_name: str = Form(default="", description="逻辑集合名；空串用默认知识库集合"),
        description: str = Form(
            default="",
            description="集合功能描述：这个集合里是什么内容、覆盖什么主题",
        ),
        retrieval_hint: str = Form(
            default="",
            description="检索时机描述：用户出现什么样的问题/表达时应该检索这个集合",
        ),
) -> DocumentUploadResponse:
    """上传文档：LlamaIndex 解析 + SentenceSplitter 分块，写入 Milvus 与 BM25，
    并保留 Postgres 文档元数据记录以支撑列表接口。

    可选的 ``description`` / ``retrieval_hint`` 会登记（upsert）到集合注册表，
    供意图识别把后续问题路由到本集合。
    """
    settings = get_settings()
    kb_collection: str = _validate_collection_name(
        collection_name
    ) or settings.milvus_kb_collection_name
    description = (description or "").strip()
    retrieval_hint = (retrieval_hint or "").strip()
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

    # 4. Postgres 元数据记录（含集合描述注册表 upsert）
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
        if description or retrieval_hint:
            await upsert_vector_collection(
                session,
                kb_collection,
                description=description or None,
                retrieval_hint=retrieval_hint or None,
            )
        await session.commit()
    except Exception as exc:
        await session.rollback()
        logger.exception("Postgres 文档元数据写入失败: {}", exc)
        # 元数据记录失败不阻断已经成功的向量入库，降级返回
        logger.warning("文档元数据写入失败，但向量已入库。doc_id={}", doc_id)

    # 5. 刷新意图路由（动态集合注册表 + 向量索引），失败不阻断上传
    try:
        milvus_groups = await rag_service.list_logical_files()
        await refresh_kb_registry(session, extra_active_names=list(milvus_groups.keys()))
        await _reset_intent_vector_index(request)
    except Exception as refresh_error:  # noqa: BLE001
        logger.warning("上传后意图路由刷新失败（不影响上传结果）：{}", refresh_error)

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=safe_name,
        status="ready",
        chunk_count=len(chunks),
        collection_name=kb_collection,
        description=description or None,
        retrieval_hint=retrieval_hint or None,
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


# ======================================================================
# 向量集合管理
# ======================================================================
@router.get("/vector/collections", response_model=KbCollectionListResponse)
async def list_vector_collections_api(
    session: AsyncSession = Depends(get_async_session),
    rag_service: RAGService = Depends(get_rag_service),
) -> KbCollectionListResponse:
    """列出 RAG 物理集合内的全部逻辑集合（含文件与描述）。"""
    try:
        collections = await _build_collection_infos(session, rag_service)
        return KbCollectionListResponse(
            physical_collection=rag_service.physical_collection_name,
            collections=collections,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("查询向量集合列表失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"查询失败: {exc!s}") from exc


@router.get(
    "/vector/collections/{collection_name}/files",
    response_model=KbCollectionInfo,
)
async def list_collection_files_api(
    collection_name: str,
    session: AsyncSession = Depends(get_async_session),
    rag_service: RAGService = Depends(get_rag_service),
) -> KbCollectionInfo:
    """列出某个逻辑集合下的全部文件（含各自向量切片数）。"""
    tag = _validate_collection_name(collection_name)
    try:
        infos = await _build_collection_infos(session, rag_service)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("查询集合文件失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"查询失败: {exc!s}") from exc
    for info in infos:
        if info.name == tag:
            return info
    raise HTTPException(status_code=404, detail=f"集合不存在或为空：{tag}")


@router.delete(
    "/vector/collections/{collection_name}/files/{filename:path}",
    response_model=CollectionFileDeleteResponse,
)
async def delete_collection_file_api(
    collection_name: str,
    filename: str,
    request: Request,
    session: AsyncSession = Depends(get_async_session),
    rag_service: RAGService = Depends(get_rag_service),
) -> CollectionFileDeleteResponse:
    """删除指定逻辑集合下某个文件的全部向量与文档元数据。

    路径参数 ``filename`` 支持中文（调用方需 URL 编码）；删除后同步收缩 BM25、
    Postgres 行（chunks 级联删除）并刷新意图集合路由。
    """
    tag = _validate_collection_name(collection_name)
    filename = (filename or "").strip()
    if not filename:
        raise HTTPException(status_code=422, detail="filename 不能为空")

    # 1. 先取 Postgres 行（拿 storage_path 做原始文件清理），再删 Milvus 向量
    try:
        doc_rows = (
            await session.execute(
                select(Document).where(
                    Document.filename == filename,
                    Document.meta["collection"].as_string() == tag,
                )
            )
        ).scalars().all()
    except Exception as exc:
        logger.exception("查询待删文档元数据失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"查询失败: {exc!s}") from exc

    try:
        deleted_ids = await rag_service.delete_vectors_by_filename(tag, filename)
    except Exception as exc:
        logger.exception("Milvus 向量删除失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"向量删除失败: {exc!s}") from exc

    if not deleted_ids and not doc_rows:
        raise HTTPException(
            status_code=404,
            detail=f"集合 {tag!r} 下未找到文件 {filename!r} 的向量或文档记录",
        )

    # 2. 删 Postgres 文档行（document_chunks 外键 CASCADE）
    deleted_documents = 0
    storage_paths = [row.storage_path for row in doc_rows if row.storage_path]
    if doc_rows:
        try:
            for row in doc_rows:
                await session.delete(row)
            await session.commit()
            deleted_documents = len(doc_rows)
        except Exception as exc:
            await session.rollback()
            logger.exception("Postgres 文档元数据删除失败（向量已删）: {}", exc)

    # 3. 原始文件尽力清理（不阻断）
    for storage_path in storage_paths:
        try:
            path = Path(storage_path)
            if path.exists():
                await asyncio.to_thread(path.unlink)
        except Exception as file_error:  # noqa: BLE001
            logger.warning("原始上传文件删除失败 {}：{}", storage_path, file_error)

    # 4. 刷新意图集合路由（集合可能已空，动态节点随之注销）
    try:
        milvus_groups = await rag_service.list_logical_files()
        await refresh_kb_registry(session, extra_active_names=list(milvus_groups.keys()))
        await _reset_intent_vector_index(request)
    except Exception as refresh_error:  # noqa: BLE001
        logger.warning("删除后意图路由刷新失败（不影响删除结果）：{}", refresh_error)

    return CollectionFileDeleteResponse(
        collection=tag,
        filename=filename,
        deleted_chunks=len(deleted_ids),
        deleted_documents=deleted_documents,
        message="删除成功" if deleted_ids or deleted_documents else "未找到可删除数据",
    )
