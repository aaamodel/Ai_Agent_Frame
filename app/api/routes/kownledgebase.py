# -*- coding: utf-8 -*-
"""LightRAG 知识图谱知识库 API。

- 上传：PDF/TXT 解析 → LightRAG 实体/关系抽取 → 按 workspace（=「图谱集合」）
  隔离落盘（JSON KV + NanoVectorDB + NetworkX graphml）；可携带集合的
  「功能描述 / 检索时机描述」，与 RAG 文档库共用一套意图动态集合路由。
- 集合管理：列出图谱集合、集合内文件、按文件级联删除（切片+三类向量+图谱）、
  清空历史遗留空 workspace。

注意：所有 ``from app.infrastructure.knowledgebase.light_rag import ...`` 必须
保持函数内懒加载——模块级导入会拉起 torch / transformers，拖慢应用启动。
"""

import asyncio
import os
import re
import shutil
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.collection_repo import (
    list_vector_collections,
    refresh_kb_registry,
    upsert_vector_collection,
)
from app.infrastructure.database.models import VectorCollection
from app.infrastructure.database.session import get_async_session
from app.query_intent.kb_collection_registry import ENGINE_GRAPH

router = APIRouter(tags=["kownledgebase"])

# 上传原文件存放根（CWD 相对；与旧接口保持同一个 raw_data）
UPLOAD_DIR = "./raw_data"
# 新接口按集合分子目录存放原文件，避免同名覆盖与 RAG 语料混杂
_GRAPH_UPLOAD_SUBDIR = "knowledgebase"
# LightRAG 物理 workspace 根（与 light_rag.WORKING_DIR 指向同一目录，
# 这里独立拼路径，纯文件系统枚举时无需 import lightrag——会拉起 torch）
_GRAPH_WORKSPACE_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "infrastructure",
        "knowledgebase",
        "raw_data",
        "lightrag_workspace",
    )
)
# 遗留空 workspace 在 URL 里的占位名（真实 workspace 名禁止使用它）
LEGACY_WORKSPACE_ALIAS = "__legacy__"

_ALLOWED_EXTS = (".txt", ".pdf")

# 与 light_rag.validate_graph_workspace_name / documents 侧保持同一套规则，
# 本地复制一份：路由层校验不能 import light_rag（会拉起 torch）。
_COLLECTION_NAME_RE = re.compile(r"^[\w\u4e00-\u9fff][\w.\u4e00-\u9fff\-]{0,127}$")
_DEFAULT_COLLECTION = "default"


# ======================================================================
# 响应模型
# ======================================================================
class GraphDocumentUploadResponse(BaseModel):
    status: str
    filename: str
    collection_name: str
    track_id: Optional[str] = None
    description: Optional[str] = None
    retrieval_hint: Optional[str] = None
    message: str


class TaskResponse(BaseModel):
    status: str
    task_id: str
    collection_name: str
    message: str


class GraphFileInfo(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunks_count: Optional[int] = None
    content_length: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_msg: Optional[str] = None


class GraphCollectionInfo(BaseModel):
    name: str
    legacy: bool = False
    description: Optional[str] = None
    retrieval_hint: Optional[str] = None
    document_count: int = 0
    files: List[GraphFileInfo] = []


class GraphFileDeleteResponse(BaseModel):
    collection: str
    filename: str
    deleted_documents: int
    details: List[Dict[str, Any]] = []
    message: str


class LegacyClearResponse(BaseModel):
    status: str
    dropped: List[str]
    message: str


# ======================================================================
# 内部辅助
# ======================================================================
def _lazy_light_rag():
    """懒加载 LightRAG 封装（避免模块级拉起 torch 等重型依赖）。"""
    from app.infrastructure.knowledgebase.light_rag import (  # noqa: WPS433
        GraphPipelineBusyError,
        clear_workspace,
        delete_documents_by_filename,
        insert_document,
        parse_local_file,
    )

    return {
        "GraphPipelineBusyError": GraphPipelineBusyError,
        "clear_workspace": clear_workspace,
        "delete_documents_by_filename": delete_documents_by_filename,
        "insert_document": insert_document,
        "parse_local_file": parse_local_file,
    }


def _safe_filename(raw_name: str) -> str:
    """取 basename 防路径穿越；空名直接 400。"""
    name = os.path.basename((raw_name or "").replace("\\", "/").strip())
    if not name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="文件名非法")
    return name


def _validate_name(name: str) -> str:
    """集合名正则校验（规则与 light_rag.validate_graph_workspace_name 一致）。"""
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


def _resolve_collection(collection_name: str) -> str:
    """校验并归一图谱集合名；'' / __legacy__ 等保留名拒绝写入。"""
    name = _validate_name(collection_name) or _DEFAULT_COLLECTION
    if name == LEGACY_WORKSPACE_ALIAS:
        raise HTTPException(
            status_code=422,
            detail=f"集合名不能为保留名：{LEGACY_WORKSPACE_ALIAS}",
        )
    return name


def _resolve_collection_api(name: str) -> Tuple[str, bool]:
    """URL 路径参数 → (workspace, is_legacy)。"""
    if name == LEGACY_WORKSPACE_ALIAS:
        return "", True
    workspace = _validate_name(name)
    if not workspace:
        raise HTTPException(status_code=404, detail="集合名不能为空")
    return workspace, False


def _raw_upload_dir(workspace: str) -> str:
    return os.path.join(UPLOAD_DIR, _GRAPH_UPLOAD_SUBDIR, workspace)


def _to_file_info(doc: Dict[str, Any]) -> GraphFileInfo:
    return GraphFileInfo(
        doc_id=str(doc.get("doc_id", "")),
        filename=str(doc.get("filename", "") or ""),
        status=str(doc.get("status", "") or ""),
        chunks_count=doc.get("chunks_count"),
        content_length=doc.get("content_length"),
        created_at=str(doc["created_at"]) if doc.get("created_at") else None,
        updated_at=str(doc["updated_at"]) if doc.get("updated_at") else None,
        error_msg=doc.get("error_msg"),
    )


def enumerate_active_graph_workspaces() -> List[str]:
    """纯文件系统枚举：含 processed 文档的具名 workspace（不 import lightrag）。

    LightRAG 的 doc_status 落盘为 ``kv_store_doc_status.json``（dict：
    doc_id -> 状态记录）。failed/pending-only 的集合不参与意图路由，
    历史遗留空 workspace（散落在根目录）也不在此枚举。
    """
    import json

    if not os.path.isdir(_GRAPH_WORKSPACE_DIR):
        return []
    active: List[str] = []
    for entry in sorted(os.listdir(_GRAPH_WORKSPACE_DIR)):
        sub = os.path.join(_GRAPH_WORKSPACE_DIR, entry)
        if not os.path.isdir(sub):
            continue
        status_file = os.path.join(sub, "kv_store_doc_status.json")
        if not os.path.isfile(status_file):
            continue
        try:
            with open(status_file, "r", encoding="utf-8") as fp:
                payload = json.load(fp)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and "data" in payload and isinstance(
            payload["data"], dict
        ):  # 兼容可能的 {"data": {...}} 包装格式
            payload = payload["data"]
        if not isinstance(payload, dict):
            continue
        if any(
            isinstance(rec, dict) and str(rec.get("status")) == "processed"
            for rec in payload.values()
        ):
            active.append(entry)
    return active


async def _active_graph_workspaces() -> List[str]:
    """有 processed 文档的具名 workspace 才参与意图路由（failed-only 不算）。"""
    return await asyncio.to_thread(enumerate_active_graph_workspaces)


def _enumerate_graph_workspaces_fs() -> List[Tuple[str, bool]]:
    """纯文件系统枚举 workspace：``[(name, is_legacy)]``（不 import lightrag）。

    判定规则与 light_rag.enumerate_graph_workspaces 保持一致：
    - 具名集合：WORKING_DIR 下含 .graphml/.json 数据文件的子目录；
    - 遗留空 workspace：根目录散落 kv_store_* / vdb_* / *.graphml。
    """
    out: List[Tuple[str, bool]] = []
    if not os.path.isdir(_GRAPH_WORKSPACE_DIR):
        return out
    for entry in sorted(os.listdir(_GRAPH_WORKSPACE_DIR)):
        sub = os.path.join(_GRAPH_WORKSPACE_DIR, entry)
        if not os.path.isdir(sub):
            continue
        try:
            has_data = any(
                f.endswith((".graphml", ".json"))
                for f in os.listdir(sub)
                if os.path.isfile(os.path.join(sub, f))
            )
        except OSError:
            has_data = False
        if has_data:
            out.append((entry, False))
    try:
        legacy_has_data = any(
            os.path.isfile(os.path.join(_GRAPH_WORKSPACE_DIR, f))
            and (
                f.endswith(".graphml")
                or f.startswith(("kv_store_", "vdb_"))
            )
            for f in os.listdir(_GRAPH_WORKSPACE_DIR)
        )
    except OSError:
        legacy_has_data = False
    if legacy_has_data:
        out.append(("", True))
    return out


def _read_workspace_docs_fs(
    workspace: str, *, legacy: bool
) -> List[Dict[str, Any]]:
    """直接解析 kv_store_doc_status.json 列出文档（纯 JSON，不加载 torch）。

    输出字段与 light_rag.list_workspace_documents 对齐；按 created_at 倒序。
    """
    import json

    if legacy:
        status_file = os.path.join(_GRAPH_WORKSPACE_DIR, "kv_store_doc_status.json")
    else:
        status_file = os.path.join(
            _GRAPH_WORKSPACE_DIR, workspace, "kv_store_doc_status.json"
        )
    if not os.path.isfile(status_file):
        return []
    try:
        with open(status_file, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, ValueError):
        return []
    if isinstance(payload, dict) and "data" in payload and isinstance(
        payload["data"], dict
    ):  # 兼容可能的 {"data": {...}} 包装格式
        payload = payload["data"]
    if not isinstance(payload, dict):
        return []

    docs: List[Dict[str, Any]] = []
    for doc_id, st in payload.items():
        if not isinstance(st, dict):
            continue
        docs.append(
            {
                "doc_id": str(doc_id),
                "filename": str(st.get("file_path", "") or ""),
                "status": str(st.get("status", "") or ""),
                "chunks_count": st.get("chunks_count"),
                "content_length": st.get("content_length"),
                "track_id": st.get("track_id"),
                "created_at": st.get("created_at"),
                "updated_at": st.get("updated_at"),
                "error_msg": st.get("error_msg"),
            }
        )
    docs.sort(key=lambda d: str(d.get("created_at") or ""), reverse=True)
    return docs


async def _refresh_graph_routing(request: Request, db_session: AsyncSession) -> None:
    """图谱集合增删后：刷新注册表（RAG+图谱合并）并重预热意图向量索引。"""
    try:
        graph_active = await _active_graph_workspaces()
        await refresh_kb_registry(db_session, graph_active_names=graph_active)
    except Exception as exc:  # noqa: BLE001 - 路由刷新失败不阻断主操作
        logger.warning("图谱集合注册表刷新失败（不影响本次操作）：{}", exc)

    # 复用 documents 路由里的索引 reset + 后台预热逻辑
    from app.api.routes.document import _reset_intent_vector_index  # noqa: WPS433

    await _reset_intent_vector_index(request)


# ======================================================================
# 上传（单文件，同步）
# ======================================================================
@router.post(
    "/documents/kownledgebase/upload",
    response_model=GraphDocumentUploadResponse,
)
async def upload_kownledgebase_document(
    request: Request,
    file: UploadFile = File(...),
    collection_name: str = Form(
        default="",
        description="知识图谱集合（LightRAG workspace）名；空串用默认集合 default",
    ),
    description: str = Form(
        default="",
        description="集合功能描述：这个图谱集合里是什么内容、覆盖什么主题",
    ),
    retrieval_hint: str = Form(
        default="",
        description="检索时机描述：用户出现什么样的问题/表达时应该检索这个图谱集合",
    ),
    db_session: AsyncSession = Depends(get_async_session),
) -> GraphDocumentUploadResponse:
    """上传 PDF/TXT：解析并织入指定图谱集合（LightRAG workspace）。

    可选 description / retrieval_hint 登记到集合注册表（engine=graph），
    供意图识别把关系/实体类问题路由到本集合，最终以
    ``knowledge_graph_search(collection=集合名)`` 透传给 Agent 编排层。
    """
    workspace = _resolve_collection(collection_name)
    safe_name = _safe_filename(file.filename)
    ext = os.path.splitext(safe_name)[-1].lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不支持的文件格式。目前仅支持 .txt 和 .pdf 文件。",
        )

    description = (description or "").strip()
    retrieval_hint = (retrieval_hint or "").strip()

    # 引擎撞名预检：同名集合若已注册为 RAG，拒绝入库（避免写入后才发现）
    existing = await db_session.get(VectorCollection, workspace)
    if existing is not None and (existing.engine or "rag") != ENGINE_GRAPH:
        raise HTTPException(
            status_code=409,
            detail=(
                f"集合 {workspace!r} 已作为文档向量库（RAG）集合存在，"
                "不能再作为知识图谱集合，请换一个集合名。"
            ),
        )

    # 1. 原文件落盘（按集合分子目录）
    save_dir = _raw_upload_dir(workspace)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, safe_name)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"保存文件到本地失败: {exc!s}"
        ) from exc
    finally:
        await file.close()

    # 2. 解析 + LightRAG 入库（此刻才懒加载 torch/lightrag，校验失败不付该成本）
    helpers = _lazy_light_rag()
    try:
        content = helpers["parse_local_file"](file_path)
        if not content:
            raise HTTPException(
                status_code=400,
                detail="文件内容为空或无法提取有效文本。",
            )
        track_id = await helpers["insert_document"](workspace, safe_name, content)
    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.exception("LightRAG 解析或写入索引失败")
        raise HTTPException(
            status_code=500,
            detail=f"LightRAG 解析或写入索引失败: {exc!s}",
        ) from exc

    # 3. 登记/更新集合描述（空值保留既有描述，与 RAG 上传同口径）
    try:
        if description or retrieval_hint:
            await upsert_vector_collection(
                db_session,
                workspace,
                description=description or None,
                retrieval_hint=retrieval_hint or None,
                engine=ENGINE_GRAPH,
            )
        await db_session.commit()
    except ValueError as exc:  # 并发撞名兜底
        await db_session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        await db_session.rollback()
        logger.exception("集合描述登记失败（图谱数据已入库）")
        raise HTTPException(
            status_code=500, detail=f"集合描述登记失败: {exc!s}"
        ) from exc

    # 4. 刷新意图集合路由（不阻断响应）
    await _refresh_graph_routing(request, db_session)

    return GraphDocumentUploadResponse(
        status="success",
        filename=safe_name,
        collection_name=workspace,
        track_id=str(track_id) if track_id else None,
        description=description or None,
        retrieval_hint=retrieval_hint or None,
        message="文件已成功上传，并完成 LightRAG 知识图谱的解析与本地固化！",
    )


# ======================================================================
# 上传（多文件，后台流水线）
# ======================================================================
async def _bg_processing_pipeline(
    items: List[Tuple[str, str]],
    workspace: str,
    retriever: Any,
) -> None:
    """后台：批量解析 + LightRAG 织入；完成后刷新意图路由。

    items: (保存路径, 原始文件名)
    """
    helpers = _lazy_light_rag()
    try:
        valid_pairs: List[Tuple[str, str]] = []
        for path, original_name in items:
            if os.path.exists(path):
                content = helpers["parse_local_file"](path)
                if content:
                    valid_pairs.append((content, original_name))

        if valid_pairs:
            from app.infrastructure.knowledgebase.light_rag import get_lightrag

            rag = get_lightrag(workspace)
            await rag.initialize_storages()
            contents = [text for text, _ in valid_pairs]
            file_paths = [name for _, name in valid_pairs]
            await rag.ainsert(
                contents,
                file_paths=file_paths,
                track_id=f"bulk_{uuid.uuid4().hex}",
            )
            logger.info(
                "🎉 [异步任务] 成功批量织入 {} 份文档到图谱集合 {}",
                len(valid_pairs),
                workspace,
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("❌ [异步任务失败] 原因: {}", exc)
    finally:
        for path, _ in items:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        # 刷新注册表与意图索引
        # 函数内延迟导入：模块顶部 from-import 会拿到 configure_session 之前的
        # None 快照；后台任务运行于 lifespan 之后，此刻取到的才是真实 maker。
        from app.infrastructure.database.session import async_session_factory

        try:
            if async_session_factory is not None:
                async with async_session_factory() as db_session:
                    graph_active = await _active_graph_workspaces()
                    await refresh_kb_registry(
                        db_session, graph_active_names=graph_active
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("批量入库后注册表刷新失败：{}", exc)
        if retriever is not None:
            try:
                retriever.reset()

                def _preheat() -> None:
                    try:
                        retriever._ensure_index()  # noqa: SLF001
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("批量入库后意图索引预热失败：{}", exc)

                await asyncio.to_thread(_preheat)
            except Exception as exc:  # noqa: BLE001
                logger.warning("批量入库后意图索引 reset 失败：{}", exc)


@router.post(
    "/documents/knowledgebase/upload-bulk",
    response_model=TaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_multiple_documents(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    collection_name: str = Form(default=""),
    description: str = Form(default=""),
    retrieval_hint: str = Form(default=""),
    db_session: AsyncSession = Depends(get_async_session),
) -> TaskResponse:
    """多文件批量上传到指定图谱集合（后台抽取，立即返回 202）。"""
    if not files or files[0].filename == "":
        raise HTTPException(status_code=400, detail="未检测到上传的文件。")

    workspace = _resolve_collection(collection_name)
    description = (description or "").strip()
    retrieval_hint = (retrieval_hint or "").strip()

    existing = await db_session.get(VectorCollection, workspace)
    if existing is not None and (existing.engine or "rag") != ENGINE_GRAPH:
        raise HTTPException(
            status_code=409,
            detail=(
                f"集合 {workspace!r} 已作为文档向量库（RAG）集合存在，"
                "不能再作为知识图谱集合，请换一个集合名。"
            ),
        )

    save_dir = _raw_upload_dir(workspace)
    os.makedirs(save_dir, exist_ok=True)
    saved_items: List[Tuple[str, str]] = []
    for upload in files:
        original_name = _safe_filename(upload.filename)
        ext = os.path.splitext(original_name)[-1].lower()
        if ext not in _ALLOWED_EXTS:
            for path, _ in saved_items:
                if os.path.exists(path):
                    os.remove(path)
            raise HTTPException(
                status_code=400,
                detail=f"文件 {original_name} 格式不支持。仅支持 .txt 和 .pdf",
            )
        unique_filename = f"{uuid.uuid4().hex}_{original_name}"
        file_path = os.path.join(save_dir, unique_filename)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
            saved_items.append((file_path, original_name))
        except Exception as exc:
            for path, _ in saved_items:
                if os.path.exists(path):
                    os.remove(path)
            raise HTTPException(status_code=500, detail=f"保存文件失败: {exc!s}") from exc
        finally:
            await upload.close()

    # 描述先登记（后台抽取完成后集合才会进入意图候选）
    if description or retrieval_hint:
        try:
            await upsert_vector_collection(
                db_session,
                workspace,
                description=description or None,
                retrieval_hint=retrieval_hint or None,
                engine=ENGINE_GRAPH,
            )
            await db_session.commit()
        except ValueError as exc:
            await db_session.rollback()
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    task_id = uuid.uuid4().hex
    retriever = getattr(request.app.state, "intent_vector_retriever", None)
    background_tasks.add_task(
        _bg_processing_pipeline, saved_items, workspace, retriever
    )

    return TaskResponse(
        status="processing",
        task_id=task_id,
        collection_name=workspace,
        message=f"已成功接收 {len(files)} 个文件，后端正在异步解析并构建图谱，请稍后查看。",
    )


# ======================================================================
# 图谱集合管理：列表 / 文件清单 / 按文件删除 / 清空遗留工作区
# ======================================================================
@router.get(
    "/knowledgebase/collections",
    response_model=List[GraphCollectionInfo],
)
async def list_graph_collections_api(
    db_session: AsyncSession = Depends(get_async_session),
) -> List[GraphCollectionInfo]:
    """列出全部知识图谱集合（LightRAG workspace）及每个集合内的文件。"""
    descriptor_rows = {
        row.name: row
        for row in await list_vector_collections(db_session)
        if (row.engine or "rag") == ENGINE_GRAPH
    }

    infos: List[GraphCollectionInfo] = []
    for name, legacy in _enumerate_graph_workspaces_fs():
        docs = _read_workspace_docs_fs(name, legacy=legacy)
        descriptor = descriptor_rows.get(name)
        infos.append(
            GraphCollectionInfo(
                name=LEGACY_WORKSPACE_ALIAS if legacy else name,
                legacy=legacy,
                description=descriptor.description if descriptor else None,
                retrieval_hint=descriptor.retrieval_hint
                if descriptor
                else None,
                document_count=len(docs),
                files=[_to_file_info(doc) for doc in docs],
            )
        )
    return infos


@router.get(
    "/knowledgebase/collections/{collection_name}/files",
    response_model=GraphCollectionInfo,
)
async def list_graph_collection_files_api(
    collection_name: str,
    db_session: AsyncSession = Depends(get_async_session),
) -> GraphCollectionInfo:
    """列出某个图谱集合内的全部文件（含 LightRAG 处理状态与切片数）。"""
    workspace, legacy = _resolve_collection_api(collection_name)

    known = {
        (LEGACY_WORKSPACE_ALIAS if item_legacy else item_name)
        for item_name, item_legacy in _enumerate_graph_workspaces_fs()
    }
    api_name = LEGACY_WORKSPACE_ALIAS if legacy else workspace
    if api_name not in known:
        raise HTTPException(
            status_code=404, detail=f"图谱集合不存在或为空：{api_name}"
        )

    docs = _read_workspace_docs_fs(workspace, legacy=legacy)
    descriptor = None
    if not legacy:
        descriptor = await db_session.get(VectorCollection, workspace)
    return GraphCollectionInfo(
        name=api_name,
        legacy=legacy,
        description=descriptor.description if descriptor else None,
        retrieval_hint=descriptor.retrieval_hint if descriptor else None,
        document_count=len(docs),
        files=[_to_file_info(doc) for doc in docs],
    )


@router.delete(
    "/knowledgebase/collections/{collection_name}/files/{filename:path}",
    response_model=GraphFileDeleteResponse,
)
async def delete_graph_collection_file_api(
    collection_name: str,
    filename: str,
    request: Request,
    db_session: AsyncSession = Depends(get_async_session),
) -> GraphFileDeleteResponse:
    """删除某图谱集合下指定文件的全部图谱数据（切片+实体/关系向量+图节点边）。

    同名文件若上传过多次，会删除匹配到的全部 doc。pipeline 正忙（抽取中）时
    返回 409，请稍后重试。
    """
    workspace, legacy = _resolve_collection_api(collection_name)
    filename = _safe_filename(filename)
    if legacy:
        raise HTTPException(
            status_code=403,
            detail=(
                "历史遗留工作区不支持按文件删除，请使用 "
                "POST /knowledgebase/maintenance/clear-legacy-workspace 整体清空。"
            ),
        )

    # 集合不存在先 404（纯文件系统判断，避免拉起 torch 并误建空 workspace）
    known_names = {
        item_name for item_name, item_legacy in _enumerate_graph_workspaces_fs()
        if not item_legacy
    }
    if workspace not in known_names:
        raise HTTPException(
            status_code=404,
            detail=f"图谱集合不存在或为空：{workspace}",
        )

    helpers = _lazy_light_rag()
    try:
        outcome = await helpers["delete_documents_by_filename"](
            workspace, filename
        )
    except helpers["GraphPipelineBusyError"] as exc:
        raise HTTPException(
            status_code=409,
            detail=f"知识图谱流水线正忙（抽取/删除进行中），请稍后重试：{exc!s}",
        ) from exc
    except Exception as exc:
        logger.exception("LightRAG 文档删除失败")
        raise HTTPException(status_code=500, detail=f"图谱删除失败: {exc!s}") from exc

    if int(outcome.get("deleted", 0)) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"图谱集合 {workspace!r} 下未找到文件 {filename!r}",
        )

    # 原始文件尽力清理（新路径 + 旧版单文件上传路径）
    for candidate in (
        os.path.join(_raw_upload_dir(workspace), filename),
        os.path.join(UPLOAD_DIR, filename),
    ):
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError as exc:
            logger.warning("原始上传文件删除失败 {}：{}", candidate, exc)

    await _refresh_graph_routing(request, db_session)

    return GraphFileDeleteResponse(
        collection=workspace,
        filename=filename,
        deleted_documents=int(outcome.get("deleted", 0)),
        details=list(outcome.get("results", [])),
        message="删除成功（切片、向量与图谱关联数据已级联清理）",
    )


@router.post(
    "/knowledgebase/maintenance/clear-legacy-workspace",
    response_model=LegacyClearResponse,
)
async def clear_legacy_workspace_api(
    request: Request,
    db_session: AsyncSession = Depends(get_async_session),
) -> LegacyClearResponse:
    """清空历史遗留空 workspace（升级前散落在工作目录根下的图谱数据）。

    新数据全部在具名 workspace（默认 default）中，不受影响。
    """
    helpers = _lazy_light_rag()
    try:
        outcome = await helpers["clear_workspace"]("", legacy=True)
    except Exception as exc:
        logger.exception("清空遗留工作区失败")
        raise HTTPException(status_code=500, detail=f"清空失败: {exc!s}") from exc

    await _refresh_graph_routing(request, db_session)
    return LegacyClearResponse(
        status="success",
        dropped=list(outcome.get("dropped", [])),
        message="历史遗留工作区已清空。",
    )
