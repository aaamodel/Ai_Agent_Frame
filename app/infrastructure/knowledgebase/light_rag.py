import os
import re
import asyncio
import threading
from typing import Any, Dict, List, Optional

from lightrag.utils import EmbeddingFunc
from pypdf import PdfReader
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from langchain_core.tools import tool
from loguru import logger

from app.config import get_settings
from app.llm_model_router.async_model_executor import run_with_attempt_budget
from app.llm_model_router.async_openai_caller import apply_thinking_dialect
from app.llm_model_router.tier_params import read_tier_params

# ==============================================================================
# 1. 核心环境与阿里百炼参数配置
# ==============================================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

light_rag_settings = get_settings()

# ==============================================================================
# 1.1 档位参数：与路由链路读**同一份**配置（模型名 / 单次预算 / 重试次数 / 思考开关）
# ==============================================================================
_LIGHTRAG_TIER: str = "standard"
"""知识图谱抽取使用的档位：与路由侧"STANDARD 主用、FAST 其次"的降级顺序一致。"""

_tier_params = read_tier_params(light_rag_settings, _LIGHTRAG_TIER)

# ⚠️ LightRAG 的 ``openai_complete_if_cache`` 上挂了自带的
# ``@retry(stop_after_attempt(3), wait=wait_exponential(min=4, max=10))``。
# 我们的重试策略由**档位配置**统一决定；若与它那层叠加，会出现"我们的 N 次 ×
# 它的 3 次"外加 4~10s 指数退避，单次调用最坏耗时被放大到不可接受。
# tenacity 经 functools.wraps 保留了 ``__wrapped__``，因此取其未装饰原函数，
# 由 ``run_with_attempt_budget`` 统一施放"单次预算 + 档位重试次数"。
_complete_raw = getattr(openai_complete_if_cache, "__wrapped__", None)
if _complete_raw is None:  # pragma: no cover - 依赖第三方实现细节，留兜底与告警
    logger.warning(
        "lightrag 的 openai_complete_if_cache 未暴露 __wrapped__，其内置重试（3 次）"
        "将与档位重试叠加；如观测到重试风暴，请下调 LLM_TIER_STANDARD_RETRIES",
    )
    _complete_raw = openai_complete_if_cache


def _resolve_lightrag_model(settings) -> str:
    """从 tier 候选池里取**单个**模型名给 LightRAG（它只接受单模型）。

    ⚠️ 必须读 ``*_parsed``（config.py 已按逗号 / JSON 数组拆好的 list），
    **不能**读原始 ``llm_tier_*`` 字符串字段：那是**候选池**，
    如 ``LLM_TIER_STANDARD=qwen3.8-flash,glm-4.7``，整串当模型名发给服务端就会报
    ``404 model_not_found: The model `qwen3.8-flash,glm-4.7` does not exist``
    （症状出现在图谱抽取的 extract LLM 阶段）。

    ⚠️ 历史实现自己判断"STANDARD 为主用、其次 FAST"，与路由侧是**两份口径**；
    现已改为委托 :func:`read_tier_params`（唯一入口）——模型名 / 单次预算 /
    重试次数 / 思考开关一并同源。
    """
    # 改为委托**唯一入口**读取：模型名 / 单次预算 / 重试次数 / 思考开关同源。
    # 历史实现只从候选池"借"了模型名，其余参数各自硬编码（超时用它自己的全局
    # 默认、思考开关写死百炼方言、凭据写死百炼），导致同一档模型在不同链路上行为不一致。
    return read_tier_params(settings, _LIGHTRAG_TIER).model


llm_model: str = _resolve_lightrag_model(light_rag_settings)
logger.info("LightRAG 使用模型：{}", llm_model)
EMBEDDING_MODEL = "text-embedding-v3"
# 百炼 text-embedding-v3 支持 dimensions 参数，这里显式锁定 1024。
EMBEDDING_DIM = 1024

# 💡 优化：确保这些目录无论在哪个路径下启动 FastAPI 都能正确对应
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(BASE_DIR, "raw_data", "lightrag_workspace")

# 确保 RAG 工作目录存在
os.makedirs(WORKING_DIR, exist_ok=True)

# ==============================================================================
# 2. 适配百炼 Qwen 的 LightRAG 核心基础函数
# ==============================================================================
async def qwen_llm_complete(
        prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs
) -> str:
    # ⚠️ 1.5.4 的调用约定：框架以 system_prompt=... 关键字传入（旧版参数名
    # lightrag_system_prompt 已废弃）。若沿用旧名，system_prompt 会落入
    # **kwargs，与下面显式传的同名参数冲突，实体抽取阶段直接报
    # 「got multiple values for keyword argument 'system_prompt'」。
    if history_messages is None:
        history_messages = []

    # 思考开关按**候选厂商方言**翻译（原先写死 enable_thinking，只对百炼成立；
    # 档位首候选若换成智谱等模型，那个参数名本身就是错的）。
    extra_body: Dict[str, Any] = {}
    apply_thinking_dialect(_tier_params.candidate, _tier_params.thinking, extra_body)

    call_kwargs: Dict[str, Any] = dict(kwargs)
    if extra_body:
        call_kwargs["extra_body"] = extra_body

    def _one_attempt() -> Any:
        """**一次**尝试：携带该档的单次预算（HTTP 层），重试由外层统一施放。"""
        return _complete_raw(
            llm_model,
            prompt,
            system_prompt=system_prompt,  # 这里透传框架传入的提示词
            history_messages=history_messages,
            api_key=_tier_params.candidate.api_key or DASHSCOPE_API_KEY,
            base_url=_tier_params.candidate.url or DASHSCOPE_BASE_URL,
            timeout=int(_tier_params.timeout_s) if _tier_params.timeout_s else None,
            **call_kwargs,
        )

    result, error = await run_with_attempt_budget(
        _one_attempt,
        timeout_s=_tier_params.timeout_s,
        retries=_tier_params.retries,
        budget_label=(
            f"{int(_tier_params.timeout_s * 1000)}ms(tier={_tier_params.tier})"
            if _tier_params.timeout_s
            else f"<no-timeout>(tier={_tier_params.tier})"
        ),
        subject=f"lightrag:{_tier_params.model}",
    )
    if result is None:
        if isinstance(error, BaseException):
            raise error
        raise RuntimeError("LightRAG 模型调用失败")
    return result

async def qwen_embedding(texts: list[str]) -> list[list[float]]:
    # ⚠️ 关键：必须调用 openai_embed.func（未装饰原函数），不能直接调 openai_embed。
    # LightRAG 1.5.4 用 @wrap_embedding_func_with_attrs(embedding_dim=1536) 把
    # openai_embed 装饰成了 EmbeddingFunc 实例：直接调用它会按 1536 维校验百炼
    # 实际返回的 1024 维向量（total_elements % 1536 != 0 直接报错），且
    # send_dimensions=False 时连 dimensions 参数都不会发给百炼——这正是历史
    # 「Embedding dimension mismatch ... 10240 / 1536」故障的根因。
    # .func 是官方文档指定的未装饰入口（见 lightrag/utils.py 装饰器 docstring），
    # 显式传 embedding_dim 后原函数会向百炼发送 dimensions=1024。
    return await openai_embed.func(
        texts,
        model=EMBEDDING_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
        embedding_dim=EMBEDDING_DIM,
    )

# 1. 包装百炼的向量函数
wrapped_embedding = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    func=qwen_embedding,
    max_token_size=2048,
    supports_asymmetric=False
)

# 2. 初始化 LightRAG 实例（供外部全局调用）
# ==============================================================================
# LightRAG 的「集合」= workspace：不同 workspace 在 WORKING_DIR 下使用各自独立
# 子目录（kv_store_*.json / vdb_*.json / graph_*.graphml / doc_status），天然隔离。
# 新上传的数据一律走具名 workspace（默认 DEFAULT_GRAPH_WORKSPACE）；空 workspace
# （""，文件直接散落在 WORKING_DIR 根下）只用于只读访问与清理历史遗留数据。
DEFAULT_GRAPH_WORKSPACE = "default"
LEGACY_GRAPH_WORKSPACE = ""  # 历史遗留：升级前全局单例使用的空 workspace

_WORKSPACE_NAME_RE = re.compile(r"^[\w\u4e00-\u9fff][\w.\u4e00-\u9fff\-]{0,127}$")


def validate_graph_workspace_name(name: str) -> str:
    """校验图谱集合（workspace）名，规则与 RAG 逻辑集合名保持一致。"""
    name = (name or "").strip()
    if name and not _WORKSPACE_NAME_RE.match(name):
        raise ValueError(
            "集合名仅允许中英文、数字、下划线、中划线与点，长度 1-128；"
            f"收到：{name!r}"
        )
    return name


def _build_lightrag(workspace: str) -> LightRAG:
    return LightRAG(
        working_dir=WORKING_DIR,
        workspace=workspace,
        llm_model_func=qwen_llm_complete,
        llm_model_name=llm_model,
        embedding_func=wrapped_embedding,
    )


# 默认单例：保持 rag_instance 导出名不变（graph_search 等旧代码继续可用），
# 但其 workspace 从空串改为具名默认集合。
rag_instance = _build_lightrag(DEFAULT_GRAPH_WORKSPACE)

_instances: Dict[str, LightRAG] = {DEFAULT_GRAPH_WORKSPACE: rag_instance}
_instances_lock = threading.Lock()


def get_lightrag(workspace: Optional[str] = None) -> LightRAG:
    """按 workspace 名取 LightRAG 实例（进程内缓存，懒创建）。

    None / 空串归一到默认集合；历史遗留空 workspace 只能通过
    ``get_legacy_lightrag()`` 显式获取，避免新数据误写进遗留目录。
    """
    name = validate_graph_workspace_name(workspace or DEFAULT_GRAPH_WORKSPACE)
    if not name:
        name = DEFAULT_GRAPH_WORKSPACE
    instance = _instances.get(name)
    if instance is not None:
        return instance
    with _instances_lock:
        instance = _instances.get(name)
        if instance is None:
            instance = _build_lightrag(name)
            _instances[name] = instance
    return instance


def get_legacy_lightrag() -> LightRAG:
    """获取历史遗留空 workspace 实例（只读列举/清空用）。"""
    instance = _instances.get(LEGACY_GRAPH_WORKSPACE)
    if instance is None:
        with _instances_lock:
            instance = _instances.get(LEGACY_GRAPH_WORKSPACE)
            if instance is None:
                instance = _build_lightrag(LEGACY_GRAPH_WORKSPACE)
                _instances[LEGACY_GRAPH_WORKSPACE] = instance
    return instance


# ==============================================================================
# 2.5 多 workspace（集合）管理：枚举 / 入库（带文件名）/ 删除 / 清空遗留
# ==============================================================================
# LightRAG 各层存储文件名前缀（用于判断遗留空 workspace 是否有数据）
_LEGACY_FILE_PREFIXES = ("kv_store_", "vdb_")


def _legacy_workspace_has_data() -> bool:
    """遗留空 workspace 的数据直接散落在 WORKING_DIR 根下（无子目录）。"""
    try:
        for entry in os.listdir(WORKING_DIR):
            if not os.path.isfile(os.path.join(WORKING_DIR, entry)):
                continue
            if entry.endswith(".graphml") or entry.startswith(_LEGACY_FILE_PREFIXES):
                return True
    except FileNotFoundError:
        return False
    return False


def enumerate_graph_workspaces() -> List[Dict[str, Any]]:
    """枚举磁盘上存在的图谱 workspace。

    Returns:
        ``[{"name": str, "legacy": bool}]``；具名集合对应 WORKING_DIR 下的子目录，
        遗留空 workspace 仅在根目录确有数据文件时以 ``name=""`` 追加。
    """
    out: List[Dict[str, Any]] = []
    try:
        entries = sorted(os.listdir(WORKING_DIR))
    except FileNotFoundError:
        entries = []
    for entry in entries:
        full = os.path.join(WORKING_DIR, entry)
        if not os.path.isdir(full):
            continue
        # workspace 子目录内至少要有一个 lightrag 数据文件才算数
        try:
            has_data = any(
                f.endswith((".graphml", ".json"))
                for f in os.listdir(full)
                if os.path.isfile(os.path.join(full, f))
            )
        except OSError:
            has_data = False
        if has_data:
            out.append({"name": entry, "legacy": False})
    if _legacy_workspace_has_data():
        out.append({"name": LEGACY_GRAPH_WORKSPACE, "legacy": True})
    return out


async def list_workspace_documents(
    workspace: str,
    *,
    legacy: bool = False,
    page_size: int = 200,
) -> List[Dict[str, Any]]:
    """列出某 workspace 内已登记的全部文档（含 failed/pending，供管理面展示）。"""
    rag = get_legacy_lightrag() if legacy else get_lightrag(workspace)
    await rag.initialize_storages()

    docs: List[Dict[str, Any]] = []
    page = 1
    while True:
        rows, total = await rag.doc_status.get_docs_paginated(
            status_filters=None,
            page=page,
            page_size=page_size,
            sort_field="created_at",
            sort_direction="desc",
        )
        for doc_id, status in rows:
            docs.append(
                {
                    "doc_id": str(doc_id),
                    "filename": str(getattr(status, "file_path", "") or ""),
                    "status": str(getattr(status, "status", "") or ""),
                    "chunks_count": getattr(status, "chunks_count", None),
                    "content_length": getattr(status, "content_length", None),
                    "track_id": getattr(status, "track_id", None),
                    "created_at": getattr(status, "created_at", None),
                    "updated_at": getattr(status, "updated_at", None),
                    "error_msg": getattr(status, "error_msg", None),
                }
            )
        if len(docs) >= int(total or 0) or not rows:
            break
        page += 1
    return docs


async def insert_document(workspace: str, filename: str, content: str) -> str:
    """把单份文档文本织入指定 workspace，并登记真实文件名。

    Returns:
        LightRAG track_id（文档本身的 doc_id 可通过文件列表按文件名反查）。
    """
    name = validate_graph_workspace_name(workspace or DEFAULT_GRAPH_WORKSPACE)
    if not name:
        name = DEFAULT_GRAPH_WORKSPACE
    rag = get_lightrag(name)
    await rag.initialize_storages()
    track_id = (
        f"upload_{asyncio.get_event_loop().time():.0f}_"
        f"{os.urandom(4).hex()}"
    )
    return await rag.ainsert(
        [content],
        file_paths=[filename],
        track_id=track_id,
    )


async def find_doc_ids_by_filename(
    workspace: str, filename: str, *, legacy: bool = False
) -> List[str]:
    """在 workspace 内按 file_path 精确匹配 doc_id（同名文件可能有多条）。"""
    docs = await list_workspace_documents(
        workspace, legacy=legacy
    )
    return [doc["doc_id"] for doc in docs if doc["filename"] == filename]


class GraphPipelineBusyError(RuntimeError):
    """LightRAG 文档 pipeline 正忙（上传/抽取进行中），删除被官方并发控制拒绝。"""


async def delete_documents_by_filename(
    workspace: str, filename: str, *, legacy: bool = False
) -> Dict[str, Any]:
    """删除某 workspace 下指定文件对应的全部文档：切片 + 三类向量 + 图谱级联。

    LightRAG ``adelete_by_doc_id`` 官方保证四层清理；多个文档共享的实体/关系会
    更新 source_id，必要时用 LLM 缓存重建部分实体。

    Raises:
        GraphPipelineBusyError: pipeline 忙（官方返回 not_allowed）。
    """
    rag = get_legacy_lightrag() if legacy else get_lightrag(workspace)
    await rag.initialize_storages()

    doc_ids = await find_doc_ids_by_filename(
        workspace, filename, legacy=legacy
    )
    results: List[Dict[str, Any]] = []
    for doc_id in doc_ids:
        result = await rag.adelete_by_doc_id(doc_id)
        item = {
            "doc_id": doc_id,
            "status": str(getattr(result, "status", "")),
            "message": str(getattr(result, "message", "")),
            "status_code": int(getattr(result, "status_code", 0) or 0),
        }
        if item["status"] == "not_allowed":
            raise GraphPipelineBusyError(item["message"])
        results.append(item)
    return {"filename": filename, "deleted": len(doc_ids), "results": results}


# 清空 workspace 时需要 drop 的全部存储属性（LLM 缓存保留：跨集合复用且无害）
_DROP_STORAGE_ATTRS = (
    "doc_status",
    "full_docs",
    "full_entities",
    "full_relations",
    "entity_chunks",
    "relation_chunks",
    "chunk_entity_relation_graph",
    "chunks_vdb",
    "entities_vdb",
    "relationships_vdb",
)


async def clear_workspace(workspace: str, *, legacy: bool = False) -> Dict[str, Any]:
    """清空整个 workspace（运维操作）：drop 全部知识数据存储。"""
    rag = get_legacy_lightrag() if legacy else get_lightrag(workspace)
    await rag.initialize_storages()
    dropped: List[str] = []
    for attr in _DROP_STORAGE_ATTRS:
        storage = getattr(rag, attr, None)
        if storage is None:
            continue
        try:
            outcome = await storage.drop()
            dropped.append(f"{attr}:{outcome}")
        except Exception as exc:  # noqa: BLE001 - 单层失败不阻塞其他层
            dropped.append(f"{attr}:error:{exc}")
    return {"workspace": workspace, "legacy": legacy, "dropped": dropped}


# ==============================================================================
# 3. 数据处理流：将本地的 PDF / TXT 转化为纯文本并入库
# ==============================================================================
def parse_local_file(file_path: str) -> str:
    """提取本地单份 PDF 或 TXT 文件的文本内容"""
    ext = os.path.splitext(file_path)[-1].lower()
    knowledgebase_text = ""

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            knowledgebase_text = f.read()
    elif ext == ".pdf":
        pdf_reader = PdfReader(file_path)
        extracted_pages = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)
        knowledgebase_text = "\n".join(extracted_pages)

    return knowledgebase_text.strip()

async def ingest_data_pipeline(data_directory: str):
    """
    【清洗与建库】扫描目标文件夹，提取文本并批量织入 LightRAG 知识图谱
    项目初始化（冷启动）：比如你手头已经有 100 份业务文档，你可以直接把它们丢进 ./raw_data 目录，然后直接在命令行运行 python light_rag.py。它会调用 ingest_data_pipeline 帮你把基础知识库一次性建好。

    做一个“同步历史文件”的专用接口：如果你希望在网页上提供一个“同步本地文件夹”的按钮，你可以专门为它写一个新的 Router 接口。代码长这样：

    """
    if not os.path.exists(data_directory):
        os.makedirs(data_directory, exist_ok=True)
        print(f"ℹ️ 自动创建了原始数据存放目录 '{data_directory}'")
        return

    all_texts = []
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.lower().endswith(('.pdf', '.txt')):
                file_path = os.path.join(root, file)
                print(f"📂 正在解析本地文件: {file_path}")
                content = parse_local_file(file_path)
                if content:
                    all_texts.append(content)

    if all_texts:
        print(f"🧠 开始向 LightRAG 注入数据并提取图谱关系（共 {len(all_texts)} 份文档）...")
        await rag_instance.initialize_storages()
        await rag_instance.ainsert(all_texts)
        print("✅ 知识图谱构建完成！")
    else:
        print("⚠️ 未在目标目录下找到有效的 PDF 或 TXT 文件。")



# ==============================================================================
# 5. 统一调度执行流程（仅在直接运行此文件时生效，不影响 FastAPI）
# ==============================================================================
async def main():
    # 保持本地测试目录和全局一致
    raw_data_dir = os.path.join(BASE_DIR, "raw_data")
    await ingest_data_pipeline(raw_data_dir)

if __name__ == "__main__":

    asyncio.run(main())