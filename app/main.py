# -*- coding: utf-8 -*-
"""FastAPI 应用入口：优化后的生命周期管理（支持并行初始化与耗时打点追踪）"""
import sys
import time
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from typing import Any, Dict, List

from app.llm_model_router.model_router import ModelRouter
from app.llm_model_router.model_router_config import AIModelProperties

# 导入期计时起点：放在所有重型依赖导入之前，用于量化 import 阶段耗时
_IMPORT_T0 = time.perf_counter()

from fastapi import FastAPI
from loguru import logger
import redis.asyncio as aioredis

from app.api.routes import chat, document, health, kownledgebase
from app.config import get_settings
from app.core.rag.rag_service import RAGService
from app.infrastructure.database.session import configure_session, init_engine
from app.infrastructure.database.models import Base
from app.core.memory.short_term import QwenChatLLMImpl
from app.core.memory.long_term import MilvusCollectionWrapper, QwenEmbeddingImpl

from app.infrastructure.embeddings.dashscope_embedding import DashScopeEmbedding
from app.infrastructure.trace.langfuse import flush_langfuse, setup_langfuse
from app.core.backends.filesystem import FilesystemBackend
from app.core.skill.manager import SkillManager

load_dotenv()


def _build_global_router() -> ModelRouter:
    """构建全局唯一 ModelRouter（单例）。

    初始化入口（两个层级）：
      ① 单模型兼容（只配 OPENAI_LLM_MODEL）：3 个 tier 都复用这个模型，
         超时走 LLM_TIER_*_TIMEOUT_MS（默认 FAST 15s / STANDARD 30s / DEEP 60s）。
      ② 多模型差异化（配 LLM_MODELS + LLM_TIER_*）：每路模型独立
         model_id / api_key / base_url / priority / supports_thinking；
         3 个 tier 独立设置候选池（按顺序降级）和各自超时。

    统一构造 AIModelProperties 分层配置后交给 ModelRouter。启动期由
    ChatTierConfigValidator 做 Fail-Fast 校验：3 个 Tier 枚举齐全、
    每个 tier 的 candidates 非空且 timeout>0、DEEP 至少 1 个 thinking 候选。
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeWarning("未配置 OPENAI_API_KEY，大模型路由功能将受限")

    # ------------------------------------------------------------------
    # 1) 解析模型候选：优先 LLM_MODELS（多模型），否则单模型（OPENAI_*）
    # ------------------------------------------------------------------
    providers: Dict[str, Any] = {}
    candidates: List[Dict[str, Any]] = []
    all_ids: List[str] = []
    parsed_llm_models = settings.llm_models_parsed  # 解析后 list[dict]
    if parsed_llm_models:
        for raw in parsed_llm_models:
            mid: str = str(raw.get("model_id") or "").strip()
            if not mid:
                raise ValueError(f"LLM_MODELS 条目缺少 model_id: {raw!r}")
            key = str(raw.get("api_key") or settings.openai_api_key).strip() or settings.openai_api_key
            base = str(raw.get("base_url") or settings.openai_api_base).strip() or settings.openai_api_base
            provider_key = str(raw.get("provider") or "openai").lower()
            supports_thinking = bool(raw.get("supports_thinking", True))
            # Provider 只写一次（多个模型共享）；单模型独立网关通过 candidate.url 覆盖
            providers.setdefault(provider_key, {
                "url": base or None,
                "api_key": key,
                "endpoints": {},
            })
            candidates.append({
                "id": mid,
                "provider": provider_key,
                "model": mid,
                "url": base or None,
                "priority": int(raw.get("priority", 0)),
                "enabled": True,
                "supports_thinking": supports_thinking,
            })
            all_ids.append(mid)
    else:
        # 单模型兜底
        mid = settings.openai_llm_model
        base = settings.openai_api_base or None
        providers = {"openai": {"url": base, "api_key": settings.openai_api_key, "endpoints": {}}}
        candidates = [{
            "id": mid,
            "provider": "openai",
            "model": mid,
            "url": base,
            "priority": 0,
            "enabled": True,
            "supports_thinking": True,
        }]
        all_ids = [mid]

    # 默认候选顺序：按 priority 升序（数字小优先）+ 原输入稳定顺序
    default_sorted_ids: List[str] = [
        m for _, m in sorted((c["priority"], c["id"]) for c in candidates)
    ]

    def _tier_candidates(parsed_ids: List[str]) -> List[str]:
        # 某 tier 未显式配置候选池时，fallback 为全部模型（按 priority 排序）
        return list(parsed_ids) or default_sorted_ids

    # ------------------------------------------------------------------
    # 2) 组装 AIModelProperties 分层配置
    # ------------------------------------------------------------------
    properties_data: Dict[str, Any] = {
        "providers": providers,
        "chat": {
            "default_model": model if (model := all_ids[0]) else None,
            "candidates": candidates,
            "default_tier": "standard",
            "deep_thinking_tier": "deep",
            "tiers": {
                "fast": {
                    "candidates": _tier_candidates(settings.llm_tier_fast_parsed),
                    "timeout_ms": settings.llm_tier_fast_timeout_ms,
                },
                "standard": {
                    "candidates": _tier_candidates(settings.llm_tier_standard_parsed),
                    "timeout_ms": settings.llm_tier_standard_timeout_ms,
                },
                "deep": {
                    "candidates": _tier_candidates(settings.llm_tier_deep_parsed),
                    "timeout_ms": settings.llm_tier_deep_timeout_ms,
                },
            },
        },
        "selection": {"failure_threshold": 5, "open_duration_ms": 60_000},
        "stream": {"message_chunk_size": 5},
    }
    properties = AIModelProperties.from_dict(properties_data)

    tiers = properties_data["chat"]["tiers"]
    logger.info(
        "构建全局 ModelRouter：注册 {} 个模型，tier=[FAST:{}/STANDARD:{}/DEEP:{}]，"
        "tier_timeout=[FAST:{}ms/STANDARD:{}ms/DEEP:{}ms]",
        len(all_ids),
        ",".join(tiers["fast"]["candidates"]),
        ",".join(tiers["standard"]["candidates"]),
        ",".join(tiers["deep"]["candidates"]),
        tiers["fast"]["timeout_ms"],
        tiers["standard"]["timeout_ms"],
        tiers["deep"]["timeout_ms"],
    )

    return ModelRouter(properties)


@asynccontextmanager
async def lifespan(app: FastAPI):
    total_start_time = time.perf_counter()
    settings = get_settings()
    print(settings.model_config)

    # ==========================================
    # 1. 基础客户端单例构建 (纯内存操作，同步执行)
    # ==========================================
    t0 = time.perf_counter()

    redis_client = aioredis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=True
    )

    # ========================================================================
    # 🌟【统一路由 · repo_map B1+B2】先构建全局 ModelRouter，再把它注入到
    #   所有 Chat LLM 适配器中，实现一个单例 + 一套配置 + 一套熔断降级 管
    #   所有 Chat 大模型调用（Orchestrator 3 场景 + 短期记忆摘要 + RAG LLM）。
    #   Embedding 继续保持独立固定单模型（符合用户要求，不进路由）。
    # ========================================================================
    global_model_router = _build_global_router()

    # B1 改造：QwenChatLLMImpl 接受 ModelRouter（不再直接传 api_key/model）
    qwen_chat_llm = QwenChatLLMImpl(model_router=global_model_router)

    # 修正：使用 settings 中的配置，避免硬编码 127.0.0.1 导致网络超时
    milvus_wrapper = MilvusCollectionWrapper(
        collection_name="agent_long_term_memory_v2",
        dim=1024,
        host=settings.milvus_host,
        port=str(settings.milvus_port)
    )
    # Embedding（E1）：固定单一模型 text-embedding-v3，不接入 ModelRouter
    qwen_embed_model = QwenEmbeddingImpl(api_key=settings.openai_api_key, model="text-embedding-v3")

    # ========================================================================
    # RAG（LlamaIndex 重构）：
    #   DashScopeEmbedding(BaseEmbedding) → MilvusIndexManager → HybridRetriever
    #   通过 RAGService 门面持有；利用 OpenAI 兼容协议对接 text-embedding-v3。
    # ========================================================================
    rag_embed_model = DashScopeEmbedding(
        model="text-embedding-v3",
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
    )
    rag_service = RAGService(
        embed_model=rag_embed_model,
        milvus_uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
        collection_name=settings.milvus_kb_collection_name,
        overwrite=settings.milvus_kb_overwrite,  # 一次性迁移重建后请改为 False
        dim=1024,
        top_k_default=10,
    )

    fs_backend = FilesystemBackend(virtual_mode=True)
    # 🎯【诊断】打印运行时工作目录与虚拟文件系统根，用于排查 skill/raw_data 相对路径错位
    logger.info(
        "[路径诊断] 进程 cwd={} | fs_backend 根={} | skills 目录={} | skills 是否存在={}",
        __import__("os").getcwd(),
        fs_backend.cwd,
        str((fs_backend.cwd / "skills").resolve()),
        (fs_backend.cwd / "skills").exists(),
    )
    # skills 真实目录是项目根下的 app/skills（虚拟模式禁止 ".." 相对路径，
    # 旧的 "../skills" 每次都会命中 invalid_path，这里改为根内相对路径）
    skill_manager = SkillManager(fs_backend=fs_backend, skills_dir="./skills")

    # ========================================================================
    # 🌟【⑬-B 新增】Agent Pipeline Prompt 模板加载器单例 + 预热
    # ------------------------------------------------------------------------
    # 让 get_pipeline() 依赖可直接读取 request.app.state.prompt_loader，
    # 避免每请求都做一次磁盘 .st 文件 IO。
    # 预热失败不阻塞启动（get_pipeline 有懒预热兜底）。
    # ========================================================================
    from app.query_intent.intent_prompt.prompt_template_loader import (
        PromptTemplateLoader,
    )
    from app.query_intent.rag_constant import (
        AGENT_QUESTION_REWRITE_PROMPT_PATH,
        AGENT_REWRITE_INTENT_COMBINED_PROMPT_PATH,
        INTENT_CLASSIFIER_PROMPT_PATH,
    )
    prompt_loader = PromptTemplateLoader()
    try:
        prompt_loader.load(INTENT_CLASSIFIER_PROMPT_PATH)
        prompt_loader.load(AGENT_QUESTION_REWRITE_PROMPT_PATH)
        prompt_loader.load(AGENT_REWRITE_INTENT_COMBINED_PROMPT_PATH)
        logger.info("  └─ Agent Pipeline Prompt 模板预热完成（3 个模板）")
    except Exception as prompt_err:
        logger.warning(
            "  └─ ⚠️  Agent Pipeline Prompt 预热失败（不阻塞启动，后续懒加载）：{}",
            prompt_err,
        )

    # 挂载基础对象到 app.state
    app.state.redis = redis_client
    app.state.qwen_chat_llm = qwen_chat_llm
    app.state.milvus_wrapper = milvus_wrapper
    app.state.embedding_model = qwen_embed_model
    app.state.fs_backend = fs_backend
    app.state.skill_manager = skill_manager
    app.state.prompt_loader = prompt_loader  # 🌟【⑬-B 新增】Pipeline Prompt 单例
    app.state.rag_service = rag_service
    # 注意：复用 lifespan 开头已构建的同一个 global_model_router 单例，
    # 不要重新调用 _build_global_router()，否则会产生两套独立的熔断器。
    app.state.model_router = global_model_router

    # RAG BM25 内存索引可在后台预热（从既有集合拉取上轮上传的切片文本）。
    asyncio.create_task(rag_service.seed_bm25())

    logger.info(f"⏱️  本地基础对象初始化完成，耗时: {time.perf_counter() - t0:.3f}s")

    # ==========================================
    # 2. 数据库引擎初始化 (PostgreSQL)
    # ==========================================
    t_db = time.perf_counter()
    engine = init_engine(settings.database_url)
    configure_session(engine)
    app.state.engine = engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info(f"⏱️  PostgreSQL 数据库建表初始化完成，耗时: {time.perf_counter() - t_db:.3f}s")

    # ==========================================
    # 3. 并行执行异步网络与文件 IO 初始化
    # ==========================================
    t_async = time.perf_counter()

    async def init_skills():
        t = time.perf_counter()
        try:
            await skill_manager.scan_and_refresh_skills()
            logger.info(
                f"  └─ 高级技能树扫描完成({len(skill_manager.state.available_skills)}个)，耗时: {time.perf_counter() - t:.3f}s"
            )
        except Exception as sk_err:
            logger.error(f"  └─ ❌ 技能树扫描异常: {sk_err}")

    async def init_redis_ping():
        t = time.perf_counter()
        await redis_client.ping()
        logger.info(f"  └─ Redis 连通性测试完成，耗时: {time.perf_counter() - t:.3f}s")

    # 使用 asyncio.gather 并发执行
    await asyncio.gather(
        init_skills(),
        init_redis_ping(),
    )
    logger.info(f"⏱️  所有异步依赖并行初始化完成，总耗时: {time.perf_counter() - t_async:.3f}s")

    # Langfuse 可观测性初始化（须在首个 LLM 调用前完成；未配置则安全关闭）
    setup_langfuse()

    logger.info(f"🚀 系统全部基础设施就绪，启动总耗时: {time.perf_counter() - total_start_time:.3f}s")

    # ==========================================
    # 3.5 【启动后台预热】意图树 + 向量索引
    # ---------------------------------------------------------------
    # 原先 IntentTreeVectorRetriever 在 get_pipeline 里每请求新建，且懒加载时
    # 对全部叶子节点重新 embedding（DB 加载 + 全量 embedding 可达 ~28s），会
    # 压住首请求。这里在启动后台线程预热一次并缓存为单例，get_pipeline 直接复用。
    # 预热失败不阻塞启动（get_pipeline 有懒加载兜底）。
    # ==========================================
    app.state.intent_vector_retriever = None

    def _preheat_intent_vector_index() -> None:
        try:
            from app.query_intent.intent_classify_resolver.intent_classify import (
                DefaultIntentClassifier,
            )
            from app.query_intent.intent_classify_resolver.intent_vector_retriever import (
                IntentTreeVectorRetriever,
            )
            from app.query_intent.rag_constant import INTENT_VECTOR_TOP_K
            from app.api.depends.dependencies import _AppModelRouterIntentLLMAdapter

            tree_provider = DefaultIntentClassifier(
                llm_service=_AppModelRouterIntentLLMAdapter(
                    model_router=global_model_router
                ),
                intent_node_mapper=None,
                prompt_template_loader=app.state.prompt_loader,
            )
            retriever = IntentTreeVectorRetriever(
                embed_model=app.state.embedding_model,
                tree_provider=tree_provider,
                top_k=INTENT_VECTOR_TOP_K,
            )
            retriever._ensure_index()
            app.state.intent_vector_retriever = retriever
            logger.info("🔆 意图树 + 向量索引后台预热完成（首请求不再阻塞）。")
        except Exception as preheat_error:  # pragma: no cover - 防御性兜底
            logger.warning(
                "意图树 + 向量索引后台预热失败（不影响启动，首次请求懒加载兜底）: {}",
                preheat_error,
            )

    asyncio.create_task(asyncio.to_thread(_preheat_intent_vector_index))

    yield

    # ==========================================
    # 4. 资源清理
    # ==========================================
    flush_langfuse()  # 冲刷 Langfuse 观测缓冲（已启用时才真正上报）
    await redis_client.close()
    await engine.dispose()
    logger.info("🛑 所有资源已安全关闭，服务顺利下线")


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.include_router(health.router, prefix=settings.api_prefix)
    application.include_router(chat.router, prefix=settings.api_prefix)
    application.include_router(document.router, prefix=settings.api_prefix)
    application.include_router(kownledgebase.router, prefix=settings.api_prefix)
    return application


logger.remove()
logger.add(sys.stderr, level="INFO", backtrace=True, diagnose=False)

app = create_app()
logger.info(
    "⏱️  app.main 模块导入+FastAPI 实例构建总耗时: {:.3f}s",
    time.perf_counter() - _IMPORT_T0,
)