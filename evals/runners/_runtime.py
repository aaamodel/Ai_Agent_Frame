# -*- coding: utf-8 -*-
"""评测运行时装配（不依赖 FastAPI 的 lifespan）。

为什么需要它：``app.main.lifespan`` 把全局单例挂在 ``app.state`` 上，由 FastAPI
依赖注入取用。评测 runner 是普通脚本，没有 ``app.state``，因此这里把
``app/main.py::lifespan`` 与 ``app/api/depends/dependencies.py::get_pipeline``
的装配逻辑**按同样顺序**复刻一份，保证评测跑的是**同一条真实链路**
（同一套 ModelRouter 熔断、同一个意图树、同一个 RAG 服务），而不是另起一套
简化实现——否则评测结果对业务没有参考价值。

⚠️ 依赖：Milvus / Redis / PostgreSQL / 模型 API Key 必须可用（见 .env）。
   跑之前建议先 ``curl http://127.0.0.1:8000/api/v1/health`` 确认服务在。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _log(message: str) -> None:
    print(f"[eval-runtime] {message}", flush=True)


# =====================================================================
# 0. 评测目标集合口径（评测与生产共用同一套数据，不再设独立评测集合）
# =====================================================================
# 物理集合：Milvus 里的真实 collection，评测默认与线上同一个
# （settings.milvus_kb_collection_name，当前 knowledge_base_v3）。
# 逻辑集合：每个向量节点 metadata['collection'] 上的标签，也是黄金集
# rag_cases.jsonl 里 ``collection`` 字段的值与线上上传接口 collection_name
# 表单的值。
#
# ⚠️ 评测**不要再拿这个常量当默认过滤条件**：一个物理集合里往往同时存在
# sales_kb / enterprise_kb / 各次上传登记的逻辑标签，写死单一标签会让检索结果
# 被整段过滤成空（表现为 Recall 恒为 0）。评测默认不按标签过滤，见 run_rag.py。
EVAL_LOGICAL_COLLECTION: str = "sales_kb"


# =====================================================================
# 1. 配置与模型路由
# =====================================================================
def build_settings() -> Any:
    """读取 pydantic-settings（等价于应用启动时的 get_settings()）。"""
    from app.config import get_settings

    return get_settings()


def build_model_router(settings: Any) -> Any:
    """复用 app.main 的全局路由构建函数，保证 tier/熔断配置与线上一致。"""
    from app.main import _build_global_router

    _log("构建全局 ModelRouter（与线上同一函数 app.main._build_global_router）")
    return _build_global_router()


# =====================================================================
# 2. RAG
# =====================================================================
def build_rag_service(
    settings: Any,
    *,
    overwrite: bool = False,
    collection_name: Optional[str] = None,
    top_k_default: int = 10,
) -> Any:
    """构建 RAGService（向量 + BM25 + RRF）。

    ⚠️ **安全约束（重要）**：``overwrite`` 恒为 ``False``。llama_index 的
    ``MilvusVectorStore(overwrite=True)`` 会 **drop 并重建集合**，已入库向量全部丢失。
    需要重建集合的唯一入口是 ``evals/tools/reingest_corpus.py --reset --confirm-reset``
    （显式双重确认），已不存在任何"配置一开就清库"的开关。

    Args:
        settings: 全局配置。
        overwrite: 是否允许重建集合。一律 False（默认），保留形参仅为显式传参的场景。
        collection_name: 覆盖物理集合名。常规评测不传（与线上同一个，
            即 settings.milvus_kb_collection_name）；仅 chunk-size A/B 实验
            传 eval_kb_512 / eval_kb_1024 这类独立物理集合。
        top_k_default: 默认召回数。
    """
    from app.core.rag.rag_service import RAGService
    from app.infrastructure.embeddings.dashscope_embedding import DashScopeEmbedding

    target_collection: str = collection_name or settings.milvus_kb_collection_name

    rag_embed_model = DashScopeEmbedding(
        model="text-embedding-v3",
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
    )
    _log(
        f"构建 RAGService：collection={target_collection} "
        f"uri=http://{settings.milvus_host}:{settings.milvus_port} overwrite={overwrite}"
    )
    return RAGService(
        embed_model=rag_embed_model,
        milvus_uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
        collection_name=target_collection,
        overwrite=overwrite,
        dim=1024,
        top_k_default=top_k_default,
    )


async def seed_bm25(rag_service: Any) -> None:
    """从 Milvus 全量重建 BM25 内存索引（等价于启动时的后台预热）。

    评测必须等它完成：否则 BM25 通道为空，Recall 会被系统性低估。
    """
    _log("预热 BM25 内存索引（seed_bm25）…")
    await rag_service.seed_bm25()
    _log("BM25 预热完成")


# =====================================================================
# 3. 文件沙箱与技能
# =====================================================================
def build_fs_and_skills(settings: Any) -> tuple:
    from app.core.backends.filesystem import FilesystemBackend
    from app.core.skill.manager import SkillManager

    fs_backend = FilesystemBackend(virtual_mode=True)
    skill_manager = SkillManager(fs_backend=fs_backend, skills_dir="./skills")
    return fs_backend, skill_manager


# =====================================================================
# 4. 意图 Pipeline（复刻 dependencies.get_pipeline）
# =====================================================================
def build_prompt_loader() -> Any:
    from app.query_intent.intent_prompt.prompt_template_loader import PromptTemplateLoader
    from app.query_intent.rag_constant import (
        AGENT_QUESTION_REWRITE_PROMPT_PATH,
        AGENT_REWRITE_INTENT_COMBINED_PROMPT_PATH,
        INTENT_CLASSIFIER_PROMPT_PATH,
    )

    loader = PromptTemplateLoader()
    loader.load(INTENT_CLASSIFIER_PROMPT_PATH)
    loader.load(AGENT_QUESTION_REWRITE_PROMPT_PATH)
    loader.load(AGENT_REWRITE_INTENT_COMBINED_PROMPT_PATH)
    _log("Pipeline Prompt 模板加载完成（3 个）")
    return loader


def build_intent_pipeline(settings: Any, model_router: Any, prompt_loader: Any) -> Any:
    """装配 改写 → 意图聚合 → 模式决策 三段流水线（与 get_pipeline 同构）。"""
    from app.api.depends.dependencies import _AppModelRouterIntentLLMAdapter
    from app.query_intent.intent_3stage_pipeline.agent_query_intent_pipeline import (
        AgentQueryIntentPipeline,
    )
    from app.query_intent.intent_3stage_pipeline.mode_decider import ModeDecider
    from app.query_intent.intent_classify_resolver.intent_classify import (
        AgentIntentAggregator,
        DefaultIntentClassifier,
    )
    from app.query_intent.intent_classify_resolver.intent_resolver import IntentResolver
    from app.query_intent.intent_classify_resolver.intent_vector_retriever import (
        IntentTreeVectorRetriever,
    )
    from app.query_intent.rag_constant import INTENT_VECTOR_TOP_K
    from app.query_intent.rewrite.combined_rewrite_intent_service import (
        AgentCombinedRewriteIntentService,
    )
    from app.query_intent.rewrite.multi_question_rewrite_service import RAGConfigProperties
    from app.query_intent.rewrite.query_rewrite import (
        QueryTermMappingCacheManager,
        QueryTermMappingService,
    )

    llm_service = _AppModelRouterIntentLLMAdapter(model_router=model_router)
    mapping_service = QueryTermMappingService(
        mapping_mapper=None,
        cache_manager=QueryTermMappingCacheManager(),
    )
    rag_cfg = RAGConfigProperties(
        query_rewrite_enabled=True,
        rerank_enabled=None,
        context_enrich_enabled=None,
        citation_enabled=None,
    )
    classifier = DefaultIntentClassifier(
        llm_service=llm_service,
        intent_node_mapper=None,
        prompt_template_loader=prompt_loader,
    )

    # 意图树向量召回：embedding 走独立的固定单模型（与线上一致，不进路由）
    vector_retriever: Optional[Any] = None
    try:
        from app.core.memory.long_term import QwenEmbeddingImpl

        intent_embed_model = QwenEmbeddingImpl(
            api_key=settings.openai_api_key, model="text-embedding-v3"
        )
        vector_retriever = IntentTreeVectorRetriever(
            embed_model=intent_embed_model,
            tree_provider=classifier,
            top_k=INTENT_VECTOR_TOP_K,
        )
        vector_retriever._ensure_index()
        _log("意图树向量索引就绪")
    except Exception as exc:  # noqa: BLE001 - 与线上一致的降级：退回全量意图清单
        _log(f"意图向量检索不可用（{type(exc).__name__}: {exc}），退回全量意图清单路径")

    rewrite_service = AgentCombinedRewriteIntentService(
        llm_service=llm_service,
        rag_config_properties=rag_cfg,
        query_term_mapping_service=mapping_service,
        prompt_template_loader=prompt_loader,
        vector_retriever=vector_retriever,
        intent_classifier=classifier,
    )
    return AgentQueryIntentPipeline(
        rewrite_service=rewrite_service,
        intent_resolver=IntentResolver(intent_classifier=classifier),
        intent_aggregator=AgentIntentAggregator(
            base_classifier=classifier, min_intent_score=0.35, top_k_per_question=3
        ),
        mode_decider=ModeDecider(),
    )


# =====================================================================
# 5. 工具注册表 / 记忆 / 编排器（工具调用评测用）
# =====================================================================
def build_tool_registry(settings: Any, fs_backend: Any, rag_service: Any, model_router: Any = None) -> Any:
    from app.core.tools.builtin.init_tools import bootstrap_tools

    return bootstrap_tools(
        db_session_factory=None,
        fs_backend=fs_backend,
        rag_service=rag_service,
        model_router=model_router,
    )


def build_memory_manager(settings: Any, model_router: Any) -> Any:
    """短期(Redis) + 长期(Milvus) 记忆（等价于 get_memory_manager）。"""
    import redis.asyncio as aioredis

    from app.core.memory.long_term import LongTermMemory, MilvusCollectionWrapper, QwenEmbeddingImpl
    from app.core.memory.manager import MemoryManager
    from app.core.memory.short_term import QwenChatLLMImpl, ShortTermMemory

    redis_client = aioredis.from_url(
        settings.redis_url, encoding="utf-8", decode_responses=True
    )
    short_term = ShortTermMemory(
        redis_client=redis_client,
        llm=QwenChatLLMImpl(model_router=model_router),
        window_size=20,
        max_tokens=4000,
    )
    long_term = LongTermMemory(
        milvus_collection=MilvusCollectionWrapper(
            collection_name="agent_long_term_memory_v2",
            dim=1024,
            host=settings.milvus_host,
            port=str(settings.milvus_port),
        ),
        embedding_model=QwenEmbeddingImpl(
            api_key=settings.openai_api_key, model="text-embedding-v3"
        ),
    )
    return MemoryManager(short_term=short_term, long_term=long_term)


def build_agent_config(settings: Any) -> Dict[str, Any]:
    """复刻 dependencies.get_agent_config（评测必须与线上同一套开关）。"""
    return {
        "enable_skill_tool_gating": settings.enable_skill_tool_gating,
        "enable_empty_result_replan": settings.enable_empty_result_replan,
        "react_max_steps": settings.react_max_steps,
        "max_replan_attempts": settings.max_replan_attempts,
        "agent_evidence_gate_enabled": settings.agent_evidence_gate_enabled,
        "agent_approval_enabled": settings.agent_approval_enabled,
        "agent_danger_tools": settings.agent_danger_tools,
        "agent_reflect_enabled": settings.agent_reflect_enabled,
        "agent_reflect_min_score": settings.agent_reflect_min_score,
        "agent_node_retry_max": settings.agent_node_retry_max,
    }


def build_orchestrator(
    settings: Any,
    model_router: Any,
    memory_manager: Any,
    tool_registry: Any,
    skill_manager: Any,
    *,
    strategy: str = "auto",
) -> Any:
    from app.core.agent.orchestrator import AgentOrchestrator
    from app.infrastructure.trace.tracer import Tracer

    return AgentOrchestrator(
        config={**build_agent_config(settings), "default_strategy": strategy},
        model_router=model_router,
        memory_manager=memory_manager,
        tool_registry=tool_registry,
        tracer=Tracer(),
        skill_manager=skill_manager,
    )


# =====================================================================
# 统一运行时容器（按需装配，避免"只跑意图评测却去连 Milvus"）
# =====================================================================
@dataclass
class EvalRuntime:
    """评测运行时：各组件按需懒构建并缓存。"""

    settings: Any = None
    # 物理 Milvus 集合覆盖：默认 None = 与线上同一个
    # （settings.milvus_kb_collection_name，当前 knowledge_base_v3）。
    # 常规 RAG 评测不需要传；逻辑集合过滤（sales_kb）在 run_rag_eval 里做。
    # 仅 chunk-size A/B 实验用它指向独立物理集合（eval_kb_512 / eval_kb_1024）。
    collection_name: Optional[str] = None
    _model_router: Any = None
    _rag_service: Any = None
    _fs_backend: Any = None
    _skill_manager: Any = None
    _prompt_loader: Any = None
    _pipeline: Any = None
    _tool_registry: Any = None
    _memory_manager: Any = None
    _orchestrator: Any = None
    extras: Dict[str, Any] = field(default_factory=dict)

    # ---- 基础 ----
    def get_settings(self) -> Any:
        if self.settings is None:
            self.settings = build_settings()
        return self.settings

    def get_model_router(self) -> Any:
        if self._model_router is None:
            self._model_router = build_model_router(self.get_settings())
        return self._model_router

    # ---- RAG ----
    def get_rag_service(self) -> Any:
        if self._rag_service is None:
            # 评测链路**强制 overwrite=False**：绝不删表重建（见 build_rag_service 说明）
            self._rag_service = build_rag_service(
                self.get_settings(),
                overwrite=False,
                collection_name=self.collection_name,
            )
        return self._rag_service

    # ---- 文件沙箱 / 技能 ----
    def get_fs_and_skills(self) -> tuple:
        if self._fs_backend is None or self._skill_manager is None:
            self._fs_backend, self._skill_manager = build_fs_and_skills(self.get_settings())
        return self._fs_backend, self._skill_manager

    # ---- 意图 Pipeline ----
    def get_prompt_loader(self) -> Any:
        if self._prompt_loader is None:
            self._prompt_loader = build_prompt_loader()
        return self._prompt_loader

    def get_pipeline(self) -> Any:
        if self._pipeline is None:
            self._pipeline = build_intent_pipeline(
                self.get_settings(), self.get_model_router(), self.get_prompt_loader()
            )
        return self._pipeline

    # ---- 工具/记忆/编排 ----
    def get_tool_registry(self) -> Any:
        if self._tool_registry is None:
            fs_backend, _ = self.get_fs_and_skills()
            self._tool_registry = build_tool_registry(
                self.get_settings(), fs_backend, self.get_rag_service(),
                model_router=self.get_model_router(),
            )
        return self._tool_registry

    def get_memory_manager(self) -> Any:
        if self._memory_manager is None:
            self._memory_manager = build_memory_manager(
                self.get_settings(), self.get_model_router()
            )
        return self._memory_manager

    def get_orchestrator(self, strategy: str = "auto") -> Any:
        if self._orchestrator is None:
            _, skill_manager = self.get_fs_and_skills()
            self._orchestrator = build_orchestrator(
                self.get_settings(),
                self.get_model_router(),
                self.get_memory_manager(),
                self.get_tool_registry(),
                skill_manager,
                strategy=strategy,
            )
        return self._orchestrator

    async def aclose(self) -> None:
        """释放 Redis 连接等资源。"""
        manager = self._memory_manager
        if manager is not None:
            try:
                await manager._stm.redis_client.close()
            except Exception:  # noqa: BLE001
                pass
