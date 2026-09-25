# -*- coding: utf-8 -*-
"""app/api/depends/dependencies.py
FastAPI 依赖注入中心：统一管理各业务路由所需的内存、持久化存储及文件技能系统单例。

本文件包含两类核心注入：
  1. 原有通用：MemoryManager / ModelRouter / ToolRegistry / RAGService / SkillManager
               / FilesystemBackend / Milvus 等。
  2. Agent Pipeline（新增·对应 repo_map ⑬）：
        _AppModelRouterIntentLLMAdapter
            → 将 app 的异步 ModelRouter + _PurposeLLMAdapter 封装为
               query_intent 包需要的同步 IntentLLMService。
        get_intent_llm_service
            → 每请求返回一个轻量 Adapter 实例。
        get_pipeline
            → 组装改写 / 分类 / 解析 / 聚合 / 模式决策 → AgentQueryIntentPipeline。
"""

import asyncio
import logging
from typing import Any, Dict, Optional

from app.config import get_settings
from app.core.memory.manager import MemoryManager
from app.core.memory.short_term import ShortTermMemory
from app.core.memory.long_term import LongTermMemory
from app.llm_model_router.model_router import ModelRouter

# ==========================================
# 🌟【新整合】引入文件系统与高级技能管理器类型声明

from app.core.skill.manager import SkillManager

# 🌟【新整合】文件系统与技能树专属依赖注入项

from fastapi import Depends, Request
from app.core.tools.builtin.init_tools import bootstrap_tools
from app.core.tools.registry import ToolRegistry
from app.core.backends.filesystem import FilesystemBackend
from app.core.rag.rag_service import RAGService

# ==========================================
# 🌟【Agent Pipeline 新增】query_intent 包导入
# ==========================================
from app.query_intent.intent_prompt.prompt_template_loader import PromptTemplateLoader
from app.query_intent.intent_service import IntentLLMService
from app.query_intent.intent_data_base import IntentChatRequest
from app.query_intent.rewrite.query_rewrite import (
    QueryTermMappingCacheManager,
    QueryTermMappingService,
)
from app.query_intent.rewrite.multi_question_rewrite_service import (
    RAGConfigProperties,
)
from app.query_intent.rewrite.combined_rewrite_intent_service import (
    AgentCombinedRewriteIntentService,
)
from app.query_intent.intent_classify_resolver.intent_classify import (
    AgentIntentAggregator,
    DefaultIntentClassifier,
)
from app.query_intent.intent_classify_resolver.intent_vector_retriever import (
    IntentTreeVectorRetriever,
)
from app.query_intent.intent_classify_resolver.intent_resolver import IntentResolver
from app.query_intent.intent_3stage_pipeline.agent_query_intent_pipeline import (
    AgentQueryIntentPipeline,
)
from app.query_intent.intent_3stage_pipeline.mode_decider import ModeDecider
from app.query_intent.rag_constant import (
    AGENT_QUESTION_REWRITE_PROMPT_PATH,
    AGENT_REWRITE_INTENT_COMBINED_PROMPT_PATH,
    INTENT_CLASSIFIER_PROMPT_PATH,
    INTENT_VECTOR_TOP_K,
)

logger = logging.getLogger(__name__)


# ==========================================
# 🌟【⑬-2】Agent Pipeline 新增：LLM 门面适配层
# ==========================================
class _AppModelRouterIntentLLMAdapter(IntentLLMService):
    """把 app 侧异步 ModelRouter + _PurposeLLMAdapter → query_intent 同步接口。

    由于 query_intent 包（改写 / 意图 / 决策）全部使用同步的
    ``IntentLLMService.chat(messages, tier)`` 接口，而 app 唯一的 LLM 通道
    ``ModelRouter.for_purpose(purpose).acomplete(...)`` 是 async，因此本适配
    器在同步 chat 内部新建/复用事件循环跑 await 结果。

    【对标】app/core/agent/orchestrator.py 的 _LLMAdapter.acomplete 方向相反
    （它是把底层包装成异步给 orchestrator 用，这里是反方向）。
    """

    def __init__(self, model_router: ModelRouter) -> None:
        """构造适配器。

        Args:
            model_router: 全局 ModelRouter 单例（来自 app.state.model_router）。
        """
        self._model_router: ModelRouter = model_router

    def chat(self, messages: Any, tier: Any = None) -> str:
        """query_intent 侧的统一 LLM 入口。

        支持两种入参风格（保证与现有两版调用签名的向后兼容）：

        1) 新版 MultiQuestionRewriteService 风格：
              messages = IntentChatRequest（含 messages/temperature/top_p/thinking）
              tier     = IntentChoiceTier.FAST / DEFAULT（此处未做差异化路由，
                         留作未来按 tier 区分模型）
        2) 老版 DefaultIntentClassifier 风格（疑问-B B-1 现状）：
              messages = {"messages": [...], "temperature": 0.1, "top_p": 0.3,
                          "thinking": False}
              tier     = None（此 case 下不传）

        两种风格都在这里统一为 _PurposeLLMAdapter.acomplete(...) 所需的
        messages 列表 + generation_kwargs。
        """
        # ---- 入参规范化：messages_list / generation_kwargs ----
        if isinstance(messages, IntentChatRequest):
            messages_list = [
                {"role": msg.role, "content": msg.content}
                for msg in (messages.messages or [])
            ]
            generation_kwargs: Dict[str, Any] = {
                key: value
                for key, value in {
                    "temperature": messages.temperature,
                    "top_p": messages.top_p,
                    "max_tokens": messages.max_tokens,
                    # thinking 字段并非所有底层 Provider 接受；仅在非 None 时透传
                    "thinking": messages.thinking,
                    # 本轮结构化输出：response_format / tools / tool_choice
                    "response_format": messages.response_format,
                    "tools": messages.tools,
                    "tool_choice": messages.tool_choice,
                }.items()
                if value is not None
            }
        elif isinstance(messages, dict):
            raw_messages = messages.get("messages") or []
            messages_list = [
                {"role": m.get("role"), "content": m.get("content")}
                for m in raw_messages
                if isinstance(m, dict)
            ]
            generation_kwargs = {
                key: value
                for key, value in {
                    "temperature": messages.get("temperature"),
                    "top_p": messages.get("top_p"),
                    "max_tokens": messages.get("max_tokens"),
                    "thinking": messages.get("thinking"),
                    # 本轮结构化输出：response_format / tools / tool_choice
                    "response_format": messages.get("response_format"),
                    "tools": messages.get("tools"),
                    "tool_choice": messages.get("tool_choice"),
                }.items()
                if value is not None
            }
        else:
            raise TypeError(
                "_AppModelRouterIntentLLMAdapter.chat(messages) 仅支持 "
                "IntentChatRequest 或 dict；实际得到："
                f"{type(messages).__name__}"
            )

        # ---- 选择 Purpose 适配器（intent_analysis 或 chat 兜底）----
        # 说明：旧代码调用了 self._model_router.for_purpose(...)，该方法在原方案 A
        # 和新重写方案驱动的 ModelRouter 中均未实现（必报 AttributeError）。
        # 统一改为 get_llm(purpose)：内部按 PURPOSE_TIER_MAP 自动命中 FAST tier
        # （适用于意图分类/改写这类高频场景）。该档位的**单次尝试**超时取自
        # LLM_TIER_FAST_TIMEOUT_MS、重试次数取自 LLM_TIER_FAST_RETRIES——
        # 两者不要再在本文件里写死数值（历史注释里的"15s"早已与实际配置不符）。
        # chat 为 STANDARD 兜底。
        try:
            purpose_adapter = self._model_router.get_llm("intent_analysis")
        except Exception:
            purpose_adapter = self._model_router.get_llm("chat")

        # ---- 同步等待异步 acomplete 返回 ----
        try:
            # 优先拿 running 事件循环；如果当前线程没有 loop，就新建一个。
            asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
            try:
                asyncio_loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio_loop = None

            if asyncio_loop is None:
                # 同步线程（asyncio.to_thread 中）：新建 loop 跑一次。
                return asyncio.run(
                    purpose_adapter.acomplete(
                        messages=messages_list, **generation_kwargs
                    )
                )

            # 意外：路由层自己同步调（没包装 to_thread）。
            # 这种情况不能阻塞 event_loop，尝试直接 asyncio.create_task 再用
            # 独立 loop 兜底，避免 RuntimeError。
            logger.warning(
                "_AppModelRouterIntentLLMAdapter.chat 命中 running_event_loop "
                "非空分支（通常不应该走到这里），回退新建 loop 执行以避免阻塞。"
            )
            return asyncio.run(
                purpose_adapter.acomplete(
                    messages=messages_list, **generation_kwargs
                )
            )
        except Exception as llm_error:
            logger.error(
                "Agent Pipeline LLM 适配调用失败：%s；返回空字符串以触发"
                "下游 fallback。",
                llm_error,
                exc_info=True,
            )
            return ""


async def get_intent_llm_service(
    request: Request,
) -> IntentLLMService:
    """FastAPI 依赖：获取适配后的同步 IntentLLMService。

    本函数每请求 new 一次 Adapter 即可，实例化非常轻，无缓存必要性。
    直接从 app.state 取全局 ModelRouter 单例，风格与 get_model_router 等一致，
    避免跨函数 Depends 导致的前向引用 NameError。
    """
    model_router: ModelRouter = request.app.state.model_router
    return _AppModelRouterIntentLLMAdapter(model_router=model_router)


async def get_filesystem_backend(request: Request) -> FilesystemBackend:
    """【新整合】FastAPI 依赖注入：获取全局唯一的安全文件沙箱后端单例。

    让后续的工具层或 API 接口能够共享同一个物理/虚拟根目录视图。
    """
    return request.app.state.fs_backend


async def get_skill_manager(request: Request) -> SkillManager:
    """【新整合】FastAPI 依赖注入：获取全局统一的高级技能树管理器单例。

    方便编排层（Orchestrator）或前端动态刷新、读取当前的 Agent 技能包元数据。
    """
    return request.app.state.skill_manager


# ==========================================
# 原有标准基础设施依赖注入项
# ==========================================

async def get_model_router(request: Request) -> ModelRouter:
    """极轻量：直接从 app.state 获取已经初始化好的全局路由器"""
    return request.app.state.model_router


async def get_rag_service(request: Request) -> RAGService:
    """
    极轻量：直接从全局 app.state 中获取已经初始化好的 RAG 核心服务。
    这样可以确保全局单例运行，且不会在每次请求时重复连接 Milvus 或加载模型。
    """
    return request.app.state.rag_service


async def get_memory_manager(request: Request) -> MemoryManager:
    """FastAPI 依赖注入：在请求进入时，动态组装并获取完整的记忆管理器。"""
    redis_client = request.app.state.redis
    qwen_llm = request.app.state.qwen_chat_llm

    # 1. 实例化短期记忆
    short_term = ShortTermMemory(
        redis_client=redis_client,
        llm=qwen_llm,
        window_size=20,
        max_tokens=4000
    )

    # 2. 组装长期记忆
    milvus_wrapper = request.app.state.milvus_wrapper
    embedding_model = request.app.state.embedding_model

    long_term = LongTermMemory(
        milvus_collection=milvus_wrapper,
        embedding_model=embedding_model
    )

    # 3. 完美装配并返回
    return MemoryManager(short_term=short_term, long_term=long_term)

async def get_tool_registry(
        request: Request,
        fs_backend: FilesystemBackend = Depends(get_filesystem_backend),
        rag_service: RAGService = Depends(get_rag_service),
        # db_session_factory = Depends(get_db_session_factory) # 如果你有数据库工厂依赖，也可以平铺在这里
) -> ToolRegistry:
    """🌟 统一的工具箱工厂依赖注入中心。

    在请求到达路由前，动态调配系统中的核心单例，组装出一套具备完整原子能力的底层工具链。

    ⚠️ 进程级单例：工具本身无请求级状态，依赖（fs_backend / rag_service /
    model_router）也都是 app.state 单例，**绝不能每请求重建**——
    KnowledgeGraphSearchTool 首次导入会拉起 lightrag / torch /
    sentence-transformers 重依赖栈（内存吃紧时实测首请求被卡住 30~40s）。
    lifespan 启动后会在后台预热本单例；预热未完成时此处兜底构建并缓存。
    """
    existing_registry: Optional[ToolRegistry] = getattr(
        request.app.state, "tool_registry", None
    )
    if existing_registry is not None:
        return existing_registry

    # 完美实现多组件在工具链中的完全闭环与高内聚
    dynamic_registry = bootstrap_tools(
        db_session_factory=None,  # 根据你实际情况传入
        fs_backend=fs_backend,  # 注入虚拟文件沙箱
        rag_service=rag_service,  # 🌟 注入在寿命周期内已联通 Milvus 的 RAG 单例
        # 表格自然语言查询工具的代码生成通道：复用 lifespan 构建的全局 ModelRouter 单例
        # （多厂商 tier/熔断/追踪），不另起 LLM 客户端。
        model_router=getattr(request.app.state, "model_router", None),
    )
    request.app.state.tool_registry = dynamic_registry
    return dynamic_registry


# ==========================================
# 🌟【⑬-4】Agent Pipeline 新增：Pipeline 工厂依赖
# ==========================================
async def get_pipeline(
    request: Request,
    llm_service: IntentLLMService = Depends(get_intent_llm_service),
) -> AgentQueryIntentPipeline:
    """FastAPI 依赖：组装 AgentQueryIntentPipeline 实例。

    风格对齐 get_tool_registry：每请求构造一份（classifier / aggregator
    内部无请求级脏状态，若后续需要优化为单例，可在 lifespan 中放入
    app.state.pipeline 并直接在此处 return。）
    """
    # 1) Prompt 模板加载器（优先 lifespan 中预热好的单例）
    prompt_loader: Optional[PromptTemplateLoader] = getattr(
        request.app.state, "prompt_loader", None
    )
    if prompt_loader is None:
        logger.debug(
            "未在 app.state.prompt_loader 找到 PromptTemplateLoader "
            "（未执行 lifespan 预热）。每请求实例化一份。"
        )
        prompt_loader = PromptTemplateLoader()
        # 懒预热：避免每次请求都读文件 IO
        try:
            prompt_loader.load(INTENT_CLASSIFIER_PROMPT_PATH)
            prompt_loader.load(AGENT_QUESTION_REWRITE_PROMPT_PATH)
            prompt_loader.load(AGENT_REWRITE_INTENT_COMBINED_PROMPT_PATH)
        except Exception as preload_error:
            logger.warning(
                "Pipeline Prompt 懒预热失败（忽略，首次请求会重试）：%s",
                preload_error,
            )

    # 2) 术语映射（首版：空实现；后续接入 QueryTermMappingMapper 时替换）
    mapping_cache_mgr = QueryTermMappingCacheManager()
    mapping_service = QueryTermMappingService(
        mapping_mapper=None,  # 【待补充-⑬-4-A】：后续接入 DB Mapper 再补
        cache_manager=mapping_cache_mgr,
    )

    # 3) RAG 配置属性（Agent 改写复用开关即可）
    rag_cfg = RAGConfigProperties(
        query_rewrite_enabled=True,
        rerank_enabled=None,
        context_enrich_enabled=None,
        citation_enabled=None,
    )

    # 4) 意图分类器（组合服务与向量检索器都复用其意图树缓存访问）
    # 【待补充-⑬-4-B】intent_node_mapper 首版 None，走 IntentTreeFactory 默认；
    #                  未来接 DB 意图树表再注入。
    classifier = DefaultIntentClassifier(
        llm_service=llm_service,
        intent_node_mapper=None,
        prompt_template_loader=prompt_loader,
    )

    # 5) 意图树向量检索器（调整二：Embedding Top-K 召回替代全量意图树塞 Prompt）
    # embedding_model 不可用时置 None，组合服务自动降级全量叶子清单路径。
    # 【性能】优先复用 lifespan 启动后台预热的单例（向量索引已就绪，免去每请求
    # 对全部叶子节点重新 embedding）；预热未完成/失败时回退原每请求自建（懒加载）。
    vector_retriever: Optional[IntentTreeVectorRetriever] = getattr(
        request.app.state, "intent_vector_retriever", None
    )
    if vector_retriever is None:
        embedding_model = getattr(request.app.state, "embedding_model", None)
        if embedding_model is not None:
            vector_retriever = IntentTreeVectorRetriever(
                embed_model=embedding_model,
                tree_provider=classifier,
                top_k=INTENT_VECTOR_TOP_K,
            )
        else:
            logger.warning(
                "app.state.embedding_model 不可用，意图向量检索未启用"
                "（组合调用将使用全量意图清单）。"
            )

    # 6) Agent 改写服务（调整一：组合「改写+意图」单次 LLM 调用；
    #    组合链路失败时自动回退父类 AgentMultiQuestionRewriteService 两段链路）
    rewrite_service = AgentCombinedRewriteIntentService(
        llm_service=llm_service,
        rag_config_properties=rag_cfg,
        query_term_mapping_service=mapping_service,
        prompt_template_loader=prompt_loader,
        vector_retriever=vector_retriever,
        intent_classifier=classifier,
    )

    # 7) Resolver + Agent 专用聚合器
    resolver = IntentResolver(intent_classifier=classifier)
    aggregator = AgentIntentAggregator(
        base_classifier=classifier,
        min_intent_score=0.35,
        top_k_per_question=3,
    )

    # 8) 模式决策器（调整三：完全规则化，纯内存计算，无 LLM / Prompt）
    mode_decider = ModeDecider()

    return AgentQueryIntentPipeline(
        rewrite_service=rewrite_service,
        intent_resolver=resolver,
        intent_aggregator=aggregator,
        mode_decider=mode_decider,
    )


# ==========================================
# 🌟【2.2 状态图】Agent GraphRunner 单例 + 节点配置
# ==========================================
def get_agent_config(request: Request) -> Dict[str, Any]:
    """节点读取的配置字典（deps.cfg 协议即 ``dict.get(key, default)``）。

    旧链路传的是 ``{"default_strategy": ...}`` 裸 dict；此处把 Settings 中
    2.2 新增开关与既有步数/预算相关键一并下发，缺省行为与旧硬编码默认值一致。
    """
    settings = get_settings()
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


async def get_agent_graph_runner(request: Request) -> Any:
    """FastAPI 依赖：获取进程级 GraphRunner 单例（懒初始化，带并发锁）。

    - checkpointer 优先 Redis（专用 ``settings.checkpoint_redis_url``，db1，
      与短期记忆/缓存的 db0 隔离；独立连接避免与业务 Redis 的
      decode_responses=True 冲突），失败自动降级 InMemorySaver；
    - 图只编译一次；审批/断点续跑依赖该单例持有的同一个 saver。
    """
    existing_runner: Optional[Any] = getattr(request.app.state, "agent_graph_runner", None)
    if existing_runner is not None:
        return existing_runner

    init_lock: Optional[asyncio.Lock] = getattr(request.app.state, "agent_graph_runner_lock", None)
    if init_lock is None:
        init_lock = asyncio.Lock()
        request.app.state.agent_graph_runner_lock = init_lock

    async with init_lock:
        # 双检：持锁期间可能已被其他请求初始化
        existing_runner = getattr(request.app.state, "agent_graph_runner", None)
        if existing_runner is not None:
            return existing_runner

        from app.core.agent.graph.builder import compile_agent_graph
        from app.core.agent.graph.checkpoint import build_checkpointer
        from app.core.agent.graph.runner import GraphRunner

        settings = get_settings()
        saver, actual_backend = await build_checkpointer(
            backend=settings.agent_checkpoint_backend,
            redis_url=settings.checkpoint_redis_url,
            ttl_seconds=settings.agent_checkpoint_ttl_seconds,
            checkpoint_prefix=settings.agent_checkpoint_prefix,
        )
        runner = GraphRunner(compile_agent_graph(saver), saver)
        request.app.state.agent_graph_runner = runner
        request.app.state.agent_checkpoint_backend = actual_backend
        logger.info("Agent GraphRunner 初始化完成（checkpointer=%s）。", actual_backend)
        return runner
