# -*- coding: utf-8 -*-
"""多模型路由器（重写方案驱动 · 对外完全兼容旧接口）。

调用侧（chat / get_llm / LLMResponse）对外形态一致，初始化直接接收「重写
model_router」分层配置（不再翻译旧 ModelConfig）：

    AIModelProperties (调用方构建: providers + chat tiers + selection)
           ↓ 启动校验
    ChatTierConfigValidator.validate()  (Fail-Fast)
           ↓ 初始化
    ┌─ AsyncModelHealthStore  ——  异步三态熔断器（含健康预选）
    ├─ AsyncModelSelector     ——  Tier/Thinking/Purpose 选择（async/await）
    └─ AsyncModelRoutingExecutor ——异步逐候选降级（超时按 tier 集中查表）
           ↓ 实际调用
    async_openai_chat_caller + AsyncOpenAI（返回标准化四元结构）

业务方导入路径：``from app.llm_model_router.model_router import
ModelRouter, LLMResponse``（旧 ``ModelConfig`` 扁平配置已废弃删除）：

    - LLMResponse: content/model_id/usage/raw/tool_calls/reasoning_content 响应 (BaseModel)
    - ModelRouter:  __init__(properties: AIModelProperties)
                    get_llm(purpose) -> _PurposeLLMAdapter
                    async chat(messages, model_preference=None, **kwargs) -> LLMResponse
    - _PurposeLLMAdapter: 鸭子类型，含 acomplete(messages, **kwargs) -> str
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
from pydantic import BaseModel

from app.llm_model_router.async_model_executor import AsyncModelRoutingExecutor
from app.llm_model_router.async_model_health import AsyncModelHealthStore
from app.llm_model_router.async_model_selector import AsyncModelSelector
from app.llm_model_router.async_openai_caller import async_build_openai_client, AsyncOpenAICallResult, \
    async_openai_chat_caller
from app.llm_model_router.model_router_config import AIModelProperties, resolve_model_id
from app.llm_model_router.model_router_enums import Tier, ModelTarget, ModelCapability


from app.llm_model_router.model_router_validator import ChatTierConfigValidator



# ---------------------------------------------------------------------------
# Purpose → Tier 映射（5 个声明式场景 → FAST/STANDARD/DEEP 分层）
# ---------------------------------------------------------------------------
# tier 覆写逻辑已废弃并删除：tier 的 candidates / timeout 现在全部由调用方在
# AIModelProperties.chat.tiers 中声明，ModelRouter 不再做任何隐式补全 / 塞入。


PURPOSE_TIER_MAP: Dict[str, "Tier"] = {
    "planner": Tier.STANDARD,      # 规划: 质量优先, 30s
    "react": Tier.FAST,            # ReAct 工具交互: 速度优先, 15s
    "reflection": Tier.DEEP,       # 反思质量门: 深度推理, 60s
    "intent_analysis": Tier.FAST,  # 意图识别/改写: 高频短平快, 15s
    "chat": Tier.STANDARD,         # 标准对话: 平衡, 30s
}

# =====================================================================
# ① 响应结构（Pydantic BaseModel，与旧协议一致）
# =====================================================================
class LLMResponse(BaseModel):
    """统一 LLM 响应结构（旧协议兼容字段）。

    Attributes:
        content: 纯文本内容（Function Calling 场景通常为空）。
        model_id: 实际命中的模型 ID。
        usage: 用量统计。
        raw: 完整原始响应 dict。
        tool_calls: Function Calling 产生的工具调用列表（[{id, type, function:{name, arguments}}]），
            无工具调用时为 None。
        reasoning_content: 思考模型的思维链文本，未开启思考或非思考模型时为 None。
    """

    content: str = ""
    model_id: str = ""
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None


# =====================================================================
# ② 旧 ModelConfig 扁平配置 dataclass 已废弃并删除：
#    ModelRouter 现直接接收 AIModelProperties 分层配置（见 __init__）
# =====================================================================


# =====================================================================
# ③ 编排层专属适配器（签名与旧版一致，仅内部自动注入 purpose_hint）
# =====================================================================
class _PurposeLLMAdapter:
    """鸭子类型：包装 ModelRouter.chat 为 orchestrator 想要的 ``acomplete(str)``。

    相比旧实现新增：**保存 purpose**，调用 ``router.chat`` 时自动注入
    ``purpose_hint=purpose`` → 触发 PURPOSE_TIER_MAP →
    Tier (FAST 15s / STANDARD 30s / DEEP 60s) 独立超时 + 候选顺序。
    """

    __slots__ = ("_router", "_purpose", "_model_preference")

    def __init__(
        self,
        router: "ModelRouter",
        purpose: Optional[str] = None,
        model_preference: Optional[str] = None,
    ) -> None:
        self._router: ModelRouter = router
        self._purpose: Optional[str] = purpose
        self._model_preference: Optional[str] = model_preference

    async def acomplete(
        self, messages: Sequence[Dict[str, str]], **kwargs: Any
    ) -> str:
        """编排层契约：Sequence[Dict[str,str]] → 纯回答字符串。"""
        formatted: List[Dict[str, Any]] = [dict(m) for m in messages]
        # 注入 purpose_hint（caller 传了就不覆盖）
        if self._purpose and "purpose_hint" not in kwargs:
            kwargs["purpose_hint"] = self._purpose
        resp: LLMResponse = await self._router.chat(
            messages=formatted,
            model_preference=self._model_preference,
            **kwargs,
        )
        return resp.content


# =====================================================================
# ④ 核心路由器（对外协议不变，内部 = 重写分层方案 异步三件套）
# =====================================================================
class ModelRouter:
    """多模型路由器（重写方案驱动 · 旧接口兼容）。

    兼容性承诺：
      ✅ ``ModelRouter(list[ModelConfig], *, failure_threshold=5,
         recovery_timeout=60.0)`` —— 与旧实现完全一致。
      ✅ ``get_llm(purpose: str) -> _PurposeLLMAdapter``：orchestrator
         planner/react/reflection 调用链零改动。
      ✅ ``async chat(messages, model_preference=None, **kwargs) -> LLMResponse``：
         chat.py 及 adapter 取 .content/.model_id/.usage 零改动。
    """

    def __init__(self, properties: AIModelProperties) -> None:
        """构建多模型路由器（直接接收重写版分层配置，不再翻译旧 ModelConfig）。

        Args:
            properties: AIModelProperties 分层配置（providers + chat tiers +
                selection）。调用方用 ``AIModelProperties.from_dict(...)`` 从
                env / 文件构建；启动期 ChatTierConfigValidator 负责 Fail-Fast
                校验（3 个 Tier 枚举齐全、tier 非空、timeout>0、DEEP 含 thinking
                候选），Router 不再对 DEEP 做任何隐式塞入。
        """
        if properties is None or properties.chat is None:
            raise ValueError("properties.chat 不能为空")

        # ----- 1) 启动期结构校验（Fail-Fast）-----
        try:
            ChatTierConfigValidator(properties).validate()
        except ValueError as verr:
            logger.error("ModelRouter ChatTier 配置校验失败: {}", verr)
            raise

        # ----- 2) 异步三件套（全链路 asyncio 原生，无线程锁切换）-----
        self._health_store: AsyncModelHealthStore = AsyncModelHealthStore(properties)
        self._selector: AsyncModelSelector = AsyncModelSelector(
            properties, self._health_store
        )
        # 超时由 Executor 依据 target.tier_name 在 tier 配置中集中查表
        self._executor: AsyncModelRoutingExecutor = AsyncModelRoutingExecutor(
            self._health_store, properties
        )

        # ----- 3) 预构建 AsyncOpenAI 客户端（按 model_id O(1) 查表）-----
        self._clients: Dict[str, Any] = {}
        if properties.chat and properties.chat.candidates:
            providers = properties.providers or {}
            for cand in properties.chat.candidates:
                if cand is None:
                    continue
                mid = resolve_model_id(cand)
                if mid in self._clients:
                    continue
                client = async_build_openai_client(cand, providers.get(cand.provider))
                if client is not None:
                    self._clients[mid] = client

    # ------------------------------------------------------------------
    # get_llm(purpose).acomplete(messages, **kwargs) -> str
    # ------------------------------------------------------------------
    def get_llm(self, purpose: str) -> Any:
        """按应用场景（Purpose）返回带 acomplete 方法的 LLM 适配器。

        ``purpose`` 会保存到 adapter 内，实际调用时自动映射到 Tier：
          planner→STANDARD, react→FAST, reflection→DEEP,
          intent_analysis→FAST, chat→STANDARD。
        """
        logger.info(f"编排层请求场景模型 -> purpose: {purpose}")
        return _PurposeLLMAdapter(
            router=self, purpose=purpose, model_preference=None
        )

    # ------------------------------------------------------------------
    # async chat(messages, model_preference=None, **kwargs) -> LLMResponse
    # async chat_with_tools(messages, tools, tool_choice, **kwargs) -> LLMResponse
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model_preference: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """纯文本对话路由：失败按 Selector 排序顺序自动降级。

        Args:
            messages: OpenAI 协议消息列表
            model_preference: 旧协议兼容参数（精确命中 model_id）
            **kwargs: temperature / max_tokens / thinking / response_format
                / purpose_hint / tier_override。
        """
        result: AsyncOpenAICallResult = await self._execute_route(
            messages, model_preference=model_preference, **kwargs
        )
        return self._to_response(result)

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Any = "auto",
        model_preference: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Function Calling 对话路由：透传 tools/tool_choice 并返回含 tool_calls 的响应。

        专供 Agent（ReAct / Planner 子任务）在需要大模型自主/强制调用工具时使用。
        tools/tool_choice 不参与 tier 选择，仅作为请求载荷透传给底层
        AsyncOpenAI SDK；除 content 外，结果中还携带 tool_calls（含严格校验后的
        JSON 参数字符串）与 reasoning_content（思考模型思维链）。

        Args:
            messages: OpenAI 协议消息列表
            tools: OpenAI tools[]（[{type: function, function: {name, description, parameters}}]）
            tool_choice: "auto" 或 {"type": "function", "function": {"name": ...}} 强制指定
            model_preference: 旧协议兼容参数（精确命中 model_id）
            **kwargs: temperature / max_tokens / thinking / response_format
                / purpose_hint / tier_override。
        """
        if not tools:
            raise ValueError("chat_with_tools 要求至少传入一个 tools 定义")
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        result: AsyncOpenAICallResult = await self._execute_route(
            messages, model_preference=model_preference, **kwargs
        )
        return self._to_response(result)

    async def _execute_route(
        self,
        messages: List[Dict[str, Any]],
        model_preference: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncOpenAICallResult:
        """统一路由执行体：解析 tier → 选候选 → 逐候选降级调用（chat 与 chat_with_tools 共用）。"""
        thinking_raw = kwargs.get("thinking")
        thinking_flag = bool(thinking_raw) if thinking_raw is not None else False

        # tier 解析优先级：tier_override > purpose_hint（来自 PURPOSE_TIER_MAP）
        #                 > Selector 默认（thinking→DEEP，否则 STANDARD）
        override_tier: Optional[Tier] = kwargs.pop("tier_override", None)
        if override_tier is None:
            purpose_hint = kwargs.get("purpose_hint")
            if purpose_hint and purpose_hint in PURPOSE_TIER_MAP:
                override_tier = PURPOSE_TIER_MAP[purpose_hint]
        # ⚠️ purpose_hint 只是路由内部的控制参数：这里解析完 tier 之后必须立即
        #    从 kwargs 里移除，否则会一路透传进 AsyncOpenAI 的
        #    chat.completions.create(**params)，触发
        #    "unexpected keyword argument 'purpose_hint'"。
        kwargs.pop("purpose_hint", None)

        # ----- 1) 异步选择候选（内部 await health_store.is_unavailable）-----
        try:
            targets: List[ModelTarget] = (
                await self._selector.select_chat_candidates(
                    thinking=thinking_flag,
                    override=override_tier,
                    preferred_model_id=model_preference,
                )
            )
        except Exception as select_err:
            logger.exception("AsyncModelSelector 选择异常，降级空列表: {}", select_err)
            targets = []

        if not targets:
            raise RuntimeError(
                "模型路由失败：Selector 未返回任何可用候选。请检查"
                " OPENAI_API_KEY / model_id / enabled / provider 配置。"
            )

        # ----- 2) 异步逐候选执行 + 降级（Executor 内：熔断许可 + 状态回写）-----
        result: AsyncOpenAICallResult = (
            await self._executor.execute_with_fallback(
                capability=ModelCapability.CHAT,
                targets=targets,
                client_resolver=self._resolve_client,
                caller=async_openai_chat_caller,
                messages=messages,
                **kwargs,
            )
        )
        return result

    def _to_response(self, result: AsyncOpenAICallResult) -> LLMResponse:
        """把内部 AsyncOpenAICallResult 转为对外 LLMResponse（含 tool_calls / reasoning_content）。"""
        return LLMResponse(
            content=result.content,
            model_id=result.model_id,
            usage=result.usage,
            raw=result.raw,
            tool_calls=getattr(result, "tool_calls", None),
            reasoning_content=getattr(result, "reasoning_content", None),
        )

    # ------------------------------------------------------------------
    # client_resolver（同步查表，Executor 接受同步 Callable）
    # ------------------------------------------------------------------
    def _resolve_client(self, target: "ModelTarget") -> Optional[Any]:
        return self._clients.get(target.id)
