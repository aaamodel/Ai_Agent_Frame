# -*- coding: utf-8 -*-
"""
文件所在目录：app/core/agent/orchestrator.py
Agent 编排器门面（2.2 状态图版）：

- 业务编排全部下沉到 ``app/core/agent/graph`` 的 StateGraph；
- 本类只保留**门面契约**：``run()`` 签名、``IntentContext`` / ``AgentResponse`` /
  ``OrchestrationMode`` 数据结构与 langfuse 观测，上层 chat.py 无感知；
- "react / plan_execute" 的新语义：同一张图里"跳过 plan + execute 自环" vs
  "plan → execute 循环"，不再存在两套引擎，也不再有"plan 失败整体重跑 react"；
  执行层降级（replan/协议降级）统一体现在 ``AgentResponse.degraded``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# 💡 项目统一日志
from loguru import logger

from app.core.skill.manager import SkillManager
from app.infrastructure.trace import Tracer
from app.core.memory.manager import MemoryManager
from app.core.tools.registry import ToolRegistry
from app.llm_model_router.model_router import ModelRouter

# Langfuse 可观测性：@observe 在未配置密钥时自动退化为 no-op（零侵入）
from langfuse import observe as langfuse_observe

# ── 2.2 状态图 ──────────────────────────────────────────────────────────
# 函数内 import GraphRunner 以彻底切断 orchestrator ↔ graph.runner 的导入环
from app.core.agent.graph.deps import GraphDeps

OrchestrationMode = Literal["react", "plan_execute"]

# ─── 🎯 SKILLS 渐进式披露系统级核心提示词 ───
# 常量单一来源在 graph 节点共享模块（prepare 节点渲染使用），此处 re-export
# 保持旧导入路径兼容。
from app.core.agent.graph.nodes._common import SKILLS_SYSTEM_PROMPT  # noqa: E402,F401


# ---------------------------------------------------------------------------
# 依赖抽象
class OrchestratorConfig:
    def get(self, key: str, default: Any = None): ...


@dataclass
class IntentContext:
    intent: str = "general"
    confidence: float = 1.0
    slots: Dict[str, Any] = field(default_factory=dict)
    preferred_mode: Optional[OrchestrationMode] = None
    allowed_tools: Optional[List[str]] = None


@dataclass
class AgentResponse:
    answer: str
    mode_used: OrchestrationMode
    success: bool
    trace_id: str
    intent: IntentContext
    steps: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    degraded: bool = False
    # ── 2.2 HITL 扩展字段（均有默认值，旧调用方/旧测试零感知）─────────────
    run_id: Optional[str] = None
    """checkpoint thread_id（f"{session_id}:{uuid}"），审批恢复/状态查询用。"""
    awaiting_approval: bool = False
    """True 表示图已在危险工具 interrupt 点挂起并持久化，answer 暂为空。"""
    approval_payloads: List[Dict[str, Any]] = field(default_factory=list)
    """interrupt 载荷（工具名/参数/子任务），供 SSE awaiting_approval 事件透传。"""


class AgentOrchestrator:
    """智能体编排门面：组装每请求 GraphDeps，委派 GraphRunner 驱动状态图。"""

    def __init__(
            self,
            config: OrchestratorConfig,
            model_router: ModelRouter,
            memory_manager: MemoryManager,
            tool_registry: ToolRegistry,
            tracer: Tracer,
            skill_manager: SkillManager,
    ) -> None:
        """初始化编排器，注入全局核心基础设施单例。"""
        self._config: OrchestratorConfig = config
        self._model_router: ModelRouter = model_router
        self._memory: MemoryManager = memory_manager
        self._tools: ToolRegistry = tool_registry
        self._tracer: Tracer = tracer
        self._skill_manager: SkillManager = skill_manager
        self._graph_runner: Optional[Any] = None  # GraphRunner（lifespan 注入 Redis 版）

    # ------------------------------------------------------------------
    # GraphRunner 接线
    # ------------------------------------------------------------------
    def set_graph_runner(self, graph_runner: Any) -> None:
        """由 lifespan/dependencies 注入带 Redis checkpointer 的 GraphRunner。"""
        self._graph_runner = graph_runner

    def _get_graph_runner(self) -> Any:
        """惰性兜底：未注入时用 InMemorySaver 编译（进程内可跑，跨进程续跑不可用）。"""
        if self._graph_runner is None:
            from langgraph.checkpoint.memory import InMemorySaver

            from app.core.agent.graph.builder import compile_agent_graph
            from app.core.agent.graph.runner import GraphRunner

            logger.warning(
                "未注入 Redis GraphRunner，使用进程内 InMemorySaver 兜底"
                "（审批/断点续跑在进程重启后失效）。"
            )
            saver: Any = InMemorySaver()
            self._graph_runner = GraphRunner(compile_agent_graph(saver), saver)
        return self._graph_runner

    def _build_deps(self) -> GraphDeps:
        """组装每请求依赖：经 RunnableConfig 注入节点，不进入 checkpoint。"""
        return GraphDeps(
            config=self._config,
            model_router=self._model_router,
            memory=self._memory,
            tools=self._tools,
            skill_manager=self._skill_manager,
            tracer=self._tracer,
        )

    # ------------------------------------------------------------------
    # 门面主入口（签名与旧版完全一致）
    # ------------------------------------------------------------------
    @langfuse_observe(name="AgentOrchestrator.run", as_type="agent", capture_input=False, capture_output=False)
    async def run(
            self,
            user_input: str,
            session_id: str,
            mode: str = "react",
            intent: Optional[IntentContext] = None,
            precomputed_memory: Optional[Any] = None,
    ) -> AgentResponse:
        """执行 Agent 编排（状态图驱动）。

        - mode/intent.preferred_mode 仅决定是否经过 plan 节点；
        - 记忆/技能/工具预算/审批闸门等全部在图节点内完成；
        - 危险工具审批开启且命中时，返回 ``awaiting_approval=True`` 的挂起响应，
          调用方需凭 ``run_id`` 走审批恢复端点续跑。
        """
        intent_context: IntentContext = intent or IntentContext()
        try:
            outcome = await self._get_graph_runner().run(
                deps=self._build_deps(),
                user_input=user_input,
                session_id=session_id,
                mode=mode,
                intent=intent_context,
                precomputed_memory=precomputed_memory,
            )
            return outcome.response
        except Exception as system_uncaught_exception:  # noqa: BLE001 - 门面兜底契约
            logger.exception("智能体编排器门面发生未捕获的严重异常")
            return AgentResponse(
                answer="",
                mode_used="react",
                success=False,
                trace_id=getattr(self._tracer, "new_trace_id", lambda: "")(),
                intent=intent_context,
                steps=[],
                error=str(system_uncaught_exception),
            )

    # ------------------------------------------------------------------
    # HITL：审批恢复 / 运行状态查询（标准档新增，供 agent_runs 路由调用）
    # ------------------------------------------------------------------
    async def resume_approval(
        self,
        *,
        run_id: str,
        approved: bool,
        comment: str = "",
    ) -> AgentResponse:
        """审批决策后从 interrupt 点恢复执行。

        每请求依赖（ToolRegistry 等）在此重建；预算/消息等运行态以 checkpoint
        中的 state 账本为准，恢复后熔断状态不丢失。
        """
        outcome = await self._get_graph_runner().resume(
            run_id=run_id,
            deps=self._build_deps(),
            approved=approved,
            comment=comment,
        )
        return outcome.response

    async def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """查询某 run 的快照状态（是否暂停/下一节点/interrupt 载荷）。"""
        return await self._get_graph_runner().get_run_status(run_id)
