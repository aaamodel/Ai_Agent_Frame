# -*- coding: utf-8 -*-
"""每请求运行时依赖容器：只走 RunnableConfig，不进 state、不参与 checkpoint 序列化。

节点内统一用法::

    from app.core.agent.graph.deps import get_deps
    deps = get_deps(config)
    deps.tools / deps.model_router / deps.memory / deps.skill_manager / deps.tracer

这些依赖（尤其 ToolRegistry）是每请求新建的，resume 请求会重建一份；
工具本身无状态，预算等运行态以 state 账本为准。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional, Set

from loguru import logger

DEPS_CONFIG_KEY = "deps"


@dataclass
class GraphDeps:
    """节点运行所需的全部外部基础设施（引用透传，图不拥有其生命周期）。"""

    config: Any                       # OrchestratorConfig（.get(key, default) 协议）
    model_router: Any
    memory: Any                       # MemoryManager
    tools: Any                        # ToolRegistry（每请求 bootstrap_tools 新建）
    skill_manager: Any                # SkillManager
    tracer: Any                       # Tracer
    background_tasks: Set[asyncio.Task] = field(default_factory=set)
    """单次 run 内的后台任务强引用登记表（防 GC 提前回收，长期记忆写等）。"""

    def cfg(self, key: str, default: Any = None) -> Any:
        """从 OrchestratorConfig 读配置的快捷方式。"""
        if self.config is None:
            return default
        return self.config.get(key, default)

    def spawn_background_task(self, coroutine: Any) -> asyncio.Task:
        """调度不阻塞主链路的后台任务：强引用防 GC + 异常仅告警（从 orchestrator 平移）。"""
        task: asyncio.Task = asyncio.create_task(coroutine)
        self.background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, finished_task: asyncio.Task) -> None:
        self.background_tasks.discard(finished_task)
        if finished_task.cancelled():
            return
        task_exception: Optional[BaseException] = finished_task.exception()
        if task_exception is not None:
            logger.warning("后台任务执行异常（已忽略，不影响主流程响应）: {}", task_exception)


def get_deps(config: Optional[dict]) -> GraphDeps:
    """从 LangGraph RunnableConfig 中取出 GraphDeps。

    Raises:
        KeyError: 调用方未注入 deps（属于接线错误，需快速暴露）。
    """
    configurable = (config or {}).get("configurable") or {}
    deps = configurable.get(DEPS_CONFIG_KEY)
    if deps is None:
        raise KeyError(
            "RunnableConfig['configurable']['deps'] 未注入：图必须由 GraphRunner "
            "通过 configurable 传入 GraphDeps，节点不可直接 import 全局单例。"
        )
    return deps


def with_deps(run_id: str, deps: GraphDeps) -> dict:
    """组装 RunnableConfig 的 configurable 段（runner/resume API 共用）。"""
    return {"configurable": {"thread_id": run_id, DEPS_CONFIG_KEY: deps}}
