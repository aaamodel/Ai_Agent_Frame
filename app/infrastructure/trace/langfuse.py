# -*- coding: utf-8 -*-
"""Langfuse 可观测性接入（Agent 编排层观测调试）。

设计要点：
  - 配置驱动开关：settings.LANGFUSE_*。公钥/私钥任一为空 → 不做任何初始化，
    langfuse 内部自动进入 disabled 状态，所有 @observe 退化为透明 no-op，
    对既有调用链路零侵入、零风险（无需改动线上行为）。
  - 一旦开启，编排层关键节点（AgentOrchestrator.run / ToolRegistry.invoke /
    底层 LLM 调用）自动产生 Trace / Span / Generation 并上报，父-子嵌套关系
    由 langfuse 基于 OpenTelemetry 的 contextvars 自动维护。
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from typing import Any, ContextManager, Dict, Iterator, Optional, Tuple

from loguru import logger

from langfuse import Langfuse, get_client


# ─────────────────────────────────────────────────────────────────────────────
# 进程级禁用 langfuse.openai 的全局 openai SDK 猴子补丁
#
# ``langfuse/openai.py`` 在**导入期**调用 register_tracing()，用 wrapt 给整个
# 进程的 openai SDK（chat.completions / embeddings 等）打全局包裹。第三方依赖
# （lightrag/llm/openai.py）检测到 LANGFUSE_* 环境变量就会 ``from
# langfuse.openai import AsyncOpenAI`` 而触发它；补丁装上后，进程内所有 openai
# 调用都会额外产生 OpenAI-generation / OpenAI-embedding span——其中会话外的
# 调用（后台预热、文档索引 embedding、抽取）各自成为**根 trace**，把
# Langfuse 的会话视图打散成"每个模型调用一条 trace"。
#
# 本项目统一使用自有埋点（@observe 会话根 + tool_invoke + llm.* generation），
# 任何模块都不应再使用 langfuse.openai 自动包裹，因此在 finder 层进程级禁用：
# lightrag 的 ``except ImportError`` 会自动回退到标准 openai.AsyncOpenAI。
# ─────────────────────────────────────────────────────────────────────────────
_BLOCK_MESSAGE = (
    "langfuse.openai 在本项目中被禁用：其 import 期 register_tracing() 会全局"
    "猴子补丁 openai SDK，使会话外的 LLM/embedding 调用各自成为根 trace。"
    "观测统一走 app.infrastructure.trace.langfuse 的 @observe 手工埋点。"
)


class _BlockLangfuseOpenAIImport:
    """meta_path finder：对 ``langfuse.openai`` 的导入直接抛 ImportError。"""

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if fullname == "langfuse.openai":
            raise ImportError(_BLOCK_MESSAGE)
        return None


def _try_unwrap_openai_patch() -> None:
    """若阻断器安装前补丁已被其他导入链抢先注册，尽力拆除 wrapt 包裹。"""
    if "langfuse.openai" not in sys.modules:
        return
    try:
        mod = importlib.import_module("langfuse.openai")
        resources = getattr(mod, "OPENAI_METHODS_V1", None)
        for resource in resources or []:
            target_mod = importlib.import_module(resource.module)
            obj = target_mod
            for part in str(resource.object).split("."):
                obj = getattr(obj, part)
            current = getattr(obj, resource.method, None)
            # 补丁可能层层包裹，逐层剥回原始函数
            seen = 0
            while current is not None and hasattr(current, "__wrapped__") and seen < 5:
                current = current.__wrapped__
                seen += 1
            if seen:
                setattr(obj, resource.method, current)
        logger.warning(
            "langfuse.openai 已抢先导入并补丁 openai SDK，已尽力拆除 {} 处包裹。",
            len(resources or []),
        )
    except Exception:  # pragma: no cover - 纯防御，失败也不应影响启动
        logger.debug("拆除 langfuse.openai 补丁时异常（忽略）。", exc_info=True)


def disable_langfuse_openai_autopatch() -> None:
    """安装进程级阻断（幂等）。在应用包导入早期执行一次即可。"""
    if not any(isinstance(finder, _BlockLangfuseOpenAIImport) for finder in sys.meta_path):
        _try_unwrap_openai_patch()
        sys.meta_path.insert(0, _BlockLangfuseOpenAIImport())


disable_langfuse_openai_autopatch()


def resolve_langfuse_keys() -> Tuple[str, str, str]:
    """从应用配置读取 Langfuse 三要素（公钥 / 私钥 / 服务地址）。

    Returns:
        元组 (public_key, secret_key, host)。
    """
    from app.config import get_settings  # 延迟导入，避免与 config 形成循环依赖

    settings = get_settings()
    public_key = (settings.langfuse_public_key or "").strip()
    secret_key = (settings.langfuse_secret_key or "").strip()
    host = (settings.langfuse_host or "https://cloud.langfuse.com").strip()
    return public_key, secret_key, host


def is_langfuse_enabled() -> bool:
    """是否已配置并启用 Langfuse 观测。"""
    public_key, secret_key, _ = resolve_langfuse_keys()
    return bool(public_key and secret_key)


# ─────────────────────────────────────────────────────────────────────────────
# 统一的会话内观测门控
#
# 所有 LLM / embedding 埋点（llm_model_router 咽喉、DashScopeEmbedding、
# LightRAG、sql_vanna）统一走下面的 span 工厂：只有当前执行流**已经处在一条
# Langfuse trace 内**（即 AgentOrchestrator.run / AgentOrchestrator.resume
# 根 span 之下）才开子 observation。
#
# 会话外的模型调用——启动预热、意图向量索引、文档批量抽取/入库 embedding、
# 后台记忆摘要等——不开任何 observation：否则它们会各自成为根 trace，
# 把 Langfuse 的会话视图打散成"每个模型调用一条 trace"（历史故障现象）。
# ─────────────────────────────────────────────────────────────────────────────
def current_trace_active() -> bool:
    """当前上下文是否已在某条 Langfuse trace（根 span）内。"""
    if not is_langfuse_enabled():
        return False
    try:
        return get_client().get_current_observation_id() is not None
    except Exception:  # pragma: no cover - 观测层异常一律视为不在 trace 内
        return False


def _observation_span(as_type: str, name: str, model: Optional[str],
                      input: Optional[Dict[str, Any]]):
    """构造一个门控的 observation contextmanager（generation/embedding 共用）。"""
    @contextlib.contextmanager
    def _cm() -> Iterator[Optional[Any]]:
        if not current_trace_active():
            yield None
            return
        try:
            cm = get_client().start_as_current_observation(
                name=name, as_type=as_type, input=input, model=model,
            )
            span = cm.__enter__()
        except Exception:  # pragma: no cover - 观测层初始化失败 → no-op
            logger.debug("Langfuse %s span 初始化异常，忽略。", as_type, exc_info=False)
            yield None
            return
        # 真实调用在 yield 出的 with 体内发生：其异常必须原样上抛，
        # 只保证 span 被关闭，绝不能吞 APIError/RateLimitError。
        try:
            yield span
        except BaseException:
            try:
                cm.__exit__(*sys.exc_info())
            except Exception:  # pragma: no cover
                logger.debug("Langfuse %s span 异常关闭失败，忽略。", as_type)
            raise
        else:
            try:
                cm.__exit__(None, None, None)
            except Exception:  # pragma: no cover
                logger.debug("Langfuse %s span 关闭失败，忽略。", as_type)

    return _cm()


def generation_span(
    name: str,
    *,
    model: Optional[str] = None,
    input: Optional[Dict[str, Any]] = None,  # noqa: A002 - 对齐 langfuse 字段名
) -> ContextManager[Optional[Any]]:
    """会话内的一次 LLM 调用 observation（``as_type=generation``）。

    会话外或观测层异常时退化为 no-op（yield None），真实调用方零分支成本。
    yield 出 span 对象，调用方可在拿到响应后 ``span.update(output=...,
    usage_details=...)`` 富化。
    """
    return _observation_span("generation", name, model, input)


def embedding_span(
    name: str,
    *,
    model: Optional[str] = None,
    input: Optional[Dict[str, Any]] = None,  # noqa: A002
) -> ContextManager[Optional[Any]]:
    """会话内的一次向量调用 observation（``as_type=embedding``），门控同上。"""
    return _observation_span("embedding", name, model, input)


def setup_langfuse() -> bool:
    """应用启动时初始化 Langfuse 单例客户端。

    Returns:
        True 表示已启用；False 表示未配置或初始化失败（观测关闭）。
    """
    public_key, secret_key, host = resolve_langfuse_keys()
    if not (public_key and secret_key):
        logger.info(
            "Langfuse 未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，观测关闭（@observe 退化为 no-op）。"
        )
        return False

    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
    os.environ.setdefault("LANGFUSE_BASE_URL", host)
    try:
        # 显式构建唯一客户端（单项目场景），get_client() 复用它
        Langfuse(public_key=public_key, secret_key=secret_key, base_url=host)
        logger.info("Langfuse 可观测性已启用，上报地址: {}", host)
    except Exception as exc:  # pragma: no cover - 初始化异常保守兜底
        logger.warning("Langfuse 客户端初始化失败（观测关闭，不影响主流程）: {}", exc)
        return False
    return True


def flush_langfuse() -> None:
    """进程退出前冲刷发送缓冲，确保已产生的观测记录全部上报。"""
    if not is_langfuse_enabled():
        return
    try:
        get_client().flush()
        logger.info("Langfuse 已冲洗发送缓冲。")
    except Exception as exc:  # pragma: no cover
        logger.warning("Langfuse flush 失败: {}", exc)