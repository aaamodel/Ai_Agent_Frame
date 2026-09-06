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

import os
from typing import Tuple

from loguru import logger

from langfuse import Langfuse, get_client


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