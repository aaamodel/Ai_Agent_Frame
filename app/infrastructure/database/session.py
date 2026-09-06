# -*- coding: utf-8 -*-
"""异步数据库引擎与会话工厂（已集成超时控制与连接池优化）。"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any

from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# 默认回退地址调整为 localhost（避免硬编码特定容器 IP 导致 TCP 握手超时）
_default_url = "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db"


def normalize_async_database_url(url: str) -> str:
    """将同步驱动 URL 转为 SQLAlchemy 异步 URL。"""
    if "+asyncpg" in url:
        return url
    u = url.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    u = u.replace("postgres://", "postgresql+asyncpg://")
    u = re.sub(
        r"^postgresql://",
        "postgresql+asyncpg://",
        u,
    )
    return u


def init_engine(database_url: str | None = None, **engine_kwargs: Any) -> AsyncEngine:
    """创建异步引擎（支持连接超时限制与池化参数优化）。"""
    raw_url = database_url or _default_url
    url = normalize_async_database_url(raw_url)

    # 🛠️ 核心优化参数设置
    default_kwargs: dict[str, Any] = {
        "echo": False,
        "pool_pre_ping": True,         # 每次获取连接前心跳检测，防止失效连接
        "pool_size": 20,                # 常用连接池大小
        "max_overflow": 10,             # 突发并发最大溢出连接数
        "pool_timeout": 5.0,            # 连接池满时获取连接的最大等待时间（秒）
        "pool_recycle": 1800,           # 30 分钟自动回收连接，防止 DB 端主动断开
        "connect_args": {
            "timeout": 5.0,             # ⚡ 核心修复：TCP 建立连接超时限制为 5 秒
            "command_timeout": 15.0     # 单条 SQL 执行超时限制为 15 秒
        },
    }

    default_kwargs.update(engine_kwargs)
    engine = create_async_engine(url, **default_kwargs)
    logger.info("数据库引擎已初始化（配置 5 秒连接超时与连接池健康检查）")
    return engine


_engine: AsyncEngine | None = None
async_session_factory: async_sessionmaker[AsyncSession] | None = None


def configure_session(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """绑定全局 session 工厂。"""
    global _engine, async_session_factory
    _engine = engine
    async_session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
        autoflush=False,
    )
    return async_session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """依赖注入用：获取异步会话（由路由层负责 commit）。"""
    if async_session_factory is None:
        raise RuntimeError("请先调用 configure_session(init_engine(...))")
    async with async_session_factory() as session:
        yield session