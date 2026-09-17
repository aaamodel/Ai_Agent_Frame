# -*- coding: utf-8 -*-
"""内置网络搜索工具（基于 DuckDuckGo 即时答案或占位 HTTP）。"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter

import asyncio
import os
from typing import Any, Optional
from tavily import AsyncTavilyClient  # 需要先安装：pip install tavily-python



class WebSearchTool(BaseTool):
    """Tavily 联网搜索通道（**豆包搜索的内部降级专用，不注册进 ToolRegistry**）。

    ⚠️ 本工具不参与意图路由，也不下发给大模型——它的唯一调用方是
    ``DoubaoWebSearchTool``：豆包搜索**空结果或调用异常**时由代码直接降级调用
    （见 ``doubao_search.DoubaoWebSearchTool._try_fallback``）。

    因为模型根本看不到它，这里**不再编写任何"使用时机"提示词**（原 SYSTEM_PROMPT 已删除）：
    "什么时候该降级"不再是模型的判断题，而是 ``DoubaoWebSearchTool`` 里的 if 分支。
    保留 3 次退避重试，作为降级通道自身的健壮性。
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__()
        # name 仅用于日志留痕；由于未注册，LLM 无法通过它发起调用。
        # 刻意不叫 "tavily_web_search"（那个"第二梯队工具名"已从全链路消失），
        # 避免日志里出现一个在意图白名单/工具清单中都不存在的工具名造成误读。
        self.name = "tavily_search_internal"
        self.description = "Tavily 联网搜索（豆包搜索的内部降级通道，模型不可见）。"
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="搜索关键词或完整问句",
                required=True,
            )
        ]

        # 优先从初始化参数获取 Key，其次读取系统环境变量
        tavily_key = api_key or os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            logger.warning("未检测到 TAVILY_API_KEY，搜索工具在执行时可能会报错，请检查配置。")

        self._client = AsyncTavilyClient(api_key=tavily_key)

    async def execute(self, **kwargs: Any) -> str:
        """执行搜索：尝试最多 3 次，失败则返回提示让大模型用自身知识库回答。"""
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ValueError("参数 query 不能为空")

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.info("开始 Tavily 搜索，query='{}'，第 {}/{} 次尝试", query, attempt, max_retries)

                # 使用最稳妥的 search 方法
                response = await self._client.search(query=query, max_results=5)

                # 提取结果并拼接成干净的文本
                results = response.get("results", [])
                if results:
                    context_list = []
                    for idx, res in enumerate(results, 1):
                        title = res.get("title", "无标题")
                        content = res.get("content", "")
                        context_list.append(f"[{idx}] 标题: {title}\n内容: {content}\n")

                    context = "\n".join(context_list)
                    logger.info("Tavily 搜索成功，query='{}'，返回内容长度={}", query, len(context))
                    return f"以下是关于「{query}」的最新联网搜索结果：\n{context}"

            except Exception as exc:
                logger.warning("Tavily 搜索第 {} 次尝试失败，原因: {}", attempt, exc)
                if attempt < max_retries:
                    await asyncio.sleep(1.0)
                else:
                    logger.error("Tavily 搜索 3 次尝试全部失败。")

        return (
            f"【系统提示】联网搜索工具目前不可用（已尝试 {max_retries} 次均失败）。"
            f"请忽略联网搜索，直接根据你（大模型）自身拥有的知识和记忆，"
            f"尽可能准确地回答用户的问题：「{query}」。在回答中无需向用户提及搜索失败的事情。"
        )