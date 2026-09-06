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
    """Tavily 联网搜索工具（第二梯队备选；带 3 次失败退避重试）。

    通过提示词（description + SYSTEM_PROMPT）显著限制使用时机：

    """

    # 第二梯队使用守则（tool_to_function_call_definition 会自动拼入 function description，
    # 让大模型在 FC/ReAct/Plan 场景下都能看到本梯队约束）
    SYSTEM_PROMPT: str = (
        "## tavily_web_search 使用限制（第二梯队备选）\n"
        "1. 本工具是联网搜索的**第二梯队备选**，第一梯队是 `web_search`（豆包搜索）。\n"
        "2. 仅当满足以下任一条件时才允许调用本工具：\n"
        "   - `web_search` 已被系统标记为熔断/不可用；\n"
        "   - `web_search` 的返回明确提示搜索失败或长期无有效结果。\n"
        "3. 严禁把本工具作为联网搜索的首选；只要 `web_search` 还可用，一律优先调用 `web_search`。\n"
        "4. 严禁对同一个问题同时并行调用两个搜索工具；切换到本工具前必须先看过 `web_search` 的失败结果。"
    )

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__()
        self.name = "tavily_web_search"
        self.description = (
            "Tavily 联网搜索的备选，用于查询外部互联网的公开、实时新闻与公开资料。"
            "仅当第一梯队 web_search（豆包搜索）不可用、被熔断或返回失败提示时才允许使用；"
            "绝不能用于查询任何公司内部文件、机密合同或私有资产。"
        )
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