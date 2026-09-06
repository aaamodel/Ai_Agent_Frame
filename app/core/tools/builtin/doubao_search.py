# -*- coding: utf-8 -*-
"""豆包搜索（火山引擎 Torchlight / Search Infinity）联网搜索工具：HTTP 直连为主，Tavily 为第二备选。

真实接口返回结构（已通过 DEBUG 日志验证）：
{
  "ResponseMetadata": {"RequestId": "...", ...},
  "Result": {
    "TotalDocCount": 20,
    "Summary": "...",   // NeedSummary=True 时服务端生成的整体综合摘要（token 极低，优先使用）
    "Documents": [
      {
        "Rank": 0,
        "Url": "...",
        "Title": "...",
        "Snippet": [{"Type": "text", "Text": "..."}],
        "DocumentInfo": {
          "ContentCharCount": 14520,
          "ContentTokenCount": 7264,
          "Filetype": "webpage",
          "PublishTime": "2023-04-15T20:49:59+08:00"
        },
        "HostInfo": {"Hostname": "抖音百科", "IconUrl": "..."}
      }
    ]
  }
}
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

import httpx
from loguru import logger

from app.core.tools.base import BaseTool, ToolParameter

from load_dotenv import load_dotenv
load_dotenv()
# 搜索工具真正故障（网络/鉴权/服务异常）时的提示
_SEARCH_TOOL_FAILURE_SUFFIX = (
    "【系统提示】联网搜索工具发生故障。"
    "请忽略联网搜索，直接根据你（大模型）自身拥有的知识和记忆，"
    "尽可能准确地回答用户的问题：「{query}」。在回答中无需向用户提及搜索失败的事情。"
)

# 接口正常但无匹配网页
_SEARCH_EMPTY_RESULT = "【系统提示】搜索接口调用成功，但未检索到匹配网页数据。"

_VALID_TIME_RANGE = {"OneDay", "OneWeek", "OneMonth", "OneYear"}


class DoubaoWebSearchTool(BaseTool):
    """豆包搜索 API 直连工具（第一梯队主）。

    区分两种情况：
    1. 工具故障：http 异常、鉴权错误、服务报错 → 自动降级 Tavily
    2. API 返回 0 条文档：接口正常，只是没有网页资料 → 不降级，直接返回空结果提示
    """

    # ===== 输出体积控制参数（可按 Agent 上下文窗口调优）=====
    # MAX_OUTPUT_TOTAL 4000-8000 ≈ 1k-2k token；MAX_SNIPPET_PER_DOC 建议 400-800；
    # KEEP_TOP_DOCS 建议 3-6。count 入参仍可传大值拉取更多候选，工具内部只精选 top N。
    MAX_SNIPPET_PER_DOC = 600   # 单篇文档摘要最大字符，防止单条 snippet 爆炸
    MAX_OUTPUT_TOTAL = 6000     # 搜索工具返回给 LLM 的整体最大字符，控制 token
    KEEP_TOP_DOCS = 5           # 只保留 Rank 靠前 N 篇文档，不全部返回

    SYSTEM_PROMPT: str = (
        "## web_search 外部信息联网搜索工具，优先使用此工具\n"
        "1. 本工具（豆包搜索）是联网搜索的**第一梯队首选通道**：需要外部互联网的实时、"
        "公开信息（新闻、行情、最新动态、公开资料）时，一律优先调用本工具。\n"
        "2. 仅在以下情况才考虑切换到第二梯队备选 `tavily_web_search`：本工具被系统标记为"
        "熔断/不可用，或返回结果明确提示搜索失败、长期无有效数据。\n"
        "3. 本工具绝不能用于查询公司内部文件、机密合同或私有资产（内部数据请用 "
        "rag_knowledge_search / knowledge_graph_search / local_excel_tool 等内部检索工具）。\n"
        "4. `count` 控制结果条数（1-10），`time_range` 可限定网页发布时间"
        "（OneDay / OneWeek / OneMonth / OneYear），时效性强的问题建议传 time_range。"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        version: str = "global",
        fallback: Optional[BaseTool] = None,
    ) -> None:
        super().__init__()
        self.name = "web_search"
        self.description = (
            "联网检索工具的首选，用于查询外部互联网的公开、"
            "实时新闻、最新动态与公开资料；仅在自身服务故障时由系统降级到备选tavily_web_search "
            "。绝对不能用于查询任何公司内部文件、机密合同或私有资产。"
        )
        self.parameters = [
            ToolParameter(name="query", type="string", description="搜索关键词或完整问句", required=True),
            ToolParameter(name="count", type="integer", description="返回结果条数，范围 1-10，默认 5", required=False),
            ToolParameter(name="time_range", type="string", description="可选时间过滤：OneDay / OneWeek / OneMonth / OneYear", required=False),
        ]

        self._api_key = api_key or os.getenv("ARK_SEARCH_API_KEY")
        if not self._api_key:
            logger.warning("未检测到 ARK_SEARCH_API_KEY，豆包搜索主通道将不可用。")

        if version == "global":
            self._url = "https://open.feedcoopapi.com/search_api/global_search"
        elif version == "custom":
            self._url = "https://open.feedcoopapi.com/search_api/web_search"
        else:
            raise ValueError("version 只能是 global / custom")

        self._fallback = fallback

    async def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            raise ValueError("参数 query 不能为空")

        count = self._clamp_count(kwargs.get("count", 5))
        time_range = self._normalize_time_range(kwargs.get("time_range"))

        doubao_max_retries = 2
        last_exception: Optional[Exception] = None
        got_empty_result = False

        for attempt in range(1, doubao_max_retries + 1):
            try:
                logger.info("开始豆包搜索，query='{}'，第 {}/{} 次尝试", query, attempt, doubao_max_retries)
                context = await self._search_via_doubao(query, count, time_range)
                if context:
                    logger.info("豆包搜索成功，query='{}'，返回内容长度={}", query, len(context))
                    return f"以下是关于「{query}」的最新联网搜索结果：\n{context}"
                # HTTP 正常、业务返回空数据：不再重试相同请求，直接标记空结果
                got_empty_result = True
                logger.warning("豆包搜索接口调用成功，但 Documents 为空，query='{}'", query)
                break
            except Exception as exc:
                last_exception = exc
                logger.warning("豆包搜索第 {} 次尝试发生异常，原因: {}", attempt, exc)

            if attempt < doubao_max_retries:
                await asyncio.sleep(1.0)

        # 分支1：接口正常，只是无网页，不走降级
        if got_empty_result:
            return _SEARCH_EMPTY_RESULT

        # 分支2：发生网络/鉴权异常，才走 fallback 降级
        if self._fallback is not None:
            logger.warning("豆包搜索【服务异常】（{}），降级到备选搜索工具 [{}]", repr(last_exception), self._fallback.name)
            try:
                fallback_result = await self._fallback.execute(query=query)
                text = str(fallback_result)
                if "【系统提示】" not in text:
                    logger.info("备选搜索成功，query='{}'，返回内容长度={}", query, len(text))
                    return text
                logger.warning("备选搜索也未返回有效结果，query='{}'", query)
            except Exception as exc:
                logger.warning("备选搜索执行失败，原因: {}", exc)

        logger.error("豆包搜索与备选通道均发生故障（最后异常: {}）。", repr(last_exception))
        return _SEARCH_TOOL_FAILURE_SUFFIX.format(query=query)

    async def _search_via_doubao(self, query: str, count: int, time_range: Optional[str]) -> str:
        """调用豆包搜索 HTTP 接口并格式化结果；0 条结果返回空串；HTTP 异常直接 raise。

        优化：不再全量输出所有文档——优先 API 综合摘要（Result.Summary），
        精选 top-N 文档片段做来源补充，单条与全局双重截断，控制总 token。
        """
        payload: dict[str, Any] = {
            "query": query,
            "count": count,
            "NeedSummary": True,
        }
        if time_range:
            payload["TimeRange"] = time_range

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        logger.debug(f"[DoubaoSearch] POST url={self._url}")
        logger.debug(f"[DoubaoSearch] request payload={payload}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self._url, json=payload, headers=headers)
            logger.debug(f"[DoubaoSearch] http status_code={resp.status_code}")
            logger.debug(f"[DoubaoSearch] raw response(first 1500): {resp.text[:1500]}")
            resp.raise_for_status()
            data = resp.json()

        # ===== 修复：按真实接口结构解析 =====
        result_obj = data.get("Result") or {}
        total_count = result_obj.get("TotalDocCount", 0)
        documents = result_obj.get("Documents") or []
        # 读取服务端生成的整体综合摘要（NeedSummary=True 时接口返回）
        api_summary = str(result_obj.get("Summary") or "").strip()

        logger.debug(f"[DoubaoSearch] TotalDocCount={total_count}, Documents len={len(documents)}, Summary len={len(api_summary)}")

        if not documents and not api_summary:
            return ""

        # 优化：不再全量输出所有文档——优先 API 综合摘要 + 精选 top-N 文档片段，控制总 token
        chunks: list[str] = []

        # 1) 综合摘要优先：信息密度最高、token 最少
        if api_summary:
            chunks.append(f"【搜索综合摘要】\n{api_summary}\n")

        # 2) 只取 Rank 靠前 KEEP_TOP_DOCS 篇文档做来源补充
        top_docs = documents[: self.KEEP_TOP_DOCS]

        for idx, doc in enumerate(top_docs, 1):
            title = doc.get("Title") or "无标题"
            url = doc.get("Url") or ""

            # Snippet 是数组，每个元素有 Type 和 Text；拼接所有 text 类型片段并单条截断
            snippet_parts: list[str] = []
            for snip in (doc.get("Snippet") or []):
                if isinstance(snip, dict) and snip.get("Type") == "text":
                    snippet_parts.append(snip.get("Text", ""))
            summary_raw = "\n".join(snippet_parts) if snippet_parts else ""
            summary = summary_raw[: self.MAX_SNIPPET_PER_DOC]

            # 发布时间在 DocumentInfo 下
            doc_info = doc.get("DocumentInfo") or {}
            publish_time = doc_info.get("PublishTime")

            # 来源站点
            host_info = doc.get("HostInfo") or {}
            hostname = host_info.get("Hostname")

            line = f"[{idx}] 标题: {title}\n链接: {url}"
            if hostname:
                line += f"\n来源: {hostname}"
            if summary:
                line += f"\n摘要: {summary}"
            if publish_time:
                line += f"\n发布时间: {publish_time}"
            chunks.append(line + "\n")

        final_text = "\n".join(chunks)

        # 3) 全局总字符截断，防止极端大返回把 token 打满
        if len(final_text) > self.MAX_OUTPUT_TOTAL:
            final_text = final_text[: self.MAX_OUTPUT_TOTAL] + "\n【结果过长，已做截断】"

        return final_text

    @staticmethod
    def _clamp_count(raw: Any) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 5
        return max(1, min(10, value))

    @staticmethod
    def _normalize_time_range(raw: Any) -> Optional[str]:
        if raw is None:
            return None
        value = str(raw).strip()
        return value if value in _VALID_TIME_RANGE else None
