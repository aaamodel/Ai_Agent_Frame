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
    """联网搜索唯一对外工具（豆包 API 直连；内部自带 Tavily 降级通道）。

    降级策略（**全部在代码层完成，不经过大模型判断**）：
    1. 调用异常（网络 / 鉴权 / 服务报错，重试 2 次后仍失败）→ 降级 ``_fallback``（Tavily）
    2. 接口正常但 0 条文档 / 无摘要 → 同样降级 ``_fallback``
    两条通道都没有有效结果时，才返回"未检索到 / 工具故障"提示，让模型改用自身知识作答。

    之所以把两档合并：豆包返回空结果时，另一家搜索引擎往往能命中（索引源不同），
    而降级只是一次内部 HTTP 调用、不额外消耗 LLM 轮次，比让模型去判断"要不要换工具"便宜且确定。
    """

    # ===== 输出体积控制参数（可按 Agent 上下文窗口调优）=====
    # MAX_OUTPUT_TOTAL 4000-8000 ≈ 1k-2k token；MAX_SNIPPET_PER_DOC 建议 400-800；
    # KEEP_TOP_DOCS 建议 3-6。count 入参仍可传大值拉取更多候选，工具内部只精选 top N。
    MAX_SNIPPET_PER_DOC = 600   # 单篇文档摘要最大字符，防止单条 snippet 爆炸
    MAX_OUTPUT_TOTAL = 6000     # 搜索工具返回给 LLM 的整体最大字符，控制 token
    KEEP_TOP_DOCS = 5           # 只保留 Rank 靠前 N 篇文档，不全部返回

    SYSTEM_PROMPT: str = "联网搜索外部公开信息（新闻/行情/最新动态/公开资料）"



    def __init__(
        self,
        api_key: Optional[str] = None,
        version: str = "global",
        fallback: Optional[BaseTool] = None,
    ) -> None:
        super().__init__()
        self.name = "web_search"
        self.description = (
            "联网搜索外部公开信息（新闻/行情/最新动态/公开资料）。"
            "不用于公司内部文件、私有数据。"
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
                # HTTP 正常、业务返回空数据：不再重试相同请求，转内部降级通道
                got_empty_result = True
                logger.warning("豆包搜索接口调用成功，但 Documents 为空，query='{}'", query)
                break
            except Exception as exc:
                last_exception = exc
                logger.warning("豆包搜索第 {} 次尝试发生异常，原因: {}", attempt, exc)

            if attempt < doubao_max_retries:
                await asyncio.sleep(1.0)

        # 统一走代码层降级：空结果与调用异常都交给内部通道，模型不参与"要不要换工具"的决策
        fallback_text = await self._try_fallback(query, got_empty_result)
        if fallback_text:
            return fallback_text

        # 两条通道都无有效结果：按失败成因返回对应提示，让模型改用自身知识作答
        if got_empty_result:
            return _SEARCH_EMPTY_RESULT
        logger.error("豆包搜索与降级通道均发生故障（最后异常: {}）。", repr(last_exception))
        return _SEARCH_TOOL_FAILURE_SUFFIX.format(query=query)

    async def _try_fallback(self, query: str, doubao_empty: bool) -> Optional[str]:
        """调用内部降级通道（Tavily）；拿不到有效结果时返回 None。

        ⚠️ 这是**纯代码层降级**：该通道（``search.WebSearchTool``）不注册进 ToolRegistry、
        也不出现在任何意图白名单里，模型既看不到也调不到它，"何时降级"由本方法决定。

        Args:
            query: 用户检索问句（原样透传，降级不再改写 query）。
            doubao_empty: True 表示豆包"接口正常但无结果"，False 表示调用异常。

        Returns:
            可用的搜索结果文本；未注入通道 / 抛异常 / 返回"【系统提示】"占位时返回 None。
        """
        if self._fallback is None:
            return None
        reason: str = "接口返回空结果" if doubao_empty else "调用发生异常"
        logger.warning(
            "豆包搜索【{}】，降级到内部通道 [{}]，query='{}'", reason, self._fallback.name, query
        )
        try:
            fallback_result = await self._fallback.execute(query=query)
        except Exception as exc:  # noqa: BLE001 - 降级通道失败只影响最终文案，不向上抛
            logger.warning("降级通道执行失败，原因: {}", exc)
            return None
        text: str = str(fallback_result)
        # 降级通道以"【系统提示】"开头表示"我也没拿到有效结果"
        if "【系统提示】" in text:
            logger.warning("降级通道也未返回有效结果，query='{}'", query)
            return None
        logger.info("降级通道搜索成功，query='{}'，返回内容长度={}", query, len(text))
        return text

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
