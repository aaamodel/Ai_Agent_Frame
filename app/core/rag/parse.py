# -*- coding: utf-8 -*-
"""文档解析与分块（LlamaIndex 重构，替代旧 file_pipe）。

- parse_bytes_to_text：按扩展名解析内存字节为纯文本（pdf 走 pypdf，其余按
  utf-8 解码），避免依赖磁盘临时文件。
- split_text：用 LlamaIndex SentenceSplitter 按 token 智能分块。
"""

from __future__ import annotations

import io
from typing import List

from llama_index.core.node_parser import SentenceSplitter

from loguru import logger


def parse_bytes_to_text(content: bytes, filename: str) -> str:
    """把上传文件字节解析为纯文本。

    Args:
        content: 上传文件原始字节。
        filename: 文件名（用于判定扩展名与日志）。

    Returns:
        解析后的纯文本；解析失败/空内容时返回空串。
    """
    name = (filename or "").lower()

    if not content:
        return ""

    try:
        if name.endswith(".pdf"):
            return _parse_pdf(content, name)

        # 其余类型（txt/md/csv/docx 原始等）按 utf-8 强解码
        text = content.decode("utf-8", errors="ignore").strip()
        # 移除可能出现的 BOM
        if text.startswith("\ufeff"):
            text = text[1:]
        return text
    except Exception as exc:
        logger.warning("文档解析失败 filename=%s err=%s", filename, exc)
        return ""


def _parse_pdf(content: bytes, filename: str) -> str:
    """用 pypdf 从内存字节解析 PDF 文本。"""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages: List[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # pragma: no cover - 单页容错
                continue
        text = "\n".join(pages).strip()
        logger.info("PDF 解析完成：{} 共 {} 页、{} 字。", filename, len(pages), len(text))
        return text
    except Exception as exc:
        logger.warning("PDF 解析失败 filename=%s err=%s", filename, exc)
        return ""


def split_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[str]:
    """用 LlamaIndex SentenceSplitter 按 token 分块。

    Args:
        text: 待分块纯文本。
        chunk_size: 每块最大 token 数。
        chunk_overlap: 相邻块重叠 token 数。

    Returns:
        分块后的文本列表（可能为空）。
    """
    text = (text or "").strip()
    if not text:
        return []
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_tokenizer_fn=None,
    )
    tokens = splitter.split_text(text)
    chunks = [c.strip() for c in tokens if c and c.strip()]
    logger.info("分块完成：{} 字 → {} 块（chunk_size={}, overlap={}）", len(text), len(chunks), chunk_size, chunk_overlap)
    return chunks