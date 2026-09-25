# -*- coding: utf-8 -*-
"""file_grep 排除 trace_to_markdown 调试落盘文件的单测。

2026-09-24 事故：file_grep_tool('线索更新') 在仓库根执行时，命中的全是
运行期自己落盘的 trace md（build_question_keywords.md 等），真实业务文档
零命中——Agent 把"自己产生的观测回声"当成了证据，并诱发后续幻觉。
grep 的两个引擎（ripgrep / 纯 Python 兜底）都必须按首行标记剔除这类文件。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.backends.filesystem import FilesystemBackend  # noqa: E402

TRACE_MD = "# 🔍 函数变量追踪报告: `build_question_keywords`\n\n一些内容\n"
NEEDLE = "线索更新频率"


def test_is_trace_artifact_marker_detection(tmp_path):
    trace_file = tmp_path / "trace.md"
    trace_file.write_text(TRACE_MD, encoding="utf-8")
    normal_file = tmp_path / "normal.md"
    normal_file.write_text("# 线索阶段流转规则\n普通业务文档\n", encoding="utf-8")

    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    assert backend._is_trace_artifact(trace_file) is True
    assert backend._is_trace_artifact(normal_file) is False
    # 非 .md 不做标记识别
    assert backend._is_trace_artifact(tmp_path / "a.txt") is False


def test_grep_excludes_trace_artifacts(tmp_path):
    # 1) trace 调试文件：关键词只出现在这种"观测回声"里时必须零命中
    trace_file = tmp_path / "build_question_keywords.md"
    trace_file.write_text(TRACE_MD + f"\n检索{NEEDLE}的子任务\n", encoding="utf-8")

    # 2) 正常业务文档：唯一允许命中的文件
    real_file = tmp_path / "线索阶段流转规则.md"
    real_file.write_text(f"# 规则\n{NEEDLE}：每月一次\n", encoding="utf-8")

    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    result = asyncio.run(backend.grep(pattern=NEEDLE, path="."))

    assert result.error is None
    matched_paths = {m["path"] for m in (result.matches or [])}
    assert len(matched_paths) == 1
    only_path = next(iter(matched_paths))
    assert only_path.endswith("线索阶段流转规则.md")
    assert "build_question_keywords.md" not in only_path


def test_grep_returns_no_matches_when_only_trace_artifacts_hit(tmp_path):
    trace_file = tmp_path / "retrieve_contexts.md"
    trace_file.write_text(TRACE_MD + f"\n{NEEDLE}\n", encoding="utf-8")

    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    result = asyncio.run(backend.grep(pattern=NEEDLE, path="."))

    # 不允许把 trace 命中当成业务证据；上层工具会据此渲染 "No matches found"
    assert result.error is None
    assert not result.matches
