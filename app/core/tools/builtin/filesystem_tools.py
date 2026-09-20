# -*- coding: utf-8 -*-
"""内置文件系统工具集：将 FilesystemBackend 封装为大模型可调用的 BaseTool 实例。

提供文件读取、目录扫描、以及高性能文本检索能力，所有接口故障均被平滑包装为大模型可读的错误文本。
"""

from typing import Any, List, Optional
import fnmatch
import json

from app.core.tools.base import BaseTool, ToolParameter
from app.core.backends.filesystem import FilesystemBackend


class FileReadTool(BaseTool):
    """读取指定路径文件的工具（大模型可用此查看具体技能的源码/配置闭环）。"""

    name: str = "file_read_tool"
    description: str = (
        "读取文本类文件（代码/配置/文档/日志等）。"
        "不用于 Excel 等二进制文件（业务数据请直接用 sales_sql_query 查库）。"
    )
    """
    description: str = (
        "【使用时机】当你判断用户问题的答案很可能存在于本地某个文本类文件（如 .py/.md/.json/.txt/.yaml/.log/.ini/.toml/.csv 等纯文本文件）中，"
        "或者你已经通过 file_list_tool / file_grep_tool 锁定了具体文件、现在需要查看它的完整内容或某段行范围时，调用本工具。"
        "典型场景：阅读某段源码、查看配置文件、读取文档正文、按行分页查看长文件。"
        "【能力】按行读取指定路径文件内容，支持 offset（起始行，0 基）和 limit（最大行数，默认 2000）分页，防止单次内容过长打爆上下文。"
        "【不要用于】读取 Excel/二进制等非文本文件——这类请改用 sales_sql_query 查业务库；本工具只面向可解码为文本的文件。"
        "【前提】请先用 file_list_tool 或 file_grep_tool 确认目标路径真实存在，避免臆造路径。"
    )
    """

    def __init__(self, fs_backend: FilesystemBackend) -> None:
        super().__init__()
        self.fs_backend = fs_backend

        # 声明工具入参规则，用于生成契约式的 OpenAI Tool Json Schema
        self.parameters = [
            ToolParameter(
                name="file_path",
                type="string",
                description="需要读取的目标文件的相对路径或绝对路径",
                required=True,
            ),
            ToolParameter(
                name="offset",
                type="integer",
                description="读取的起始行数索引（0 表示从第一行开始），默认为 0",
                required=False,
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="单次最大读取的行数，默认为 2000 行",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> str:
        """执行文件流安全读取操作。"""
        file_path = kwargs.get("file_path")
        if not file_path:
            return "Error: Missing required parameter 'file_path'."

        # 防御 LLM 偶尔将数值写为字符串格式的情况，做一层稳健的类型强制转换
        try:
            offset = int(kwargs.get("offset", 0))
            limit = int(kwargs.get("limit", 2000))
        except (ValueError, TypeError):
            return "Error: Parameters 'offset' and 'limit' must be valid integers."

        # 调用后端核心接口
        result = await self.fs_backend.read(file_path=file_path, offset=offset, limit=limit)

        # 平滑处理错误，把异常当成 Observation 传递给 LLM
        if result.error:
            return f"Error: Failed to read file '{file_path}'. Reason: {result.error}"

        if not result.file_data or not result.file_data.get("content"):
            return f"--- File '{file_path}' is empty or offset out of bounds ---"

        return result.file_data["content"]


class FileListTool(BaseTool):
    """列出目录内容的工具（支持递归深度、文件名过滤与结果截断）。"""

    name: str = "file_list_tool"
    description: str = (
        "列出目录内容，支持递归多层（depth，默认 3 层）与文件名过滤（pattern，如 '*.xlsx'）。"
        "不确定文件在哪、或读文件前确认路径时用；按**内容**搜索请用 file_grep_tool。"
    )

    # ── 深度 / 体积控制 ──────────────────────────────────────────────────
    # 默认 3 层：足够覆盖 raw_data/sales_intel/*.xlsx 这类 2~3 层目录结构，
    # 让模型**一次调用**就定位到目标文件（旧实现只列 1 层，实测一条用例要连调
    # 3 次 file_list_tool 一层层往下摸：raw_data → raw_data/sales_intel → 才到文件）。
    DEFAULT_DEPTH: int = 3
    MAX_DEPTH: int = 8
    DEFAULT_MAX_ENTRIES: int = 200
    MAX_MAX_ENTRIES: int = 1000
    # 字符级兜底：条目数已受 max_entries 约束，但超长文件名/深缩进仍可能撑爆上下文
    MAX_OUTPUT_CHARS: int = 8000

    # 递归时**不进入也不列出**的噪音目录。
    # ⚠️ 底层 fs_backend.ls 不做任何过滤（原样返回目录下所有条目），从仓库根以 depth=3
    # 递归必然会掉进 .venv/Lib/site-packages/... 这类目录，把结果撑成纯噪音。
    # 注意：只对**下降过程**生效；模型显式传 path=".venv" 时仍会正常列出（保留逃生口）。
    SKIP_DIR_NAMES: tuple = (
        ".git", ".venv", "venv", "__pycache__", "node_modules",
        ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".tox", ".eggs", "site-packages", "dist", "build",
    )

    @classmethod
    def _is_noise_dir(cls, name: str) -> bool:
        """噪音目录判定：显式名单 + 一切点开头的目录。"""
        return name in cls.SKIP_DIR_NAMES or name.startswith(".")

    def __init__(self, fs_backend: FilesystemBackend) -> None:
        super().__init__()
        self.fs_backend = fs_backend
        self.parameters = [
            ToolParameter(
                name="path",
                type="string",
                description="【重要】目标目录的路径。请务必将此参数命名为 'path'，严禁使用 'directory' 或 'dir' 作为参数名。",
                required=True,
            ),
            ToolParameter(
                name="depth",
                type="integer",
                description=(
                    f"递归层数，默认 {self.DEFAULT_DEPTH}（1=只看当前目录），最大 {self.MAX_DEPTH}。"
                    "不知道文件在哪时用默认值，一次就能看到多层结果。"
                ),
                required=False,
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="可选：文件名通配过滤（如 '*.xlsx'、'客户*'），只过滤文件、不影响目录展开。",
                required=False,
            ),
            ToolParameter(
                name="max_entries",
                type="integer",
                description=f"最多返回的条目数，默认 {self.DEFAULT_MAX_ENTRIES}；超出会截断并提示。",
                required=False,
            ),
        ]

    # ------------------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_int(raw: Any, default: int, low: int, high: int) -> int:
        """把模型传进来的整数参数夹到合法区间（容忍字符串/空值/垃圾值）。"""
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return max(low, min(high, value))

    @staticmethod
    def _name_of(entry_path: str) -> str:
        """从虚拟路径里取末级名字（'/raw_data/x.xlsx' → 'x.xlsx'）。"""
        text = str(entry_path or "").rstrip("/")
        return text.rsplit("/", 1)[-1] or text

    async def execute(self, **kwargs: Any) -> str:
        """扫描目录并输出目录树快照（depth=1 时退回单层明细表）。"""
        path = kwargs.get("path")
        if path is None:
            return "Error: Missing required parameter 'path'."

        depth: int = self._clamp_int(
            kwargs.get("depth"), self.DEFAULT_DEPTH, 1, self.MAX_DEPTH
        )
        max_entries: int = self._clamp_int(
            kwargs.get("max_entries"), self.DEFAULT_MAX_ENTRIES, 1, self.MAX_MAX_ENTRIES
        )
        pattern: str = str(kwargs.get("pattern") or "").strip()

        # depth=1：保留原「单层明细表」（带类型/大小/修改时间），向后兼容；
        # depth>1：输出目录树（不带 size/mtime——递归场景下这些字段纯属占 token）。
        if depth <= 1:
            return await self._single_level(path, pattern, max_entries)
        return await self._directory_tree(path, depth, pattern, max_entries)

    async def _single_level(self, path: str, pattern: str, max_entries: int) -> str:
        result = await self.fs_backend.ls(path=path)
        if result.error:
            return f"Error: Failed to list directory '{path}'. Reason: {result.error}"

        entries: List[dict] = list(result.entries or [])
        # 噪音目录（.venv/__pycache__/…）不列出、不计数
        kept: List[dict] = []
        skipped: int = 0
        for entry in entries:
            if entry.get("is_dir") and self._is_noise_dir(self._name_of(entry.get("path", ""))):
                skipped += 1
                continue
            kept.append(entry)
        entries = kept
        if pattern:
            entries = [
                e for e in entries
                if e.get("is_dir") or fnmatch.fnmatch(self._name_of(e.get("path", "")), pattern)
            ]
        if not entries:
            return f"Directory '{path}' is empty" + (f" (pattern='{pattern}')" if pattern else "") + "."

        truncated: bool = len(entries) > max_entries
        entries = entries[:max_entries]

        output_lines = [
            f"### Directory Listing for: {path}",
            "",
            "| Type | Size (Bytes) | Last Modified | Path |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for entry in entries:
            item_type = "DIR" if entry.get("is_dir") else "FILE"
            size = entry.get("size", "-")
            modified_at = entry.get("modified_at", "-")
            item_path = entry.get("path", "")
            output_lines.append(f"| {item_type} | {size} | {modified_at} | {item_path} |")
        if truncated:
            output_lines.append(
                f"\n⚠️ 已截断：仅显示前 {max_entries} 条，请用 pattern 或更具体的 path 收窄。"
            )
        if skipped:
            output_lines.append(f"（已跳过 {skipped} 个噪音目录：.venv/__pycache__/.git 等）")
        return "\n".join(output_lines)

    async def _directory_tree(
        self, root: str, depth: int, pattern: str, max_entries: int
    ) -> str:
        """按 depth 递归展开目录树；条目数与总字符数双重截断。"""
        head: str = f"### Directory tree: {root}  (depth={depth}"
        if pattern:
            head += f", pattern='{pattern}'"
        head += ")"

        lines: List[str] = [head, ""]
        state: dict = {"count": 0, "dirs": 0, "files": 0, "skipped": 0, "truncated": False}

        async def walk(dir_path: str, level: int) -> None:
            if state["truncated"]:
                return
            result = await self.fs_backend.ls(path=dir_path)
            if result.error:
                lines.append("  " * level + f"[无法读取: {result.error}]")
                return

            entries: List[dict] = list(result.entries or [])
            dirs: List[dict] = []
            for e in entries:
                if not e.get("is_dir"):
                    continue
                # 噪音目录不列出也不下降（否则从仓库根递归必掉进 .venv/site-packages）
                if self._is_noise_dir(self._name_of(e.get("path", ""))):
                    state["skipped"] += 1
                    continue
                dirs.append(e)
            files: List[dict] = [e for e in entries if not e.get("is_dir")]
            if pattern:
                files = [
                    e for e in files
                    if fnmatch.fnmatch(self._name_of(e.get("path", "")), pattern)
                ]
            dirs.sort(key=lambda e: self._name_of(e.get("path", "")).casefold())
            files.sort(key=lambda e: self._name_of(e.get("path", "")).casefold())

            for entry in dirs:
                if state["count"] >= max_entries:
                    state["truncated"] = True
                    return
                lines.append("  " * level + self._name_of(entry.get("path", "")) + "/")
                state["count"] += 1
                state["dirs"] += 1
                if level + 1 < depth:
                    await walk(str(entry.get("path", "")), level + 1)

            for entry in files:
                if state["count"] >= max_entries:
                    state["truncated"] = True
                    return
                lines.append("  " * level + self._name_of(entry.get("path", "")))
                state["count"] += 1
                state["files"] += 1

        await walk(root, 0)

        lines.append("")
        lines.append(
            f"共 {state['dirs']} 个目录 / {state['files']} 个文件（depth={depth}）"
        )
        if state["truncated"]:
            lines.append(
                f"⚠️ 已截断：达到 max_entries={max_entries}。请用 pattern（如 '*.xlsx'）"
                "收窄文件类型，或直接把 path 指向更具体的子目录。"
            )
        if pattern:
            lines.append(f"（已按 pattern='{pattern}' 过滤文件，目录仍完整展开）")
        if state["skipped"]:
            lines.append(
                f"（已跳过 {state['skipped']} 个噪音目录：.venv/__pycache__/.git/node_modules 等；"
                "确实要看它们请直接把这些目录名作为 path 传入）"
            )

        text: str = "\n".join(lines)
        if len(text) > self.MAX_OUTPUT_CHARS:
            text = (
                text[: self.MAX_OUTPUT_CHARS]
                + f"\n⚠️ 结果过长已截断（原长 {len(text)} 字符）：请用 pattern 或更具体的 path 收窄范围。"
            )
        return text


class FileGrepTool(BaseTool):
    """在文件中检索关键字的工具。"""

    name: str = "file_grep_tool"
    description: str = (
        "跨文件文本检索，定位关键字/函数/变量出现在哪些文件的哪些行。"
        "仅支持文本文件，不支持 Excel；读取内容请用 file_read_tool。"
    )
    """
    description: str = (
        "【使用时机】当你不知道目标内容在哪个文件里，只知道一个关键字/类名/方法名/变量名/硬编码文本，需要在本地多个文件中快速定位它出现在哪些文件的哪些行时，调用本工具。"
        "典型场景：查找某个函数的定义位置、搜索某段配置项、定位某个业务关键词出现的所有位置。"
        "【能力】跨文件全文精确检索，底层优先使用 ripgrep，可在海量源文件中秒级返回 文件:行号:内容 的匹配视图。"
        "【使用顺序建议】先用本工具定位到候选文件，再用 file_read_tool 精读该文件相关行范围。"
        "【不要用于】读取文件内容本身（那是 file_read_tool 的职责）；pattern 为文本字面量，无需正则转义。"
    )
    """
    def __init__(self, fs_backend: FilesystemBackend) -> None:
        super().__init__()
        self.fs_backend = fs_backend
        self.parameters = [
            ToolParameter(
                name="pattern",
                type="string",
                description="要进行全文精确匹配的文本字面量（不需要对特殊符号进行正则转义）",
                required=True,
            ),
            ToolParameter(
                name="path",
                type="string",
                description="限定检索的目标根目录路径，如果不提供则默认在当前根目录 '.' 下进行全局搜索",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> str:
        """调用底层的 ripgrep / Python 双检索引擎进行高速代码检索。"""
        pattern = kwargs.get("pattern")
        path = kwargs.get("path", "")

        if not pattern:
            return "Error: Missing required parameter 'pattern'."

        result = await self.fs_backend.grep(pattern=pattern, path=path)

        if result.error:
            return f"Error: Grep operation failed under root '{path}'. Reason: {result.error}"

        if not result.matches:
            return f"No matches found for pattern '{pattern}' in directory '{path}'."

        # 将匹配行整合为标准编译器风格的代码映射视图，便于 LLM 进行上下文追踪
        output_lines = [f"### Grep Results for pattern: '{pattern}'", ""]
        for match in result.matches:
            output_lines.append(f"{match['path']}:{match['line']}: {match['text']}")

        return "\n".join(output_lines)