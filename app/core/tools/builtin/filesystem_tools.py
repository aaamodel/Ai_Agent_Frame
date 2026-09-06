# -*- coding: utf-8 -*-
"""内置文件系统工具集：将 FilesystemBackend 封装为大模型可调用的 BaseTool 实例。

提供文件读取、目录扫描、以及高性能文本检索能力，所有接口故障均被平滑包装为大模型可读的错误文本。
"""

from typing import Any, Optional
import json

from app.core.tools.base import BaseTool, ToolParameter
from app.core.backends.filesystem import FilesystemBackend


class FileReadTool(BaseTool):
    """读取指定路径文件的工具（大模型可用此查看具体技能的源码/配置闭环）。"""

    name: str = "file_read_tool"
    description: str = (
        "读取指定路径文件的工具。支持配置分页参数（offset 和 limit），"
        "用于按需查看大文件或长代码的指定行范围，防止单次输入过长打爆上下文。"
    )

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
    """列出目录下文件列表的工具。"""

    name: str = "file_list_tool"
    description: str = "列出指定工作目录下的所有子文件与子目录（非递归扫描）。"

    def __init__(self, fs_backend: FilesystemBackend) -> None:
        super().__init__()
        self.fs_backend = fs_backend
        self.parameters = [
            ToolParameter(
                name="path",
                type="string",
                description="【重要】目标目录的路径。请务必将此参数命名为 'path'，严禁使用 'directory' 或 'dir' 作为参数名。",
                required=True,
            )
        ]

    async def execute(self, **kwargs: Any) -> str:
        """扫描并格式化输出当前目录结构快照。"""
        path = kwargs.get("path")
        if path is None:
            return "Error: Missing required parameter 'path'."

        result = await self.fs_backend.ls(path=path)

        if result.error:
            return f"Error: Failed to list directory '{path}'. Reason: {result.error}"

        if not result.entries:
            return f"Directory '{path}' is empty."

        # 为大模型构建极其直观易读的 Markdown 表格文本
        output_lines = [
            f"### Directory Listing for: {path}",
            "",
            "| Type | Size (Bytes) | Last Modified | Path |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for entry in result.entries:
            item_type = "DIR" if entry.get("is_dir") else "FILE"
            size = entry.get("size", "-")
            modified_at = entry.get("modified_at", "-")
            item_path = entry.get("path", "")
            output_lines.append(f"| {item_type} | {size} | {modified_at} | {item_path} |")

        return "\n".join(output_lines)


class FileGrepTool(BaseTool):
    """在文件中检索关键字的工具。"""

    name: str = "file_grep_tool"
    description: str = (
        "跨文件全网文本关键字检索工具。底层优先使用高吞吐量的 ripgrep 引擎，"
        "能在海量源文件中秒级定位特定的类、方法、变量或硬编码文本。"
    )

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