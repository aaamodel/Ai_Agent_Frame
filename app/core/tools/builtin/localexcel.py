# -*- coding: utf-8 -*-
import os
import json
import difflib
from pathlib import Path
from typing import Any, List, Optional
import openpyxl
from app.core.tools.base import BaseTool, ToolParameter


class LocalExcelTool(BaseTool):
    """读写本地 Excel 文件的工具"""

    name: str = "local_excel_tool"
    description: str = "用于读取或写入本地电脑上的 Excel (.xlsx) 表格文件。"

    def __init__(self, base_dir: Optional[str] = None) -> None:
        super().__init__()
        # 规范的数据根目录（通常是项目根）。注入后可用它解析相对路径，
        # 避免依赖进程 cwd；未注入时回退到本文件推断的仓库根。
        self._base_dir: Optional[Path] = Path(base_dir) if base_dir else None
        # 严格匹配你的 ToolParameter 契约
        self.parameters = [
            ToolParameter(name="file_path", type="string", description="本地 Excel 文件的绝对路径", required=True),
            ToolParameter(name="action", type="string", description="操作类型: 'read' (读取数据) 或 'write' (写入数据)",
                          required=True),
            ToolParameter(name="sheet_name", type="string", description="工作表名称，默认使用 'Sheet1'", required=False),
            ToolParameter(name="cell", type="string",
                          description="单元格坐标 (如 'A1', 'B2')。读取时若为空则返回全表预览，写入时必填",
                          required=False),
            ToolParameter(name="value", type="string",
                          description="准备写入单元格的内容字符串 (仅在 action='write' 时有效)", required=False)
        ]

    # ------------------------------------------------------------------
    # 路径接地：把相对路径稳定解析到规范根，且模型臆造路径时给出真实文件清单
    # ------------------------------------------------------------------
    def _canonical_base(self) -> Path:
        """本地 Excel 数据文件的规范根目录。"""
        if self._base_dir is not None:
            return self._base_dir
        # 本文件位于 app/core/tools/builtin/localexcel.py → parents[4] = 项目根
        return Path(__file__).resolve().parents[4]

    def _resolve_physical_path(self, file_path: str) -> Optional[Path]:
        """把模型传入的路径解析为真实存在的物理路径；找不到返回 None。"""
        given = Path(file_path)
        if given.is_absolute():
            return given if given.exists() else None
        base = self._canonical_base()
        candidates: List[Path] = [base / file_path, Path(os.getcwd()) / file_path]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _grounding_with_real_files(self, file_path: str) -> str:
        """文件不存在时，返回真实可用的 Excel 文件清单，引导模型下一轮用真实路径。"""
        base = self._canonical_base()
        target_dir: Path = base / "raw_data" / "sales_intel"
        if not target_dir.exists():
            target_dir = base / "raw_data"
        xlsx_files: List[Path] = []
        if target_dir.exists():
            xlsx_files = sorted(
                p
                for p in target_dir.glob("**/*.xlsx")
                if not p.name.startswith("~$")  # 过滤 Excel 临时锁文件
            )
        if not xlsx_files:
            return (
                f"错误：找不到指定的 Excel 文件: {file_path}，"
                f"且当前未能定位到数据目录 {target_dir}，请先使用 file_list_tool / file_grep_tool 探查实际数据文件。"
            )

        lines: List[str] = [
            f"错误：找不到指定的 Excel 文件: {file_path}。"
            "以下是你当前真实可用的 Excel 文件（请严格按绝对路径使用，禁止臆造、改写或拼接文件名）：",
        ]
        wanted_name = Path(file_path).name
        best_match: Optional[Path] = max(
            xlsx_files,
            key=lambda fp: difflib.SequenceMatcher(None, wanted_name, fp.name).ratio(),
        )
        best_ratio = difflib.SequenceMatcher(None, wanted_name, best_match.name).ratio()
        if best_ratio >= 0.5:
            lines.append(f"- 与你要找的最相近的可能是: {best_match.resolve()}（相似度 {best_ratio:.2f}）")
        for fp in xlsx_files:
            rel = fp.relative_to(base).as_posix() if fp.is_relative_to(base) else str(fp)
            lines.append(f"\t绝对路径: {fp.resolve()} | 相对路径: {rel}")
        return "\n".join(lines)

    async def execute(self, **kwargs: Any) -> str:
        file_path = kwargs.get("file_path")
        action = kwargs.get("action")
        sheet_name = kwargs.get("sheet_name", "Sheet1")
        cell = kwargs.get("cell")
        value = kwargs.get("value")

        if not file_path:
            return "错误：未提供文件路径 file_path"

        try:
            # --- 读取逻辑 ---
            if action == "read":
                physical_path = self._resolve_physical_path(file_path)
                if physical_path is None:
                    return self._grounding_with_real_files(file_path)
                file_path = str(physical_path)

                wb = openpyxl.load_workbook(file_path, data_only=True)
                sheet = wb[sheet_name] if sheet_name in wb.sheetnames else wb.active

                # 情况 A：读取特定单元格
                if cell:
                    return f"文件 [{os.path.basename(file_path)}] 表 [{sheet.title}] 单元格 {cell} 的内容为: {sheet[cell].value}"

                # 情况 B：无特定单元格，默认读取前 50 行做全表矩阵预览
                matrix_data = []
                for row in sheet.iter_rows(max_row=50, values_only=True):
                    if any(row):  # 过滤全空行
                        matrix_data.append([str(c) if c is not None else "" for c in row])
                return json.dumps(matrix_data, ensure_ascii=False, indent=2)

            # --- 写入逻辑 ---
            elif action == "write":
                if not cell:
                    return "错误：写入操作必须指定具体的单元格位置 cell (例如 'A1')"

                # 相对路径稳定落到规范根，避免写入到错误的相对位置
                target = Path(file_path)
                if not target.is_absolute():
                    target = self._canonical_base() / file_path
                file_path = str(target)

                # 文件存在则加载，不存在则新建
                if os.path.exists(file_path):
                    wb = openpyxl.load_workbook(file_path)
                else:
                    wb = openpyxl.Workbook()

                sheet = wb[sheet_name] if sheet_name in wb.sheetnames else wb.create_sheet(title=sheet_name)
                sheet[cell] = value
                wb.save(file_path)
                return f"成功：已在文件 [{os.path.basename(file_path)}] 的 [{sheet.title}] 表 {cell} 位置写入值: {value}"

            else:
                return f"错误：不支持的 action 类型 '{action}'"

        except Exception as e:
            return f"执行 Excel 工具期间崩溃: {str(e)}"