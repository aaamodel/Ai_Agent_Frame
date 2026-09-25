# -*- coding: utf-8 -*-
"""销售分析报表导出工具。

把销售分析结论生成为格式化 Excel 文件，落盘到 ``outputs/sales_reports/``。
属于"对外可分发文件"类写操作，默认纳入 ``agent_danger_tools`` 人工审批名单：
审批通过才真正生成文件，拒绝则磁盘无任何产物。
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import openpyxl
from openpyxl.styles import Alignment, Font

from app.core.tools.base import BaseTool, ToolParameter

# Windows/Unix 文件名非法字符 + 控制字符
_UNSAFE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]+')


def _safe_filename(raw: str) -> str:
    """把模型传入的文件名/标题清洗成单段安全文件名（防路径穿越）。"""
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", str(raw or "")).strip(" ._")
    return cleaned[:80] or "sales_report"


# 数据不充分时拒绝导出。实测事故：agent 因为没取到 8 月数据，把
# "本次导出任务未能生成有效分析内容……关键指标无可用指标……"这种**数据缺失说明**
# 当成报表正文传了进来，白烧一次人工审批（写操作），落盘文件对用户毫无价值。
INSUFFICIENT_MARKERS: tuple = (
    "数据缺失", "无可用指标", "数据不充分", "无法生成", "未能生成",
    "数据不存在", "暂无数据", "未获取到", "无法获取", "无可用数据",
)

# 与上面的缺失话术**联合**判定：只有"出现缺失话术" **且** "正文里几乎没有数字"
# 才拦截。单看话术会误拦正常报表（某项指标写 N/A 但整体是有效分析）。
_MIN_SUBSTANTIVE_NUMBERS: int = 3
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")

# "标题党正文"收尾词：正文只有一行、没有任何结构标点，且以这些抽象名词收尾，
# 基本可断定是 planner 看不到前序结果时写的占位语（实测："2026年7月销售及
# 负责人产品销量排名数据"→ 导出只有一行占位语的空报表，还白烧一次人工审批）。
_ECHO_TAIL_PATTERN = re.compile(r"(数据|结果|情况|内容|信息|报表|报告)$")
_STRUCTURE_PUNCT_PATTERN = re.compile(r"[：:；;，,。\n]")


class SalesReportExportTool(BaseTool):
    """将销售分析结论导出为格式化 .xlsx 报表（写操作，需人工审批）。"""

    name: str = "sales_report_export_tool"
    description: str = (
        "将销售分析结论导出为 .xlsx 报表，落盘到 outputs/sales_reports/。"
        "仅在用户明确要求导出/下载/归档报表时调用；写操作，需人工审批。"
    )

    def __init__(self, base_dir: Optional[str] = None) -> None:
        super().__init__()
        self._base_dir: Path = Path(base_dir) if base_dir else Path(__file__).resolve().parents[4]
        self.parameters = [
            ToolParameter(
                name="report_title",
                type="string",
                description="报表标题",
                required=True,
            ),
            ToolParameter(
                name="content",
                type="string",
                description="报表正文纯文本，建议按 结论/指标/原因/建议 分段",
                required=True,
            ),
            ToolParameter(
                name="file_name",
                type="string",
                description="自定义文件名（不含扩展名，禁止带路径）；留空自动生成",
                required=False,
            ),
        ]

    @staticmethod
    def _content_is_substantive(content: str) -> bool:
        """判断报表正文是否含实质业务数据（用于拦住"数据缺失说明"被导出）。

        判定口径是**两个条件同时成立**才拦：
          1. 正文出现缺失话术（"数据缺失"/"无可用指标"/"未能生成" 等）；
          2. 正文里的数字少于 ``_MIN_SUBSTANTIVE_NUMBERS`` 个。
        只看话术会误拦正常报表（某项指标写 N/A、整体仍是有效分析）；
        只看数字又拦不住 "本次导出任务未能生成有效分析内容（关键指标无可用指标）"
        这类纯说明文本。
        """
        markers_hit: List[str] = [marker for marker in INSUFFICIENT_MARKERS if marker in content]
        if not markers_hit:
            return True
        return len(_NUMBER_PATTERN.findall(content)) >= _MIN_SUBSTANTIVE_NUMBERS

    @staticmethod
    def _content_is_title_echo(content: str) -> bool:
        """判断正文是否只是"标题党"占位语（单行 + 无结构标点 + 抽象名词收尾）。

        真报表正文至少会有分行/冒号/逗号组织的指标行；planner 凭空写的占位语
        形态固定为"××年×月××排名数据"这类单行短语。宁可误拦一次让执行层
        重新用前序结论组参，也不能让空报表通过人工审批落盘。
        """
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) != 1:
            return False
        only_line = lines[0]
        if _STRUCTURE_PUNCT_PATTERN.search(only_line):
            return False
        return bool(_ECHO_TAIL_PATTERN.search(only_line))

    async def execute(self, **kwargs: Any) -> str:
        title = str(kwargs.get("report_title") or "").strip()
        content = str(kwargs.get("content") or "").strip()
        file_name = str(kwargs.get("file_name") or "").strip()

        # FC 模型常把换行双重转义成字面量 "\n"（实测 GLM-4.7 的 tool arguments
        # 即如此），不归一化会导致整篇报表挤在一个单元格、splitlines 失效。
        content = (
            content.replace("\r\n", "\n").replace("\r", "\n")
            .replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
        )

        if not title:
            return "错误：缺少必填参数 report_title（报表标题）"
        if not content:
            return "错误：缺少必填参数 content（报表正文），禁止导出空报表"
        # 数据充分性闸门：拦住"数据缺失说明"被当报表导出（写操作会白烧一次人工审批）
        if not self._content_is_substantive(content):
            return (
                "错误：报表正文缺少实质业务数据，已拒绝导出（未落盘、未消耗审批）。"
                "请先完成取数再导出：用 sales_sql_query 查询，"
                "并按 filter_column+filter_value 精确定位目标期间（例如 月份=2026-08），"
                "拿到具体指标数值后再汇总成报表正文。"
            )
        # "标题党正文"闸门：拦住 planner 凭标题写的单行占位语（取不到前序结果时
        # 的典型故障），要求把前序查询的真实结论/指标组织成正文后再导出。
        if self._content_is_title_echo(content):
            return (
                "错误：报表正文只是标题性占位语（单行、无具体指标与结论），"
                "已拒绝导出（未落盘、未消耗审批）。请把 sales_sql_query 查到的"
                "真实排名/指标按行整理进 content（包含人员、产品、数值等），"
                "不要只写“××数据/××结果”这类短语。"
            )

        try:
            out_dir = self._base_dir / "outputs" / "sales_reports"
            out_dir.mkdir(parents=True, exist_ok=True)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = _safe_filename(file_name) if file_name else _safe_filename(title)
            target = out_dir / f"{stem}_{stamp}.xlsx"

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "销售分析报表"
            ws.column_dimensions["A"].width = 100

            title_font = Font(bold=True, size=14)
            meta_font = Font(size=9, color="808080")
            wrap = Alignment(wrap_text=True, vertical="top")

            ws["A1"] = title
            ws["A1"].font = title_font
            ws["A2"] = f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ws["A2"].font = meta_font

            row = 4
            line_count = 0
            for line in content.splitlines():
                text = line.strip()
                if not text:
                    continue
                cell = ws.cell(row=row, column=1, value=text)
                cell.alignment = wrap
                line_count += 1
                row += 1

            wb.save(target)
            return (
                f"成功：销售分析报表《{title}》已导出，共 {line_count} 行正文。\n"
                f"文件路径：{target.resolve()}"
            )

        except Exception as e:
            return f"导出销售报表期间崩溃: {str(e)}"
