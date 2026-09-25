# -*- coding: utf-8 -*-
"""销售报表导出工具的正文闸门测试。

重点回归 2026-09-23 事故：planner 看不到前序取数结果，给 content 写了单行
标题性占位"2026年7月销售及负责人产品销量排名数据"，非空且通过必填校验，
零 LLM 直达工具 → 导出只有一行占位语的空报表，还白烧一次人工审批。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.tools.builtin.sales_report import SalesReportExportTool  # noqa: E402


def test_title_echo_detection_catches_online_incident_placeholder():
    tool = SalesReportExportTool(base_dir=str(Path(__file__).parent))
    # 线上事故里的原文（单行、无结构标点、以"数据"收尾）
    assert tool._content_is_title_echo("2026年7月销售及负责人产品销量排名数据") is True
    assert tool._content_is_title_echo("请参考8月份销售业绩结果") is True


def test_title_echo_allows_real_multiline_report():
    tool = SalesReportExportTool(base_dir=str(Path(__file__).parent))
    real_body = (
        "2026年7月销售/负责人产品销量排名：\n"
        "1. 张伟-企业知识库：4单\n"
        "2. 王强-工单系统：4单\n"
        "3. 李娜-智能客服平台：3单\n"
        "结论：张伟与王强并列第一。"
    )
    assert tool._content_is_title_echo(real_body) is False
    # 单行但带冒号+指标的也算有效行
    assert tool._content_is_title_echo("总销量：12台，环比增长8%") is False


def test_execute_rejects_title_echo_without_writing_file(tmp_path: Path) -> None:
    tool = SalesReportExportTool(base_dir=str(tmp_path))

    result = asyncio.run(tool.execute(
        report_title="2026年7月销售/负责人产品销量排名报告",
        content="2026年7月销售及负责人产品销量排名数据",
        file_name="占位报表",
    ))

    assert result.startswith("错误")
    assert "占位" in result
    assert not (tmp_path / "outputs" / "sales_reports").exists() or not any(
        (tmp_path / "outputs" / "sales_reports").iterdir()
    )


def test_execute_writes_real_report(tmp_path: Path) -> None:
    tool = SalesReportExportTool(base_dir=str(tmp_path))
    body = "排名如下：\n张伟-企业知识库：4单\n王强-工单系统：4单\n李娜-智能客服平台：3单"

    result = asyncio.run(tool.execute(
        report_title="2026年7月销售排名报告",
        content=body,
        file_name=None,
    ))

    assert result.startswith("成功")
    files = list((tmp_path / "outputs" / "sales_reports").glob("*.xlsx"))
    assert len(files) == 1


def test_execute_normalizes_literal_escaped_newlines(tmp_path: Path) -> None:
    """FC 模型（实测 GLM-4.7）常把换行双重转义成字面量 \\n，需归一化为真换行，
    否则整篇报表挤在一个单元格。"""
    import openpyxl

    tool = SalesReportExportTool(base_dir=str(tmp_path))
    body = "排名如下：\\n张伟：4单\\n王强：4单\\n李娜：3单"  # 字面 \n

    result = asyncio.run(tool.execute(
        report_title="2026年7月销售排名报告", content=body, file_name=None,
    ))

    assert result.startswith("成功")
    target = next((tmp_path / "outputs" / "sales_reports").glob("*.xlsx"))
    ws = openpyxl.load_workbook(target).active
    values = [c.value for row in ws.iter_rows() for c in row if c.value]
    assert any("张伟：4单" == v for v in values)  # 每行独立成单元格
    assert not any("\\n" in str(v) for v in values)  # 无字面量残留
