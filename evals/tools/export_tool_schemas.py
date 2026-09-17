# -*- coding: utf-8 -*-
"""一次性取数工具：导出「已注册工具」的真实 JSON Schema。

用途（对齐手册第 2.2 节）：写工具调用黄金集时，``expected_tool`` / ``key_args``
必须照**真实注册的工具名与参数名**写，不能凭记忆编。本脚本产出这份对照表。

两种模式：

1. 默认（**首选**）：调用 ``app/core/tools/builtin/init_tools.py::bootstrap_tools``
   构建真实 ToolRegistry，再逐个 ``tool_to_function_call_definition`` 导出 →
   100% 反映运行时真实 schema。

2. ``--offline``：不 import 项目重型依赖，直接输出**代码核验过的静态目录**
   （来源：各工具类源码中的 ``self.name`` / ``ToolParameter(...)`` 声明，
   与 ``app/query_intent/rag_constant.py::REGISTERED_ENABLED_TOOL_NAMES`` 对齐）。
   输出里会带 ``"source": "static-fallback"`` 标记，避免与真实 schema 混淆。

用法::

    python evals/tools/export_tool_schemas.py --out evals/golden/_tool_schemas.json
    python evals/tools/export_tool_schemas.py --offline   # 无中间件/依赖时可用
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_OUT: Path = _REPO_ROOT / "evals" / "golden" / "_tool_schemas.json"


# =====================================================================
# 模式二：静态目录（代码核验，离线可用）
# =====================================================================
STATIC_TOOL_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "rag_knowledge_search",
        "description": "混合检索知识库（向量 + BM25 + RRF），答案来自公司私有文档。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "检索查询"},
            "top_k": {"type": "integer", "description": "召回条数"},
            "collection_names": {"type": "array", "description": "知识库集合白名单"},
        }, "required": ["query"]},
        "source_file": "app/core/tools/builtin/rag_search.py",
    },
    {
        "name": "knowledge_graph_search",
        "description": "知识图谱检索，适合「谁负责/什么关系/上下游」类问题。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "实体+关系描述"},
        }, "required": ["query"]},
        "source_file": "app/core/tools/builtin/graph_search.py",
    },
    {
        "name": "web_search",
        "description": "联网搜索（豆包 API 直连；内部自带 Tavily 降级通道，模型不可见）。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "搜索关键词或完整问句"},
            "count": {"type": "integer", "description": "返回结果条数 1-10，默认 5"},
            "time_range": {"type": "string", "description": "OneDay/OneWeek/OneMonth/OneYear"},
        }, "required": ["query"]},
        "source_file": "app/core/tools/builtin/doubao_search.py",
    },
    {
        "name": "local_excel_read_tool",
        "description": "本地表格只读（pandas 摘要/过滤/聚合，不审批）。",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "表格路径（绝对或相对项目根）"},
            "sheet_name": {"type": "string", "description": "工作表名；留空返回全部 sheet 摘要"},
            "cell": {"type": "string", "description": "可选单元格，如 D7"},
            "filter_column": {"type": "string", "description": "按列过滤的列名"},
            "filter_value": {"type": "string", "description": "过滤值（包含匹配）"},
            "group_by": {"type": "string", "description": "分组聚合的分组列"},
            "agg_column": {"type": "string", "description": "聚合目标列"},
            "agg_func": {"type": "string", "description": "sum/mean/count/max/min/median"},
            "head_rows": {"type": "integer", "description": "展示行数，默认 5"},
        }, "required": ["file_path"]},
        "source_file": "app/core/tools/builtin/localexcel.py",
    },
    {
        "name": "local_excel_query_tool",
        "description": "自然语言直接查询/统计本地 Excel/CSV（全量数据 pandas 计算，只读不审批）：筛选、分组、透视、TopN、环比、跨列运算均支持。",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "表格路径（绝对或相对项目根）"},
            "query": {"type": "string", "description": "自然语言问题/计算需求"},
            "sheet_name": {"type": "string", "description": "工作表名；多 sheet 文件必填，单 sheet 可省"},
        }, "required": ["file_path", "query"]},
        "source_file": "app/core/tools/builtin/excel_query.py",
    },
    {
        "name": "local_excel_write_tool",
        "description": "本地表格写入（危险工具，走人工审批）：语义定位更新（推荐）/ 单格 / 批量。",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "目标表格路径"},
            "sheet_name": {"type": "string", "description": "工作表名（语义模式多 sheet 必填）"},
            "filter_column": {"type": "string", "description": "语义模式：定位行的条件列名"},
            "filter_value": {"type": "string", "description": "语义模式：定位行的条件值（须唯一命中）"},
            "target_column": {"type": "string", "description": "语义模式：要修改的列名"},
            "new_value": {"type": "string", "description": "语义模式：写入的新值"},
            "cell": {"type": "string", "description": "单格模式：目标单元格，如 C5（已确认坐标时用）"},
            "value": {"type": "string", "description": "单格模式：写入值"},
            "rows": {"type": "string", "description": "批量模式：JSON 数组（对象数组或首行为列名的二维数组）"},
            "write_mode": {"type": "string", "description": "append（默认）/ replace"},
        }, "required": ["file_path"]},
        "source_file": "app/core/tools/builtin/localexcel.py",
    },
    {
        "name": "sales_report_export_tool",
        "description": "销售分析报表导出为 xlsx（危险工具，走人工审批）。",
        "parameters": {"type": "object", "properties": {
            "report_title": {"type": "string", "description": "报表标题"},
            "content": {"type": "string", "description": "报表正文"},
            "file_name": {"type": "string", "description": "输出文件名"},
        }, "required": ["report_title", "content"]},
        "source_file": "app/core/tools/builtin/sales_report.py",
    },
    {
        "name": "feishu_bitable_tool",
        "description": "飞书多维表格操作（查询/新增/更新/删除）。",
        "parameters": {"type": "object", "properties": {
            "app_id": {"type": "string", "description": "飞书应用 App ID"},
            "app_secret": {"type": "string", "description": "飞书应用 App Secret"},
            "app_token": {"type": "string", "description": "多维表格 app_token"},
            "table_id": {"type": "string", "description": "数据表 table_id（tbl 开头）"},
            "action": {"type": "string", "description": "操作类型"},
            "fields_json": {"type": "string", "description": "字段 JSON"},
        }, "required": ["app_id", "app_secret", "app_token", "table_id", "action"]},
        "source_file": "app/core/tools/builtin/feishu.py",
    },
    {
        "name": "file_read_tool",
        "description": "读取沙箱内文件内容。",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "文件路径"},
            "offset": {"type": "integer", "description": "起始行"},
            "limit": {"type": "integer", "description": "读取行数"},
        }, "required": ["file_path"]},
        "source_file": "app/core/tools/builtin/filesystem_tools.py",
    },
    {
        "name": "file_list_tool",
        "description": "浏览目录 / 定位文件（递归 depth 层，可按 pattern 过滤文件名）。",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "目录路径"},
            "depth": {"type": "integer", "description": "递归层数，默认 3，最大 8"},
            "pattern": {"type": "string", "description": "文件名通配过滤，如 *.xlsx"},
            "max_entries": {"type": "integer", "description": "最多返回条目数，默认 200"},
        }, "required": ["path"]},
        "source_file": "app/core/tools/builtin/filesystem_tools.py",
    },
    {
        "name": "file_grep_tool",
        "description": "在文件内容里按关键词/正则检索。",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string", "description": "检索关键词或正则"},
            "path": {"type": "string", "description": "检索范围路径"},
        }, "required": ["pattern", "path"]},
        "source_file": "app/core/tools/builtin/filesystem_tools.py",
    },
]


# =====================================================================
# 模式一：真实注册表
# =====================================================================
def export_from_registry(out_path: Path) -> int:
    """用真实 ToolRegistry 导出 schema；失败时抛异常由调用方降级。"""
    from app.core.backends.filesystem import FilesystemBackend
    from app.core.tools.base import tool_to_function_call_definition
    from app.core.tools.builtin.init_tools import bootstrap_tools

    registry = bootstrap_tools(fs_backend=FilesystemBackend(virtual_mode=True))
    definitions: List[Dict[str, Any]] = []
    for tool in registry.get_all_tools():
        definition = tool_to_function_call_definition(tool)
        definition["source"] = "runtime-registry"
        definitions.append(definition)

    payload: Dict[str, Any] = {
        "generated_by": "evals/tools/export_tool_schemas.py",
        "source": "runtime-registry",
        "tool_count": len(definitions),
        "tools": definitions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[export_tool_schemas] 从真实 ToolRegistry 导出 {len(definitions)} 个工具 -> {out_path}")
    return len(definitions)


def export_static(out_path: Path) -> int:
    """输出静态（代码核验）目录。"""
    payload: Dict[str, Any] = {
        "generated_by": "evals/tools/export_tool_schemas.py --offline",
        "source": "static-fallback",
        "warning": "本文件由静态目录生成，未连接运行时注册表；schema 以各工具源码为准。",
        "tool_count": len(STATIC_TOOL_CATALOG),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item["description"],
                    "parameters": item["parameters"],
                },
                "source_file": item["source_file"],
                "source": "static-fallback",
            }
            for item in STATIC_TOOL_CATALOG
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"[export_tool_schemas] 输出静态目录 {len(STATIC_TOOL_CATALOG)} 个工具 -> {out_path}")
    return len(STATIC_TOOL_CATALOG)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出已注册工具的 JSON Schema")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 JSON 路径")
    parser.add_argument("--offline", action="store_true", help="强制使用静态目录")
    args = parser.parse_args()

    if args.offline:
        export_static(args.out)
        return

    try:
        export_from_registry(args.out)
    except Exception as exc:  # noqa: BLE001 - 依赖缺失/环境未就绪时优雅降级
        print(f"[export_tool_schemas] 真实注册表不可用（{type(exc).__name__}: {exc}）")
        print("[export_tool_schemas] 降级为静态目录（--offline 等价行为）")
        export_static(args.out)


if __name__ == "__main__":
    main()
