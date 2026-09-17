# -*- coding: utf-8 -*-
"""自然语言表格查询工具（全量数据 + pandas 代码生成 + AST 安全护栏 + 报错自修复）。

为什么需要它（实测事故驱动）：
    旧的 local_excel_read_tool 是"结构摘要 + 固定算子（filter/groupby）"模式，
    模型只能看到前 5 行预览。2026-08 销售复盘事故中，模型看到预览里没有 8 月，
    就误判"数据不存在"，反复换路径重读预览空转 10+ 轮；多表关联/透视/环比等
    计划外取数方式也会持续以新形式卡住（固定算子的覆盖面永远追不上业务问题）。

设计（与主流 code-interpreter 表格分析同一模式：全量数据在内存 → LLM 生成
pandas 表达式 → 执行 → 报错带着 traceback 自修复）：
    1. **全量数据**：整 sheet 载入 pandas，模型看到真实中文列名/类型/取值样例，
       "预览误判"与"臆造英文列名"两类问题从机制上消失；
    2. **LLM 走自家 ModelRouter**（purpose_hint=react, thinking=False）：
       tier 选择/熔断降级/多厂商方言/Langfuse 追踪全部复用，不另起 LLM 通道；
    3. **AST 安全护栏**：禁 import、禁 dunder、禁文件/网络 IO 方法、禁对 df 写入，
       执行命名空间只给白名单 builtins + pd/np/math/df；
    4. **报错自修复两轮**：执行报错把错误原样回传让模型改代码（上游
       llama-index-experimental 的 PandasQueryEngine 恰恰**没有**自修复，
       且该包已被官方标记 deprecated/no longer maintained，故在此项目内实现精简版）。

本工具只读，不触发人工审批；写入仍走 local_excel_write_tool。
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import math
from typing import Any, Dict, List, Optional, Tuple

from app.core.tools.base import BaseTool, ToolParameter
from app.core.tools.builtin.localexcel import (
    _BaseExcelTool,
    _preferred_excel_engine,
    LocalExcelReadTool,
)


# 代码执行最多 3 次 LLM 调用（首次生成 + 2 次自修复）
_MAX_REPAIR_ROUNDS: int = 2
_MAX_OUTPUT_CHARS: int = 6000
_PREVIEW_ROWS: int = 15
_SAMPLE_VALUES_PER_COL: int = 8

# 执行命名空间允许的 builtins（只留纯计算/容器类，杜绝文件/网络/反射）。
# 注意：必须用标准库 builtins 模块取对象——模块全局里的 __builtins__ 可能是
# dict 也可能是 module（主模块 vs 被导入模块行为不同），直接 getattr 会炸。
_ALLOWED_BUILTINS: Dict[str, Any] = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "bytearray", "bytes", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "hash", "hex", "int",
        "isinstance", "issubclass", "len", "list", "map", "max", "min", "oct", "ord",
        "pow", "print", "range", "repr", "reversed", "round", "set", "slice", "sorted",
        "str", "sum", "tuple", "type", "zip",
    )
}

# 名字级黑名单：反射/执行/退出类 builtin 即使被模型写出也不提供
_FORBIDDEN_NAMES = frozenset({
    "__import__", "open", "eval", "exec", "compile", "input", "breakpoint",
    "exit", "quit", "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "help", "copyright", "credits", "license",
})

# pandas/numpy 上禁止触碰的 IO / 持久化方法（精确名单，避免误伤 tolist 这类正常方法）
_FORBIDDEN_ATTRS = frozenset({
    "read_csv", "read_excel", "read_json", "read_html", "read_xml", "read_parquet",
    "read_feather", "read_orc", "read_stata", "read_sas", "read_spss", "read_sql",
    "read_sql_table", "read_sql_query", "read_gbq", "read_hdf", "read_pickle",
    "read_table", "read_fwf", "read_clipboard", "read_fwf", "load", "loads",
    "to_csv", "to_excel", "to_json", "to_html", "to_xml", "to_parquet", "to_feather",
    "to_orc", "to_stata", "to_hdf", "to_pickle", "to_sql", "to_gbq", "to_clipboard",
    "savefig", "save", "dump", "dumps", "socket", "connect", "system", "popen",
    "remove", "unlink", "rmdir", "mkdir", "rename", "replace",
})


class _UnsafeCodeError(Exception):
    """生成的 pandas 代码未通过 AST 安全检查。"""


class _PandasCodeGuard(ast.NodeVisitor):
    """对 LLM 生成的 pandas 代码做只读安全检查。

    拦截四类风险：
      1. import 任何模块（pd/np/math 已预置，无需导入）；
      2. dunder 属性访问（``__class__``/``__globals__`` 等提权通道）；
      3. 反射/文件/网络/持久化相关的名字与方法；
      4. 任何写操作：对 df/子帧的下标/属性赋值、aug-assign、del、变量名顶替 df。
    """

    def __init__(self) -> None:
        self._assign_targets: List[str] = []

    def check(self, source: str) -> None:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            self.visit(node)
        # 普通中间变量允许赋值（如 tmp = df[...]; tmp.groupby(...)），
        # 但整个代码块不允许把 df 这个名字重新绑定。
        if "df" in self._assign_targets:
            raise _UnsafeCodeError("禁止重新赋值变量 df（数据框只读）")

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        raise _UnsafeCodeError("禁止 import；pd/np/math 已可用")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        raise _UnsafeCodeError("禁止 import；pd/np/math 已可用")

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in _FORBIDDEN_NAMES:
            raise _UnsafeCodeError(f"禁止使用 {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        attr = node.attr
        if attr.startswith("_"):
            raise _UnsafeCodeError(f"禁止访问私有属性 {attr!r}")
        if attr in _FORBIDDEN_ATTRS:
            raise _UnsafeCodeError(f"禁止文件/网络/持久化操作 .{attr}()")
        self.generic_visit(node)

    def _check_store_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._assign_targets.append(target.id)
            return
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            raise _UnsafeCodeError("禁止修改数据框/对象（df 及其列是只读的）")
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._check_store_target(elt)
            return
        if isinstance(target, ast.Starred):
            self._check_store_target(target.value)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            self._check_store_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.target is not None:
            self._check_store_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        raise _UnsafeCodeError("禁止增量赋值（数据框只读）")

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        raise _UnsafeCodeError("禁止 del（数据框只读）")


def _strip_code_fence(raw: str) -> str:
    """剥 LLM 常输出的 ```python ... ``` 围栏。"""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # 去首行围栏
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _execute_pandas_code(code: str, df: Any) -> str:
    """安全执行生成的 pandas 代码，返回结果文本。

    - AST 校验不通过/运行报错：抛异常给自修复循环；
    - 最后一条是表达式 → eval 取值（推荐路径）；
    - 否则执行全部语句，捕获 print 输出兜底。
    """
    import pandas as pd
    import numpy as np

    _PandasCodeGuard().check(code)
    tree = ast.parse(code)
    body: List[ast.stmt] = tree.body
    if not body:
        raise _UnsafeCodeError("生成的代码为空")

    safe_globals: Dict[str, Any] = {
        "__builtins__": dict(_ALLOWED_BUILTINS),
        "pd": pd,
        "np": np,
        "math": math,
    }
    safe_locals: Dict[str, Any] = {"df": df}

    stdout_buf = io.StringIO()
    if isinstance(body[-1], ast.Expr):
        prefix = ast.Module(body=body[:-1], type_ignores=[])
        with contextlib.redirect_stdout(stdout_buf):
            exec(compile(prefix, "<pandas_query>", "exec"), safe_globals, safe_locals)
            value = eval(  # noqa: S307 - 已过 AST 白名单护栏，命名空间无危险 builtins
                compile(ast.Expression(body[-1].value), "<pandas_query>", "eval"),
                safe_globals, safe_locals,
            )
    else:
        with contextlib.redirect_stdout(stdout_buf):
            exec(compile(tree, "<pandas_query>", "exec"), safe_globals, safe_locals)
            value = None

    if value is None:
        printed = stdout_buf.getvalue().strip()
        if printed:
            return printed
        return "（代码执行成功但没有产出结果；请让最后一行是一个取值表达式）"

    # DataFrame/Series 渲染时放宽截断，保证小表结果完整可见
    with pd.option_context("display.max_rows", 200, "display.max_columns", None,
                           "display.max_colwidth", 200, "display.width", 4000):
        return str(value)


def _build_table_context(df: Any, sheet_name: str) -> str:
    """构造给代码生成模型的表上下文：形状 + 列类型 + 低基数列取值样例 + 数据预览。

    取值样例是关键护栏：模型看到"月份"列真实长这样
    ``2025-09 / 2026-08``，就不会臆造 "2026年8月" / "Aug 2026" 等格式。
    """
    import pandas as pd

    rows, cols = df.shape
    lines: List[str] = [f"工作表: {sheet_name}（{rows} 行 × {cols} 列）", "列清单（列名 → 类型 → 取值样例）:"]
    for column in df.columns:
        series = df[column]
        label = f"{column} ({series.dtype})"
        if pd.api.types.is_numeric_dtype(series):
            lines.append(f"  - {label}")
            continue
        try:
            uniques = series.dropna().astype(str).unique().tolist()
        except Exception:  # noqa: BLE001 - 取值样例只是辅助信息，取不到不阻断
            uniques = []
        if 0 < len(uniques) <= _SAMPLE_VALUES_PER_COL * 2:
            shown = "、".join(uniques[:_SAMPLE_VALUES_PER_COL])
            lines.append(f"  - {label}，全部取值: {shown}")
        elif uniques:
            shown = "、".join(uniques[:_SAMPLE_VALUES_PER_COL])
            lines.append(f"  - {label}，样例值: {shown}")
        else:
            lines.append(f"  - {label}")

    preview_n = min(_PREVIEW_ROWS, rows) if rows else 0
    lines.append(f"\n前 {preview_n} 行数据（共 {rows} 行，未展示的行同样存在，不要据预览判断数据缺失）:")
    with pd.option_context("display.max_rows", _PREVIEW_ROWS, "display.max_columns", None,
                           "display.max_colwidth", 60, "display.width", 4000):
        lines.append(df.head(preview_n).to_string())
    return "\n".join(lines)


_CODEGEN_SYSTEM = """你是严谨的 pandas 数据分析专家。用户会给你一个已加载的 pandas DataFrame（变量名 `df`）的结构、样例和一个自然语言问题，你要输出**可直接执行的 pandas Python 代码**来回答问题。

铁律：
1. 只输出 Python 代码本身：不要 Markdown 围栏、不要解释、不要注释以外的任何文字。
2. 最后一行必须是一个**表达式**，它的求值结果就是给用户的答案（数字/字符串/DataFrame/Series 均可）。
3. 数据框只读：禁止修改 df、禁止重新赋值 df；可用中间变量（如 `tmp = df[...]`）。
4. 禁止 import（pd/np/math 已可用）、禁止任何文件/网络操作。
5. 严格按"列清单"里给出的真实中文列名与真实取值格式书写；筛选前看清取值样例的精确格式（如月份写作 '2026-08' 就不要写成 '2026年8月' 或 '8月'）。
6. 比率/百分比列可能是带 '%' 的字符串，计算前先按需做数值转换（如 `str.rstrip('%').astype(float) / 100`）。
7. 透视、分组聚合、多步计算都可以；结果只保留回答问题所需的列/行，并按业务语义排序。
8. 数值结果请用 round(x, 2) 之类保留合理小数位，避免出现 1.8000000000000007 这样的浮点尾巴。
"""

_REPAIR_SYSTEM = """你刚才输出的 pandas 代码执行失败。请根据错误信息修正后，**重新输出完整的可执行代码**（不是 diff、不是解释）。
仍然遵守：只输出代码；最后一行是结果表达式；df 只读；禁 import/文件/网络；使用真实中文列名。

你上一版代码：
```python
{bad_code}
```

执行错误（原样）：
{error}
"""


class LocalExcelQueryTool(_BaseExcelTool):
    """自然语言查询本地表格：全量数据 pandas 取数（只读，不审批）。"""

    name: str = "local_excel_query_tool"
    description: str = (
        "用一句中文问题直接查询/统计本地 Excel/CSV（底层在**全量数据**上执行 pandas 计算，"
        "不是前若干行预览）：筛选、分组聚合、透视、TopN、环比同比、多条件计算、跨列运算都支持。"
        "多 sheet 文件必须给 sheet_name，单 sheet 可省略。"
    )

    def __init__(self, base_dir: Optional[str] = None, model_router: Any = None) -> None:
        super().__init__(base_dir)
        self._model_router = model_router
        self.parameters = [
            ToolParameter(name="file_path", type="string",
                          description="本地表格文件路径（相对项目根或绝对路径）", required=True),
            ToolParameter(name="query", type="string",
                          description="自然语言问题/计算需求，如：2026年8月的赢单金额、赢单数、赢单率分别是多少？",
                          required=True),
            ToolParameter(name="sheet_name", type="string",
                          description="工作表名；文件有多个 sheet 时必填，单 sheet 文件可省略",
                          required=False),
        ]

    # ------------------------------------------------------------------
    # 表加载 / sheet 选择
    # ------------------------------------------------------------------
    def _load_sheets(self, path) -> Dict[str, Any]:
        import pandas as pd

        suffix: str = path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            return {path.stem: pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")}
        read_kwargs: Dict[str, Any] = {"sheet_name": None}
        engine = _preferred_excel_engine()
        if engine:
            read_kwargs["engine"] = engine
        return pd.read_excel(path, **read_kwargs)

    def _select_sheet(
        self, sheets: Dict[str, Any], sheet_name: str
    ) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        """返回 (DataFrame, 真实sheet名, 错误文本)。"""
        if len(sheets) == 1:
            only = next(iter(sheets))
            if sheet_name and sheet_name != only:
                # 单 sheet 文件：容忍近名（模型偶尔会把文件名当 sheet 名）
                resolved, _ = LocalExcelReadTool._ground_name(sheet_name, list(sheets), "sheet 名")
                if resolved is None:
                    return sheets[only], only, None  # 只有一个表，直接用并在调用处注明
            return sheets[only], only, None
        if not sheet_name:
            return None, None, (
                f"该文件有 {len(sheets)} 个 sheet，请用 sheet_name 指定要查询哪一个："
                f"{', '.join(str(n) for n in sheets)}"
            )
        if sheet_name in sheets:
            return sheets[sheet_name], sheet_name, None
        resolved, _note = LocalExcelReadTool._ground_name(sheet_name, list(sheets), "sheet 名")
        if resolved is None:
            return None, None, _note
        return sheets[str(resolved)], str(resolved), None

    # ------------------------------------------------------------------
    # LLM 代码生成（走全局 ModelRouter）
    # ------------------------------------------------------------------
    async def _codegen(self, system: str, user: str) -> str:
        if self._model_router is None:
            raise RuntimeError("local_excel_query_tool 未注入 model_router，无法生成查询代码")
        resp = await self._model_router.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            purpose_hint="react",      # FAST tier：代码生成要快要便宜
            thinking=False,            # 纯代码生成，关思考
            temperature=0.0,
            max_tokens=800,
        )
        return _strip_code_fence(getattr(resp, "content", "") or "")

    async def execute(self, **kwargs: Any) -> str:
        file_path = kwargs.get("file_path")
        query = str(kwargs.get("query") or "").strip()
        sheet_name = str(kwargs.get("sheet_name") or "").strip()
        if not file_path:
            return "错误：未提供文件路径 file_path"
        if not query:
            return "错误：未提供查询问题 query"
        if self._model_router is None:
            return (
                "错误：local_excel_query_tool 未注入 model_router，自然语言查询不可用；"
                "请改用 local_excel_read_tool 的 filter_column/group_by 能力，或联系运维检查工具初始化配置。"
            )

        try:
            physical_path = self._resolve_physical_path(file_path)
            if physical_path is None:
                return self._grounding_with_real_files(file_path)

            sheets = self._load_sheets(physical_path)
            if not sheets:
                return f"错误：文件 [{physical_path.name}] 中没有任何可读工作表。"
            frame, real_sheet, err = self._select_sheet(sheets, sheet_name)
            if err or frame is None:
                return f"错误：{err}"

            context = _build_table_context(frame, real_sheet)
            user_msg = f"{context}\n\n问题：{query}"
            code = await self._codegen(_CODEGEN_SYSTEM, user_msg)

            last_error: str = ""
            for attempt in range(_MAX_REPAIR_ROUNDS + 1):
                try:
                    output = _execute_pandas_code(code, frame)
                    break
                except (SyntaxError, _UnsafeCodeError, Exception) as exc:  # noqa: BLE001
                    # 任何执行/护栏错误都转成自修复信号（错误文本不回灌文件内容）
                    last_error = f"{type(exc).__name__}: {exc}"[:800]
                    if attempt >= _MAX_REPAIR_ROUNDS:
                        return (
                            f"查询执行失败，已自动修正 {_MAX_REPAIR_ROUNDS} 次仍未成功。\n"
                            f"最后错误：{last_error}\n最后生成的代码：\n{code[:1200]}\n"
                            "建议：换一种问法（明确 sheet/列名/条件），或改用 local_excel_read_tool 先核对列结构。"
                        )
                    code = await self._codegen(
                        _REPAIR_SYSTEM.format(bad_code=code[:1200], error=last_error),
                        user_msg,
                    )
            else:  # pragma: no cover - 防御性分支
                return "错误：查询未能产出结果"

            result = (
                f"## 查询：{query}\n工作表 [{real_sheet}]（来源 {physical_path.name}）\n\n"
                f"结果：\n{output}"
            )
            if len(result) > _MAX_OUTPUT_CHARS:
                result = result[:_MAX_OUTPUT_CHARS] + (
                    f"\n\n（结果过长已截断，原长 {len(result)} 字符；请在 query 里收窄条件或只取关键列）"
                )
            return result

        except Exception as e:  # noqa: BLE001 - 工具边界不抛异常
            return f"表格自然语言查询工具期间崩溃: {type(e).__name__}: {e}"
