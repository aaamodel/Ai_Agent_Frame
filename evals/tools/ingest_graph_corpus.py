# -*- coding: utf-8 -*-
"""图谱语料入库工具：把 ``rag_data_graph/`` 的跨实体关系语料织入 LightRAG 图谱 workspace。

## 为什么单独一个工具（和 reingest_corpus 什么关系）

``reingest_corpus.py`` 灌的是**向量库（Milvus）**，走 ``RAGService.ingest_texts``；
本脚本灌的是**知识图谱（LightRAG workspace）**，走 ``light_rag.insert_document``。
两者的存储、集合概念（逻辑集合 vs workspace）、检索通道（rag_knowledge_search vs
knowledge_graph_search）完全不同，所以不做成一个脚本。

它补的是 ``evals/golden/DATA_ISSUES.md`` F13 记录的空缺：意图树
``knowledge-entity-relation`` 节点声明走图谱，但此前**没有任何图谱语料**，
该通道在评测里无法被有效验证。

## 集合口径

图谱集合 = LightRAG 的 **workspace**（``WORKING_DIR/<workspace>/`` 各自独立）。
意图识别没有下"图谱集合硬约束"时，``knowledge_graph_search`` 的 ``collection``
留空 → 落到 ``DEFAULT_GRAPH_WORKSPACE = "default"``，所以本工具默认也写 default。

## 安全边界（本脚本不做的事）

1. **不删除**：不改用 ``clear_workspace()`` / ``delete_documents_by_filename()``，
   不覆盖既有 workspace 数据；
2. **幂等**：已入库的同名文件（按 ``file_path`` 匹配）直接跳过，可安全重跑；
   需要替换已入库文件时，先走 ``DELETE /knowledgebase/collections/{workspace}/files/{filename}``；
3. ``--dry-run`` 完全离线：只列文件与字数，不连图谱、不消耗 LLM。

用法::

    python -m evals.tools.ingest_graph_corpus --dry-run      # 先空跑
    python -m evals.tools.ingest_graph_corpus                # 入库默认集合 default
    python -m evals.tools.ingest_graph_corpus --workspace sales_graph
    python -m evals.tools.ingest_graph_corpus --only-file 01_销售跨实体关系图谱.txt

⚠️ 用仓库上级 venv 运行（系统 Python 缺 langfuse/asyncpg 等依赖）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 图谱语料目录（与 rag_data/、rag_data_enterprise/ 并列；**不在 RAG 向量语料收录范围内**）
GRAPH_CORPUS_DIR: Path = _REPO_ROOT / "rag_data_graph"
CORPUS_SUFFIXES = {".md", ".txt"}
# 与 light_rag.DEFAULT_GRAPH_WORKSPACE 一致（意图未指定集合时工具落到这里）
DEFAULT_WORKSPACE = "default"


def _log(message: str) -> None:
    print(f"[graph-ingest] {message}", flush=True)


def _collect_files(
    src_dir: Path,
    recursive: bool = True,
    only_files: Optional[List[str]] = None,
) -> List[Path]:
    """收集图谱语料文件。

    同一份语料通常同时存在 ``.md``（人读）与 ``.txt``（LightRAG 传统入口只认
    .pdf/.txt）两个镜像，**按文件名主干去重**，只入库一次（优先 .txt）。
    ``README*`` 是说明文档，不入库。
    """
    pattern: str = "**/*" if recursive else "*"
    wanted = {str(n).strip().casefold() for n in (only_files or []) if str(n).strip()}
    by_stem: Dict[str, Path] = {}

    for path in sorted(src_dir.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in CORPUS_SUFFIXES:
            continue
        if path.name.casefold().startswith("readme"):
            continue
        if wanted and path.name.casefold() not in wanted:
            continue
        existing = by_stem.get(path.stem)
        if existing is None or (
            path.suffix.lower() == ".txt" and existing.suffix.lower() != ".txt"
        ):
            by_stem[path.stem] = path.resolve()
    return sorted(by_stem.values())


def _ensure_dashscope_key() -> None:
    """把百炼 key 桥接给 ``light_rag``。

    ``light_rag`` 在**模块级**执行 ``os.getenv("DASHSCOPE_API_KEY")``，而本项目的
    ``.env`` 里只有 ``OPENAI_API_KEY``（本就是同一个百炼 key、同一个
    compatible-mode 端点），且 ``.env`` 只在 uvicorn 启动时由 ``load_dotenv()``
    灌进 ``os.environ``。命令行直接跑本脚本时若不桥接，LLM 抽取会因 api_key 为空而失败。
    必须在 ``import light_rag`` **之前**调用（模块级已取值）。
    """
    if os.getenv("DASHSCOPE_API_KEY"):
        return
    from evals.runners._runtime import build_settings

    key: str = str(build_settings().openai_api_key or "").strip()
    if key:
        os.environ["DASHSCOPE_API_KEY"] = key
        _log("DASHSCOPE_API_KEY 未设置，已用配置中的 OPENAI_API_KEY 桥接（同一百炼 key）。")
    else:
        _log("⚠️ 既没有 DASHSCOPE_API_KEY，配置里也没有 OPENAI_API_KEY，LLM 抽取会失败。")


async def _existing_filenames(workspace: str) -> set:
    """列出该 workspace 已入库文档的 ``file_path``（即入库时传的文件名），用于幂等跳过。"""
    from app.infrastructure.knowledgebase.light_rag import list_workspace_documents

    docs = await list_workspace_documents(workspace)
    return {str(doc.get("filename") or "").strip() for doc in docs}


async def ingest_into_graph(
    *,
    src_dir: Path,
    workspace: str,
    recursive: bool = True,
    only_files: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """把语料织入图谱 workspace，返回统计信息。"""
    files: List[Path] = _collect_files(src_dir, recursive, only_files)
    if not files:
        _log(f"⚠️ 在 {src_dir} 下没有找到可入库的图谱语料（.md/.txt）。")
        return {"workspace": workspace, "files": 0, "ingested": 0, "skipped": 0}

    total_chars: int = sum(
        len(path.read_text(encoding="utf-8", errors="ignore")) for path in files
    )
    _log(f"语料目录 {src_dir}：{len(files)} 份文件 / {total_chars} 字")

    if dry_run:
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            _log(f"  - {path.name}: {len(text)} 字 -> workspace [{workspace}]")
        _log("[dry-run] 未连接图谱、未写入任何数据。")
        return {"workspace": workspace, "files": len(files), "ingested": 0, "skipped": 0}

    _ensure_dashscope_key()
    from app.infrastructure.knowledgebase.light_rag import insert_document

    existing: set = await _existing_filenames(workspace)
    if existing:
        _log(f"workspace [{workspace}] 已入库 {len(existing)} 份文档，同名文件将跳过。")

    ingested: int = 0
    skipped: int = 0
    for path in files:
        if path.name in existing:
            _log(f"  跳过（已入库同名文件）：{path.name}")
            skipped += 1
            continue
        content: str = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            _log(f"  跳过（内容为空）：{path.name}")
            skipped += 1
            continue
        await insert_document(workspace, path.name, content)
        ingested += 1
        _log(f"  已织入 workspace [{workspace}]：{path.name}（{len(content)} 字）")

    _log(
        f"完成：新增 {ingested} 份 / 跳过 {skipped} 份 -> workspace [{workspace}]。"
        "（实体抽取由 LLM 在后台完成，稍后可查 GET /knowledgebase/collections）"
    )
    return {
        "workspace": workspace,
        "files": len(files),
        "ingested": ingested,
        "skipped": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把 rag_data_graph 的跨实体关系语料织入 LightRAG 图谱 workspace"
    )
    parser.add_argument(
        "--workspace",
        default=DEFAULT_WORKSPACE,
        help=f"图谱集合（LightRAG workspace），默认 {DEFAULT_WORKSPACE}"
             "（意图未指定集合时 knowledge_graph_search 用的就是这个）",
    )
    parser.add_argument("--src-dir", type=Path, default=GRAPH_CORPUS_DIR, help="语料目录")
    parser.add_argument(
        "--only-file",
        default=None,
        help="只入库文件名精确匹配的语料（逗号分隔多个，如 01_销售跨实体关系图谱.txt）",
    )
    parser.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出会入库的文件与字数，不连图谱、不消耗 LLM",
    )
    args = parser.parse_args()

    only_files: Optional[List[str]] = (
        [name.strip() for name in args.only_file.split(",") if name.strip()]
        if args.only_file
        else None
    )
    result: Dict[str, Any] = asyncio.run(
        ingest_into_graph(
            src_dir=args.src_dir,
            workspace=args.workspace,
            recursive=not args.no_recursive,
            only_files=only_files,
            dry_run=args.dry_run,
        )
    )
    if not args.dry_run:
        _log(
            f"汇总：文件 {result['files']} / 新增 {result['ingested']} / "
            f"跳过 {result['skipped']}"
        )


if __name__ == "__main__":
    main()
