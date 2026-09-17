# -*- coding: utf-8 -*-
"""评测语料入库工具：把 ``app/rag_data`` 的语料按指定 chunk size 灌进知识库。

## 集合口径（评测即线上，不再使用独立评测集合）

- **物理集合**：Milvus 里的真实 collection，默认与线上同一个
  （``settings.milvus_kb_collection_name``，当前 ``knowledge_base_v3``），
  可用 ``--physical-collection`` 覆盖（chunk-size A/B 实验传 eval_kb_* ）；
- **逻辑集合**：每个向量节点 ``metadata['collection']`` 上的标签，默认
  ``sales_kb``（``--collection`` 可改）。它与黄金集 rag_cases.jsonl 的
  ``collection`` 字段、上传接口表单 collection_name 完全同口径。

⚠️ 因此本脚本灌进去的数据**线上 Agent 立刻能检索到**——这是当前的有意决策，
评测与生产共用一套知识库数据。

## 为什么不能复用线上上传接口

``POST /documents/upload`` 的 doc_id 用 ``uuid4()``，且每次只处理一个文件、需要
Postgres 记录元数据。评测需要的是：**批量、确定性 ID、可指定逻辑标签**。所以这里
直接用 ``RAGService.ingest_texts``（与线上同一个写入实现），只把 ID 规则和标签
换掉。注意：确定性 doc_id（``doc-xxxx``）与线上 uuid4 文档若同标签共存，检索会
出现同内容重复切片，二选一即可。

## 语料纯净度（默认排除评测资产）

默认**不收录** ``app/rag_data/other/`` 与文件名命中 ``*评测问题集*`` / ``*golden*``
的文件——它们是评测参考资产，不是答案语料。若被灌进检索库，黄金集里的问题会
直接命中"题目 + 期望要点"本身，**Recall@5 系统性虚高且看不出来**。排除规则
定义在 ``evals/corpus_rules.py``（单一真源，``dump_corpus.py`` 共用）。

原始文件不会被修改或删除；需要复现旧口径做对照时用 ``--include-excluded``。

⚠️ **单位陷阱**：``chunk_size`` 走 LlamaIndex ``SentenceSplitter``，单位是
**token 而非字符**；512 token 在中英混排下约 400~700 个汉字。dry-run 输出的
``avg_chunk_chars`` 用来反向验证参数是否生效。

## 删表重建保护（本脚本是全仓库唯一会"删表重建"的入口）

1. ``--reset``（删表重建）必须配合 ``--confirm-reset``；
2. ⚠️ ``--reset`` 删的是**整个物理集合**：灌进默认物理集合时，里面其他逻辑标签
   的数据也会一起被清空；
3. 支持 ``--dry-run``：只统计会切成多少片，不连任何服务。

用法::

    # 1) 空跑，先看分块结果（不需要 Milvus）
    python -m evals.tools.reingest_corpus --dry-run

    # 2) 真正入库（默认物理集合 knowledge_base_v3 + 逻辑标签 sales_kb）
    python -m evals.tools.reingest_corpus --chunk-size 512 --reset --confirm-reset

    # 3) chunk-size A/B 实验：写入独立物理集合（不影响线上）
    python -m evals.tools.reingest_corpus --collection eval_kb_512 \
        --physical-collection eval_kb_512 --chunk-size 512 --reset --confirm-reset
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.corpus_rules import exclusion_reason, resolve_rag_data_dir  # noqa: E402
from evals.doc_ids import make_doc_id  # noqa: E402

# 语料根目录：自动探测（仓库根 rag_data 优先，兼容旧位置 app/rag_data）。
# 写死旧路径会导致"文件=0 总片数=0"→ 集合永远是空的 → RAG 评测 Recall 恒为 0。
RAG_DATA_DIR: Path = resolve_rag_data_dir(_REPO_ROOT)
CORPUS_SUFFIXES = {".md", ".txt"}

# 默认 overlap 比例：与 document.py 的 512/64 保持同一比例（1/8），
# 这样 512 vs 1024 的对比只改"片长"、不改"重叠相对量"，是更干净的单变量实验。
DEFAULT_OVERLAP_RATIO: float = 0.125


def _collect_files(
    src_dir: Path,
    recursive: bool = True,
    *,
    include_excluded: bool = False,
    only_files: Optional[List[str]] = None,
) -> List[Path]:
    """收集可入库语料文件。

    ⚠️ 默认**排除评测资产**（``app/rag_data/other/`` 下的评测问题集等）——
    它们一旦被灌进检索库，黄金集的问题会直接命中"题目+答案要点"本身，
    造成 Recall@5 系统性虚高。排除规则见 ``evals/corpus_rules.py``。

    原始文件不会被改动或删除；需要"什么都收"的旧口径做对照时传
    ``include_excluded=True``（CLI: ``--include-excluded``）。

    ``only_files`` 是文件名白名单（精确匹配、忽略大小写），用于**单独补录**
    某几份缺失文档而不动同目录其他文件（CLI: ``--only-file``）。
    """
    pattern: str = "**/*" if recursive else "*"
    wanted: set = {
        str(name).strip().casefold() for name in (only_files or []) if str(name).strip()
    }
    files: List[Path] = []
    for path in sorted(src_dir.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in CORPUS_SUFFIXES:
            continue
        if wanted and path.name.casefold() not in wanted:
            continue
        if not include_excluded and exclusion_reason(path, src_dir) is not None:
            continue
        # 统一转绝对路径：--src-dir 传相对路径时，下游 relative_to() 会抛 ValueError
        files.append(path.resolve())
    return files


def list_excluded_files(src_dir: Path, recursive: bool = True) -> List[tuple]:
    """列出会被排除的语料文件及原因（``[(path, reason), ...]``），供 dry-run 展示。"""
    pattern: str = "**/*" if recursive else "*"
    excluded: List[tuple] = []
    for path in sorted(src_dir.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in CORPUS_SUFFIXES:
            continue
        reason = exclusion_reason(path, src_dir)
        if reason:
            excluded.append((path.resolve(), reason))
    return excluded


def plan_chunks(
    src_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    recursive: bool = True,
    include_excluded: bool = False,
    only_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """离线计算每份语料会切成多少片（``--dry-run`` 用，不连任何服务）。"""
    from app.core.rag.parse import split_text

    files: List[Path] = _collect_files(
        src_dir,
        recursive=recursive,
        include_excluded=include_excluded,
        only_files=only_files,
    )
    plan: List[Dict[str, Any]] = []
    total_chunks: int = 0
    for path in files:
        text: str = path.read_text(encoding="utf-8", errors="ignore")
        chunks: List[str] = split_text(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        total_chunks += len(chunks)
        plan.append(
            {
                "filename": path.name,
                "document_id": make_doc_id(path.name),
                "chars": len(text),
                "chunks": len(chunks),
                # 平均片长：能直观看出 1024 是否真的切得更长（验证参数生效）
                "avg_chunk_chars": round(len(text) / len(chunks), 1) if chunks else 0.0,
                "first_chunk_head": (chunks[0][:60] if chunks else ""),
            }
        )
    return {
        "src_dir": str(src_dir),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "files": len(files),
        "total_chunks": total_chunks,
        "plan": plan,
    }


async def ingest_into_collection(
    *,
    src_dir: Path,
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
    reset: bool,
    physical_collection: Optional[str] = None,
    recursive: bool = True,
    include_excluded: bool = False,
    only_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """把语料批量写入，返回统计信息。

    Args:
        collection_name: 逻辑集合标签（写进每个节点的 metadata['collection']，
            与上传接口 collection_name 同口径），默认 sales_kb。
        physical_collection: 物理 Milvus 集合；None = 与线上同一个
            （settings.milvus_kb_collection_name）。
    """
    from app.core.rag.parse import split_text
    from evals.runners._runtime import build_rag_service, build_settings

    settings: Any = build_settings()
    target_physical: str = physical_collection or str(
        settings.milvus_kb_collection_name
    )

    # ⚠️ overwrite=True 只能由 --reset 显式触发；它会 drop 整个**物理**集合。
    rag_service: Any = build_rag_service(
        settings,
        overwrite=reset,
        collection_name=target_physical,
        top_k_default=10,
    )

    files: List[Path] = _collect_files(
        src_dir,
        recursive=recursive,
        include_excluded=include_excluded,
        only_files=only_files,
    )
    ingested: List[Dict[str, Any]] = []

    for path in files:
        text: str = path.read_text(encoding="utf-8", errors="ignore")
        chunks: List[str] = split_text(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if not chunks:
            ingested.append({"filename": path.name, "chunks": 0, "node_ids": 0})
            continue
        document_id: str = make_doc_id(path.name)
        metadatas: List[Dict[str, Any]] = [
            {
                "document_id": document_id,
                "chunk_index": index,
                "filename": path.name,
                # 逻辑集合标签：与线上上传接口 Form collection_name 完全同口径
                "collection": collection_name,
                # 以下两个是评测专有字段，便于事后复核"这批数据是哪组参数灌的"
                "eval_chunk_size": chunk_size,
                "eval_chunk_overlap": chunk_overlap,
            }
            for index in range(len(chunks))
        ]
        node_ids: List[str] = await rag_service.ingest_texts(chunks, metadatas)
        ingested.append(
            {"filename": path.name, "document_id": document_id,
             "chunks": len(chunks), "node_ids": len(node_ids)}
        )
        print(
            f"[reingest] {path.name} -> {len(chunks)} 片 (doc_id={document_id})",
            flush=True,
        )

    return {
        "collection": collection_name,
        "physical_collection": target_physical,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "files": len(ingested),
        "total_chunks": sum(item["chunks"] for item in ingested),
        "detail": ingested,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把评测语料按指定 chunk size 灌进知识库（默认物理集合 + sales_kb 标签）"
    )
    parser.add_argument(
        "--collection",
        default="sales_kb",
        help="逻辑集合标签（写进 metadata['collection']），默认 sales_kb；"
             "与黄金集 collection 字段、上传接口 collection_name 同口径。",
    )
    parser.add_argument(
        "--physical-collection",
        default=None,
        help="物理 Milvus 集合名；默认与线上同一个"
             "（settings.milvus_kb_collection_name）。A/B 实验可指向 eval_kb_*。",
    )
    parser.add_argument("--chunk-size", type=int, required=True,
                        help="分块长度，单位是 **token**（LlamaIndex SentenceSplitter 口径），"
                             "不是字符数。中英混排时 512 token 大约对应 400~700 个汉字。")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help=f"分块重叠；默认 = chunk_size * {DEFAULT_OVERLAP_RATIO}（保持与 512/64 同比例）",
    )
    parser.add_argument("--src-dir", type=Path, default=RAG_DATA_DIR, help="语料目录")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="⚠️ 删除并重建**物理**集合（其中所有逻辑标签的数据都会清空）",
    )
    parser.add_argument(
        "--confirm-reset",
        action="store_true",
        help="--reset 的二次确认开关",
    )
    parser.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    parser.add_argument(
        "--only-file",
        default=None,
        help="只入库文件名精确匹配的语料（逗号分隔多个），用于**单独补录**缺失文档；"
        "不传则入库目录下全部文件。示例："
        "--only-file 01_人事制度与员工手册.md",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help=(
            "连评测资产一起收录（默认排除 app/rag_data/other/ 与评测问题集类文件）。"
            "⚠️ 仅在需要复现「什么都收」的旧口径做对照实验时使用；"
            "正常评测必须保持默认排除，否则黄金集问题会直接命中问题集本身，Recall 虚高。"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="只统计分块，不写入")
    args = parser.parse_args()

    recursive: bool = not args.no_recursive
    include_excluded: bool = args.include_excluded
    only_files: Optional[List[str]] = (
        [name.strip() for name in args.only_file.split(",") if name.strip()]
        if args.only_file
        else None
    )
    excluded_preview = list_excluded_files(args.src_dir, recursive=recursive)
    chunk_overlap: int = (
        args.chunk_overlap
        if args.chunk_overlap is not None
        else max(0, int(args.chunk_size * DEFAULT_OVERLAP_RATIO))
    )

    # 解析物理集合（默认与线上同一个——评测/生产共用数据是当前有意决策）
    try:
        from evals.runners._runtime import build_settings

        default_physical: str = str(build_settings().milvus_kb_collection_name)
    except Exception:  # noqa: BLE001 - 配置不可读时不阻塞 dry-run
        default_physical = ""
    target_physical: str = args.physical_collection or default_physical

    # ---- 唯一保护：删表（整个物理集合）重建需二次确认 ----
    if args.reset and not args.confirm_reset and not args.dry_run:
        raise SystemExit(
            "❌ --reset 会删除并重建整个物理集合（其中所有逻辑标签的数据都会清空），\n"
            "   需要同时加 --confirm-reset 才执行。\n"
            f"   物理集合：{target_physical}，逻辑标签：{args.collection}"
        )

    if excluded_preview and not include_excluded:
        print(
            f"[reingest] 已排除 {len(excluded_preview)} 份非答案语料（评测资产）："
        )
        for path, reason in excluded_preview:
            print(f"  · {path.relative_to(_REPO_ROOT)} —— {reason}")

    if args.dry_run:
        plan: Dict[str, Any] = plan_chunks(
            args.src_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=chunk_overlap,
            recursive=recursive,
            include_excluded=include_excluded,
            only_files=only_files,
        )
        print(
            f"[reingest][dry-run] 物理集合={target_physical or '(配置不可读)'} "
            f"逻辑标签={args.collection} chunk_size={args.chunk_size} "
            f"overlap={chunk_overlap} 文件={plan['files']} 总片数={plan['total_chunks']}"
        )
        for item in plan["plan"]:
            print(
                f"  - {item['filename']}: {item['chars']} 字符 -> {item['chunks']} 片 "
                f"(平均 {item['avg_chunk_chars']} 字符/片, doc_id={item['document_id']})"
            )
        return

    if args.reset:
        print(
            f"⚠️ 即将删除并重建整个物理集合 [{target_physical}]"
            f"（含所有逻辑标签），再以标签 [{args.collection}] 写入语料。",
            flush=True,
        )

    result: Dict[str, Any] = asyncio.run(
        ingest_into_collection(
            src_dir=args.src_dir,
            collection_name=args.collection,
            physical_collection=target_physical or None,
            chunk_size=args.chunk_size,
            chunk_overlap=chunk_overlap,
            reset=args.reset,
            recursive=recursive,
            include_excluded=include_excluded,
            only_files=only_files,
        )
    )
    print(
        f"[reingest] 完成：物理集合={result['physical_collection']} "
        f"逻辑标签={result['collection']} "
        f"chunk_size={result['chunk_size']} overlap={result['chunk_overlap']} "
        f"文件={result['files']} 总片数={result['total_chunks']}"
    )


if __name__ == "__main__":
    main()
