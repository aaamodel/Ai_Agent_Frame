# -*- coding: utf-8 -*-
"""一次性取数工具：导出知识库语料清单（doc_id / 文件名 / 首 200 字）。

两种模式（对齐《评测-压测-监控落地手册》第 2.2 节）：

1. ``--from-files``（默认，**离线可用**）
   扫描本地语料目录（默认 ``app/rag_data``），输出「文件名 / 字符数 / 首 200 字」清单。
   用途：写黄金集时照着真实语料挑文档；也能在 Milvus 未启动时先推进标注工作。

2. ``--from-milvus``（需要 Milvus 在线）
   走 LlamaIndex ``MilvusVectorStore`` 枚举集合内全部节点，按 ``document_id`` 聚合，
   输出**真实 document_id** 与文件名、切片数的对应表。

3. ``--fill-rag-cases``（需要先跑过 ``--from-milvus``）
   把 milvus 清单里的真实 ``document_id`` 回填进 ``evals/golden/rag_cases.jsonl``
   的 ``expected_doc_id`` 字段（按 ``expected_doc_name`` 匹配）。

    ⚠️ 这一步是 RAG 评测**可辩护性**的关键：``expected_doc_id`` 不能编，
    必须来自真实入库语料。回填前请人工核对文件名与文档内容是否真的对得上。

4. ``--fill-from-names``（**离线可用，推荐**）
   按 ``evals/doc_ids.py::make_doc_id`` 的**确定性派生规则**填 ``expected_doc_id``：

       document_id = "doc-" + md5(文件名 utf-8)[:12]

   为什么这条路径是推荐的默认：线上 ``/documents/upload`` 用 ``uuid4()``，
   每次入库的 doc_id 都不同，导致 ``expected_doc_id`` **无法离线标注、也无法跨
   批次复用**。而评测语料统一用 ``reingest_corpus.py`` 入库（默认物理集合、
   ``sales_kb`` 逻辑标签），doc_id 由文件名派生 → 同一份黄金集在常规评测与
   512 / 1024 A/B 实验里都成立。此时文件名本身就确定了 doc_id，
   无需连 Milvus 就能回填。

用法::

    # 离线：列出语料
    python evals/tools/dump_corpus.py --from-files --out evals/golden/_corpus_manifest.csv

    # 离线：按确定性规则回填 expected_doc_id（推荐）
    python evals/tools/dump_corpus.py --fill-from-names

    # 在线：导出真实 doc_id（需 Milvus 已启动且语料已入库）
    python evals/tools/dump_corpus.py --from-milvus --out evals/golden/_corpus_dump.csv

    # 在线：用真实 doc_id 回填（评测线上真实集合时用这条）
    python evals/tools/dump_corpus.py --fill-rag-cases --corpus-csv evals/golden/_corpus_dump.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# 允许 `python evals/tools/dump_corpus.py` 直接运行（把仓库根加入 sys.path）
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 与 reingest_corpus 共用同一真源：自动探测仓库根 rag_data / 旧位置 app/rag_data
from evals.corpus_rules import resolve_rag_data_dir  # noqa: E402

RAG_DATA_DIR: Path = resolve_rag_data_dir(_REPO_ROOT)
RAG_CASES_PATH: Path = _REPO_ROOT / "evals" / "golden" / "rag_cases.jsonl"

CORPUS_SUFFIXES = {".md", ".txt", ".pdf"}


def _iter_corpus_files(
    src_dir: Path, recursive: bool = True, *, include_excluded: bool = False
) -> List[Path]:
    """列出语料文件；默认排除评测资产（与 reingest_corpus 共用同一套规则）。

    保持两处口径一致很重要：``_corpus_manifest.csv`` 是"这个集合里到底灌了
    哪些文档"的凭据，如果它把被排除的文件也列进去，就会误导 RAG 评测的排查。
    """
    from evals.corpus_rules import exclusion_reason

    pattern = "**/*" if recursive else "*"
    out: List[Path] = []
    for path in sorted(src_dir.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in CORPUS_SUFFIXES:
            continue
        if not include_excluded and exclusion_reason(path, src_dir) is not None:
            continue
        out.append(path)
    return out


# =====================================================================
# 模式一：扫描本地目录
# =====================================================================
def _read_text_preview(path: Path, limit: int = 200) -> Dict[str, Any]:
    """读取文本预览；pdf 跳过正文（仅记录大小）。"""
    if path.suffix.lower() == ".pdf":
        return {"chars": "", "preview": "[PDF：请用在线模式导出正文预览]"}
    try:
        text: str = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # noqa: BLE001
        return {"chars": "", "preview": f"[读取失败: {exc}]"}
    flat: str = " ".join(text.split())
    return {"chars": str(len(text)), "preview": flat[:limit]}


def dump_from_files(
    src_dir: Path,
    out_path: Path,
    recursive: bool = True,
    *,
    include_excluded: bool = False,
) -> int:
    """扫描目录写出语料清单 CSV，返回文件数（默认排除评测资产）。"""
    if not src_dir.exists():
        raise SystemExit(f"语料目录不存在：{src_dir}")
    files: List[Path] = _iter_corpus_files(
        src_dir, recursive=recursive, include_excluded=include_excluded
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["doc_key", "relative_path", "suffix", "char_count", "first_200_chars"])
        for path in files:
            info = _read_text_preview(path)
            writer.writerow([
                path.name,
                str(path.relative_to(src_dir)).replace("\\", "/"),
                path.suffix.lower(),
                info["chars"],
                info["preview"],
            ])
    print(f"[dump_corpus] 扫描 {len(files)} 份语料 -> {out_path}")
    return len(files)


# =====================================================================
# 模式二：从 Milvus 导出真实 document_id
# =====================================================================
def _iter_nodes_from_milvus(uri: str, collection_name: str) -> List[Any]:
    """枚举集合内全部 TextNode（复用 LlamaIndex 的存储解析，避免手写字段名）。

    与 ``app/infrastructure/vectordb/milvus_store.py::get_all`` 同思路，
    但不依赖 embedding 模型（导出脚本不需要向量化能力）。
    """
    try:
        from llama_index.vector_stores.milvus import MilvusVectorStore
        from llama_index.vector_stores.milvus.base import MILVUS_ID_FIELD
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "需要 llama-index-vector-stores-milvus：pip install -r requirements.txt"
        ) from exc

    store = MilvusVectorStore(
        uri=uri, collection_name=collection_name, dim=1024, overwrite=False
    )
    client = store.client
    pk_field: str = str(MILVUS_ID_FIELD)

    node_ids: List[str] = []
    offset: int = 0
    batch: int = 1000
    while True:
        rows = client.query(
            collection_name=collection_name,
            output_fields=[pk_field],
            limit=batch,
            offset=offset,
        )
        if not rows:
            break
        node_ids.extend(str(r.get(pk_field)) for r in rows if r.get(pk_field))
        if len(rows) < batch:
            break
        offset += len(rows)

    nodes: List[Any] = []
    for start in range(0, len(node_ids), 1000):
        fetched = store.get_nodes(node_ids[start:start + 1000]) or []
        nodes.extend(n for n in fetched if n is not None)
    return nodes


def dump_from_milvus(out_path: Path, collection_name: Optional[str] = None) -> int:
    """从 Milvus 导出「document_id / 文件名 / 切片数 / 首 200 字」CSV，返回文档数。"""
    try:
        from app.config import get_settings
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("请在仓库根目录运行（需要 import app.config）") from exc

    settings = get_settings()
    target_collection: str = collection_name or settings.milvus_kb_collection_name
    uri: str = f"http://{settings.milvus_host}:{settings.milvus_port}"

    print(f"[dump_corpus] 连接 Milvus {uri} 集合 {target_collection} ...")
    nodes = _iter_nodes_from_milvus(uri, target_collection)
    if not nodes:
        raise SystemExit(
            f"集合 {target_collection} 为空或不可达。请先启动 Milvus 并上传语料"
            "（POST /api/v1/documents/upload，collection_name=sales_kb）。"
        )

    grouped: Dict[tuple, Dict[str, Any]] = {}
    for node in nodes:
        meta: Dict[str, Any] = dict(getattr(node, "metadata", None) or {})
        document_id: str = str(meta.get("document_id") or "")
        filename: str = str(meta.get("filename") or "")
        collection: str = str(meta.get("collection") or target_collection)
        key = (collection, document_id or filename)
        entry = grouped.setdefault(key, {
            "collection": collection,
            "document_id": document_id,
            "filename": filename,
            "chunk_count": 0,
            "first_200_chars": "",
        })
        entry["chunk_count"] += 1
        if not entry["first_200_chars"] and getattr(node, "text", None):
            entry["first_200_chars"] = " ".join(str(node.text).split())[:200]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["collection", "document_id", "filename", "chunk_count", "first_200_chars"])
        for entry in grouped.values():
            writer.writerow([
                entry["collection"], entry["document_id"], entry["filename"],
                entry["chunk_count"], entry["first_200_chars"],
            ])
    print(f"[dump_corpus] 集合 {target_collection} 共 {len(nodes)} 节点 / {len(grouped)} 文档 -> {out_path}")
    return len(grouped)


# =====================================================================
# 模式三：回填 expected_doc_id
# =====================================================================
def fill_rag_cases(corpus_csv: Path, cases_path: Path = RAG_CASES_PATH) -> Dict[str, int]:
    """把真实 document_id 回填进 rag_cases.jsonl（按 expected_doc_name 匹配）。"""
    if not corpus_csv.exists():
        raise SystemExit(f"语料清单不存在：{corpus_csv}（先跑 --from-milvus）")

    name_to_id: Dict[str, str] = {}
    with corpus_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = (row.get("filename") or "").strip()
            document_id = (row.get("document_id") or "").strip()
            if filename and document_id:
                name_to_id[filename] = document_id

    if not name_to_id:
        raise SystemExit("语料清单里没有非空的 filename/document_id，无法回填。")

    records: List[Dict[str, Any]] = []
    with cases_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(json.loads(text))

    filled: int = 0
    unmatched: List[str] = []
    for record in records:
        name = str(record.get("expected_doc_name") or "")
        document_id = name_to_id.get(name)
        if document_id:
            record["expected_doc_id"] = document_id
            filled += 1
        else:
            unmatched.append(name)

    with cases_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[dump_corpus] 已回填 {filled}/{len(records)} 条 expected_doc_id -> {cases_path}")
    if unmatched:
        print(f"[dump_corpus] 未匹配到 doc_id 的文档名（请人工核对是否已入库）: {sorted(set(unmatched))}")
    return {"filled": filled, "total": len(records), "unmatched": len(set(unmatched))}


# =====================================================================
# 模式四：离线按确定性规则回填 expected_doc_id
# =====================================================================
def fill_rag_cases_from_names(
    cases_path: Path = RAG_CASES_PATH,
    *,
    known_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """按 ``make_doc_id`` 的确定性规则离线回填 ``expected_doc_id``。

    Args:
        cases_path: ``rag_cases.jsonl`` 路径。
        known_names: 可选的"真实存在"的文件名白名单；提供时会对黄金集里
            指向不存在文件的标注报错（防止标注了一个已删除的文档）。

    Returns:
        ``{"filled", "total", "missing_files"}``。
    """
    from evals.doc_ids import make_doc_id

    records: List[Dict[str, Any]] = []
    with cases_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                records.append(json.loads(text))

    known: set[str] = {str(n) for n in (known_names or [])}
    filled: int = 0
    missing: List[str] = []
    for record in records:
        name: str = str(record.get("expected_doc_name") or "").strip()
        if not name:
            missing.append("(空 expected_doc_name)")
            continue
        record["expected_doc_id"] = make_doc_id(name)
        filled += 1
        if known and name not in known:
            missing.append(name)

    with cases_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[dump_corpus] 按确定性规则回填 {filled}/{len(records)} 条 expected_doc_id "
        f"-> {cases_path}"
    )
    if missing:
        print(
            "[dump_corpus] ⚠️ 以下 expected_doc_name 在语料目录里找不到对应文件，"
            f"请核对：{sorted(set(missing))}"
        )
    return {"filled": filled, "total": len(records), "missing_files": sorted(set(missing))}


# =====================================================================
# CLI
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="导出知识库语料清单 / 回填 RAG 黄金集 doc_id")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-files", action="store_true", help="扫描本地语料目录（离线可用）")
    mode.add_argument("--from-milvus", action="store_true", help="从 Milvus 导出真实 document_id")
    mode.add_argument("--fill-rag-cases", action="store_true", help="把 doc_id 回填进 rag_cases.jsonl")
    mode.add_argument(
        "--fill-from-names",
        action="store_true",
        help="离线按 make_doc_id 确定性规则回填 expected_doc_id（推荐）",
    )

    parser.add_argument("--src-dir", type=Path, default=RAG_DATA_DIR, help="本地语料目录")
    parser.add_argument("--out", type=Path, default=None, help="输出 CSV 路径")
    parser.add_argument("--collection", default=None, help="Milvus 集合名（默认取配置）")
    parser.add_argument("--corpus-csv", type=Path, default=None, help="--fill-rag-cases 用的语料清单")
    parser.add_argument("--cases", type=Path, default=RAG_CASES_PATH, help="rag_cases.jsonl 路径")
    parser.add_argument("--no-recursive", action="store_true", help="--from-files 时不递归子目录")
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="连评测资产一起列出（默认排除 other/ 与评测问题集类文件，与 reingest 口径一致）",
    )
    args = parser.parse_args()

    if args.from_files:
        out = args.out or (_REPO_ROOT / "evals" / "golden" / "_corpus_manifest.csv")
        dump_from_files(
            args.src_dir,
            out,
            recursive=not args.no_recursive,
            include_excluded=args.include_excluded,
        )
    elif args.from_milvus:
        out = args.out or (_REPO_ROOT / "evals" / "golden" / "_corpus_dump.csv")
        dump_from_milvus(out, collection_name=args.collection)
    elif args.fill_from_names:
        # 顺带做一次"标注指向的文件是否真的存在"的体检
        existing: List[str] = sorted(
            p.name for p in _iter_corpus_files(
                RAG_DATA_DIR, include_excluded=args.include_excluded
            )
        )
        fill_rag_cases_from_names(args.cases, known_names=existing)
    else:
        corpus_csv = args.corpus_csv or (_REPO_ROOT / "evals" / "golden" / "_corpus_dump.csv")
        fill_rag_cases(corpus_csv, args.cases)


if __name__ == "__main__":
    main()
