# -*- coding: utf-8 -*-
"""黄金集纯净度校验（离线可跑，不需要 Milvus / Redis / 模型 Key）。

## 为什么必须有这个脚本

D1 缺陷的教训：意图黄金集 15 条里有 10 条的 query 与 ``intent_tree.py`` 的
``examples`` **逐字相同**，而 ``examples`` 同时进了向量索引和 LLM prompt ——
等于拿训练集当测试集，准确率天然虚高。

这类问题**不会报错、不会变红、只在面试被问时才暴露**。所以正确做法不是
"这次改干净了"，而是**把判据固化成每次可跑的检查**：任何人往黄金集里
加一条与示例重合的 query，脚本立刻拦下来。

## 检查项

===========  ==================================================  =======
编号          检查内容                                             级别
===========  ==================================================  =======
C1            意图集 query 与 ``intent_tree.examples`` 归一化后完全相同    FAIL
C1'           意图集 query 与 examples 高度相似（>= 0.80）               WARN
C2            意图集 ``expected_intent`` 是否为意图树中真实存在的叶子 id      FAIL
C3            RAG 集 query 与语料库小标题高度相似（>= 0.75）                WARN
C3'           RAG 集 query 与「销售助手评测问题集.md」问题列高度相似          WARN
C4            RAG 集 ``expected_doc_id`` 与 md5(文件名)[:12] 规则一致       FAIL
C4'           RAG 集 ``expected_doc_name`` 对应的语料文件是否真实存在         FAIL
C5            工具集引用的工具名是否都在真实注册白名单内                      FAIL
C5'           工具集是否引用了 feishu（暂不可用）                          FAIL
C6            工具集 ``key_args`` 的键是否是该工具 schema 的合法参数名         FAIL
===========  ==================================================  =======

用法::

    python -m evals.tools.check_golden_purity            # 全量检查
    python -m evals.tools.check_golden_purity --json out.json

退出码：有 FAIL 返回 1，仅有 WARN 返回 0（WARN 让人看得见，但不拦 CI）。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evals.corpus_rules import resolve_rag_data_dir  # noqa: E402

GOLDEN_DIR: Path = _REPO_ROOT / "evals" / "golden"
# 与 reingest_corpus / dump_corpus 共用同一真源（语料已从 app/rag_data 迁到 rag_data）
RAG_DATA_DIR: Path = resolve_rag_data_dir(_REPO_ROOT)
TOOL_SCHEMAS_PATH: Path = GOLDEN_DIR / "_tool_schemas.json"

# 相似度阈值（difflib.SequenceMatcher 的 ratio）
_SIM_FAIL_EXAMPLES: float = 0.92   # 与 examples 这么像，基本是抄的 → 视为 FAIL
_SIM_WARN_EXAMPLES: float = 0.80   # 明显雷同 → WARN
_SIM_WARN_HEADING: float = 0.75    # 与语料小标题/问题集问题雷同 → WARN

# 暂不可用的工具（用户明确指示剔除）
_UNAVAILABLE_TOOLS: Set[str] = {"feishu_bitable_tool"}


# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------
def _norm(text: Any) -> str:
    """归一化：全角转半角、去所有空白与中英标点、小写。

    用于把「请假流程是怎样的？」与「请假流程是怎样的?」判为同一句。
    """
    raw: str = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(
        ch for ch in raw
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def _sim(a: Any, b: Any) -> float:
    """归一化后的序列相似度（0~1）。"""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        records.append(json.loads(line))
    return records


def make_doc_id(filename: str) -> str:
    """与 evals/doc_ids.py 同一规则：doc- + md5(basename)[:12]。"""
    base: str = Path(str(filename)).name
    return "doc-" + hashlib.md5(base.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------
# 数据源加载
# ---------------------------------------------------------------------
def load_examples_map() -> Tuple[Dict[str, List[str]], Optional[str], Set[str]]:
    """加载意图树 examples。

    Returns:
        (``{node_id: [examples...]}``, 导入错误信息或 None, 全部叶子节点 id 集合)
    """
    try:
        from app.query_intent.intent_classify_resolver.intent_tree import (
            IntentTreeFactory,
        )
    except Exception as exc:  # noqa: BLE001 - 缺依赖时该项跳过而不是崩溃
        return {}, f"{type(exc).__name__}: {exc}", set()

    roots = IntentTreeFactory.build_intent_tree()
    examples_map: Dict[str, List[str]] = {}
    leaf_ids: Set[str] = set()

    def walk(nodes: Sequence[Any]) -> None:
        for node in nodes:
            children = list(getattr(node, "children", None) or [])
            if children:
                walk(children)
            else:
                leaf_ids.add(str(node.id))
                if getattr(node, "examples", None):
                    examples_map[str(node.id)] = [str(e) for e in node.examples]

    walk(roots)
    return examples_map, None, leaf_ids


def load_corpus_headings() -> List[Tuple[str, str]]:
    """收集语料库（app/rag_data 下 md/txt）的各级标题。

    Returns:
        ``[(文件名, 标题文本), ...]``
    """
    out: List[Tuple[str, str]] = []
    if not RAG_DATA_DIR.exists():
        return out
    for path in sorted(RAG_DATA_DIR.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                # 去掉 markdown 加粗/引号等残留
                title = title.replace("**", "").replace("`", "").strip("\"'“”")
                if title:
                    out.append((path.name, title))
    return out


def load_question_set_items() -> List[str]:
    """从「销售助手评测问题集.md」抽取表格里的「问题」列。

    该文件被放在 app/rag_data/other/ 下，一旦被灌进检索库就形成污染源；
    同时它的问题也可能与 RAG 黄金集撞车，两项都要检查。
    """
    target: Path = RAG_DATA_DIR / "other" / "销售助手评测问题集.md"
    if not target.exists():
        return []
    items: List[str] = []
    for line in target.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        # 形如 | Q1 | 问题 | 数据源 | 要点 |
        if len(cells) >= 2 and cells[0].upper().startswith("Q"):
            items.append(cells[1])
    return items


def load_tool_schema() -> Tuple[Set[str], Dict[str, Set[str]], Optional[str]]:
    """加载工具 schema 静态快照。

    Returns:
        (全部工具名, ``{工具名: 合法参数名集合}``, 错误信息或 None)
    """
    if not TOOL_SCHEMAS_PATH.exists():
        return set(), {}, f"缺少 {TOOL_SCHEMAS_PATH.name}（请先跑 evals/tools/export_tool_schemas.py）"
    try:
        payload: Dict[str, Any] = json.loads(TOOL_SCHEMAS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return set(), {}, f"{type(exc).__name__}: {exc}"

    names: Set[str] = set()
    params: Dict[str, Set[str]] = {}
    for item in payload.get("tools") or []:
        fn: Dict[str, Any] = item.get("function") or {}
        name: str = str(fn.get("name") or "")
        if not name:
            continue
        names.add(name)
        props = ((fn.get("parameters") or {}).get("properties") or {})
        params[name] = {str(k) for k in props}
    return names, params, None


# ---------------------------------------------------------------------
# 检查
# ---------------------------------------------------------------------
def check_intent_cases(
    failures: List[str], warnings: List[str],
) -> Dict[str, Any]:
    cases = _load_jsonl(GOLDEN_DIR / "intent_cases.jsonl")
    examples_map, err, leaf_ids = load_examples_map()
    all_examples: List[Tuple[str, str]] = [
        (node_id, ex) for node_id, exs in examples_map.items() for ex in exs
    ]

    exact_hits: List[str] = []
    near_hits: List[str] = []
    bad_ids: List[str] = []

    for case in cases:
        query = str(case.get("query") or "")
        nq = _norm(query)

        for node_id, ex in all_examples:
            if nq == _norm(ex):
                exact_hits.append(f"{case['id']} ↔ {node_id}.examples「{ex}」")
                break
            sim = _sim(query, ex)
            if sim >= _SIM_FAIL_EXAMPLES:
                near_hits.append(
                    f"{case['id']} 与 {node_id}.examples「{ex}」相似度 {sim:.2f}（视同抄写）"
                )
                break
            if sim >= _SIM_WARN_EXAMPLES:
                near_hits.append(
                    f"[WARN] {case['id']} 与 {node_id}.examples「{ex}」相似度 {sim:.2f}"
                )
                break

        expected = str(case.get("expected_intent") or "")
        if leaf_ids and expected not in leaf_ids:
            bad_ids.append(f"{case['id']} expected_intent={expected} 不是意图树叶子 id")

    for item in exact_hits:
        failures.append(f"C1 意图集与 examples 逐字重合：{item}")
    for item in near_hits:
        (failures if not item.startswith("[WARN]") else warnings).append(
            f"C1' {item.replace('[WARN] ', '')}" if not item.startswith("[WARN]") else f"C1' {item}"
        )
    for item in bad_ids:
        failures.append(f"C2 {item}")

    return {
        "cases": len(cases),
        "examples_total": len(all_examples),
        "examples_map_loaded": bool(examples_map),
        "import_error": err,
        "exact_overlap": len(exact_hits),
        "near_overlap": len(near_hits),
        "bad_expected_intent": len(bad_ids),
    }


def check_rag_cases(failures: List[str], warnings: List[str]) -> Dict[str, Any]:
    cases = _load_jsonl(GOLDEN_DIR / "rag_cases.jsonl")
    corpus_files: Set[str] = {
        p.name for p in RAG_DATA_DIR.glob("**/*")
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}
    } if RAG_DATA_DIR.exists() else set()
    headings = load_corpus_headings()
    question_items = load_question_set_items()

    heading_warns: List[str] = []
    qset_warns: List[str] = []
    id_bad: List[str] = []
    name_missing: List[str] = []

    for case in cases:
        query = str(case.get("query") or "")
        doc_name = case.get("expected_doc_name")
        doc_id = case.get("expected_doc_id")

        # ---- C4：doc_id 与 md5 规则一致性 ----
        if doc_name:
            expect_id = make_doc_id(str(doc_name))
            if str(doc_id or "") != expect_id:
                id_bad.append(
                    f"{case['id']} expected_doc_id={doc_id} 应为 {expect_id}（{doc_name}）"
                )
            if str(doc_name) not in corpus_files:
                name_missing.append(f"{case['id']} 期望文档 {doc_name} 不在 app/rag_data 中")

        # ---- C3：与语料小标题雷同 ----
        best_head = max(
            ((_sim(query, t), f, t) for f, t in headings),
            key=lambda x: x[0],
            default=(0.0, "", ""),
        )
        if best_head[0] >= _SIM_WARN_HEADING and len(_norm(best_head[2])) >= 6:
            heading_warns.append(
                f"{case['id']}「{query}」≈ {best_head[1]} 标题「{best_head[2]}」"
                f"（{best_head[0]:.2f}）"
            )

        # ---- C3'：与销售助手评测问题集撞车 ----
        best_q = max(((_sim(query, q), q) for q in question_items), key=lambda x: x[0], default=(0.0, ""))
        if best_q[0] >= _SIM_WARN_HEADING and len(_norm(best_q[1])) >= 6:
            qset_warns.append(
                f"{case['id']}「{query}」≈ 问题集「{best_q[1]}」（{best_q[0]:.2f}）"
            )

    for item in id_bad:
        failures.append(f"C4 {item}")
    for item in name_missing:
        failures.append(f"C4' {item}")
    for item in heading_warns:
        warnings.append(f"C3 {item}")
    for item in qset_warns:
        warnings.append(f"C3' {item}")

    return {
        "cases": len(cases),
        "answerable": sum(1 for c in cases if not c.get("unanswerable")),
        "unanswerable": sum(1 for c in cases if c.get("unanswerable")),
        "doc_id_mismatch": len(id_bad),
        "missing_doc": len(name_missing),
        "heading_overlap": len(heading_warns),
        "questionset_overlap": len(qset_warns),
        "corpus_files": len(corpus_files),
    }


def check_tool_cases(failures: List[str], warnings: List[str]) -> Dict[str, Any]:
    cases = _load_jsonl(GOLDEN_DIR / "tool_cases.jsonl")
    tool_names, tool_params, err = load_tool_schema()

    unknown: List[str] = []
    feishu_refs: List[str] = []
    bad_keys: List[str] = []
    unavailable_refs: List[str] = []
    covered: Set[str] = set()

    for case in cases:
        expected = str(case.get("expected_tool") or "")
        acceptable = [str(t) for t in (case.get("acceptable_tools") or [])]
        used = [expected, *acceptable]
        covered.update(used)

        for name in used:
            if name in _UNAVAILABLE_TOOLS:
                unavailable_refs.append(f"{case['id']} 引用了暂不可用工具 {name}")
            if name.casefold().startswith("feishu") or "bitable" in name.casefold():
                feishu_refs.append(f"{case['id']} 引用了飞书工具 {name}")
            if tool_names and name not in tool_names:
                unknown.append(f"{case['id']} 工具 {name} 不在 schema 快照中")

        # ---- C6：key_args 键名合法性（只校验 expected_tool）----
        key_args: Dict[str, Any] = case.get("key_args") or {}
        if key_args and expected in tool_params:
            legal = tool_params[expected]
            for key, spec in key_args.items():
                aliases: List[str] = []
                if isinstance(spec, dict) and "aliases" in spec:
                    aliases = [str(a) for a in (spec.get("aliases") or [])]
                candidates = {key, *aliases}
                if legal and not (candidates & legal):
                    bad_keys.append(
                        f"{case['id']} key_args 键 {sorted(candidates)} 均不是 {expected} 的合法参数"
                        f"（合法：{sorted(legal)}）"
                    )

    for item in unknown:
        failures.append(f"C5 {item}")
    for item in feishu_refs:
        failures.append(f"C5' {item}")
    for item in unavailable_refs:
        failures.append(f"C5' {item}")
    for item in bad_keys:
        failures.append(f"C6 {item}")

    return {
        "cases": len(cases),
        "tool_schema_loaded": bool(tool_names),
        "schema_error": err,
        "unknown_tool_refs": len(unknown),
        "feishu_refs": len(feishu_refs),
        "bad_key_args": len(bad_keys),
        "covered_tools": sorted(covered),
        "uncovered_tools": sorted(tool_names - covered - _UNAVAILABLE_TOOLS) if tool_names else [],
    }


# ---------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="黄金集纯净度校验（离线）")
    parser.add_argument("--json", type=Path, default=None, help="把结果同时写成 JSON")
    args = parser.parse_args()

    failures: List[str] = []
    warnings: List[str] = []

    intent_summary = check_intent_cases(failures, warnings)
    rag_summary = check_rag_cases(failures, warnings)
    tool_summary = check_tool_cases(failures, warnings)

    print("=" * 74)
    print("黄金集纯净度校验")
    print("=" * 74)

    print("\n【意图集】")
    print(f"  样本 {intent_summary['cases']} 条；意图树 examples 共 {intent_summary['examples_total']} 条")
    if intent_summary["import_error"]:
        print(f"  ⚠️ 意图树未能导入，C1/C2 已跳过：{intent_summary['import_error']}")
    else:
        print(f"  与 examples 逐字重合：{intent_summary['exact_overlap']} 条（期望 0）")
        print(f"  与 examples 近似重合：{intent_summary['near_overlap']} 条（期望 0）")
        print(f"  expected_intent 非法：{intent_summary['bad_expected_intent']} 条（期望 0）")

    print("\n【RAG 集】")
    print(f"  样本 {rag_summary['cases']} 条"
          f"（可答 {rag_summary['answerable']} / 应拒答 {rag_summary['unanswerable']}）"
          f"；语料 {rag_summary['corpus_files']} 篇")
    print(f"  doc_id 与 md5 规则不符：{rag_summary['doc_id_mismatch']} 条（期望 0）")
    print(f"  期望文档不存在：{rag_summary['missing_doc']} 条（期望 0）")
    print(f"  与语料小标题雷同：{rag_summary['heading_overlap']} 条（越少越好）")
    print(f"  与销售助手评测问题集雷同：{rag_summary['questionset_overlap']} 条（越少越好）")

    print("\n【工具集】")
    print(f"  样本 {tool_summary['cases']} 条")
    print(f"  引用了不存在的工具：{tool_summary['unknown_tool_refs']} 条（期望 0）")
    print(f"  引用飞书工具：{tool_summary['feishu_refs']} 条（期望 0）")
    print(f"  key_args 参数名非法：{tool_summary['bad_key_args']} 条（期望 0）")
    if tool_summary["uncovered_tools"]:
        print(f"  未被覆盖的可用工具：{', '.join(tool_summary['uncovered_tools'])}")

    if warnings:
        print(f"\n{'-' * 74}\n提示（WARN {len(warnings)} 条，不阻断）\n{'-' * 74}")
        for item in warnings:
            print(f"  · {item}")

    if failures:
        print(f"\n{'-' * 74}\n❌ 违规（FAIL {len(failures)} 条）\n{'-' * 74}")
        for item in failures:
            print(f"  ✗ {item}")
    else:
        print(f"\n✅ 无 FAIL 违规，黄金集纯净度检查通过。")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "failures": failures,
                    "warnings": warnings,
                    "intent": intent_summary,
                    "rag": rag_summary,
                    "tool": tool_summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n结果已写出：{args.json}")

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
