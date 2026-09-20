# -*- coding: utf-8 -*-
"""集合枚举约束 + 集合元数据单一权威字段的专项测试。

覆盖两组能力：

    agent/kb-collection-routing      —— 集合参数受约束取值、越界静默过滤与回落
    platform/kb-collection-metadata  —— retrieval_hint 退役、description 唯一承载路由语义

背景（2026-09-18 实测）：Planner 给 `rag_knowledge_search` 传了 `product_docs` /
`sales_policies` / `pricing_guide` 三个**并不存在**的集合名，两个子任务全部空召回、
工具被硬熔断、整轮降级收尾。当时该参数的说明里还写着一个示例 `如 ['hr_docs']`——
`hr_docs` 本身也不是真实集合。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, List

import pytest

_REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "app").is_dir())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.rag.collection_guard import (  # noqa: E402
    clear_fallbacks,
    partition_collection_names,
    recent_fallbacks,
    record_fallback,
)
from app.core.rag.rag_service import RAGService  # noqa: E402
from app.core.tools.base import BaseTool, ToolParameter  # noqa: E402
from app.core.tools.builtin.rag_search import RagSearchTool  # noqa: E402
from app.query_intent.kb_collection_registry import (  # noqa: E402
    KbCollectionDescriptor,
    KbCollectionRegistry,
)

# ── 真实集合（取自向量库实际枚举）────────────────────────────────────────────
DESCRIBED = [
    ("sales_kb", "销售智能与销售运营知识集合（sales_kb / sales_intel）。覆盖：产品与定价、客户画像与ICP、"
                 "销售方法论与商机评估、销售分析与指标诊断、目标考核、成功案例、竞品情报与应对、"
                 "客户反馈、市场活动、销售培训、折扣权限与异议话术"),
    ("Finance_Invoice_Manual", "财务与发票信息手册。覆盖：开票信息查询路径与字段构成、发票开具与红冲流程、"
                               "报销流程与时限、差旅标准、付款与收款、预算与成本中心"),
    ("Human_Resources_Employee", "人事制度与员工手册。覆盖：入职材料与流程、试用期与转正规则、"
                                 "考勤与假期、离职申请时限与工作交接六步流程"),
]
#: description 为 null 的系统兜底桶（3 篇未分类文档）
UNDESCRIBED = ("_untagged",)


@pytest.fixture(autouse=True)
def _registry() -> Any:
    """每个用例前灌入一份可控的集合快照，用后清空——注册表是进程级全局状态。"""
    KbCollectionRegistry.replace(
        [KbCollectionDescriptor(name=name, description=desc, engine="rag")
         for name, desc in DESCRIBED]
        + [KbCollectionDescriptor(name=name, description="", engine="rag") for name in UNDESCRIBED]
    )
    clear_fallbacks()
    yield
    KbCollectionRegistry.clear()
    clear_fallbacks()


def _rag_tool() -> RagSearchTool:
    return RagSearchTool(None)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# 1.1 参数模型：enum / items 必须能导出，且不影响存量工具
# ═══════════════════════════════════════════════════════════════════════════
def test_default_export_is_byte_identical_to_before() -> None:
    """未设置 enum/items 的参数，导出结果必须与改动前**逐字一致**。

    否则每个工具都会平白多出 schema 体积，而 `build_dynamic_*` 之外的工具完全用不到。
    """

    class _Plain(BaseTool):
        name = "plain"

        def __init__(self) -> None:
            super().__init__()
            self.parameters = [
                ToolParameter(name="q", type="string", description="问题", required=True),
                ToolParameter(name="k", type="integer", description="条数", required=False),
            ]

        async def execute(self, **kwargs: Any) -> str:
            return ""

    assert _Plain().schema_parameters() == {
        "type": "object",
        "properties": {
            "q": {"type": "string", "description": "问题"},
            "k": {"type": "integer", "description": "条数"},
        },
        "required": ["q"],
    }


def test_enum_and_items_are_exported_when_set() -> None:
    class _Constrained(BaseTool):
        name = "constrained"

        def __init__(self) -> None:
            super().__init__()
            self.parameters = [
                ToolParameter(
                    name="collections", type="array", description="集合",
                    required=False, items={"type": "string", "enum": ["a", "b"]},
                ),
                ToolParameter(name="mode", type="string", description="模式",
                              required=False, enum=["x", "y"]),
            ]

        async def execute(self, **kwargs: Any) -> str:
            return ""

    schema = _Constrained().schema_parameters()
    assert schema["properties"]["collections"]["items"] == {"type": "string", "enum": ["a", "b"]}
    assert schema["properties"]["mode"]["enum"] == ["x", "y"]


# ═══════════════════════════════════════════════════════════════════════════
# 1.2 / 2.2 rag 工具：非集合参数不得被牵连
# ═══════════════════════════════════════════════════════════════════════════
def test_rag_tool_other_params_carry_no_extra_keys() -> None:
    schema = _rag_tool().schema_parameters()
    for param in ("query", "top_k"):
        assert set(schema["properties"][param]) == {"type", "description"}


# ═══════════════════════════════════════════════════════════════════════════
# 2.1 注册表口径：只有具备描述的集合
# ═══════════════════════════════════════════════════════════════════════════
def test_described_excludes_blank_descriptions() -> None:
    names = [row.name for row in KbCollectionRegistry.described()]
    assert names == [name for name, _ in DESCRIBED]
    assert UNDESCRIBED[0] not in names


def test_described_preserves_registration_order() -> None:
    KbCollectionRegistry.replace([
        KbCollectionDescriptor(name="z_last", description="有描述", engine="rag"),
        KbCollectionDescriptor(name="a_first", description="也有描述", engine="rag"),
    ])
    assert KbCollectionRegistry.describe_names() == ["z_last", "a_first"]


def test_is_valid_uses_full_registry_not_enum_scope() -> None:
    """越界过滤的口径是"注册表内全部集合"，而不是"可显式选择的集合"。

    两者混用会把 `_untagged` 这种"存在但无描述"的集合误判成幻觉并丢弃——它有真实文档，
    丢掉它等于把 3 篇文档从检索里凭空抹掉。
    """
    assert KbCollectionRegistry.is_valid("_untagged") is True
    assert KbCollectionRegistry.is_valid("sales_kb") is True
    assert KbCollectionRegistry.is_valid("pricing_guide") is False


# ═══════════════════════════════════════════════════════════════════════════
# 2.2 枚举 = 有描述集合全集；每个取值逐条带描述；不含注册表外的名字
# ═══════════════════════════════════════════════════════════════════════════
def test_enum_equals_described_set() -> None:
    prop = _rag_tool().schema_parameters()["properties"]["collection_names"]
    assert prop["items"]["enum"] == [name for name, _ in DESCRIBED]


def test_enum_lives_on_items_not_on_the_array_itself() -> None:
    """⚠️ 挂在数组同一层的 enum 会被解读为"整个数组只能等于这几个值之一"。"""
    prop = _rag_tool().schema_parameters()["properties"]["collection_names"]
    assert "enum" not in prop
    assert prop["items"]["type"] == "string"


def test_every_enum_value_has_a_description_line() -> None:
    prop = _rag_tool().schema_parameters()["properties"]["collection_names"]
    description = prop["description"]
    for name, text in DESCRIBED:
        assert f"- {name}：" in description
        assert text[:20] in description


def test_no_name_outside_registry_appears_anywhere() -> None:
    """参数说明里不得出现任何不存在的集合名——包括作为"示例"。"""
    text = _rag_tool().schema_parameters()["properties"]["collection_names"]["description"]
    for bogus in ("hr_docs", "product_docs", "sales_policies", "pricing_guide"):
        assert bogus not in text
    assert "_untagged" not in text, "无描述的集合不得出现在说明里"


def test_empty_registry_yields_no_enum_and_no_empty_entries() -> None:
    """注册表为空时不发枚举、也不留下无描述的空条目（比发一个空 enum 诚实）。

    注意：基础说明文本里本来就有"下列可用集合"这句话，因此这里判的是**有没有真的列出清单**
    （是否存在"- "条目），而不是有没有出现某个词。
    """
    KbCollectionRegistry.clear()
    prop = _rag_tool().schema_parameters()["properties"]["collection_names"]
    assert "enum" not in prop.get("items", {})
    assert "- " not in prop["description"], "不得留下无描述的空条目"


# ═══════════════════════════════════════════════════════════════════════════
# 2.3 枚举随注册表刷新，无需重建工具实例
# ═══════════════════════════════════════════════════════════════════════════
def test_enum_refreshes_without_rebuilding_the_tool() -> None:
    tool = _rag_tool()
    before = tool.schema_parameters()["properties"]["collection_names"]["items"]["enum"]
    assert "New_Collection" not in before

    KbCollectionRegistry.replace([
        KbCollectionDescriptor(name="New_Collection", description="新上传的集合", engine="rag"),
    ])
    after = tool.schema_parameters()["properties"]["collection_names"]["items"]["enum"]
    assert after == ["New_Collection"], "新增集合必须在下一个请求即可选，无需重启或重建工具"


# ═══════════════════════════════════════════════════════════════════════════
# 3.1 越界过滤
# ═══════════════════════════════════════════════════════════════════════════
def test_partition_drops_out_of_registry_names() -> None:
    kept, dropped = partition_collection_names(
        ["sales_kb", "pricing_guide", "sales_policies"], KbCollectionRegistry.is_valid
    )
    assert kept == ["sales_kb"]
    assert dropped == ["pricing_guide", "sales_policies"]


def test_partition_keeps_undescribed_but_real_collection() -> None:
    kept, dropped = partition_collection_names(["_untagged"], KbCollectionRegistry.is_valid)
    assert kept == ["_untagged"] and dropped == []


def test_partition_dedupes_and_ignores_blanks() -> None:
    kept, dropped = partition_collection_names(
        [" sales_kb ", "sales_kb", "", None, "   "], KbCollectionRegistry.is_valid
    )
    assert kept == ["sales_kb"] and dropped == []


# ═══════════════════════════════════════════════════════════════════════════
# 3.2 两条回落路径（走真实 retrieve_contexts）
# ═══════════════════════════════════════════════════════════════════════════
class _FakeRetriever:
    def __init__(self, hits: List[Any]) -> None:
        self._hits = hits

    def retrieve(self, query: str) -> List[Any]:
        return list(self._hits)


def _service(hits: List[Any]) -> RAGService:
    service = RAGService.__new__(RAGService)
    service._retriever = _FakeRetriever(hits)  # type: ignore[attr-defined]
    service.top_k_default = 5  # type: ignore[attr-defined]
    return service


def _hits() -> List[Any]:
    from llama_index.core.schema import NodeWithScore, TextNode

    return [
        NodeWithScore(node=TextNode(text="销售记录：客户A 成交 12 万", metadata={"collection": "sales_kb"}), score=0.9),
        NodeWithScore(node=TextNode(text="财务手册：开票流程", metadata={"collection": "Finance_Invoice_Manual"}), score=0.8),
    ]


@pytest.mark.asyncio
async def test_partially_valid_names_limit_retrieval_to_valid_ones() -> None:
    service = _service(_hits())
    out = await service.retrieve_contexts("q", collection_names=["sales_kb", "pricing_guide"])
    assert [item.metadata["collection"] for item in out] == ["sales_kb"]


@pytest.mark.asyncio
async def test_all_invalid_names_fall_back_to_unrestricted_retrieval() -> None:
    """全部越界 → 不限定集合检索，且请求必须正常完成（不失败、不空召回）。"""
    service = _service(_hits())
    out = await service.retrieve_contexts(
        "q", collection_names=["pricing_guide", "product_docs"]
    )
    assert len(out) == 2, "应回落为全库检索，而不是因幻觉直接空召回"


@pytest.mark.asyncio
async def test_valid_names_are_not_reported_as_fallback() -> None:
    service = _service(_hits())
    await service.retrieve_contexts("q", collection_names=["sales_kb"])
    assert recent_fallbacks() == []


# ═══════════════════════════════════════════════════════════════════════════
# 3.3 留痕：静默回落的必要配套
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.asyncio
async def test_fallback_records_the_dropped_values() -> None:
    service = _service(_hits())
    await service.retrieve_contexts("q", collection_names=["sales_kb", "pricing_guide", "nope"])

    records = recent_fallbacks()
    assert records, "越界必须留痕——否则幻觉被静默吸收、无法统计"
    assert set(records[0].dropped) == {"pricing_guide", "nope"}
    assert records[0].kept == ("sales_kb",)
    assert records[0].fell_back_to_all is False


@pytest.mark.asyncio
async def test_fallback_record_marks_full_fallback() -> None:
    service = _service(_hits())
    await service.retrieve_contexts("q", collection_names=["completely_made_up"])
    assert recent_fallbacks()[0].fell_back_to_all is True


def test_record_fallback_is_queryable() -> None:
    record = record_fallback(["a", "bogus"], ["a"], ["bogus"], fell_back_to_all=False)
    assert record.dropped == ("bogus",)
    assert recent_fallbacks()[0].dropped == ("bogus",)


# ═══════════════════════════════════════════════════════════════════════════
# 4.2 / 4.3 / 4.4 retrieval_hint 全链路退役
# ═══════════════════════════════════════════════════════════════════════════
def test_descriptor_has_single_route_semantic_field() -> None:
    fields = set(KbCollectionDescriptor.__dataclass_fields__)
    assert "description" in fields
    assert "retrieval_hint" not in fields


def test_intent_text_renders_only_one_semantic_block() -> None:
    text = KbCollectionDescriptor(name="sales_kb", description="销售智能与销售运营知识集合").intent_text()
    assert "适用检索时机" not in text
    assert "销售智能与销售运营知识集合" in text


def test_pydantic_schemas_dropped_the_field() -> None:
    from app.models.agent_schemas import DocumentUploadResponse, KbCollectionInfo

    assert "retrieval_hint" not in DocumentUploadResponse.model_fields
    assert "retrieval_hint" not in KbCollectionInfo.model_fields


def test_upload_endpoints_no_longer_declare_the_form_field() -> None:
    from app.api.routes.document import upload_document
    from app.api.routes.kownledgebase import (
        upload_kownledgebase_document,
        upload_multiple_documents,
    )

    from fastapi.params import Form

    for endpoint in (upload_document, upload_multiple_documents, upload_kownledgebase_document):
        params = inspect.signature(endpoint).parameters
        form_fields = {
            name for name, param in params.items()
            if isinstance(param.default, Form)
        }
        assert "retrieval_hint" not in form_fields, f"{endpoint.__name__} 仍声明了该字段"
        assert "description" in form_fields, "description 必须保留——它是唯一的路由语义字段"


def test_upsert_signature_dropped_the_field() -> None:
    from app.infrastructure.database.collection_repo import upsert_vector_collection

    assert "retrieval_hint" not in inspect.signature(upsert_vector_collection).parameters


def test_orm_model_dropped_the_column() -> None:
    from app.infrastructure.database.models import VectorCollection

    assert "retrieval_hint" not in VectorCollection.__table__.columns


def test_startup_ddl_drops_the_legacy_column_idempotently() -> None:
    source = Path(_REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "DROP COLUMN IF EXISTS retrieval_hint" in source, (
        "缺少幂等 DDL 时，列会在已有库中留存而代码已不读它——成为下一次误用的入口"
    )


def test_collection_asset_json_dropped_the_field() -> None:
    import json

    asset = _REPO_ROOT / "向量化数据库的集合.json"
    data = json.loads(asset.read_text(encoding="utf-8-sig"))
    assert data["collections"], "资产文件不应被清空"
    for collection in data["collections"]:
        assert "retrieval_hint" not in collection
        assert "name" in collection


# ═══════════════════════════════════════════════════════════════════════════
# 4.3③ 存量调用方多传该字段不得 422
# ═══════════════════════════════════════════════════════════════════════════
def test_extra_undeclared_form_field_is_ignored_by_fastapi() -> None:
    """框架层保证：未声明的多余 Form 字段会被忽略，不会 422。

    这是"删除表单字段"对存量调用方安全的**机制**依据——所以此处用最小 app 验证该机制，
    而不是去起真实 endpoint（那会牵出 Milvus / Postgres 依赖）。
    """
    from fastapi import FastAPI, Form
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/probe")
    async def probe(name: str = Form(default="")) -> dict:
        return {"name": name}

    response = TestClient(app).post(
        "/probe", data={"name": "sales_kb", "retrieval_hint": "旧的检索时机文本"}
    )
    assert response.status_code == 200
    assert response.json() == {"name": "sales_kb"}
