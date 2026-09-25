# -*- coding: utf-8 -*-
"""长期记忆：向量库存储与按会话召回。"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from app.models.agent_schemas import MemoryItem
from openai import OpenAI
from loguru import logger

from pymilvus import MilvusClient, DataType

import http
import dashscope
from dashscope import TextEmbedding

from app.infrastructure.trace.langfuse import embedding_span

@runtime_checkable
class LTMEmbedProtocol(Protocol):
    """嵌入模型接口。"""

    def embed_query(self, text: str) -> list[float]:
        ...


@runtime_checkable
class LTMCollectionProtocol(Protocol):
    """Milvus Collection 最小接口。"""

    def insert(self, data: Any, **kwargs: Any) -> Any:
        ...

    def search(
        self,
        data: list[list[float]],
        anns_field: str,
        param: dict[str, Any],
        limit: int,
        expr: str | None = None,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        ...

    def delete(self, expr: str, **kwargs: Any) -> Any:
        ...

    def flush(self, **kwargs: Any) -> Any:
        ...


# -*- coding: utf-8 -*-
"""
基础设施层：长期记忆协议的具体实现（Milvus 仓储与 OpenAI Embedding）。
"""

# -*- coding: utf-8 -*-




class QwenEmbeddingImpl:
    """
    嵌入模型实现类：对接阿里云百炼平台（DashScope）的通义千问向量模型。
    实现 LTMEmbedProtocol 契约。
    """

    def __init__(self, api_key: str, model: str = "text-embedding-v3"):
        """
        :param api_key: 阿里云百炼平台的 API Key (DASHSCOPE_API_KEY)
        :param model: 向量模型名称，推荐使用最新通用模型 'text-embedding-v3'
                      也可根据业务选择 'text-embedding-v1' 或 'text-embedding-v2'
        """
        self.api_key = api_key
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        """将文本转化为通义千问稠密向量"""
        try:
            # Langfuse 门控：会话内召回时挂 embedding observation，会话外建库 no-op
            with embedding_span("embedding.longterm_qwen", model=self.model):
                # 调用百炼平台的文本向量服务
                response = TextEmbedding.call(
                    model=self.model,
                    input=text,
                    api_key=self.api_key
                )

            # 百炼平台标准状态码校验
            if response.status_code == http.HTTPStatus.OK:
                # text-embedding-v3 返回的 embeddings 是一个包含 dict 的 list
                # 结构为: [{'embedding': [0.1, 0.2, ...], 'text_index': 0}]
                return response.output['embeddings'][0]['embedding']
            else:
                logger.error(
                    f"Qwen Embedding 请求失败: 状态码={response.status_code}, "
                    f"错误码={response.code}, 错误信息={response.message}"
                )
                raise RuntimeError(f"DashScope Error: {response.message}")

        except Exception as e:
            logger.error(f"Qwen Embedding 转化过程中发生异常: {e}")
            raise


class OpenAIEmbeddingImpl:
    """
    嵌入模型实现类：对接 OpenAI 兼容的 Embedding API。
    实现 LTMEmbedProtocol 契约。
    """

    def __init__(self, api_key: str, base_url: str, model: str = "text-embedding-3-small"):
        """
        :param api_key: 大模型 API 密钥
        :param base_url: 大模型 API 基础路径 (例如 https://api.deepseek.com/v1)
        :param model: 向量模型名称
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        """将文本转化为稠密向量"""
        try:
            with embedding_span("embedding.longterm_openai", model=self.model):
                response = self.client.embeddings.create(
                    input=[text],
                    model=self.model
                )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding 转化失败: {e}")
            raise


# -*- coding: utf-8 -*-
"""
基础设施层：长期记忆协议的具体实现（采用最新 MilvusClient 现代标准）。
"""




class MilvusCollectionWrapper:
    """
    Milvus 集合包装类：处理连接、自动建表、创建索引（基于新版 MilvusClient 接口）。
    实现 LTMCollectionProtocol 契约。
    """

    def __init__(
            self,
            collection_name: str,
            dim: int,
            host: str = "127.0.0.1",
            port: str = "19530"
    ):
        self.collection_name = collection_name
        self.dim = dim

        # 使用新版推荐的 MilvusClient，一行代码搞定连接
        endpoint = f"http://{host}:{port}"
        self.client = MilvusClient(uri=endpoint)

        # 自动化建表和索引
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """检查并创建符合长期记忆期望的简易 Schema 集合"""
        if self.client.has_collection(self.collection_name):
            return

        logger.info(f"Milvus 中未找到集合 {self.collection_name}，开始使用 MilvusClient 自动创建...")

        # 使用 MilvusClient 的 create_schema / add_field 体系
        schema = self.client.create_schema(auto_id=False, description="Agent 长期记忆存储库")

        # ✨ 修复点：将下面所有的 dtype 替换为 datatype
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="session_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="meta", datatype=DataType.VARCHAR, max_length=65535)

        # 配置索引参数
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="L2",
            params={"M": 8, "efConstruction": 64}
        )

        # 创建集合并建立索引
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info(f"Milvus 集合 {self.collection_name} 初始化成功！")

    # ---------------------------------------------------------------------------
    # 下面 4 个方法将 MilvusClient 的原生方法完美适配为原脚手架要求的 Protocol 格式
    # ---------------------------------------------------------------------------
    def insert(self, data: Any, **kwargs: Any) -> Any:
        # LongTermMemory 传过来的是 [{'pk': ..., 'embedding': ...}] 格式，MilvusClient 原生支持
        return self.client.insert(collection_name=self.collection_name, data=data)

    def search(
            self,
            data: list[list[float]],
            anns_field: str,
            param: dict[str, Any],
            limit: int,
            expr: str | None = None,
            output_fields: list[str] | None = None,
            **kwargs: Any,
    ) -> Any:

        raw_res = self.client.search(
            collection_name=self.collection_name,
            data=data,
            anns_field=anns_field,
            search_params=param,
            limit=limit,
            filter=expr,
            output_fields=output_fields
        )



        return  raw_res

    def delete(self, expr: str, **kwargs: Any) -> Any:
        return self.client.delete(collection_name=self.collection_name, filter=expr)

    def flush(self, **kwargs: Any) -> Any:
        # 新版 MilvusClient 默认自动 flush，这里直接跳过即可
        pass


# ---------------------------------------------------------------------------
# 长期记忆写入侧：结构化记忆条目（**0 次额外 LLM 调用**）
# ---------------------------------------------------------------------------
# 设计口径（回答"不调 LLM 该怎么抽取"）：
#   记忆条目 = 【问题全文】+【规则判定的结论句】+【结构化 sidecar】+【全文存档】。
#   - 问题全文：无损保留。它就是召回键的语义来源，信息密度最高、成本最低（≤200 字）。
#   - 结论句：**不做字符级截断**，改为句级选择——按可解释的打分规则从长答案里挑出
#     "最像结论"的整句。字符截断（取头/取中）是在写入时替未来某次查询做决定，而未来
#     查什么当时并不知道，任何固定位置都必然在某些 query 上丢关键信息；句级选择至少
#     保证**语义单位完整、规则可复现、可被单测断言**。
#   - 短答案直接全文保留（无损），只有超长答案才触发句级选择。
#   - sidecar：intent / tools / trace_id 等**本来就已结构化**的信号，零成本落库，
#     供 Milvus expr 做结构化预过滤（如"只在同一 intent 分区内召回"）。
#   - 全文**不进向量库**：Milvus 召回会把 meta 原样回传，存 8k 全文等于每次请求都白付
#     一笔带宽；且该内容永不被渲染。长期记忆是"结论索引"，全文归档归 DB，用 trace_id join。
#     想省钱又想不丢信息，唯一不矛盾的解法就是这个分层：留指针，不留内容。
LTM_QUESTION_MAX_CHARS: int = 200
"""写入长期记忆时保留的问题上限（问题全文，不截断语义）。"""

LTM_CONCLUSION_MAX_CHARS: int = 200
"""结论句上限（只对单句生效；触顶说明规则选错了句子，兜底硬切）。"""

_CONCLUSION_TARGET_CHARS: int = 60
"""结论句的目标长度带：超出部分开始按梯度扣分（越啰嗦越不像结论）。"""

_CONCLUSION_SHORT_SENTENCE_CHARS: int = 80
"""短句阈值：只有不超过此长度的句子里的数字才被视为"硬事实结论"并加分。"""

LTM_META_VERSION: int = 2
"""记忆条目结构版本。1 = 旧的"问题+答案前缀"；2 = 问题+结论句+sidecar。"""

# 句末切分：中英文终止符 或 换行
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])|\n+")
# 去噪：代码块 / Markdown 表格行 / 行首符号（列表项、强调符、标题符、序号）
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_LINE_LEAD = re.compile(r"^\s*(?:[-*+>#]+|\d+[.、)]|\*\*)\s*")
_WHITESPACE = re.compile(r"[ \t\u3000]+")
# 问题前缀（兼容 v1 "用户问题:" / 更早的 "用户提问:"）
_QUESTION_PREFIX_SPLIT = re.compile(r"^(?:用户问题|用户提问|问)\s*[:：]\s*")

# 结论候选句的标记词（命中即加权，是可解释信号而非 LLM 判断）
_CONCLUSION_MARKERS: tuple = (
    "因此", "所以", "综上", "结论", "总结", "如上", "由此可见",
    "建议", "答案是", "应当", "需要", "说明", "意味着", "关键",
)
# 拒答/不确定语义：这类句子信息量最高（"查不到"本身就是结论），必须加权保留
_UNCERTAIN_MARKERS: tuple = (
    "未找到", "未匹配", "没有找到", "未检索到", "无法确定", "无法回答",
    "没有依据", "缺乏依据", "不确定", "建议咨询", "暂无法", "不在知识库",
)


def extract_conclusion(
    answer: str,
    *,
    max_chars: int = LTM_CONCLUSION_MAX_CHARS,
) -> str:
    """从答案中按规则选出"最像结论"的**整句**（0 次 LLM 调用）。

    打分模型的两条取舍：

    1. **内容标记是主信号（±3），位置只是弱先验（±2）。**
       首句确实常是直接回答（BLUF 写法），但长分析型答案的首句往往是数据罗列。
       若给首句 +3，一个"121 字、塞满数字的华东区明细"就会压过真正的建议句——
       实测过，所以把位置降权。
    2. **长度惩罚必须够重，且数字加分只在短句里成立。**
       结论句的价值在于"紧凑、可复用"；长句里的数字是**论据**而非结论，不能加分。
       扣分按超出 60 字的部分每 40 字扣 1（上限 -3），形成"越啰嗦越不像结论"的梯度。

    参与打分的维度（全部可解释、可复现、可单测断言）：
      +3 含结论标记词            +3 含拒答/不确定语义（"查不到"本身就是结论）
      +2 首句                    +2 末句
      +1 含数字且短句（<=80 字）
      -min(3, 超出 60 字的部分 // 40)
    并列时取更靠前的句子。

    Args:
        answer: 智能体最终答案。
        max_chars: 单句长度兜底上限（正常选句不会触顶）。

    Returns:
        结论句纯文本；答案为空时返回空串；短答案（<= max_chars）原样返回。
    """
    text: str = (answer or "").strip()
    if not text:
        return ""
    cleaned: str = _TABLE_ROW.sub("", _CODE_FENCE.sub(" ", text)).strip()
    if not cleaned:
        return ""
    # 短答案不截断：无损优先，句级选择只在"必须压缩"时才出手
    if len(cleaned) <= max_chars:
        return cleaned

    sentences: list[str] = [
        _LINE_LEAD.sub("", sentence).strip() for sentence in _SENTENCE_SPLIT.split(cleaned)
    ]
    sentences = [s for s in sentences if len(s) >= 8]
    if not sentences:
        return cleaned[:max_chars].strip()

    last_index: int = len(sentences) - 1
    best_sentence: str = sentences[0]
    best_score: float = float("-inf")
    for index, sentence in enumerate(sentences):
        score: float = 0.0
        if any(marker in sentence for marker in _CONCLUSION_MARKERS):
            score += 3
        if any(marker in sentence for marker in _UNCERTAIN_MARKERS):
            score += 3
        if index == 0:
            score += 2
        if index == last_index:
            score += 2
        if len(sentence) <= _CONCLUSION_SHORT_SENTENCE_CHARS and any(
            char.isdigit() for char in sentence
        ):
            score += 1
        score -= min(3, max(0, len(sentence) - _CONCLUSION_TARGET_CHARS) // 40)
        if score > best_score:
            best_sentence, best_score = sentence, score

    return best_sentence[:max_chars]


@dataclass(frozen=True)
class LongTermDigest:
    """一条长期记忆：可注入部分（content）与存档/过滤部分（metadata）。"""

    content: str
    """**唯一会被渲染进提示词**的文本：问题全文 + 结论句。"""

    metadata: dict
    """结构化 sidecar + 全文存档；只用于过滤与回查，不参与渲染。"""

    @property
    def question(self) -> str:
        """从 content 中取出问题部分（供调用方做去重/日志）。"""
        first_line: str = self.content.split("\n", 1)[0]
        return _QUESTION_PREFIX_SPLIT.sub("", first_line).strip()


def build_ltm_digest(
    query: str,
    answer: str,
    *,
    mode: Optional[str] = None,
    trace_id: Optional[str] = None,
    intent: Optional[str] = None,
    tools: Optional[list[str]] = None,
) -> LongTermDigest:
    """把一问一答转成一条结构化长期记忆（**0 次额外 LLM 调用**）。

    **为什么不调用 LLM 做摘要（刻意的工程选择，不是图省事）：**

    1. 记忆的召回键是 query 的语义向量，"答案全文"对"下次能否召回"几乎没有边际
       贡献——真正有价值的是「问过什么 + 结论是什么」这两行结构。
    2. 落库记录可能被 ``long_term_mem_block`` 注入，不做上限时提示词体积随会话数
       线性膨胀（评测中单轮 4 万 token 的成因之一）。
    3. 写入发生在主链路 / 评测窗口内，多一次 LLM 调用 = 每轮对话多付**一整次 prompt**
       的钱，收益仅是"更漂亮的一段话"——净负收益。

    行业口径也一致：Mem0 的 ADD/UPDATE/DELETE 抽取、Zep/Graphiti 的实体抽取、
    ChatGPT 的记忆提炼确实用 LLM，但**全部挂在后台异步任务或离线批处理**上，且解决的
    是"从未结构化对话里挖事实"。这里天然就是结构化的（intent / 工具名都是现成字段），
    确定性规则即可。若确需 LLM 级摘要，应在离线回填流程里做，而不是放在在线写入路径。

    Args:
        query: 用户原始问题。
        answer: 智能体最终答案。
        mode: 编排模式（react / plan_execute），落 sidecar 供过滤。
        trace_id: 链路 ID；与短期消息的 trace_id 对齐后可做**跨系统去重**。
        intent: 识别到的意图，落 sidecar 供"同意图分区召回"。
        tools: 本轮实际调用过的工具名，落 sidecar 供"按工具过滤"。

    Returns:
        :class:`LongTermDigest`：``content`` 进提示词，``metadata`` 存档与过滤。
    """
    question: str = _WHITESPACE.sub(" ", (query or "").strip())[:LTM_QUESTION_MAX_CHARS]
    conclusion: str = extract_conclusion(answer)
    content: str = (
        f"用户问题: {question}\n结论: {conclusion}" if conclusion else f"用户问题: {question}"
    )

    metadata: dict[str, Any] = {"ltm_version": LTM_META_VERSION}
    if mode:
        metadata["mode"] = str(mode)
    if trace_id:
        metadata["trace_id"] = str(trace_id)
    if intent:
        metadata["intent"] = str(intent)
    # 保序去重：同一工具在一轮里可被调用多次，重复名在 sidecar 里没有信息量
    tool_names: list[str] = list(
        dict.fromkeys(str(t).strip() for t in (tools or []) if str(t).strip())
    )
    if tool_names:
        metadata["tools"] = tool_names[:8]
    # 只记体量与指针，**不把答案全文塞进向量库**：Milvus 每次召回都会把 meta 原样
    # 回传，6 条 × 8k 字符的"永不渲染内容"会白白吃掉每次请求的带宽与延迟。
    # 长期记忆的职责是"可召回的结论索引"，全文归档属于 DB / 对象存储，用 trace_id join。
    metadata["answer_chars"] = len((answer or "").strip())
    return LongTermDigest(content=content, metadata=metadata)


def resolve_max_distance() -> float:
    """读取召回相似度门限（L2 距离，越小越相似）。

    默认 ``0`` = **不过滤**（保持既有行为）。刻意不做成"猜一个阈值"：L2 距离的绝对
    量纲取决于 embedding 模型的归一化方式，拍脑袋设阈值要么形同虚设、要么把记忆全滤空。
    正确顺序是**先测量再定**——``recall`` 会把每次召回的命中距离打日志，跑几轮真实
    会话后按实际分布设一个能切掉"明显不相关尾巴"的值，再写进配置。

    Returns:
        非负浮点门限；``>0`` 时召回只保留 distance <= 门限的条目。
    """
    raw: str = os.getenv("LTM_MAX_DISTANCE", "0") or "0"
    try:
        value: float = float(raw)
    except (TypeError, ValueError):
        logger.warning("LTM_MAX_DISTANCE={!r} 无法解析为浮点，按不过滤处理", raw)
        return 0.0
    return value if value > 0 else 0.0


class LongTermMemory:
    """长期记忆：基于向量数据库的持久化记忆。"""

    vector_field: str = "embedding"
    content_field: str = "content"
    session_field: str = "session_id"
    pk_field: str = "pk"
    meta_field: str = "meta"

    metric_param: dict[str, Any] = {"metric_type": "L2", "params": {"nprobe": 16}}

    def __init__(self, milvus_collection: Any, embedding_model: Any) -> None:
        """
        :param milvus_collection: Milvus Collection，需包含向量、文本、会话 ID 等字段
        :param embedding_model: 含 ``embed_query`` 的嵌入模型
        """
        self._coll = milvus_collection
        self._embed = embedding_model

    def _ensure_embed(self) -> None:
        if not isinstance(self._embed, LTMEmbedProtocol):
            raise TypeError("embedding_model 需实现 embed_query")

    def _ensure_coll(self) -> None:
        if not isinstance(self._coll, LTMCollectionProtocol):
            raise TypeError("milvus_collection 需支持 insert/search/delete")

    async def store(self, session_id: str, content: str, metadata: dict[str, Any]) -> str:
        """写入一条长期记忆，返回 memory_id。"""
        self._ensure_embed()
        self._ensure_coll()

        memory_id = str(uuid.uuid4())
        meta = dict(metadata)
        meta["memory_id"] = memory_id

        def _sync() -> None:
            vec = self._embed.embed_query(content)
            row = {
                self.pk_field: memory_id,
                self.vector_field: vec,
                self.content_field: content,
                self.session_field: session_id,
                self.meta_field: json.dumps(meta, ensure_ascii=False),
            }
            # pymilvus 2.4+ 支持实体字典列表，字段名需与 Collection Schema 一致
            self._coll.insert([row])
            try:
                self._coll.flush()
            except Exception as fe:
                logger.warning("flush 失败（可忽略）: {}", fe)

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("长期记忆写入失败: {}", e)
            raise RuntimeError(f"store 失败: {e}") from e

        return memory_id

    async def recall(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        *,
        max_distance: Optional[float] = None,
    ) -> list[MemoryItem]:
        """按语义在指定会话内召回记忆。

        ⚠️ **架构提醒**：这里的 ``expr`` 带了 ``session_id == ...`` 过滤，也就是只能
        召回**本会话**的历史。而本会话历史同时在短期记忆（Redis）里 —— 两处内容高度
        重叠，等于同一批信息被注入两次。真正的"跨会话"记忆需要放开这个过滤（例如改成
        ``user_id`` 维度），否则本方法对短期记忆是纯冗余，这一点在调用方（去重逻辑）
        已经做了兜底，但根因在这里。

        Args:
            query: 当前问题文本。
            session_id: 会话 ID（当前实现下同时充当召回分区键）。
            top_k: 召回条数上限。
            max_distance: L2 距离门限（越小越相似）。``None`` 时读 ``LTM_MAX_DISTANCE``；
                非正值表示不过滤（保持既有行为，避免未测量就误杀召回）。
        """
        self._ensure_embed()
        self._ensure_coll()
        threshold: float = resolve_max_distance() if max_distance is None else float(max_distance)

        def _sync() -> list[MemoryItem]:
            vec = self._embed.embed_query(query)
            # 转义单引号，避免 expr 注入
            sid = session_id.replace("'", "\\'")
            expr = f'{self.session_field} == "{sid}"'
            out = self._coll.search(
                data=[vec],
                anns_field=self.vector_field,
                param=self.metric_param,
                limit=top_k,
                expr=expr,
                output_fields=[self.pk_field, self.content_field, self.meta_field],
            )
            items: list[MemoryItem] = []
            hits = out[0] if out else []

            # 命中距离分布：用于"先测量再定阈值"。阈值没定之前，这行日志就是唯一依据。
            distances: list[float] = [
                float(hit.get("distance", 0.0) or 0.0) for hit in hits
            ]
            if distances:
                logger.debug(
                    "长期记忆召回命中 {} 条，distance min={:.4f} max={:.4f}（门限={:.4f}，"
                    "0 表示不过滤）",
                    len(distances), min(distances), max(distances), threshold,
                )

            # 此时 hit 就是标准的 Python dict
            for hit in hits:
                entity = hit.get("entity") or {}

                rid = str(entity.get(self.pk_field) or hit.get("id") or "")
                text = str(entity.get(self.content_field) or "")

                meta_raw = entity.get(self.meta_field) or "{}"
                try:
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
                except json.JSONDecodeError:
                    meta = {}

                score = float(hit.get("distance", 0.0) or 0.0)

                # 相似度门限：切掉"明显不相关但被 top_k 强行凑数"的尾巴。
                # top_k 是无条件返回 N 条的，会话一长必然后半截是噪声。
                if threshold > 0 and score > threshold:
                    continue

                items.append(
                    MemoryItem(id=rid, content=text, score=score, metadata=meta),
                )
            return items

        try:
            return await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("长期记忆召回失败: {}", e)
            raise RuntimeError(f"recall 失败: {e}") from e

    async def forget(self, memory_id: str) -> None:
        """按主键删除一条记忆。"""
        self._ensure_coll()

        def _sync() -> None:
            mid = memory_id.replace("'", "\\'")
            expr = f'{self.pk_field} == "{mid}"'
            self._coll.delete(expr)
            try:
                self._coll.flush()
            except Exception as fe:
                logger.warning("flush 失败（可忽略）: {}", fe)

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("长期记忆删除失败: {}", e)
            raise RuntimeError(f"forget 失败: {e}") from e

    async def forget_session(self, session_id: str) -> None:
        """按会话批量删除该会话下的全部长期记忆。

        用途：会话级数据删除（用户要求"忘掉这段对话"）与**评测隔离**——
        评测跑完清掉自己的记忆分区，否则下一轮召回会命中上一轮的落库记录，
        单轮 token 成本会随着评测次数逐次膨胀，指标失去可比性。
        """
        self._ensure_coll()

        def _sync() -> None:
            sid = session_id.replace("'", "\\'")
            expr = f'{self.session_field} == "{sid}"'
            self._coll.delete(expr)
            try:
                self._coll.flush()
            except Exception as fe:
                logger.warning("flush 失败（可忽略）: {}", fe)

        try:
            await asyncio.to_thread(_sync)
        except Exception as e:
            logger.exception("长期记忆按会话删除失败: {}", e)
            raise RuntimeError(f"forget_session 失败: {e}") from e
