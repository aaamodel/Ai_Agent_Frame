# merged_query_processing.py
# ======================================================================
# 合并了以下5个文件：
# - rewrite_result.py
# - query_term_mapping_util.py
# - query_term_mapping_cache_manager.py
# - query_term_mapping_service.py
# - query_rewrite_service.py
# ======================================================================

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from app.query_intent.intent_data_base import AgentChatContext, IntentChatMessage
from app.query_intent.intent_dto import AgentRewriteResult
from app.query_intent.intent_entity import QueryTermMappingDO
from app.query_intent.intent_mapper import QueryTermMappingMapper

logger = logging.getLogger(__name__)


# ==============================
# 1. RewriteResult (from rewrite_result.py)
# ==============================
@dataclass
class RewriteResult:
    rewritten_question: str = ""
    sub_questions: list[str] = field(default_factory=list)


# ==============================
# 2. QueryTermMappingUtil (from query_term_mapping_util.py)
# ==============================
class QueryTermMappingUtil:

    @staticmethod
    def apply_mapping(text: str, source_term: str, target_term: str) -> str:
        if text is None or text == "" or source_term is None or source_term == "":
            return text

        sb = []
        idx = 0
        length = len(text)
        source_len = len(source_term)
        target_len = len(target_term)

        while idx < length:
            hit = text.find(source_term, idx)
            if hit < 0:
                sb.append(text[idx:length])
                break

            sb.append(text[idx:hit])

            already_target = (
                target_term is not None
                and hit + target_len <= length
                and text.startswith(target_term, hit)
            )

            if already_target:
                sb.append(text[hit:hit + target_len])
                idx = hit + target_len
            else:
                sb.append(target_term)
                idx = hit + source_len

        return "".join(sb)


# ==============================
# 3. QueryTermMappingCacheManager (from query_term_mapping_cache_manager.py)
# ==============================
CACHE_KEY = "ragent:query-term:mappings"
CACHE_EXPIRE_DAYS = 7


@dataclass
class QueryTermMappingCacheManager:
    string_redis_template: Optional[object] = None
    object_mapper: Optional[object] = None

    def get_mappings_from_cache(self) -> Optional[List[object]]:
        if self.string_redis_template is None:
            logger.warning("string_redis_template is None，跳过 Redis 缓存读取")
            return None
        try:
            cache_json = self.string_redis_template.ops_for_value().get(CACHE_KEY)
            if cache_json is None:
                logger.info("术语映射缓存不存在，需要从数据库加载")
                return None
            # 反序列化 JSON
            data = json.loads(cache_json)
            if isinstance(data, list):
                return data
            logger.warning("缓存数据格式异常，预期列表，实际: %s", type(data))
            return None
        except Exception as e:
            logger.exception("从 Redis 读取术语映射缓存失败")
            return None

    def save_mappings_to_cache(self, mappings: List[object]) -> None:
        if self.string_redis_template is None:
            logger.warning("string_redis_template is None，无法保存缓存")
            return
        try:
            # 将对象转为字典（假设有 as_dict 方法）或直接序列化（如果对象是 dict 或可序列化）
            # 这里简单处理：如果对象是 QueryTermMappingDO，我们可以转为 dict
            serializable = []
            for m in mappings:
                if hasattr(m, 'as_dict'):
                    serializable.append(m.as_dict())
                elif hasattr(m, '__dict__'):
                    serializable.append(m.__dict__)
                else:
                    serializable.append(m)  # 假设已经是可序列化类型
            cache_json = json.dumps(serializable, ensure_ascii=False)
            self.string_redis_template.ops_for_value().set(
                CACHE_KEY, cache_json, CACHE_EXPIRE_DAYS, "DAYS"
            )
            logger.info("术语映射已保存到 Redis 缓存，共 %d 条规则", len(mappings))
        except Exception as e:
            logger.exception("保存术语映射到 Redis 缓存失败")

    def clear_cache(self) -> None:
        if self.string_redis_template is None:
            logger.warning("string_redis_template is None，无法清除缓存")
            return
        try:
            deleted = self.string_redis_template.delete(CACHE_KEY)
            if deleted:
                logger.info("术语映射缓存已清除")
            else:
                logger.info("术语映射缓存不存在，无需清除")
        except Exception as e:
            logger.exception("清除术语映射缓存失败")


# ==============================
# 4. QueryTermMappingService (from query_term_mapping_service.py)
# ==============================
@dataclass
class QueryTermMappingService:
    mapping_mapper: Optional[QueryTermMappingMapper] = None
    cache_manager: Optional[QueryTermMappingCacheManager] = None

    def normalize(self, text: str) -> str:
        if text is None or text == "":
            return text

        mappings = self._load_mappings()
        if len(mappings) == 0:
            return text

        result = text
        for mapping in mappings:
            if mapping.enabled is None or mapping.enabled == 0:
                continue
            if mapping.match_type is not None and mapping.match_type != 1:
                continue
            source = mapping.source_term
            target = mapping.target_term
            if source is None or source == "" or target is None or target == "":
                continue
            result = QueryTermMappingUtil.apply_mapping(result, source, target)

        if text != result:
            logger.info("查询归一化：original='%s', normalized='%s'", text, result)
        return result

    def _load_mappings(self) -> List[QueryTermMappingDO]:
        # 1. 尝试从缓存读取
        cached = None
        if self.cache_manager is not None:
            cached = self.cache_manager.get_mappings_from_cache()
        if cached is not None:
            # 如果缓存数据是 dict 列表，尝试转换为 QueryTermMappingDO
            if cached and isinstance(cached[0], dict):
                try:
                    # 假设 QueryTermMappingDO 可以通过字典构造
                    converted = [QueryTermMappingDO(**item) for item in cached]
                    logger.info("从缓存加载术语映射，共 %d 条", len(converted))
                    return converted
                except Exception as e:
                    logger.warning("缓存数据转换为对象失败，忽略缓存: %s", e)
                    cached = None
            else:
                # 如果已经是对象列表，直接返回（假定类型正确）
                logger.info("从缓存加载术语映射，共 %d 条", len(cached))
                return cached

        # 2. 从数据库加载
        if self.mapping_mapper is None:
            logger.warning("mapping_mapper is None，无法从数据库加载术语映射，返回空映射")
            return []

        try:
            db_list = self.mapping_mapper.select_list(
                lambda q: q.eq(QueryTermMappingDO.enabled, 1)
            )

            # 排序：优先级高（数值小）优先，源词长优先
            def sort_key(m):
                priority = getattr(m, 'priority', 0)
                source_term = getattr(m, 'source_term', '')
                return (-priority, -len(source_term) if source_term else 0)

            db_list.sort(key=sort_key)

            # 保存到缓存
            if self.cache_manager is not None:
                self.cache_manager.save_mappings_to_cache(db_list)

            logger.info("术语映射规则从数据库加载完成，共 %d 条规则", len(db_list))
            return db_list
        except Exception as e:
            logger.exception("从数据库加载术语映射失败")
            return []


# ==============================
# 5. QueryRewriteService (from query_rewrite_service.py)
# ==============================
class QueryRewriteService(ABC):
    """RAG 侧查询改写抽象：仅负责 rewritten 字符串与子问题拆分。

    保持原有签名不变，RAG 链路继续调用 rewrite / rewrite_with_split。
    """

    @abstractmethod
    def rewrite(self, user_question: str) -> str:
        """单条问题改写（RAG 专用），返回改写后的问题字符串。"""
        raise NotImplementedError

    def rewrite_with_split(
        self,
        user_question: str,
        history: Optional[List[IntentChatMessage]] = None,
    ) -> RewriteResult:
        """RAG 侧改写 + 子问题拆分。

        默认行为：若未传 history，直接调用 rewrite 并包一层 RewriteResult。
        子类可覆写为 LLM 改写。
        """
        if history is None:
            rewritten_question = self.rewrite(user_question)
            return RewriteResult(
                rewritten_question=rewritten_question,
                sub_questions=[rewritten_question],
            )
        # 保持原有兼容：未覆写时调用单参数版本避免死循环
        return self.rewrite_with_split(user_question)


# ==============================
# 6. AgentQueryRewriteService （Agent 编排侧新增抽象）
# ==============================
class AgentQueryRewriteService(ABC):
    """Agent 编排侧改写服务抽象。

    与 RAG 侧 QueryRewriteService 的关键区别：
      - 返回 AgentRewriteResult（除 rewritten / split 外，还含复杂度分析、
        工具建议、explicit_plan_hint）。
      - 入参是完整的 AgentChatContext，能直接读取：
            session_id / available_tool_ids / conversation_history
        便于改写 Prompt 做工具名建议与指代消解。
    """

    @abstractmethod
    def rewrite_for_agent(
        self,
        agent_chat_context: AgentChatContext,
    ) -> AgentRewriteResult:
        """对用户问题做 Agent 编排侧改写。

        Args:
            agent_chat_context: 一次 Agent 对话请求级上下文（包含原始问题、
                会话 ID、工具快照、短期记忆历史）。

        Returns:
            AgentRewriteResult：改写问题、拆分、复杂度分析、工具建议、
                步骤提示原文。LLM 失败时返回 fallback 对象（所有字段为默认
                值或 original 问题），不抛出异常。
        """
        raise NotImplementedError