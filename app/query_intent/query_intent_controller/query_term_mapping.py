# ============================================================================
# merged_query_term_mapping_admin.py
# 合并了以下3个文件：
# - query_term_mapping_admin_service.py
# - query_term_mapping_admin_service_impl.py
# - query_term_mapping.py (原合并的请求/响应/控制器)
# ============================================================================

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any
from abc import ABC, abstractmethod

from app.query_intent.intent_data_base import ClientException, IntentPage, IntentResult
from app.query_intent.intent_entity import QueryTermMappingDO
from app.query_intent.intent_mapper import QueryTermMappingMapper
from app.query_intent.intent_utils import IntentSuccessResults

from app.query_intent.rewrite.query_rewrite import QueryTermMappingCacheManager


# -------------------- 数据类 (Request / VO) --------------------
@dataclass
class QueryTermMappingPageRequest(IntentPage):
    keyword: Optional[str] = None


@dataclass
class QueryTermMappingCreateRequest:
    source_term: Optional[str] = None
    target_term: Optional[str] = None
    match_type: Optional[int] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None
    remark: Optional[str] = None


@dataclass
class QueryTermMappingUpdateRequest:
    source_term: Optional[str] = None
    target_term: Optional[str] = None
    match_type: Optional[int] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None
    remark: Optional[str] = None


@dataclass
class QueryTermMappingVO:
    id: Optional[str] = None
    source_term: Optional[str] = None
    target_term: Optional[str] = None
    match_type: Optional[int] = None
    priority: Optional[int] = None
    enabled: Optional[bool] = None
    remark: Optional[str] = None
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None


# -------------------- 服务接口 --------------------
class QueryTermMappingAdminService(ABC):

    @abstractmethod
    def create(self, request_param: QueryTermMappingCreateRequest) -> str:
        pass

    @abstractmethod
    def update(self, id: str, request_param: QueryTermMappingUpdateRequest) -> None:
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        pass

    @abstractmethod
    def query_by_id(self, id: str) -> QueryTermMappingVO:
        pass

    @abstractmethod
    def page_query(self, request_param: QueryTermMappingPageRequest) -> IntentPage:
        pass


# -------------------- 服务实现 --------------------
class BizChangeLogContext:
    """
    占位类，实际由外部依赖提供。
    保留原实现中的引用，不修改。
    """
    def put(self, biz_no: str, before: Optional[Any], after: Optional[Any]) -> None:
        pass


class QueryTermMappingAdminServiceImpl(QueryTermMappingAdminService):

    def __init__(
        self,
        query_term_mapping_mapper: QueryTermMappingMapper,
        query_term_mapping_cache_manager: QueryTermMappingCacheManager,
        biz_change_log_context: BizChangeLogContext,
    ):
        self.query_term_mapping_mapper = query_term_mapping_mapper
        self.query_term_mapping_cache_manager = query_term_mapping_cache_manager
        self.biz_change_log_context = biz_change_log_context

    def create(self, request_param: QueryTermMappingCreateRequest) -> str:
        assert request_param is not None, ClientException("请求不能为空")
        source_term = (request_param.source_term or "").strip() or None
        target_term = (request_param.target_term or "").strip() or None
        assert source_term, ClientException("原始词不能为空")
        assert target_term, ClientException("目标词不能为空")

        record = QueryTermMappingDO()
        record.source_term = source_term
        record.target_term = target_term
        record.match_type = request_param.match_type if request_param.match_type is not None else 1
        record.priority = request_param.priority if request_param.priority is not None else 0
        if request_param.enabled is not None:
            record.enabled = 1 if request_param.enabled else 0
        else:
            record.enabled = 1
        record.remark = (request_param.remark or "").strip() or None

        self.query_term_mapping_mapper.insert(record)
        self.query_term_mapping_cache_manager.clear_cache()
        self.biz_change_log_context.put(str(record.id), None, record)
        return str(record.id)

    def update(self, id: str, request_param: QueryTermMappingUpdateRequest) -> None:
        assert request_param is not None, ClientException("请求不能为空")
        record = self._load_by_id(id)
        before = QueryTermMappingDO(**record.__dict__) if record else None

        if request_param.source_term is not None:
            source_term = (request_param.source_term or "").strip() or None
            assert source_term, ClientException("原始词不能为空")
            record.source_term = source_term
        if request_param.target_term is not None:
            target_term = (request_param.target_term or "").strip() or None
            assert target_term, ClientException("目标词不能为空")
            record.target_term = target_term
        if request_param.match_type is not None:
            record.match_type = request_param.match_type
        if request_param.priority is not None:
            record.priority = request_param.priority
        if request_param.enabled is not None:
            record.enabled = 1 if request_param.enabled else 0
        if request_param.remark is not None:
            record.remark = (request_param.remark or "").strip() or None

        self.query_term_mapping_mapper.update_by_id(record)
        self.query_term_mapping_cache_manager.clear_cache()
        self.biz_change_log_context.put(
            id, before, self.query_term_mapping_mapper.select_by_id(id)
        )

    def delete(self, id: str) -> None:
        record = self._load_by_id(id)
        before = QueryTermMappingDO(**record.__dict__) if record else None
        self.query_term_mapping_mapper.delete_by_id(record.id)
        self.query_term_mapping_cache_manager.clear_cache()
        self.biz_change_log_context.put(id, before, None)

    def query_by_id(self, id: str) -> QueryTermMappingVO:
        record = self._load_by_id(id)
        return self._to_vo(record)

    def page_query(self, request_param: QueryTermMappingPageRequest) -> IntentPage:
        keyword = (request_param.keyword or "").strip() or None
        page = IntentPage(current=request_param.current, size=request_param.size)
        result = self.query_term_mapping_mapper.select_page(
            page,
            {
                "keyword": keyword,
            },
        )
        records = [self._to_vo(r) for r in (result.records or [])]
        return IntentPage(
            current=result.current,
            size=result.size,
            total=result.total,
            records=records,
        )

    def _load_by_id(self, id: str) -> QueryTermMappingDO:
        record = self.query_term_mapping_mapper.select_by_id(id)
        assert record is not None, ClientException("映射规则不存在")
        return record

    def _to_vo(self, record: QueryTermMappingDO) -> QueryTermMappingVO:
        return QueryTermMappingVO(
            id=str(record.id),
            source_term=record.source_term,
            target_term=record.target_term,
            match_type=record.match_type,
            priority=record.priority,
            enabled=record.enabled is not None and record.enabled == 1,
            remark=record.remark,
            create_time=record.create_time,
            update_time=record.update_time,
        )


# -------------------- 控制器 --------------------
class QueryTermMappingController:

    def __init__(self, query_term_mapping_admin_service: QueryTermMappingAdminService):
        self.query_term_mapping_admin_service = query_term_mapping_admin_service

    def page_query(self, request_param: QueryTermMappingPageRequest) -> IntentResult:
        return IntentSuccessResults.success(self.query_term_mapping_admin_service.page_query(request_param))

    def query_by_id(self, id: str) -> IntentResult:
        return IntentSuccessResults.success(self.query_term_mapping_admin_service.query_by_id(id))

    def create(self, request_param: QueryTermMappingCreateRequest) -> IntentResult:
        return IntentSuccessResults.success(self.query_term_mapping_admin_service.create(request_param))

    def update(self, id: str, request_param: QueryTermMappingUpdateRequest) -> IntentResult:
        self.query_term_mapping_admin_service.update(id, request_param)
        return IntentSuccessResults.success()

    def delete(self, id: str) -> IntentResult:
        self.query_term_mapping_admin_service.delete(id)
        return IntentSuccessResults.success()