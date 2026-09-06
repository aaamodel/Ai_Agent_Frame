from abc import abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic
from abc import ABC

from app.query_intent.intent_entity import ConversationMessageDO, IntentNodeDO, QueryTermMappingDO

T = TypeVar("T")


class BaseMapper(ABC, Generic[T]):
    @abstractmethod
    def insert(self, entity: T) -> int:
        ...

    @abstractmethod
    def delete_by_id(self, id: Any) -> int:
        ...

    @abstractmethod
    def delete_by_map(self, column_map: Dict[str, Any]) -> int:
        ...

    @abstractmethod
    def delete_batch_ids(self, id_list: List[Any]) -> int:
        ...

    @abstractmethod
    def update_by_id(self, entity: T) -> int:
        ...

    @abstractmethod
    def select_by_id(self, id: Any) -> Optional[T]:
        ...

    @abstractmethod
    def select_batch_ids(self, id_list: List[Any]) -> List[T]:
        ...

    @abstractmethod
    def select_by_map(self, column_map: Dict[str, Any]) -> List[T]:
        ...

    @abstractmethod
    def select_one(self, query_wrapper: Any) -> Optional[T]:
        ...

    @abstractmethod
    def select_count(self, query_wrapper: Any) -> int:
        ...

    @abstractmethod
    def select_list(self, query_wrapper: Any) -> List[T]:
        ...

class ConversationMessageMapper(BaseMapper[ConversationMessageDO], ABC):
    pass


class IntentNodeMapper(BaseMapper[IntentNodeDO], ABC):
    pass


class QueryTermMappingMapper(BaseMapper[QueryTermMappingDO], ABC):
    pass
