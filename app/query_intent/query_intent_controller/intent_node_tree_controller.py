# ============================================================================
# intent_tree_management.py
# 合并了以下3个文件：
# - intent_node_request.py
# - intent_node_tree_vo.py
# - intent_tree_controller.py
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Optional

from app.query_intent.intent_data_base import IntentResult
from app.query_intent.intent_utils import IntentSuccessResults


# -------------------- Request DTOs --------------------
@dataclass
class IntentNodeBatchRequest:
    ids: Optional[List[str]] = None


@dataclass
class IntentNodeCreateRequest:
    kb_id: Optional[str] = None
    collection_names: Optional[List[str]] = None
    intent_code: Optional[str] = None
    name: Optional[str] = None
    level: Optional[int] = None
    parent_code: Optional[str] = None
    description: Optional[str] = None
    examples: Optional[List[str]] = None
    mcp_tool_id: Optional[str] = None
    top_k: Optional[int] = None
    kind: Optional[int] = None
    sort_order: Optional[int] = None
    enabled: Optional[int] = None
    prompt_snippet: Optional[str] = None
    prompt_template: Optional[str] = None
    param_prompt_template: Optional[str] = None


@dataclass
class IntentNodeUpdateRequest:
    name: Optional[str] = None
    level: Optional[int] = None
    parent_code: Optional[str] = None
    description: Optional[str] = None
    examples: Optional[List[str]] = None
    mcp_tool_id: Optional[str] = None
    collection_name: Optional[str] = None
    collection_names: Optional[List[str]] = None
    top_k: Optional[int] = None
    kind: Optional[int] = None
    sort_order: Optional[int] = None
    enabled: Optional[int] = None
    prompt_snippet: Optional[str] = None
    prompt_template: Optional[str] = None
    param_prompt_template: Optional[str] = None


# -------------------- View Object --------------------
@dataclass
class IntentNodeTreeVO:
    id: Optional[str] = None
    intent_code: Optional[str] = None
    name: Optional[str] = None
    level: Optional[int] = None
    parent_code: Optional[str] = None
    description: Optional[str] = None
    examples: Optional[str] = None
    collection_name: Optional[str] = None
    collection_names: Optional[List[str]] = None
    top_k: Optional[int] = None
    kind: Optional[int] = None
    sort_order: Optional[int] = None
    enabled: Optional[int] = None
    mcp_tool_id: Optional[str] = None
    prompt_snippet: Optional[str] = None
    prompt_template: Optional[str] = None
    param_prompt_template: Optional[str] = None
    children: Optional[List["IntentNodeTreeVO"]] = field(default_factory=list)


# -------------------- Service and Controller --------------------
class IntentTreeService:
    def get_full_tree(self) -> List[IntentNodeTreeVO]:
        return []

    def create_node(self, request_param: IntentNodeCreateRequest) -> str:
        return ""

    def update_node(self, id: str, request_param: IntentNodeUpdateRequest) -> None:
        pass

    def delete_node(self, id: str) -> None:
        pass

    def batch_enable_nodes(self, ids: List[str]) -> None:
        pass

    def batch_disable_nodes(self, ids: List[str]) -> None:
        pass

    def batch_delete_nodes(self, ids: List[str]) -> None:
        pass


class IntentTreeController:

    def __init__(self, intent_tree_service: IntentTreeService):
        self.intent_tree_service = intent_tree_service

    def tree(self) -> IntentResult:
        return IntentSuccessResults.success(self.intent_tree_service.get_full_tree())

    def create_node(self, request_param: IntentNodeCreateRequest) -> IntentResult:
        return IntentSuccessResults.success(self.intent_tree_service.create_node(request_param))

    def update_node(self, id: str, request_param: IntentNodeUpdateRequest) -> None:
        self.intent_tree_service.update_node(id, request_param)

    def delete_node(self, id: str) -> None:
        self.intent_tree_service.delete_node(id)

    def batch_enable(self, request_param: IntentNodeBatchRequest) -> None:
        self.intent_tree_service.batch_enable_nodes(request_param.ids)

    def batch_disable(self, request_param: IntentNodeBatchRequest) -> None:
        self.intent_tree_service.batch_disable_nodes(request_param.ids)

    def batch_delete(self, request_param: IntentNodeBatchRequest) -> None:
        self.intent_tree_service.batch_delete_nodes(request_param.ids)