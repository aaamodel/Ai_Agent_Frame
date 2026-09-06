from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class IntentNodeDO:
    id: Optional[str] = field(default=None)
    kb_id: Optional[str] = field(default=None)
    intent_code: Optional[str] = field(default=None)
    name: Optional[str] = field(default=None)
    level: Optional[int] = field(default=None)
    parent_code: Optional[str] = field(default=None)
    description: Optional[str] = field(default=None)
    examples: Optional[str] = field(default=None)
    collection_name: Optional[str] = field(default=None)
    collection_names: Optional[List[str]] = field(default=None)
    mcp_tool_id: Optional[str] = field(default=None)
    top_k: Optional[int] = field(default=None)
    kind: Optional[int] = field(default=None)
    sort_order: Optional[int] = field(default=None)
    prompt_snippet: Optional[str] = field(default=None)
    prompt_template: Optional[str] = field(default=None)
    param_prompt_template: Optional[str] = field(default=None)
    enabled: Optional[int] = field(default=None)
    create_by: Optional[str] = field(default=None)
    update_by: Optional[str] = field(default=None)
    create_time: Optional[datetime] = field(default=None)
    update_time: Optional[datetime] = field(default=None)
    deleted: Optional[int] = field(default=None)



@dataclass
class SourceRef:
    index: Optional[int] = field(default=None)
    doc_id: Optional[str] = field(default=None)
    doc_name: Optional[str] = field(default=None)
    source_type: Optional[str] = field(default=None)
    file_type: Optional[str] = field(default=None)
    url: Optional[str] = field(default=None)
    excerpt: Optional[str] = field(default=None)

@dataclass
class GroundingChunk:
    doc_name: Optional[str] = field(default=None)
    text: Optional[str] = field(default=None)

@dataclass
class ConversationMessageDO:
    id: Optional[str] = field(default=None)
    conversation_id: Optional[str] = field(default=None)
    user_id: Optional[str] = field(default=None)
    role: Optional[str] = field(default=None)
    content: Optional[str] = field(default=None)
    thinking_content: Optional[str] = field(default=None)
    thinking_duration: Optional[int] = field(default=None)
    sources: Optional[List[SourceRef]] = field(default=None)
    retrieved_chunks: Optional[List[GroundingChunk]] = field(default=None)
    recommended_questions: Optional[List[str]] = field(default=None)
    reply_to_message_id: Optional[str] = field(default=None)
    message_status: Optional[str] = field(default=None)
    create_time: Optional[datetime] = field(default=None)
    update_time: Optional[datetime] = field(default=None)
    deleted: Optional[int] = field(default=None)



@dataclass
class QueryTermMappingDO:
    id: Optional[str] = field(default=None)
    domain: Optional[str] = field(default=None)
    source_term: Optional[str] = field(default=None)
    target_term: Optional[str] = field(default=None)
    match_type: Optional[int] = field(default=None)
    priority: Optional[int] = field(default=None)
    enabled: Optional[int] = field(default=None)
    remark: Optional[str] = field(default=None)
    create_by: Optional[str] = field(default=None)
    update_by: Optional[str] = field(default=None)
    create_time: Optional[datetime] = field(default=None)
    update_time: Optional[datetime] = field(default=None)
