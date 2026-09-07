# intent_core.py
# ============================================================
# 核心数据模型、枚举、常量
# ============================================================

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# -------------------- 类型别名（结构化输出参数用，避免循环导入 typing_extensions）--------------------
OpenAIResponseFormat = Dict[str, Any]
OpenAITool = Dict[str, Any]
OpenAIToolChoice = Any  # Literal["auto", "none", "required"] 或 {"type": "function", "function": {...}}


# -------------------- 异常 --------------------
class ClientException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# -------------------- 聊天消息模型 --------------------
class MessageStatus(Enum):
    NORMAL = "NORMAL"


@dataclass
class IntentChatMessage:
    role: str
    content: str

    @staticmethod
    def user(content: str) -> "IntentChatMessage":
        return IntentChatMessage(role="user", content=content)

    @staticmethod
    def assistant(content: str) -> "IntentChatMessage":
        return IntentChatMessage(role="assistant", content=content)

    @staticmethod
    def system(content: str) -> "IntentChatMessage":
        return IntentChatMessage(role="system", content=content)


@dataclass
class IntentChatRequest:
    """query_intent 侧统一的 LLM 聊天请求 DTO。


      - response_format：OpenAI 协议 response_format 字典（json_schema / json_object），
        详见 llm_schemas.pydantic_to_openai_response_format。
      - tools：function calling 工具列表；见 llm_schemas.build_function_tool_def。
      - tool_choice：function calling 选择策略；见 llm_schemas.build_tool_choice_required。
    """
    messages: List[IntentChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    thinking: Optional[bool] = None
    # 本轮新增：结构化输出 / function calling 参数
    response_format: Optional[OpenAIResponseFormat] = None
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[OpenAIToolChoice] = None

    @staticmethod
    def builder() -> "ChatRequestBuilder":
        return ChatRequestBuilder()


class ChatRequestBuilder:
    def __init__(self):
        self._messages = None
        self._temperature = None
        self._top_p = None
        self._max_tokens = None
        self._thinking = None
        self._response_format: Optional[OpenAIResponseFormat] = None
        self._tools: Optional[List[OpenAITool]] = None
        self._tool_choice: Optional[OpenAIToolChoice] = None

    def messages(self, messages: List[IntentChatMessage]) -> "ChatRequestBuilder":
        self._messages = messages
        return self

    def temperature(self, temperature: float) -> "ChatRequestBuilder":
        self._temperature = temperature
        return self

    def top_p(self, top_p: float) -> "ChatRequestBuilder":
        self._top_p = top_p
        return self

    def max_tokens(self, max_tokens: int) -> "ChatRequestBuilder":
        self._max_tokens = max_tokens
        return self

    def thinking(self, thinking: bool) -> "ChatRequestBuilder":
        self._thinking = thinking
        return self

    def response_format(self, response_format: OpenAIResponseFormat) -> "ChatRequestBuilder":
        """填入 OpenAI response_format 字典（通常由 llm_schemas.pydantic_to_openai_response_format 构造）。"""
        self._response_format = response_format
        return self

    def tools(self, tools: List[OpenAITool]) -> "ChatRequestBuilder":
        """填入 function calling 定义的 tools 数组（见 llm_schemas.build_function_tool_def）。"""
        self._tools = tools
        return self

    def tool_choice(self, tool_choice: OpenAIToolChoice) -> "ChatRequestBuilder":
        """填入 tool_choice："auto" / "required" / {"type":"function", ...}。"""
        self._tool_choice = tool_choice
        return self

    def build(self) -> IntentChatRequest:
        return IntentChatRequest(
            messages=self._messages,
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
            thinking=self._thinking,
            response_format=self._response_format,
            tools=self._tools,
            tool_choice=self._tool_choice,
        )




@dataclass
class IntentResult:
    code: Optional[int] = None
    message: Optional[str] = None
    data: Optional[Any] = None


# -------------------- 枚举 --------------------
class IntentKind(Enum):
    KB = 0
    SYSTEM = 1
    MCP = 2

    @property
    def code(self) -> int:
        return self.value

    @staticmethod
    def from_code(code: int | None) -> "IntentKind | None":
        if code is None:
            return None
        for e in IntentKind:
            if e.code == code:
                return e
        return None

    def __str__(self) -> str:
        return self.name


class IntentLevel(Enum):
    DOMAIN = 0
    CATEGORY = 1
    TOPIC = 2

    @property
    def code(self) -> int:
        return self.value

    @staticmethod
    def from_code(code: int | None) -> "IntentLevel | None":
        if code is None:
            return None
        for e in IntentLevel:
            if e.code == code:
                return e
        return None

    def __str__(self) -> str:
        return self.name


class IntentChoiceTier(Enum):
    FAST = "FAST"
    DEFAULT = "DEFAULT"


# -------------------- 分页模型 --------------------
@dataclass
class IntentPage:
    current: int = 1
    size: int = 10
    total: int = 0
    records: List[Any] = field(default_factory=list)


# -------------------- Agent 编排侧：路由级上下文（新增）--------------------
@dataclass
class AgentChatContext:
    """Agent 编排 Pipeline 的一次请求级上下文。

    承载 /chat/with_agent 路由在调用 Pipeline.run(...) 之前所准备好的、
    与"单条用户提问 + 当前会话 + 工具箱快照"相关的全部不可变输入，
    供改写、意图识别、模式决策三个阶段共享读取，避免各层重复入参。

    Attributes:
        original_user_question: 路由收到的用户原始问题原文。
        session_id: 会话唯一 ID（与 AgentOrchestrator.run 的 session_id 一致）。
        available_tool_ids: 当前请求下 ToolRegistry.list_tool_names() 的快照，
            用于改写阶段给出可靠的 suggested_tools，同时做白名单过滤。
        conversation_history: 短期记忆历史（最近 20 条以内），用于指代消解与
            省略补全。注意：改写阶段只读其中的 USER 消息用于续问还原。
        available_skills: 当前请求高级技能快照（{name, description, ...} 字典列表），
            用于改写/意图阶段注入技能清单让 LLM 挑选编排层可用技能。
    """

    original_user_question: str
    session_id: str
    available_tool_ids: List[str] = field(default_factory=list)
    conversation_history: List[IntentChatMessage] = field(default_factory=list)
    available_skills: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """字段有效性兜底：空值规范化，防止下游 NPE。"""
        if not self.session_id:
            # 路由层一般会在调用前保证 session_id；此处仅防御性兜底。
            object.__setattr__(
                self,
                "session_id",
                f"fallback_session_{abs(hash(self.original_user_question or ''))}",
            )
        if self.available_tool_ids is None:
            self.available_tool_ids = []
        if self.conversation_history is None:
            self.conversation_history = []
        if self.available_skills is None:
            self.available_skills = []


