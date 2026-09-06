from __future__ import annotations

# ============================================================
# 既有 RAG 侧常量（保持不变）
# ============================================================
INTENT_MIN_SCORE: float = 0.35
MAX_INTENT_COUNT: int = 3
MULTI_CHANNEL_KEY: str = "multi_channel"
INTENT_CLASSIFIER_PROMPT_PATH: str = "app/query_intent/prompts/intent-classifier.st"
GUIDANCE_PROMPT_PATH: str = "app/query_intent/prompts/guidance-prompt.st"
GUIDANCE_AMBIGUITY_CHECK_PROMPT_PATH: str = "app/query_intent/prompts/guidance-ambiguity-check.st"
QUERY_REWRITE_AND_SPLIT_PROMPT_PATH: str = "app/query_intent/prompts/user-question-rewrite.st"


# ============================================================
# Agent 编排侧：问题改写 + 意图识别 + 模式决策（新增）
# ============================================================

# Agent 改写 + 复杂度分析 Prompt 模板路径（对应 AgentMultiQuestionRewriteService）
AGENT_QUESTION_REWRITE_PROMPT_PATH: str = (
    "app/query_intent/prompts/agent-question-rewrite.st"
)

# Agent「改写 + 意图识别」合并单次调用 Prompt 模板路径（调整一：
# 对应 AgentCombinedRewriteIntentService，一次 LLM 调用同时产出改写与意图打分）
AGENT_REWRITE_INTENT_COMBINED_PROMPT_PATH: str = (
    "app/query_intent/prompts/agent-rewrite-intent-combined.st"
)

# 意图树向量检索默认召回候选数（调整二：Top-K 替代全量意图清单塞 Prompt）
INTENT_VECTOR_TOP_K: int = 8

# Agent 编排模式决策 Prompt 模板路径（对应 ModeDecider._llm_decide）
ORCHESTRATION_MODE_DECIDER_PROMPT_PATH: str = (
    "appquery_intent/prompts/orchestration-mode-decider.st"
)

# ---- 模式决策阈值（规则判断层：低于阈值不切 plan_execute）----
# 预估执行步骤数 >= 该阈值 → 规则层倾向 plan_execute
MODE_DECISION_STEP_THRESHOLD: int = 3

# 预估工具调用次数 >= 该阈值 → 规则层倾向 plan_execute
MODE_DECISION_TOOL_THRESHOLD: int = 2

# 规则层 + LLM 层均无法给出可靠判断时的保守兜底模式
MODE_DECISION_DEFAULT_MODE: str = "react"

# LLM 决策模式的最低置信度；低于该值回退到 MODE_DECISION_DEFAULT_MODE
MODE_DECISION_LLM_MIN_CONFIDENCE: float = 0.55

# ---- 真实 ToolRegistry 工具名（与 app/core/tools/builtin/init_tools.py
#      中所有"未被注释"的 registry.register(...) 注册语句 1:1 对齐）----
# 用途：
#   1) Pipeline.fill_allowed_tools() 做白名单交叉过滤时的对照参考；
#   2) P2 prompt 示例 suggested_tools 字面量必须从此集合取，避免 LLM 幻觉。
# 数据来源（已代码核验）：
#   local_excel_tool      ← LocalExcelTool.name
#   feishu_bitable_tool   ← FeishuBitableTool.name
#   web_search            ← DoubaoWebSearchTool.name（豆包搜索 API 直连，联网搜索第一梯队）
#   tavily_web_search     ← WebSearchTool.name（Tavily，联网搜索第二梯队备选，
#                            仅当 web_search 不可用时使用，使用时机由提示词约束）
#   rag_knowledge_search  ← RagSearchTool.name
#   knowledge_graph_search← KnowledgeGraphSearchTool.name
#   write_todos           ← AgentTodoPlannerTool.name
#   file_read_tool        ← FileReadTool.name
#   file_list_tool        ← FileListTool.name
#   file_grep_tool        ← FileGrepTool.name
REGISTERED_ENABLED_TOOL_NAMES: list[str] = [
    "local_excel_tool",
    "feishu_bitable_tool",
    "web_search",
    "rag_knowledge_search",
    "knowledge_graph_search",
    "write_todos",
    "file_read_tool",
    "file_list_tool",
    "file_grep_tool",
]

# ---- 联网搜索双梯队工具名（问题改写 / 模式决策 / Agent 执行提示词中引用）----
# 第一梯队：豆包搜索（doubao_search.py），外部实时公开信息首选通道
WEB_SEARCH_PRIMARY_TOOL_NAME: str = "web_search"
# 第二梯队：Tavily 备选（search.py），仅当第一梯队不可用/熔断/失败时使用
WEB_SEARCH_FALLBACK_TOOL_NAME: str = "tavily_web_search"

# ---- 原子基础设施工具兜底集合（与 orchestrator.py:L125 完全一致）----
# 即使 allowed_tools 被 Pipeline 大幅缩窄，也必须保证这 3 个工具存在
# 以便 Skills System 读取 SKILL.md 文件（progressive disclosure）。
PIPELINE_INFRASTRUCTURE_TOOL_SET: list[str] = [
    "file_read_tool",
    "file_list_tool",
    "file_grep_tool",
]
