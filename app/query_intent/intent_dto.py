from __future__ import annotations


from app.query_intent.intent_classify_resolver.intent_model import NodeScore
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


OrchestrationModeLiteral = Literal["react", "plan_execute"]


@dataclass
class IntentCandidate:
    sub_question_index: int
    node_score: NodeScore


class Status(Enum):
    SUCCESS = "SUCCESS"
    EMPTY = "EMPTY"
    FAILED = "FAILED"


@dataclass
class RecommendedQuestionsPayload:
    status: Status
    questions: List[str] = field(default_factory=list)

    @staticmethod
    def success(questions: Optional[List[str]] = None) -> RecommendedQuestionsPayload:
        if questions is None or len(questions) == 0:
            return RecommendedQuestionsPayload.empty()
        return RecommendedQuestionsPayload(status=Status.SUCCESS, questions=questions)

    @staticmethod
    def empty() -> RecommendedQuestionsPayload:
        return RecommendedQuestionsPayload(status=Status.EMPTY, questions=[])

    @staticmethod
    def failed() -> RecommendedQuestionsPayload:
        return RecommendedQuestionsPayload(status=Status.FAILED, questions=[])


@dataclass
class IntentGroup:
    mcp_intents: list[NodeScore] = field(default_factory=list)
    kb_intents: list[NodeScore] = field(default_factory=list)


@dataclass
class SubQuestionIntent:
    sub_question: str = ""
    node_scores: list[NodeScore] = field(default_factory=list)


# =============================================================================
# Agent 编排 Pipeline DTO（新增 5 个）
# =============================================================================


@dataclass
class TaskComplexityAnalysis:
    """Agent 改写阶段输出的「任务复杂度分析」结果。

    该 DTO 是 ModeDecider 做规则层判断的核心输入。所有字段由 LLM 改写 Prompt
    产出，并经过 rewrite service 做整数/布尔范围约束。

    Attributes:
        estimated_steps: 预估执行步骤数（1~10）。
        estimated_tool_calls: 预估工具调用次数（0~10）。
        has_multi_step_dependency: 是否存在明确的步骤先后依赖（必须先 A 后 B）。
        has_external_data_dependency: 是否需要调用外部数据源
            （API / DB / MCP / 文件 / 向量库 / 图谱）。
        need_creative_output: 是否需要创造性输出
            （写方案 / 写报告 / 写文案 / 头脑风暴）。
        reasoning_notes: LLM 给出的复杂度判断理由，用于日志追踪与调试。
    """

    estimated_steps: int = 1
    estimated_tool_calls: int = 0
    has_multi_step_dependency: bool = False
    has_external_data_dependency: bool = False
    need_creative_output: bool = False
    reasoning_notes: str = ""


# =============================================================================
# 本轮目标（agent_goal）归一化 —— 主链路 / 降级链路 / 全部注入点共用的唯一口径
# =============================================================================
AGENT_GOAL_MAX_CHARS: int = 60
"""本轮目标长度上限（一句话量级）。与提示词侧约束同源，见 design.md D6。"""

_NULL_LIKE_GOAL_VALUES = frozenset({"", "null", "none"})


def is_agent_goal_missing(raw: Any) -> bool:
    """目标是否**缺失或为空**（含 ``"null"`` / ``"none"`` 这类字面量）。

    与 :func:`normalize_agent_goal` 共用同一套空值判定：生产侧据此**标记"模型未
    提炼出验收标准"**，两处若各写一份就会出现"标记为成功、实际却回退了"的口径不一致。
    """
    return str(raw or "").strip().lower() in _NULL_LIKE_GOAL_VALUES


def normalize_agent_goal(raw: Any, fallback_question: str) -> str:
    """把改写阶段产出的目标归一为「非空 + 限长」的一句话。

    步骤（顺序不可换）：空值判定 → 回退 → 硬截断。

    ⚠️ 为什么必须**集中在这一处**、而不是各注入点各自兜底：
        三个注入点（ReAct 系统段 / 规划提示词 / 台账首行）若各自实现兜底，
        极易漂移成"一处回退到改写后的问题、另一处注入空串"，于是同一轮里
        模型看到的目标并不一致。design.md D5 要求单一真源。

    ⚠️ 截断是**硬截断**，不是"智能压缩"：压缩是提示词侧要求模型自己完成的
        写作要求（design.md D6）；代码层只保证"任何情况下都不会注入超长文本"。
        回退值本身也要截断（spec：回退为改写后问题，「必要时截断」），
        否则一条 400 字的 rewrite 会直接变成 400 字的目标。

    Args:
        raw: LLM 产出的目标（可能为 None / 空串 / "null" 字面量 / 超长）。
        fallback_question: 回退值——**改写后的问题**（调用方保证它是"当前问题"，
            而不是原始问题，否则多轮指代场景下语义会错）。

    Returns:
        目标文本；仅当 raw 与 fallback_question 都为空时返回空串（调用方
        此时按"无目标"处理，MUST NOT 因此中断链路）。
    """
    text: str = "" if is_agent_goal_missing(raw) else str(raw or "").strip()
    if not text:
        text = str(fallback_question or "").strip()
    if len(text) > AGENT_GOAL_MAX_CHARS:
        text = text[:AGENT_GOAL_MAX_CHARS]
    return text


@dataclass
class AgentRewriteResult:
    """Agent 改写阶段的完整输出。

    Attributes:
        rewritten_question: 改写后的问题（指代消解、省略补全、最小必要改写）。
            若为空则上游回退为 original_user_question。
        should_split: 是否需要拆分为多个子问题。
        sub_questions: 拆分后的子问题列表；若 should_split 为 False 则为空。
        complexity_analysis: 任务复杂度分析结果（见 TaskComplexityAnalysis）。
        suggested_tools: 推荐使用的工具名列表；每一项必须是已在
            ToolRegistry 实际注册的 name，Pipeline 会对其做白名单再过滤。
        explicit_plan_hint: 用户问题中明确的步骤提示原文（精简后）。
        例如"先查销售数据再查库存最后汇总"
        → "先查销售数据,再查库存,最后汇总"。没有则为 None。
    precomputed_intent_scores: （调整一 · 可选）组合调用
        （AgentCombinedRewriteIntentService）已经产出的逐问题意图打分：
        问题文本 → NodeScore 列表。非 None 时 Pipeline Stage2 走
        aggregate_for_agent_precomputed 零 LLM 聚合；None 时回退原
        resolve_for_agent 的独立 LLM 意图识别链路。
        agent_goal: 本轮要交付的最终产物/结论形态（一句话，≤60 字）。
            解析层经 :func:`normalize_agent_goal` 归一（strip → 截断 →
            空值回退为 rewritten_question），故**由解析层产出时恒非空**；
            手工构造（如规则兜底）也走同一函数，读取侧再有兜底不阻断链路。
    """

    rewritten_question: str = ""
    agent_goal: str = ""
    should_split: bool = False
    sub_questions: List[str] = field(default_factory=list)
    complexity_analysis: TaskComplexityAnalysis = field(
        default_factory=TaskComplexityAnalysis
    )
    suggested_tools: List[str] = field(default_factory=list)
    explicit_plan_hint: Optional[str] = None
    precomputed_intent_scores: Optional[Dict[str, List[NodeScore]]] = None
    # 【新增 · 技能感知】改写阶段 LLM 依据注入的技能清单（名称+描述）挑选出的
    # 与当前问题最相关的技能名。编排层据此号令可用工具（与 Pipeline 注入并集）。
    suggested_skills: List[str] = field(default_factory=list)
    #: 历史污染兜底后，存活子问题在模型原始 sub_questions 中的 1 基序号
    #: （组合链路的 intent_classifications.question_index 需按此对位）；
    #: 未发生剔除时为 None。
    sub_question_source_indexes: Optional[List[int]] = None


@dataclass
class AgentIntents:
    """意图聚合阶段输出。

    与 RAG 专用的 SubQuestionIntent / IntentGroup 不同，这里面向 Agent 编排：
    统一按 IntentKind 汇总命中的 NodeScore，并给出"主意图文本"用于后续
    IntentContext.intent 字段填写。

    Attributes:
        primary_intent_text: 主意图（通常是置信度最高的 1 个 KB/MCP/SYSTEM
            节点的 display_name；若无命中则为 "general"）。
        aggregated_confidence: 聚合后的综合置信度（0~1）。
        kb_hit_count: 命中 KB 类意图节点的数量。
        mcp_hit_count: 命中 MCP 类意图节点的数量。
        sys_hit_count: 命中 SYSTEM 类意图节点的数量。
        kb_node_scores: 命中 KB 节点的 NodeScore 明细。
        mcp_node_scores: 命中 MCP 节点的 NodeScore 明细。
        sys_node_scores: 命中 SYSTEM 节点的 NodeScore 明细。
        per_sub_question_intents: （若拆分）按子问题维度的意图明细列表，
            顺序与 sub_questions 一一对应，便于后续精细化路由。
        raw_slots: 分类器/解析器提取到的槽位字典（实体、参数、时间范围等）。
            直接作为 IntentContext.slots 的起点。
    """

    primary_intent_text: str = "general"
    aggregated_confidence: float = 0.0
    kb_hit_count: int = 0
    mcp_hit_count: int = 0
    sys_hit_count: int = 0
    kb_node_scores: List[NodeScore] = field(default_factory=list)
    mcp_node_scores: List[NodeScore] = field(default_factory=list)
    sys_node_scores: List[NodeScore] = field(default_factory=list)
    per_sub_question_intents: List[SubQuestionIntent] = field(default_factory=list)
    raw_slots: Dict[str, Any] = field(default_factory=dict)

    related_skill_names: List[str] = field(default_factory=list)


@dataclass
class ModeDecision:
    """模式决策阶段最终结果。

    决策流程（调整三后完全规则化，无 LLM 调用）：
      1) 显式步骤提示（explicit_plan_hint 非空）→ 最高优先级 plan_execute
      2) 规则阈值层（step / tool threshold）→ 命中即产出初判
      3) 静态意图偏好层（intent_prefer_mode）→ 最高分节点 prefer_mode
      4) 兜底默认 react

    Attributes:
        mode: 最终决策模式，严格取值 "react" | "plan_execute"。
        confidence: 最终置信度（0~1）。
        reason: 一句话决策理由，用于日志与前端 debug 面板展示。
        decision_source: 决策来源，用于追踪走的是哪一条分支。
            enum-literal: "explicit_hint" | "rule_threshold" | "intent_prefer_mode"
            | "llm"（历史保留）| "fallback_default"。
        initial_plan_hint: 当 mode=plan_execute 时，LLM 产出的 3~5 条
            步骤建议字符串；其他情况下为 None。按疑问-F 推荐 F-1，该字段会
            被写进 IntentContext.slots["initial_plan_hint"] 供 prompt 侧
            注入到 PlannerAgent 的 user_message 末尾。
        first_tool_hint: 当 mode=react 时，LLM 建议的第一个工具名；其他情
            况为 None。用于 ReActAgent 首次 Thought 前做冷启动 hint（可
            选使用；ReAct 内部零侵入，仅在 slots 中保留）。
    """

    mode: OrchestrationModeLiteral = "react"
    confidence: float = 0.0
    reason: str = ""
    decision_source: Literal[
        "explicit_hint",
        "rule_threshold",
        "intent_prefer_mode",
        "llm",
        "fallback_default",
    ] = "fallback_default"
    initial_plan_hint: Optional[str] = None
    first_tool_hint: Optional[str] = None

    # 【新增 · 技能感知】编排层本次运行可调用的技能名白名单（由改写阶段 LLM
    # 选出的 suggested_skills 透传）。供 orchestrator 用技能 allowed-tools 号令工具。
    eligible_skill_names: List[str] = field(default_factory=list)


@dataclass
class AgentQueryIntentPipelineOutput:
    """AgentQueryIntentPipeline.run() 的完整返回。

    该 DTO 是 改写 → 意图聚合 → 模式决策 三段流水线的汇总产物，
    下游 chat.py 会把它交给 build_orchestrator_input() 组装为
    (mode: str, intent: IntentContext) 二元组，再喂给 AgentOrchestrator.run。

    Attributes:
        rewrite_result: 改写阶段输出。
        intents_result: 意图聚合阶段输出。
        mode_decision: 模式决策阶段输出。
        final_user_input: 最终送入 AgentOrchestrator.run(user_input=...)
            的字符串；若 rewritten_question 为空则回退为 original。
        allowed_tools_final: 经过 "改写 suggested_tools ∪ MCP 意图命中工具
            ∪ infrastructure 兜底" 三层合并后的白名单。
        merged_slots: 经过各层合并后的最终 slots 字典
            （raw_slots + initial_plan_hint + first_tool_hint 等）。
        original_user_question: 原始问题原文，用于日志与断言。
        session_id: 对应 AgentChatContext.session_id。
    """

    rewrite_result: AgentRewriteResult = field(default_factory=AgentRewriteResult)
    intents_result: AgentIntents = field(default_factory=AgentIntents)
    mode_decision: ModeDecision = field(default_factory=ModeDecision)
    final_user_input: str = ""
    allowed_tools_final: List[str] = field(default_factory=list)
    merged_slots: Dict[str, Any] = field(default_factory=dict)
    original_user_question: str = ""
    session_id: str = ""
    # 【新增 · 技能感知】改写阶段 LLM 最终选定的技能名白名单（去重、去空），
    # 供编排层号令可用工具；空则编排层回退自身技能命中判断。
    selected_skills_final: List[str] = field(default_factory=list)
