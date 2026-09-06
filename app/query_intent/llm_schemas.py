# -*- coding: utf-8 -*-
"""llm_schemas.py —— 大模型结构化输出的 Pydantic v2 Schema 定义与 OpenAI 协议转换工具。

使用说明
--------
本模块服务于「一次性调用（SINGLE_SHOT）改「关闭思考 + 严格 JSON schema / function
calling」」的改造目标（详见 json_calltype_analysis.txt）。使用模式：

1) 纯 JSON 输出（S1 / S2 / S3 / S4 / S5 / S7 / MID-1 / S8 fallback）
    from app.query_intent import AgentRewriteSchema, pydantic_to_openai_response_format
    schema_dict = pydantic_to_openai_response_format(AgentRewriteSchema)
    # 把 schema_dict 填入 IntentChatRequest.response_format 或 acomplete(response_format=...)

2) Function calling 协议（S6 MCP 参数 / S8 ToolRouter 动态枚举）
    from app.query_intent import build_function_tool_def, build_tool_choice_required
    tools_list = [build_function_tool_def("select_relevant_tools", "工具路由描述", params_cls)]
    choice = build_tool_choice_required("select_relevant_tools")
    # 把 tools_list 填入 IntentChatRequest.tools；choice 填入 IntentChatRequest.tool_choice

本模块按 Pydantic v2（model_json_schema() 返回 $defs/definitions 结构）编写，
适配 OpenAI `/chat/completions` 兼容的「response_format.json_schema / tools / tool_choice」
三套参数。
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Literal, Optional, Type, get_args

from pydantic import BaseModel, Field, ValidationError


# ==============================================================================
# 工具函数：Pydantic → OpenAI 协议字典
# ==============================================================================
def _strip_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """就地去除 Pydantic model_json_schema 里的 $defs / $ref，输出内联结构。

    OpenAI response_format / tools[].parameters 不支持 JSON Schema 的 $ref 引用，
    因此这里做一次轻量内联。
    """
    defs = schema.pop("$defs", None) or {}

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                ref_key = node["$ref"].split("/")[-1]
                inlined = copy.deepcopy(defs.get(ref_key, node))
                # 展开后可能还有嵌套 $ref，递归
                return _walk(inlined)
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(schema)


def pydantic_to_openai_response_format(
    pydantic_cls: Type[BaseModel],
    name_override: Optional[str] = None,
    description_override: Optional[str] = None,
) -> Dict[str, Any]:
    """把 Pydantic v2 类转换为 OpenAI response_format={"type": "json_schema", ...} 的字典。

    Args:
        pydantic_cls: 要结构化输出的 Pydantic 类。
        name_override: 覆盖 OpenAI json_schema.name（默认用类名小写蛇形）。
        description_override: 覆盖 json_schema.description（默认取类 docstring）。

    Returns:
        可直接填入 `response_format` 参数的字典。
    """
    raw_schema = pydantic_cls.model_json_schema(mode="serialization")
    inline_schema = _strip_refs(raw_schema)

    # 去掉 "title" 顶层冗余字段（避免和 Pydantic 默认生成的中文 Title 混）
    inline_schema.pop("title", None)

    schema_name = name_override or _to_snake_case(pydantic_cls.__name__)
    schema_desc: Optional[str] = description_override or (
        (pydantic_cls.__doc__ or "").strip() or None
    )

    payload: Dict[str, Any] = {"name": schema_name, "schema": inline_schema, "strict": True}
    if schema_desc:
        payload["description"] = schema_desc
    return {"type": "json_schema", "json_schema": payload}


def coerce_llm_json_to_schema(
    schema_cls: Type[BaseModel],
    cleaned_text: str,
    wrap_key: Optional[str] = None,
) -> Optional[BaseModel]:
    """LLM 输出形状兜底：原始 ``model_validate_json`` 失败后的二次解析防线。

    背景：qwen 系模型在 ``response_format=json_schema`` 未被服务端严格执行时，
    会偶发输出两种偏移形状，导致「Input should be an object」校验失败：

      1. **数组包裹对象** ``[{...}]``：单对象被包进数组（改写/合并链路常见）
         → 取首个对象元素重新校验；
      2. **顶层数组** ``[{...}, ...]``：schema 本身的业务数据就是数组
         （意图打分旧版 Prompt 格式残留）→ 用 ``wrap_key`` 包装成
         ``{wrap_key: [...]}`` 后重新校验。

    Args:
        schema_cls: 目标 Pydantic 模型类。
        cleaned_text: 原始响应文本；若仍包含 markdown 代码围栏，
            会在直接解析失败后自动剥离围栏重试一次。
        wrap_key: 顶层数组的包装键。指定时数组整体包装为
            ``{wrap_key: array}``；为 None 时数组取首个对象元素。

    Returns:
        校验通过的模型实例；文本非法或形状仍不兼容时返回 None
        （由调用方记录日志并走降级逻辑）。
    """
    try:
        data: Any = json.loads(cleaned_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        # 直接解析失败：尝试剥离 markdown 代码围栏后重试一次
        fenced_match = re.search(r"```(?:\w+)?\n?([\s\S]*?)\n?```", cleaned_text or "")
        if not fenced_match:
            return None
        try:
            data = json.loads(fenced_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            return None

    if isinstance(data, list):
        if wrap_key is not None:
            data = {wrap_key: data}
        elif data and isinstance(data[0], dict):
            data = data[0]
        else:
            return None

    if not isinstance(data, dict):
        return None

    try:
        return schema_cls.model_validate(data)
    except ValidationError:
        return None


def build_function_tool_def(
    function_name: str,
    function_description: str,
    parameters_cls: Type[BaseModel],
) -> Dict[str, Any]:
    """把 Pydantic 参数类转成 OpenAI tools[0] 单条 function 定义。

    用于 S6（MCP 参数提取，动态参数）和 S8（ToolRouter，动态工具名 enum）。
    """
    raw_schema = parameters_cls.model_json_schema(mode="serialization")
    inline_schema = _strip_refs(raw_schema)
    inline_schema.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": function_description,
            "parameters": inline_schema,
        },
    }


def build_tool_choice_required(function_name: str) -> Dict[str, Any]:
    """构造 tool_choice={"type":"function","function":{"name":...}}，强制模型走指定函数。"""
    return {"type": "function", "function": {"name": function_name}}


def build_dynamic_tool_names_enum_schema(
    tool_names: List[str],
    class_name: str = "DynamicToolNames",
    min_selected: int = 0,
    max_selected: Optional[int] = None,
) -> Type[BaseModel]:
    """S8 ToolRouter 专用：按「当前候选工具名列表」动态构造带 enum 约束的 Pydantic 类。

    说明：不在 Python 类型层用 `Literal[a, b, c]`（动态 unpack 易出错），
    而是用 Field(json_schema_extra={"enum": ...}) 把约束注入到 JSON schema，
    这样 Pydantic model_json_schema() 会正确生成 OpenAI 能识别的 enum 数组。
    生成的类字段：selected_tools: List[str]（JSON schema 层约束 enum）。
    """
    tool_names_copy: List[str] = (
        [name for name in tool_names if isinstance(name, str) and name.strip()]
        if tool_names
        else []
    )

    field_extra: Dict[str, Any] = {"enum": tool_names_copy} if tool_names_copy else {}
    if max_selected is not None:
        selected_field = Field(
            default_factory=list,
            min_length=min_selected,
            max_length=max_selected,
            description="选中的工具名数组，必须全部来自候选工具清单的 enum 值。",
            json_schema_extra=field_extra,
        )
    else:
        selected_field = Field(
            default_factory=list,
            min_length=min_selected,
            description="选中的工具名数组，必须全部来自候选工具清单的 enum 值。",
            json_schema_extra=field_extra,
        )

    # 标准 type() 构造 BaseModel 子类：Pydantic 会自动注入 ModelMetaclass
    namespace: Dict[str, Any] = {
        "__annotations__": {"selected_tools": List[str]},
        "selected_tools": selected_field,
        "__doc__": "大模型路由结果：选中的工具名数组，工具名必须来自候选工具清单（JSON schema enum 约束）。",
    }
    dynamic_cls: Type[BaseModel] = type(class_name, (BaseModel,), namespace)
    return dynamic_cls


def build_mcp_parameters_schema(
    tool_name: str,
    parameters_schema: Dict[str, Any],
    class_name: Optional[str] = None,
) -> Type[BaseModel]:
    """S6 MCP 参数提取专用：按某个工具自己的 parameters JSON schema 动态构造 Pydantic 类。

    Args:
        tool_name: 工具名（仅用于生成默认 class_name 蛇形）。
        parameters_schema: 工具函数声明的 OpenAI 兼容 parameters JSON schema 字典
            （必须是 {type: object, properties: {...}, required: [...]} 结构）。
        class_name: 若显式给出，使用该类名；否则根据 tool_name 生成。

    Returns:
        可直接用于 build_function_tool_def / model_validate_json 的 Pydantic 类。
    """
    final_class_name = class_name or _to_pascal_case(f"{tool_name}_parameters")

    # 从 properties 逐个字段转 Pydantic 注解
    properties: Dict[str, Any] = parameters_schema.get("properties") or {}
    required_set = set(parameters_schema.get("required") or [])

    annotations: Dict[str, Any] = {}
    fields_dict: Dict[str, Any] = {}

    for field_key, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        field_type = _json_schema_type_to_python(prop)
        field_meta_kwargs: Dict[str, Any] = {}
        if "description" in prop:
            field_meta_kwargs["description"] = prop["description"]
        # 只转发 Pydantic 支持的常见约束（不支持的忽略，不会抛错）
        for k in ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "enum"):
            if k in prop:
                snake_key = {
                    "minimum": "ge",
                    "maximum": "le",
                    "minLength": "min_length",
                    "maxLength": "max_length",
                    "minItems": "min_length",
                    "maxItems": "max_length",
                }.get(k, k)
                field_meta_kwargs[snake_key] = prop[k]

        if field_key in required_set:
            annotations[field_key] = field_type
            fields_dict[field_key] = Field(**field_meta_kwargs) if field_meta_kwargs else Field()
        else:
            annotations[field_key] = Optional[field_type]  # type: ignore[assignment]
            fields_dict[field_key] = Field(default=None, **field_meta_kwargs)

    namespace = {
        "__annotations__": annotations,
        "__doc__": f"MCP 工具[{tool_name}]的参数 schema（动态生成，用于 function calling 校验）。",
        **fields_dict,
    }
    # 标准 type() 构造 Pydantic 子类（自动走 BaseModel metaclass）
    dynamic_cls = type(final_class_name, (BaseModel,), namespace)
    return dynamic_cls


# ==============================================================================
# 内部助手
# ==============================================================================
def _to_snake_case(name: str) -> str:
    out: List[str] = []
    for idx, char in enumerate(name):
        if char.isupper() and idx > 0:
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def _to_pascal_case(snake_name: str) -> str:
    parts = [p for p in snake_name.replace("-", "_").split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or snake_name


def _json_schema_type_to_python(prop: Dict[str, Any]) -> Any:
    """粗糙但足够的 JSON Schema type → Python/Pydantic 注解映射。"""
    type_raw = prop.get("type")
    enum_values = prop.get("enum")
    if enum_values:
        enum_types = {type(v) for v in enum_values}
        if len(enum_types) == 1:
            return Literal[tuple(enum_values)]  # type: ignore[valid-type]
        return Any

    if isinstance(type_raw, list):
        # "type": ["string", "null"] 这类；此处直接转 Optional[str]
        non_null = [t for t in type_raw if t != "null"]
        if len(non_null) == 1:
            base = _json_schema_type_map(non_null[0], prop)
            return Optional[base]  # type: ignore[return-value]
        return Any

    if isinstance(type_raw, str):
        return _json_schema_type_map(type_raw, prop)

    # 没声明 type：按 object 兜底（MCP 经常省略顶层 type）
    return Dict[str, Any]


def _json_schema_type_map(type_name: str, prop: Dict[str, Any]) -> Any:
    if type_name == "string":
        if prop.get("format") in ("date-time", "datetime"):
            return str
        return str
    if type_name in ("integer", "number"):
        return int if type_name == "integer" else float
    if type_name == "boolean":
        return bool
    if type_name == "array":
        item_schema = prop.get("items") or {}
        item_type = _json_schema_type_to_python(item_schema) if isinstance(item_schema, dict) else Any
        return List[item_type]  # type: ignore[valid-type]
    if type_name == "object":
        return Dict[str, Any]
    if type_name == "null":
        return type(None)
    return Any


# ==============================================================================
# S1：Agent 改写 Schema（agent-question-rewrite.st · 6 顶层字段）
# ==============================================================================
class TaskComplexitySchema(BaseModel):
    """Agent 改写 Stage1 输出的复杂度分析子对象（严格上下界 1~10 / 0~10）。"""

    estimated_steps: int = Field(ge=1, le=10, description="预估完成任务所需步骤数，1~10。")
    estimated_tool_calls: int = Field(ge=0, le=10, description="预估需要调用工具的总次数，0~10。")
    has_multi_step_dependency: bool = Field(description="是否存在「第 N 步依赖第 N-1 步结果」的明确依赖。")
    has_external_data_dependency: bool = Field(description="是否需要查询外部数据源（RAG/工具/飞书/网页）。")
    need_creative_output: bool = Field(description="是否要求撰写/润色/创作类输出（不只是事实查询）。")
    reasoning_notes: str = Field(
        default="",
        max_length=300,
        description="复杂度判断的简短推理说明，建议控制在 300 字内。",
    )


class AgentRewriteSchema(BaseModel):
    """Agent Pipeline Stage1 改写结果：含问题重构+拆分+复杂度+工具建议+步骤提示。

    对应 prompts/agent-question-rewrite.st 的 JSON 输出，顶层固定 6 个 key。
    """

    rewrite: str = Field(min_length=1, max_length=400, description="改写后的规范化用户问题（用于后续 Agent 处理）。")
    should_split: bool = Field(description="是否建议拆分为多个子问题独立处理。")
    sub_questions: List[str] = Field(
        default_factory=list,
        description="拆分后的子问题列表；若不拆分则只包含 rewrite 本身 1 条。",
        min_length=0,
        max_length=10,
    )
    complexity_analysis: TaskComplexitySchema = Field(description="结构化任务复杂度分析。")
    suggested_tools: List[str] = Field(
        default_factory=list,
        description="建议调用的工具名数组（必须来自 Prompt 中明确列出的可用工具白名单）。",
        min_length=0,
        max_length=10,
    )
    suggested_skills: List[str] = Field(
        default_factory=list,
        description="建议调用的高级技能名数组（必须逐字来自 Prompt 中列出的技能清单的 name 字段，"
                    "禁止自造；若问题与任何技能无关则为空数组）。",
        min_length=0,
        max_length=20,
    )
    explicit_plan_hint: Optional[str] = Field(
        default=None,
        max_length=200,
        description="若用户问题中存在「先…再…最后…」等显式步骤描述，回填原文摘要；否则为 null。",
    )


# ==============================================================================
# S2：意图分类 Schema（intent_classify_resolver-classifier.st · 顶层数组 / 兼容 results 包装已废弃）
# ==============================================================================
class IntentClassifyItemSchema(BaseModel):
    """单条叶子意图打分结果。"""

    id: str = Field(min_length=1, max_length=128, description="叶子意图节点 ID（必须与 Prompt 意图清单完全一致）。")
    score: float = Field(ge=0.0, le=1.0, description="该意图命中概率，0~1。")
    reason: Optional[str] = Field(
        default=None,
        max_length=200,
        description="简短打分原因，控制在 200 字内。",
    )


class IntentClassifyListSchema(BaseModel):
    """意图分类顶层容器（OpenAI response_format 不允许顶层直接为 array，因此用 wrapper 类）。

    在 Prompt 中同步告知模型：输出格式为 {\"results\": [{...}, ...]}；
    因此 parse_scores 简化为读取本容器的 results 字段即可，无需顶层 array/results 双兼容分支。
    """

    results: List[IntentClassifyItemSchema] = Field(
        default_factory=list,
        description="叶子意图打分列表，按分数从高到低排序。",
        max_length=20,
    )


# ==============================================================================
# S1+S2 合并：改写 + 意图识别一体化 Schema（调整一 · agent-rewrite-intent-combined.st）
# ==============================================================================
class QuestionIntentScoresSchema(BaseModel):
    """单条问题的意图打分批次（按 question_index 索引改写结果中的问题）。"""

    question_index: int = Field(
        ge=0,
        le=10,
        description=(
            "问题索引：0=改写后的主问题（rewrite 字段）；"
            "1~N=should_split=true 时按序对应的第 N 个子问题。"
        ),
    )
    results: List[IntentClassifyItemSchema] = Field(
        default_factory=list,
        description="该问题的叶子意图打分列表（id 必须与候选意图清单逐字一致），按分数从高到低排序。",
        max_length=10,
    )


class AgentRewriteIntentCombinedSchema(AgentRewriteSchema):
    """「改写 + 意图识别」单次 LLM 调用的组合输出容器（调整一核心）。

    继承 AgentRewriteSchema 的 6 个顶层字段（rewrite/should_split/sub_questions/
    complexity_analysis/suggested_tools/explicit_plan_hint），额外追加
    intent_classifications 逐问题意图打分列表——一次网络往返同时产出
    Stage1（改写）与 Stage2（意图）两段结果，Pipeline 据此削减 1 次 LLM 调用。
    """

    intent_classifications: List[QuestionIntentScoresSchema] = Field(
        default_factory=list,
        max_length=11,
        description=(
            "逐问题意图打分列表：必须包含 question_index=0（改写后主问题）的打分；"
            "若 should_split=true，还需按序包含每个子问题（question_index=1~N）的打分。"
        ),
    )


# ==============================================================================
# S3：模式决策灰区 LLM Schema（orchestration-mode-decider.st · 5 顶层字段）
# ==============================================================================
class ModeDecisionSchema(BaseModel):
    """灰区模式决策 LLM 的强制输出：mode 只允许「react」或「plan_execute」两枚举。"""

    mode: Literal["react", "plan_execute"] = Field(description="最终推荐的编排模式，严格二选一。")
    confidence: float = Field(ge=0.0, le=1.0, description="对该模式决策的置信度，0~1。")
    reason: str = Field(min_length=1, max_length=400, description="决策理由说明，控制在 400 字内。")
    initial_plan_hint: Optional[str] = Field(
        default=None,
        max_length=300,
        description="若 mode=plan_execute：给出一段精炼的初始规划提示语；否则应为 null。",
    )
    first_tool_hint: Optional[str] = Field(
        default=None,
        max_length=64,
        description="若 mode=react：推荐首个调用的工具名（必须来自可用工具白名单）；否则应为 null。",
    )


# ==============================================================================
# S4：RAG 查询改写 Schema（user-question-rewrite.st · 3 顶层字段）
# ==============================================================================
class RagRewriteSchema(BaseModel):
    """RAG 原管线查询改写+拆分结果。"""

    rewrite: str = Field(min_length=1, max_length=400, description="改写后的规范化查询语句。")
    should_split: bool = Field(default=False, description="是否应拆分为多个子查询。")
    sub_questions: List[str] = Field(
        default_factory=list,
        description="拆分后的子查询列表；若 should_split=false，包含 rewrite 本身。",
        min_length=0,
        max_length=10,
    )


# ==============================================================================
# S5：歧义澄清检查 Schema（guidance-ambiguity-check.st · 3 顶层字段）
# ==============================================================================
class AmbiguityCheckSchema(BaseModel):
    """歧义澄清检查结果：是否需要反问用户 + 命中的歧义意图 ID。"""

    ambiguous: bool = Field(description="当前问题是否因命中多个候选意图而存在歧义。")
    category_ids: List[str] = Field(
        default_factory=list,
        description="若 ambiguous=true：回填候选歧义意图 ID 数组；否则为空数组。",
        max_length=10,
    )
    reason: str = Field(
        default="",
        max_length=300,
        description="做出歧义判定（或非歧义判定）的简短理由说明。",
    )


# ==============================================================================
# S7：Reflection 反思审查 Schema（reflection.py · 7 字段质量报告）
# ==============================================================================
class ReflectionReportSchema(BaseModel):
    """反思审查报告：质量分+完成度+幻觉判断+改进建议。"""

    quality_score: int = Field(
        ge=0,
        le=100,
        description="综合质量得分，0（极差）到 100（完美）的整数。",
    )
    is_complete: bool = Field(description="回答是否覆盖了用户全部核心子问题。")
    likely_hallucination: bool = Field(description="是否怀疑回答包含无依据的虚构事实或来源。")
    hallucination_reasons: List[str] = Field(
        default_factory=list,
        description="怀疑幻觉的具体理由；若无疑似幻觉，为空数组。",
        max_length=10,
    )
    completeness_notes: str = Field(
        default="",
        max_length=400,
        description="完整性问题的说明（如缺失哪些关键要点）。",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="面向助手的可执行改进建议。",
        max_length=10,
    )
    summary: str = Field(
        min_length=1,
        max_length=200,
        description="一句话中文总结本次审查结论。",
    )


# ==============================================================================
# S8：ToolRouter 静态 Schema（作为 response_format fallback；动态 function calling 见 build_dynamic_tool_names_enum_schema）
# ==============================================================================
class SelectRelevantToolsSchema(BaseModel):
    """ToolRouter 语义路由的静态 schema（当 llm_client 不支持 tools 字段时走 response_format 兜底）。"""

    selected_tools: List[str] = Field(
        default_factory=list,
        description="选中的工具名数组，必须全部来自 Prompt 中给出的候选工具清单。",
        max_length=10,
    )


# ==============================================================================
# MID-1：Planner 初始计划生成 Schema（planner.py PLAN_SYSTEM_PROMPT）
# ==============================================================================
class SubTaskSchema(BaseModel):
    """Planner 计划中的单条 SubTask 声明（与 planner.py SubTask dataclass 1:1 对齐）。"""

    id: str = Field(min_length=1, max_length=64, description="子任务唯一 ID，例如 task_1。")
    title: str = Field(min_length=1, max_length=120, description="子任务标题（简短可读）。")
    description: str = Field(min_length=1, max_length=500, description="子任务完整描述与要求。")
    action_type: Literal["tool", "reasoning"] = Field(
        description="子任务类型：tool=需调用工具；reasoning=纯推理/总结/写作即可完成。",
    )
    tool_name: Optional[str] = Field(
        default=None,
        max_length=64,
        description="若 action_type=tool：指定要调用的工具名；否则为 null。",
    )
    tool_args_hint: Optional[str] = Field(
        default=None,
        max_length=400,
        description="若 action_type=tool：给出工具参数的 JSON 文本或自然语言提示；否则为 null。",
    )


class PlanGenerateSchema(BaseModel):
    """Planner 初始计划 / 重计划 replan 的顶层输出容器（共享同一 schema，C-1=MID-1，C-2=REPLAN 也复用）。"""

    subtasks: List[SubTaskSchema] = Field(
        default_factory=list,
        description="规划出的子任务列表，按执行顺序排列。",
        min_length=0,
        max_length=20,
    )


__all__ = [
    # 公共工具
    "pydantic_to_openai_response_format",
    "build_function_tool_def",
    "build_tool_choice_required",
    "build_dynamic_tool_names_enum_schema",
    "build_mcp_parameters_schema",
    # S1 Agent 改写
    "TaskComplexitySchema",
    "AgentRewriteSchema",
    # S2 意图分类
    "IntentClassifyItemSchema",
    "IntentClassifyListSchema",
    # S1+S2 合并（调整一）
    "QuestionIntentScoresSchema",
    "AgentRewriteIntentCombinedSchema",
    # S3 模式决策
    "ModeDecisionSchema",
    # S4 RAG 改写
    "RagRewriteSchema",
    # S5 歧义澄清
    "AmbiguityCheckSchema",
    # S7 Reflection
    "ReflectionReportSchema",
    # S8 ToolRouter
    "SelectRelevantToolsSchema",
    # MID-1 Planner 计划
    "SubTaskSchema",
    "PlanGenerateSchema",
]
