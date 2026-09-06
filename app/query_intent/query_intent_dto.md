# query_intent DTO 转换与流转深入梳理

> 覆盖范围：`/chat/with_agent` 路由上，"问题改写 → 意图聚合 → 模式决策 → 白名单/slots 合并 → 组装 IntentContext → AgentOrchestrator.run" 的完整边界流转。
> 本文档以实际代码为准（字段名 1:1 对齐源码），可与 `repo_map.txt`、`业务逻辑梳理.md` 互相对照。

## 1. 涉及文件与核心对象速览

| 文件 | 关键类型 / 常量 |
|---|---|
| `app/query_intent/intent_dto.py` | `AgentIntents`、`AgentRewriteResult`、`TaskComplexityAnalysis`、`ModeDecision`、`AgentQueryIntentPipelineOutput`、`OrchestrationModeLiteral`、`SubQuestionIntent` |
| `app/query_intent/intent_classify_resolver/intent_model.py` | `NodeScore`（意图树叶子节点打分实体，KB/MCP/SYSTEM 三族） |
| `app/query_intent/intent_classify_resolver/intent_resolver.py` | `IntentResolver.resolve_for_agent(...) -> AgentIntents` |
| `app/query_intent/intent_classify_resolver/intent_classify.py` | `AgentIntentAggregator`（聚合 + `collect_mcp_tool_ids_from_intents`） |
| `app/query_intent/rewrite/query_rewrite.py` | `AgentQueryRewriteService.rewrite_for_agent(...)` |
| `app/query_intent/intent_question_rewrite_orchestration/mode_decider.py` | `ModeDecider.decide_orchestration_mode(...)` |
| `app/query_intent/intent_question_rewrite_orchestration/agent_query_intent_pipeline.py` | `AgentQueryIntentPipeline`：`run()` / `build_orchestrator_input()` / `_finalize_allowed_tools()` / `_merge_slots()` |
| `app/query_intent/intent_data_base.py` | `AgentChatContext`、`IntentChatMessage`、`IntentChatRequest` |
| `app/query_intent/rag_constant.py` | `REGISTERED_ENABLED_TOOL_NAMES`、`PIPELINE_INFRASTRUCTURE_TOOL_SET`、`MODE_DECISION_*` 阈值 |
| `app/core/agent/orchestrator.py` | `IntentContext`、`AgentOrchestrator.run()`、`_tool_names()` |
| `app/api/routes/chat.py` | `/chat/with_agent` 路由（调度主流程） |

---

## 2. 整体流转图

```
HTTP POST /chat/with_agent (request.query / request.session_id / request.strategy)
        │
        │ ① 会话兜底 + 快照
        ▼
registered_tool_snapshot = ToolRegistry.list_tool_names()
short_term history ──► List[IntentChatMessage]（改写参考）
        │
        │ ② asyncio.to_thread(pipeline.run, query, snapshot, session_id, history)
        ▼
┌─────────────── AgentQueryIntentPipeline.run ─────────────────────────────┐
│ AgentChatContext(original_user_question, session_id,                    │
│                  available_tool_ids=快照, conversation_history)          │
│      └► _execute_three_stage_pipeline(agent_context)                     │
│            Stage1 rewrite ──► AgentRewriteResult                         │
│            Stage2 intents  ──► AgentIntents（resolve_for_agent）          │
│            Stage3 mode     ──► ModeDecision（ModeDecider 四层）           │
│            Stage4 merge    ──► _finalize_allowed_tools + _merge_slots    │
└───────────────────────────────────────────────────────────────────────────┘
        │
        │ ③ AgentQueryIntentPipelineOutput
        │    (rewrite_result / intents_result / mode_decision /
        │     final_user_input / allowed_tools_final / merged_slots / …)
        ▼
strategy 归一化 ──► force_override_mode（仅 react|plan_execute 有效）
        │
        ▼
pipeline.build_orchestrator_input(output, force_override_mode)
        │ 产出三元组 (final_mode, intent_context_payload, effective_user_input)
        ▼
IntentContext(**intent_context_payload)        // 4~5 个 key 与 dataclass 1:1
        │
        │ ④ AgentOrchestrator.run(user_input=effective_user_input, mode=final_mode, intent=…)
        ▼
orchestrator.run 内部：if intent.preferred_mode: mode = intent.preferred_mode
        │         └► _tool_names(intent)（对 allowed_tools 做“注册中心 ∩ ”二次过滤 + 基建工具补）
        ▼
ReAct / Plan-Execute 引擎执行
```

---

## 3. 边界 ③：改写与意图聚合 → AgentIntents（Stage1+Stage2）

### 3.1 输入 AgentChatContext（intent_data_base.py）
`Pipeline.run()` 先把原始参数包成 `AgentChatContext`：

| AgentChatContext 字段 | 来源 |
|---|---|
| `original_user_question` | `request.query` |
| `session_id` | 兜底后的会话 ID |
| `available_tool_ids` | `ToolRegistry.list_tool_names()`（**注册中心全量快照**，非最终白名单） |
| `conversation_history` | 短期记忆转出的 `IntentChatMessage` 列表 |

### 3.2 Stage1：改写（_stage_1_rewrite）
- 调用 `rewrite_service.rewrite_for_agent(agent_context) -> AgentRewriteResult`。
- 关键字段：`rewritten_question`（改写后主问题）、`sub_questions`、`should_split`、
  `complexity_analysis`（六维：`estimated_steps`/`estimated_tool_calls`/`has_multi_step_dependency`/`has_external_data_dependency`/`need_creative_output`/`reasoning_notes`）、`suggested_tools`、`explicit_plan_hint`。
- 兜底：改写抛异常或 `rewritten_question` 为空 → 回退 `original_user_question`；`sub_questions` 为空时收敛为单元素。

### 3.3 Stage2：意图聚合（resolve_for_agent）
- 以改写结果为主问题 + 子问题，调用意图解析器聚合输出 `AgentIntents`。
- `AgentIntents` 内部把命中的 `NodeScore` 按族分组：
  - 计数：`kb_hit_count` / `mcp_hit_count` / `sys_hit_count`
  - 明细：`kb_node_scores` / `mcp_node_scores` / `sys_node_scores`
  - 跨子问题：`per_sub_question_intents: List[SubQuestionIntent]`
  - 主意图文本：`primary_intent_text`（通常取置信度最高节点 `display_name`，无命中 `"general"`）
  - 聚合置信度：`aggregated_confidence`（仅 debug/回填用）
  - 槽位底池：`raw_slots: Dict[str, Any]`
- 兜底：本阶段异常 → `AgentIntents(primary_intent_text="general")`。

> ⚠️ 代码里没有单独的 `system_node_scores` 字段名，而是 **`sys_node_scores`**（与 `sys_hit_count` 对齐），引用时注意。

---

## 4. 边界 ④：AgentIntents → ModeDecision（Stage3 模式决策）

### 4.1 入口与四层判定

`ModeDecider.decide_orchestration_mode(rewrite_result, intents_result, available_tool_ids)`（mode_decider.py:87）：

```
Layer1 explicit_hint ─► 命中即 plan_execute（conf=0.92，decision_source="explicit_hint"）
        │ 未命中
        ▼
Layer2 rule_threshold ─► 快速判 react/plan_execute
        │   规则已定 且 不在灰区(_need_llm_double_check=False) → 直接返回
        │   规则已定 但 落在灰区 → 进入 Layer3 LLM 二次精判
        ▼
Layer3 llm（灰区才触发）：
        │   llm_service/prompt 未配置 → 返回规则结果（fallback）
        │   LLM 调用/解析失败 → 返回规则结果
        │   LLM 置信度 < llm_min_confidence(0.55) → 返回规则结果
        ▼
规则无稳定判断 + LLM 未启用/失败 → ModeDecision(mode=default=react,
        decision_source="fallback_default")
```

### 4.2 各层判定细节

| 层 | 判定逻辑 | 产出 decision_source / confidence |
|---|---|---|
| Layer1 | `rewrite_result.explicit_plan_hint` 非空（“先…再…最后…/步骤1..N”） | `explicit_hint`；conf `0.92`；`initial_plan_hint=该原文` |
| Layer2 强触发 | `estimated_steps >= 3` 或 `estimated_tool_calls >= 2` 或 `has_multi_step_dependency` | `rule_threshold`；conf `0.82`；mode=plan_execute |
| Layer2 默认 | 否则（含弱信号：纯对话/创造性单步） | `rule_threshold`；conf `0.70`；mode=react |
| Layer2 灰区 | `|steps-3|<=1` 或 `|tools-2|<=1`，或"有依赖但规则判 react"的矛盾 | 不直接返回，进入 Layer3 |
| Layer3 | LLM（temperature0.1/top_p0.3/thinking=False + `ModeDecisionSchema` json_schema） | `llm`；conf 由 LLM 给；`initial_plan_hint`(plan)/`first_tool_hint`(react) 由 LLM 给 |
| 兜底 | 上述任一失败 | `fallback_default`；mode=react |

- 阈值来自 `rag_constant.py`：`MODE_DECISION_STEP_THRESHOLD=3`、`MODE_DECISION_TOOL_THRESHOLD=2`、`MODE_DECISION_LLM_MIN_CONFIDENCE=0.55`、`MODE_DECISION_DEFAULT_MODE="react"`。

### 4.3 ModeDecision 字段消费去向

| ModeDecision 字段 | 消费点 |
|---|---|
| `mode` | → `final_mode`（除非被 strategy 强覆盖）；写入 `slots.pipeline_mode_meta.mode` |
| `confidence` | → debug（`pipeline_mode_meta.confidence` / `pipeline_original_mode_decision.confidence`） |
| `reason` | → debug meta |
| `decision_source` | → debug meta（四枚举追踪走的是哪条分支） |
| `initial_plan_hint` | 仅 plan_execute：→ `merged_slots["initial_plan_hint"]` → 再被 orchestrator 读入 `planner_initial_hint_block` |
| `first_tool_hint` | 仅 react：→ `merged_slots["first_tool_hint"]` → 再被 orchestrator 读入 `pipeline_first_tool_hint_line` |

> 注意区分两个 debug 槽位：
> - `pipeline_mode_meta`：`_merge_slots` 阶段写入的**本次决策**完整 meta（mode/confidence/reason/decision_source）。
> - `pipeline_original_mode_decision`：`build_orchestrator_input` 阶段写入的**原始（未被 strategy 覆盖前）**决策 meta，仅在有强覆盖语义时才有区分价值。

---

## 5. 边界 ⑤：三段产物 → 白名单合并 + slots 汇总

### 5.1 MCP 命中工具抽取
`AgentIntentAggregator.collect_mcp_tool_ids_from_intents(intents_result)` 把 `mcp_node_scores` 里各命中节点的 `mcp_tool_id` 拉平为列表；异常时按空列表处理。

### 5.2 `_finalize_allowed_tools`（agent_query_intent_pipeline.py:355）
算法 = **有序并集 + 交集过滤 + 基建保底**：

```
allowed_tool_universe = set(REGISTERED_ENABLED_TOOL_NAMES) ∪ request 快照(available_tool_ids)
ordered = []   # 保序、去重
for source in (rewrite.suggested_tools, mcp_hit_tool_ids, PIPELINE_INFRASTRUCTURE_TOOL_SET):
    for t in source:
        清洗后 t ∈ universe 且未出现 → 追加
若 ordered 为空 → 保底塞入 PIPELINE_INFRASTRUCTURE_TOOL_SET（file_read/file_list/file_grep）
return ordered
```

要点：
- 三段来源的**优先级顺序**：LLM 改写推荐 > 意图树 MCP 命中 > 基建工具。
- `REGISTERED_ENABLED_TOOL_NAMES`（rag_constant.py:57，含 web_search / rag_knowledge_search / knowledge_graph_search / write_todos / file_* 等）是**配置期白名单快照**；
- 交集用“`REGISTERED_ENABLED ∪ request 快照`”，是为了同时防 LLM 幻觉和防注册中心与常量漂移。

### 5.3 `_merge_slots`（agent_query_intent_pipeline.py:402）产出的最终 slots 键

~~~~| 键 | 内容 | 写入条件 |
|---|---|---|
| `raw_slots` 展开~~~~ | 意图解析器槽位（实体/时间等） | 恒有（底池） |
| `explicit_plan_hint` | 改写阶段显式步骤原文 | 非空才写 |
| `initial_plan_hint` | ModeDecision 计划冷启动 hint | `mode==plan_execute` 且有值 |
| `first_tool_hint` | ModeDecision 首工具 hint | `mode==react` 且有值 |
| `pipeline_mode_meta` | {mode, confidence, reason, decision_source} | 恒有 |
| `pipeline_complexity_meta` | 六维复杂度（含 None 防御） | 恒有 |
| `pipeline_rewrite_meta` | {should_split, sub_questions_count, suggested_tools} | 恒有 |
| `pipeline_mcp_hit_tool_ids` | MCP 命中工具列表 | 恒有 |

### 5.4 AgentQueryIntentPipelineOutput 汇总
`final_user_input = rewritten_question（非空） else original_user_question`；其余字段即上文三段产物 + 白名单 + slots + 原始问题 + session_id。

### 5.5 全程容错
`_execute_three_stage_pipeline` 每一段都有 try/except 与 fallback（默认 react + general 意图 + 基建工具白名单），**Pipeline 永不向上抛异常**；最外层还有 `fallback_output` 兜底整链异常。

---

## 6. 边界 ⑥：Pipeline 输出 → 三元组 → IntentContext

### 6.1 `build_orchestrator_input`（agent_query_intent_pipeline.py:93）

入参：`pipeline_output` + `force_override_mode`（chat.py 由 `request.strategy` 归一化而来，仅 `react|plan_execute` 会强覆盖）。

处理顺序：
1. `original_mode = output.mode_decision.mode`；`final_mode = force_override_mode 若合法 else original_mode`。
2. `merged_slots = dict(output.merged_slots)`，然后：
   - `setdefault("pipeline_original_mode_decision", {mode/confidence/reason/decision_source})`（只写一次，保留原始决策）；
   - 若有强覆盖 → 写 `mode_override_by_strategy = force_override_mode`；
   - **清理与新模式不匹配的 hint**：覆盖成 react → 删除 `initial_plan_hint`（保留 `first_tool_hint`）；覆盖成 plan_execute → 删除 `first_tool_hint`（保留 `initial_plan_hint`）。这是为了防 ReAct 误读 Planner hint 或反之。
3. `confidence` 回填：取 `intents_result.aggregated_confidence`，非法/越界 clamp 到 [0,1] 或兜底 1.0（供追踪面板真实展示）。
4. 组装 `intent_context_payload`（5 键）：
   ```
   {
     "preferred_mode": final_mode,
     "intent":        primary_intent_text or "general",
     "confidence":    effective_confidence,
     "allowed_tools": list(allowed_tools_final),
     "slots":         merged_slots,
   }
   ```
5. `effective_user_input = final_user_input or original_user_question`。
6. 返回 `(final_mode, intent_context_payload, effective_user_input)`。

### 6.2 chat.py 装配（chat.py:175-207）

```python
final_mode, intent_payload_dict, effective_user_input = pipeline.build_orchestrator_input(...)
intent_context = IntentContext(**intent_payload_dict)     # 5 key 与 dataclass 字段 1:1
agent_orchestrator.run(user_input=effective_user_input,
                       session_id=active_session_id,
                       mode=final_mode,
                       intent=intent_context)
```

> `AgentOrchestrator` 的 config 目前只放 `{"default_strategy": request.strategy}`（供 trace/debug 参考），路由参数真正生效的是显式传的 `mode` 与 `intent.preferred_mode`。

### 6.3 Orchestrator.run 内部如何消费 IntentContext

- `orchestrator.py:131`：`if intent.preferred_mode: mode = intent.preferred_mode`（最终 mode 以 intent 内字段为准，二者在正常链路中值一致）。
- `orchestrator._tool_names(intent)`（orchestrator.py:100）：
  ```
  allowed = 注册中心全量名
  if intent.allowed_tools: allowed = [t for t in intent.allowed_tools if t in 注册中心]   # 二次交集
  infra = {file_read_tool, file_list_tool, file_grep_tool} 若被裁掉则强制补回
  ```
  即：即使 Pipeline 的白名单里有脏工具名，orchestrator 仍会用**运行时注册中心**再过滤一次，并保证基建工具不被高级技能误杀。
- 槽位消费示例：
  - `_run_react`：读 `intent.slots.get("first_tool_hint")` 组装 `pipeline_first_tool_hint_line`（需在 `allowed_tools` 白名单内才注入）。
  - `_run_plan_execute`：读 `intent.slots.get("initial_plan_hint")` 组装 `planner_initial_hint_block` 注入 Planner 冷启动。

---

## 7. 常见疑问与易错点

1. **字段命名**：`AgentIntents` 是 `sys_node_scores` 不是 `system_node_scores`；`slots` 里既有 `pipeline_mode_meta`（本次决策）也有 `pipeline_original_mode_decision`（原始决策，被覆盖时才有差异）。
2. **白名单是两段式过滤**：Pipeline 先做“并集 ∩ (REGISTERED_ENABLED ∪ 快照)”，Orchestrator 再做“∩ 运行时注册中心 + 基建保底”。两段都不可省。
3. **confidence 有两个来源**：`ModeDecision.confidence`（模式决策置信度，进 debug meta）与 `IntentContext.confidence`（回填意图聚合 `aggregated_confidence`，默认 1.0）。别混用。
4. **strategy 强覆盖 ≠ 修改决策**：覆盖只会改 `final_mode` 并清理不匹配 hint，原始 `mode_decision` 保留在 `slots`（`pipeline_original_mode_decision` / `pipeline_mode_meta`）供 debug。
5. **Pipeline 是同步、零侵入执行层**：async 路由用 `asyncio.to_thread` 包装；Pipeline 不 import 也不改 ReAct/Planner/Reflection。
6. **灰区才花 LLM 钱**：显式步骤或规则强命中（steps≥3 / tools≥2 / 有依赖）时，多数请求在 Layer1/2 直接返回，不触发 Layer3。
7. **hint 注入链路**：`initial_plan_hint/first_tool_hint` 不通过形参侵入 Agent，而是经 `slots` 中转，再由 orchestrator 拼装成提示段——改动最小、可审计。
