# repo_map_calibrated.md — 8 边界静态链路全量审计 + 代码骨架校准

> 生成时间：2026-09-02（承接 8 边界链路图 + 真实代码逐行核对）
>
> 审计范围：`/chat/with_agent` → Pipeline 三阶段 → Orchestrator → ReAct/Planner → Reflection 共 8 条边界。
>
> 校验项：**参数传入 · 类实例初始化 · 参数校验 · 字段名 1:1 对齐 · 业务逻辑前后矛盾**，每条边界给出 PASS/FAIL + 详情 + 文件:行号。

---

## 第一部分 · 8 条边界 PASS/FAIL 全量审计表

符号：✅ PASS（链路图描述 == 真实代码实现，逐字段核对）、⚠️ PARTIAL（实现了但字段名/兜底/交集顺序有偏差）、❌ FAIL（没实现或与链路图矛盾）、【存疑-#】需要用户拍板（列在第四部分）。

### 边界 ①：`chat.py → Pipeline.run → AgentChatContext 对象`
链路图要求字段：`original_user_question` / `session_id` / `available_tool_ids`（9 个 tool_registry 快照）/ `conversation_history`（来自 ShortTermMemory→IntentChatMessage）。

| 字段 | PASS? | 详情（文件:行号） |
|---|---|---|
| original_user_question | ✅ | [pipeline.run L83-88](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/orchestration/agent_query_intent_pipeline.py#L83-L88) 从 `user_question=request.query` 直接赋值；[AgentChatContext L218](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/intent_data_base.py#L218) 必填字段无默认值，**不接受 None**，上游 chat.py 传 `request.query` 是字符串（Pydantic 默认空串）。✅ |
| session_id | ✅ | chat.py L112 做了 `request.session_id or session_xxx` 兜底 → pipeline.run L85 赋值 → AgentChatContext L225-231 还有 `__post_init__` 二次 fallback，双层保险。✅ |
| available_tool_ids（tool_registry.list_tool_names 快照） | ✅ | chat.py L115 `registered_tool_snapshot = tool_registry.list_tool_names()` → pipeline.run L66/L86 传参。AgentChatContext L232-233 再做 None→[] 兜底。✅ 9 个 REGISTERED 工具名与 rag_constant.py L62-72 完全一致。 |
| conversation_history（Memory → IntentChatMessage 转换） | ⚠️ **PARTIAL** | **转换确实发生了**（chat.py L120-139：role 小写化 → 仅保留 `user`/`assistant` 两类 → `IntentChatMessage.user(...)` / `.assistant(...)`）。但链路图说「来自 MemoryManager.get_context().short_term_messages」是对的，而**改写阶段 AgentChatContext 只拿到了最近 USER 消息用于指代消解（AgentChatContext 文档 L215 写的），但 chat.py 把 assistant 消息也一并传入了**。这不导致异常，但存在「LLM 改写阶段读取到 Assistant 答案后发生数据泄露」的理论风险（见存疑-1）。 |
| dataclass 初始化（必填字段漏填 NPE） | ✅ | AgentChatContext L218-219 `original_user_question`/`session_id` 是**无默认值**的必填；chat.py 都显式赋值了，上游不传会直接在构造期爆 TypeError，**不会静默 NPE**。✅ |
| 类初始化是否含未提供的参数（AgentMultiQuestionRewriteService 等 4 组件） | ✅ | 见【边界 ⑤ 依赖注入】与 chat.py L94 `pipeline: AgentQueryIntentPipeline = Depends(get_pipeline)`，所有 4 个 dataclass 字段都在 get_pipeline 内显式 new。✅ |

### 边界 ②：Stage1 `AgentMultiQuestionRewriteService.rewrite_for_agent(ctx)` → `AgentRewriteResult`
链路图：加载 agent-question-rewrite.st 模板 + 渲染 3 变量（历史/工具/规则 hint）→ LLM 输出 JSON → `_parse_agent_rewrite()` 做「白名单交集 + 上下界 clamp + 规则 fallback」。

| 字段/动作 | PASS? | 详情（文件:行号） |
|---|---|---|
| rewrite_for_agent 签名接收 AgentChatContext | ✅ | [multi_question_rewrite_service.py L197-214](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/rewrite/multi_question_rewrite_service.py#L197-L214)。ctx.original_user_question / ctx.conversation_history / ctx.available_tool_ids 都被实际读取。✅ |
| 返回 AgentRewriteResult（含 6 字段：rewritten/sub_questions/complexity/suggested_tools/explicit_plan_hint） | ✅ | DTO intent_dto.py L89-L112 含 `complexity_analysis: TaskComplexityAnalysis`（默认 factory，不会 NPE）。`_parse_agent_rewrite()` L525-532 完整返回 6 字段填充。✅ |
| Prompt 渲染 3 变量：历史/工具/规则 hint | ✅ | LLM 调用代码实际加载了 AGENT_QUESTION_REWRITE_PROMPT_PATH；并在 system+user prompt 中注入 3 类变量（对应 LLM 调用法实际 code 已落地，和 skeleton 对齐）。✅ |
| response_format=json_schema 约束（前一轮改造 S1） | ✅ | S1 改造已过：rewrite 传 thinking=False + AgentRewriteSchema response_format + model_validate_json 解析。✅ |
| **上下界 clamp**（estimated_steps 1-10 / estimated_tool_calls 0-10） | ✅ | L482-498 真代码：`max(1,min(10,int(x)))` × 2 个字段，boo l 字段强制 `bool()` 转换；reasoning_notes strip。✅ |
| **suggested_tools 白名单交集**（∩ REGISTERED ∪ request 快照） | ✅ | L500-513 真代码：`allowed_tool_union=REGISTERED_ENABLED ∪ available_tool_ids`；然后逐个 append 只有 `candidate in allowed_tool_union` 才进列表。✅ 动态白名单交集 ✓ |
| explicit_plan_hint 两级兜底（LLM 优先 → 规则正则抽取） | ✅ | L515-523 真代码：LLM 为 null/"" → fallback `pre_rule_plan_hint` = `_rule_extract_explicit_plan_hint(raw_question)`。✅ |
| **should_split / sub_questions 空值兜底** | ✅ | L472-480：sub_questions 空 → should_split=False + 自动把主问题塞进 sub_questions[0]。✅ Pipeline 后续不会拿到空数组。 |

### 边界 ③：Stage2 `IntentResolver.resolve_for_agent(rewrite_result, aggregator=)` → `AgentIntents`
链路图：rewritten_question 作主查询、sub_questions 作 per_sub_question_intents → `DefaultIntentClassifier.classify_targets` 跑三通道 → 返回 AgentIntents（含 primary_intent_text / aggregated_confidence / kb&mcp&sys 三组 NodeScore / per_sub_question_intents / raw_slots）。

| 字段/动作 | PASS? | 详情（文件:行号） |
|---|---|---|
| resolve_for_agent 签名：**第二个 aggregator 参数可选**，实际链路图中 aggregator 由 Pipeline 传入（Pipeline L189-192） | ✅ | [intent_resolver.py L242-280](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/intent/intent_resolver.py#L242-L280) 三分支：传了 aggregator → 复用；没传且 classifier 是 Default → 就地 new；没传且 classifier 类型不明 → 再兜底兼容构造。Pipeline 侧 `aggregator=self.intent_aggregator` 显式传。✅ 不会进入分支 2/3。 |
| 输出对象：`AgentIntents` | ✅ | L293-305 直接 `effective_aggregator.aggregate_for_agent(primary_question=…, sub_questions=…)` 返回 AgentIntents。✅ |
| **字段名 1:1 对齐链路图**：primary_intent_text / aggregated_confidence / kb&mcp&system_node_scores / per_sub_question_intents / raw_slots | ✅ vs ❌ 小 FAIL：**链路图写 "primary_intent_text" 和 "aggregated_confidence"，真实 intent_dto.py L139/L140 字段名就是这两个** ✅。kb/mcp/sys 三组 NodeScore 列表：链路图写 `kb_node_scores`，DTO L144-146 确实是 `kb_node_scores` 而不是 `kb_intents`（原 repo_map.txt 旧骨架写的是 kb_intents，**那个是旧名，当前真实 DTO 已修正为 xxx_node_scores**）。per_sub_question_intents 字段存在（L147）。raw_slots 存在（L148）。✅ 【但是】原 repo_map.txt L223-230 的 3 处字段名（primary_intent_label / primary_confidence / slots）在**真实 DTO 里已改名**（primary_intent_text / aggregated_confidence / raw_slots），**旧 repo_map.txt 这一段已经过时，标记为【待移除】**。 |
| IntentClassi fier.classify_targets 调用签名（疑问-B：是否真的 dict 兼容） | ✅（真实代码已在 Adapter 层兼容） | `_AppModelRouterIntentLLMAdapter.chat()` 位于 [dependencies.py L78-L193](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/app/api/depends/dependencies.py#L78-L193)，已实现 IntentChatRequest / dict 两条分支，classify_targets 走 dict 不会报错。**疑问-B 选择 B-1（现状有效，不必重写 classify_targets）**，但在 Dependencies 内的路由 "intent_analysis" purpose 是【存疑-3】：真实 Provider 上 "intent_analysis" purpose 声明了吗？如果没有会 catch Exception 回退 "chat"，有容错但性能慢一次 for_purpose round-trip。 |

### 边界 ④：Stage3 `ModeDecider.decide_orchestration_mode(rewrite, intents, tool_ids)` → `ModeDecision`
链路图：四层决策 explicit_hint → rule_threshold →（灰区 LLM）→ fallback_default；产出 mode/confidence/**decision_source**/initial_plan_hint/first_tool_hint。

| 字段/动作 | PASS? | 详情（文件:行号） |
|---|---|---|
| 方法签名与三层决策结构 | ✅ | [mode_decider.py L87-133](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/orchestration/mode_decider.py#L87-L133)：L1 显式提示 → L2 规则 → 灰区 LLM double_check → L3 LLM fallback + fallback_default。链路图写的是四层，真实代码就是四层。✅ |
| decision_source 字段（链路图要求 Literal 四选一） | ✅ | ModeDecision L179-181：`Literal["explicit_hint","rule_threshold","llm","fallback_default"]`，**就是链路图写的四种值**。每条返回点都显式赋了 decision_source 字面量（已在 plan()/replan() 对应代码中核实，非空）。✅ |
| mode 字面值（Literal react/plan_execute） | ✅ | OrchestrationModeLiteral（intent_dto.py L10）= Literal["react","plan_execute"]。和 orchestrator.py OrchestrationMode Literal 字符完全相等。✅ |
| initial_plan_hint / first_tool_hint 字段 | ✅ | ModeDecision L182-183：`Optional[str]` 两字段均定义；ModeDecider LLM 分支（_decide_by_llm_with_fallback）实际填充这两个字段。✅ |
| ModeDecider.estimated_steps 访问会 NPE 吗？（complexity_analysis 是 Optional 但 AgentRewriteResult 默认有） | ✅ | intent_dto.py L108：`complexity_analysis = field(default_factory=TaskComplexityAnalysis)` —**不是 Optional**（与旧 repo_map.txt 写的 Optional 描述不符，但实际是 default_factory 保证一定有对象）。所以 ModeDecider 里 `rewrite.complexity_analysis.estimated_steps` 永不为 None。✅ 这一点旧 repo_map.txt 的 TaskComplexityAnalysis 字段前 Optional 描述【待移除】。 |

### 边界 ⑤：三产物 → `_finalize_allowed_tools` + `_merge_slots` 汇总
链路图白名单算法：
```
并集 = rewrite_suggested_tools ∪ MCP ∪ PIPELINE_INFRASTRUCTURE_TOOL_SET(3项)
再做：并集 ∩ (REGISTERED_ENABLED ∪ request_snapshot_tools) = allowed_tools_final
```
slots 算法：`raw_slots + explicit_plan_hint + initial_plan_hint/first_tool_hint + pipeline_mode_meta + pipeline_complexity_meta + pipeline_rewrite_meta + pipeline_mcp_hit_tool_ids`。

| 字段/动作 | PASS? | 详情（文件:行号） |
|---|---|---|
| **并集顺序**：rewrite → MCP → INFRA（链路图写的 ∪ 三源） | ✅ | [agent_query_intent_pipeline.py L344-349](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/orchestration/agent_query_intent_pipeline.py#L344-L349) `for source in (rewrite_suggested_tools, mcp_hit_tool_ids, PIPELINE_INFRASTRUCTURE_TOOL_SET)`。✅ 顺序和链路图完全一致、去重、保序。 |
| **交集算法**：链路图写 `并集 ∩ (REGISTERED ∪ request_snapshot)` | ✅ | L338-342：`allowed_tool_universe = set(REGISTERED_ENABLED_TOOL_NAMES) | {request_snapshot_tools 去空字符串去空格}`。L356：`if cleaned_name not in allowed_tool_universe: continue`。**完全对应链路图算法**。✅ |
| **防御：交集结果空 → 至少保留 INFRA 3 项**（链路图未写但业务上必须） | ✅ | L362-366 防御性补足：`ordered_result` 空 → 写入 3 个 infrastructure。✅ |
| MCP 工具提取：`intent_aggregator.collect_mcp_tool_ids_from_intents(intents_result)` | ✅ **【存疑-4 结论：已实现 · 非缺口】**（2026-09-02 运行时验证：`python -c "from query_intent.intent.intent_classify import AgentIntentAggregator; print('collect_mcp_tool_ids_from_intents' in dir(AgentIntentAggregator))"` 返回 **True**；方法定义在 [intent_classify.py L492-L505](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/intent/intent_classify.py#L492-L505)，遍历 `agent_intents.mcp_node_scores` → 取 `node_score.node.mcp_tool_id` → 去重保序。✅ MCP 源白名单通路存在。**原【待补充-⑤-X】骨架作废，不要落地**，下文同步标注。 |
| slots：pipeline_mode_meta（含 mode/confidence/reason/decision_source） | ✅ | `_merge_slots()` L396-401：键名就是链路图 `pipeline_mode_meta`，字段值全对。✅ |
| slots：pipeline_complexity_meta（链路图写 complexity_meta，真实是 pipeline_complexity） | ⚠️ **小 FAIL：键名不一致**。链路图写的是 `pipeline_complexity_meta`，真实代码 L402 写的是 `pipeline_complexity`（少了 `_meta` 后缀）。**业务上不影响下游消费（没人读取这个 key 做强判断），但前后命名不一致导致未来 frontend debug 面板可能查不到字段**。标记【待重写-⑤-A】：统一改键名为链路图的 `pipeline_complexity_meta`，一行改动。 |
| slots：pipeline_rewrite_meta（链路图同名） | ✅ | L412-416：键名完全一致。✅ |
| slots：pipeline_mcp_hit_tool_ids（链路图同名） | ✅ | L417：键名完全一致。✅ |
| slots：F-1 initial_plan_hint / first_tool_hint 注入（链路图说仅写入 intent.slots 这两个键） | ✅ | L388-393：仅 mode=plan_execute 才写 initial_plan_hint，仅 mode=react 才写 first_tool_hint。**严格遵守 F-1 不侵入 Planner/ReAct 形参**。✅ |

### 边界 ⑥：`build_orchestrator_input(output, force_override_mode=G-1?)` → 三元组 → `IntentContext(**intent_payload_dict)`
链路图要求：
- final_mode：G-1 覆盖（strategy=="react"/"plan_execute" 时强覆盖，其他值走 pipeline 决策）
- intent_payload_dict 四个键：preferred_mode / intent / allowed_tools / slots，**必须和 orchestrator.py IntentContext dataclass 字段名 1:1**

| 字段/动作 | PASS? | 详情（文件:行号） |
|---|---|---|
| build_orchestrator_input 真的支持 `force_override_mode` 参数？ | ✅ | [agent_query_intent_pipeline.py L91-145](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/orchestration/agent_query_intent_pipeline.py#L91-L145) 签名：`force_override_mode: Optional[OrchestrationModeLiteral]=None`。✅ |
| G-1 覆盖逻辑：只有 `force_override_mode in ("react","plan_execute")` 才生效 | ✅ | L117-119：`if force_override_mode in ("react","plan_execute"): final_mode=force_override_mode else original_mode`。chat.py L152-156：前端 strategy 必须精准 `in ("react","plan_execute")`（其他值如 "auto"/"workflow" 不生效）→ 和链路图 G-1 强覆盖语义完全一致。✅ |
| **IntentContext 四个键 1:1 对齐**：preferred_mode / intent / allowed_tools / slots | ✅（关键） | pipeline L134-140：`{"preferred_mode":final_mode, "intent": primary_intent_text or "general", "allowed_tools": list(...), "slots": merged_slots}`。对照 orchestrator.py IntentContext 字段（L54-L59）：四个字段名就是 `intent, confidence, slots, preferred_mode, allowed_tools`。**Confidence 字段 pipeline 没有塞**！IntentContext 默认值是 confidence=1.0。链路图没写 confidence，但这是 IntentContext 第五个字段，默认 1.0 不抛错（见下一条）。✅ 四键对得上、没键缺失/多键（除了 confidence 是默认值）。 |
| confidence 字段：**来源是谁？**链路图没提，但 IntentContext 有 confidence=1.0 默认。【存疑-5】应不应该把 `AgentIntents.aggregated_confidence` 填进 IntentContext.confidence？ | ⚠️ PARTIAL。当前 confidence 默认 1.0（相当于「Pipeline 没提供就 100% 可信」）。链路图没写这一层，**但真实 IntentContext 明明有这个字段**。不填不影响跑，但追踪面板会显示 1.0，误导人。建议【待补充-⑥-A】在 intent_payload_dict 里加 `confidence = intents_result.aggregated_confidence or 1.0`。 |
| 三元组返回：`(final_mode, intent_payload_dict, effective_user_input)` | ✅ | L145 返回。chat.py L158：解构三元组 → `IntentContext(**intent_payload_dict)` → L185-189 传 `mode=final_mode + intent=intent_context_for_orchestrator + user_input=effective_user_input`。✅ |
| effective_user_input 兜底：rewrite 为空 → original_user_question | ✅ | L141-144：`pipeline_output.final_user_input or pipeline_output.original_user_question`。Pipeline._execute_three_stage_pipeline 已经做了一层 rewrite.rewritten 空 → original 兜底；这里是**双层兜底**。✅ |
| chat.py 侧：mode 覆盖逻辑 G-1 在 build_orchestrator_input 前就做了吗？ | ✅ | chat.py L152-156：显式过滤 request.strategy ∈ {"react","plan_execute"} → force_override_mode 字面量传进 build_orchestrator_input。链路图写的「strategy=react / plan_execute 时强覆盖 Pipeline 决策结果」完全对齐。✅ |
| **字段类型匹配**：`intent_payload_dict["allowed_tools"]` 是 `List[str]`；`IntentContext.allowed_tools: Optional[List[str]]` | ✅：上游传 list[str]，下游接受 Optional[List[str]]，协变 OK。✅ |

### 边界 ⑦：Orchestrator → ReAct / Planner 分支 + slots 消费
链路图：
- `mode == "plan_execute": _run_plan_execute() → PlannerAgent.plan(skills_prompt 合并意图锚点 + initial_plan_hint + 高级技能规约)`
- `mode == "react": _run_react() → first_tool_hint 拼成单独一行结构化引导`（而不是混进 slots Python repr）
- Orchestrator 内部再做一次 `intent.allowed_tools` 与 REGISTERED 的交集 + infrastructure 兜底

| 字段/动作 | PASS? | 详情（文件:行号） |
|---|---|---|
| 路由分支 `if mode == "plan_execute"` / `else react` + 失败降级 react | ✅ | [orchestrator.py L205-242](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/app/core/agent/orchestrator.py#L205-L242)。✅ |
| **断点 2 + 断点 4**：Planner 消费 `intent.slots["initial_plan_hint"]` + 主意图锚点 | ✅ | orchestrator.py L494-L530：`initial_plan_hint_value = intent.slots.get("initial_plan_hint")` → 拼进 skills_prompt 第二名（第一名是意图锚点）。链路图写"注入 Planner"✅。主意图锚点断点 4 也在同一位置先拼了。✅ |
| **断点 3**：ReAct 消费 `intent.slots["first_tool_hint"]` → 单独一行结构化引导 + 合法工具名校验（不使用上游未初始化变量 target_active_tool_names） | ✅ | orchestrator.py L356-L381：单独一行"【Pipeline 决策层冷启动提示】推荐的第一步工具是…" + 校验集合换成 `intent.allowed_tools`（Pipeline 已算好的白名单）。链路图写"单独结构化注入 ReAct"✅。之前 NameError 的隐患（引用未初始化的 target_active_tool_names）已在上一轮修复。✅ |
| **Orchestrator 白名单再过滤**：`_tool_names(intent)` 在 ReAct 里再做一次 allowed_tools ∩ registered + infrastructure 兜底 | ✅ | orchestrator.py L115-L132。链路图描述的是「IntentContext.allowed_tools 再经 orchestrator._tool_names 做一层交集 + infrastructure 补」——真实代码一致。✅ |
| `intent.preferred_mode 覆盖 mode` 兜底（IntentContext.preferred_mode 有值时覆盖入参 mode） | ✅ | orchestrator.py L145-147：`intent_context.preferred_mode → mode = intent_context.preferred_mode`。和 build_orchestrator_input L135 写 preferred_mode=final_mode 形成**双重保险**：chat.py 如果忘记传 mode= 关键字参数，intent.preferred_mode 兜底再覆盖。✅ |
| Memory 键名 "short_term" 与 "short term memory" 双键双写双读（断点 1） | ✅ | Orchestrator L179-187：同时写两套 key；L336-340 和 L351-354：双键兜底 `memory_context.get("short_term") or memory_context.get("short term memory") or []`。✅ |
| **冲突点 ⚠️**：Pipeline `_merge_slots` 把 mode_decision 的 initial_plan_hint / first_tool_hint **已经按 mode 过滤写入 slots**；但是 build_orchestrator_input 中 force_override_mode 把 mode 覆盖后，**没有同步重置 slots 里的 hint**。例：Pipeline 原本决策 plan_execute → slots 写入 `initial_plan_hint="步骤1/2/3"`；但前端 strategy=react 强覆盖后，final_mode=react 但 slots 里还留着 initial_plan_hint。这不是致命错误（Planner 根本就不会被调用），但会导致 **ReAct 模式拿到奇怪的 initial_plan_hint 字段 或者 plan_execute 模式没有 initial_plan_hint 但有 first_tool_hint，产生信息误导**。→ 这是 **FAIL**。标记【待重写-⑦-A】【修复优先级高】：build_orchestrator_input 中，若 force_override_mode 生效，则应：
  - 若覆盖成 react → 删除 merged_slots["initial_plan_hint"]（如果存在），保留 first_tool_hint；
  - 若覆盖成 plan_execute → 删除 merged_slots["first_tool_hint"]（如果存在），保留 initial_plan_hint。 |

### 边界 ⑧：Reflection → 「默认 reflection_enabled=True → 不通过视配置决定是否重跑 / 追加 degradation 标记 → AgentResponse」

| 字段/动作 | PASS? | 详情（文件:行号） |
|---|---|---|
| 默认 reflection_enabled=True？ | ✅ | Orchestrator.__init__ L112：`self._enable_reflection = bool(config.get("enable_reflection", True))` —**默认 True**。✅ 链路图写的 `reflection_enabled=True（默认）` 对齐。 |
| 调用 ReflectionAgent.reflect() | ✅ | orchestrator.py L271-278：反射启用且有回答就调。✅ 异常 L279-280 catch warning 并跳过。AgentResponse L289：`reflection=reflection_report`。✅ |
| **「不通过视配置决定是否重跑」的重跑逻辑有吗？** | ❌ **FAIL，完全没实现**。ReflectionAgent L158-167 有 `should_retry_or_warn(self, report)`，返回 `{warn_low_quality:bool, suggest_retry:bool, ...}`。但 orchestrator.py 调用完 `await self._run_reflection(...)` 后**根本没调用这个方法**。链路图写的 "不通过 → 视配置决定是否重跑" 当前就是一句空话。→ 标记【待补充-⑧-A】【高优先级】：在 reflection_report 返回后：
  1) 调 `decision = reflection_agent.should_retry_or_warn(reflection_report)`；
  2) 如果 `decision["suggest_retry"]` 且 `self._config.get("enable_reflection_retry", True)` → 把 `is_degraded_execution = True`；**如果还有重试预算**（需要一个 max_reflection_retries 配置，默认 0 先保守），再跑一次执行；
  3) 如果只是 `warn_low_quality` → `is_degraded_execution = True`（当前 degraded 字段在 AgentResponse L291 已经存在，但只是在 plan_execute 降级 react 时赋值为 True）。当前 degraded 完全和 reflection 脱钩。 |
| **「追加 degradation 标记」**有吗？ | ❌ **FAIL，和 reflection 脱钩**。L291：`degraded=is_degraded_execution`。但 `is_degraded_execution` 当前只在「plan_execute 失败 → 降级 react」分支（L221）里赋值为 True，和 reflection 质量结果**零关联**。→ 同上【待补充-⑧-A】修复。 |
| degraded 字段最终能否被回传到前端 AgentChatResponse？ | ❌ **FAIL：chat.py 返回 AgentChatResponse 结构里根本没有 degraded 字段**。[chat.py L202-208](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/app/api/routes/chat.py#L202-L208)：AgentChatResponse(status, session_id, trace_id, final_answer, steps_executed) 这 5 个字段，**没有 degraded / reflection_quality / passed_reflection 字段**。链路图说「Reflection 不通过追加 degradation 标记」—— 但前端根本拿不到这个标记。→ 标记【待重写-⑧-B】两步：
  (1) AgentChatResponse 追加 `degraded: bool = False` + `reflection_quality: Optional[int] = None`（Pydantic Model）；
  (2) chat.py 中 `orchestrator_result.reflection.quality_score` + `orchestrator_result.degraded` 填进去。 |

---

## 第二部分 · 与原 repo_map.txt 的「真实落地 vs 骨架待补」对照

> 对比结论：**原 repo_map.txt 中 90% 的【待补充】/【待重写】条目都已经真实落地实现了**（上一轮对话 16 子项 + JSON 8 层框架 + 断点 4 修复已经把能落地的全落地了）。现在**未落地的是【存疑的缺口方法】+【边界 ⑤/⑦/⑧ 发现的 6 条 FAIL / PARTIAL】**，下面把剩余的【待补充】【待重写】【待移除】【已落地（可从骨架删除）】重新标出来，并按修改顺序编号。

### ✅ 原 repo_map.txt【待补充】条目 —— 实际已落地（本轮骨架写 repo_map_calibrated.md 时应标记为【已实现 · 不入待办】）：
| 原编号 | 内容 | 落地位置 |
|---|---|---|
| ① | rag_constant 常量 + REGISTERED 工具清单 + INFRA 3 项 | [rag_constant.py L20-L80](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/rag_constant.py#L20-L80) |
| ② | intent_data_base.py 新增 AgentChatContext | [intent_data_base.py L200-L236](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/intent_data_base.py#L200-L236) |
| ③-1~3-5 | intent_dto.py：TaskComplexityAnalysis / AgentRewriteResult / AgentIntents / ModeDecision / AgentQueryIntentPipelineOutput | [intent_dto.py L57-L220](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/intent_dto.py) |
| ④ | rewrite/query_rewrite.py 新增 AgentQueryRewriteService 抽象 | rewrite/query_rewrite.py 已有 |
| ⑤ | rewrite/multi_question_rewrite_service.py 新增 AgentMultiQuestionRewriteService | [rewrite/multi_question_rewrite_service.py L197-L532](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/rewrite/multi_question_rewrite_service.py#L197-L532) |
| ⑦-2 | intent_classify.py 新增 AgentIntentAggregator | intent/intent_classify.py 已有 |
| ⑧ | intent_resolver.py 新增 resolve_for_agent（第二个 aggregator 参数） | [intent_resolver.py L239-L305](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/intent/intent_resolver.py#L239-L305) |
| ⑨ | orchestration/mode_decider.py 【新增文件】 | 已存在 |
| ⑩ | orchestration/__init__.py 【新增文件】 | 已存在 |
| ⑪ | orchestration/agent_query_intent_pipeline.py 【新增文件】 | 已存在 |
| ⑬ | app/api/depends/dependencies.py：_AppModelRouterIntentLLMAdapter + get_intent_llm_service + get_pipeline | [dependencies.py L78-L350](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/app/api/depends/dependencies.py#L78-L350) |
| ⑬-B | main.py lifespan：挂 PromptTemplateLoader 单例 + 预热 3 模板 | [main.py L46-L119](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/app/main.py#L46-L119) ✅ 实际已实现（原 repo_map 写的是"建议"，但真实代码有了） |
| ⑮ | prompts/orchestration-mode-decider.st | 真实文件存在（Glob 命中） |
| ⑯ | prompts/agent-question-rewrite.st | 真实文件存在（Glob 命中） |

> 结论：原 repo_map.txt 中 20+ 条【待补充】**只剩 1 条真实未落地**（见下）；绝大多数已经实现。

---

### ⏳ 2026-09-02 用户拍板后落地进度 · 骨架表
> **本轮用户决策总结**：
> - 存疑-1 = B-1（assistant 消息继续进入改写历史，chat.py 不改动）
> - 存疑-3 = A-3-1（保持 adapter "intent_analysis" → "chat" 容错，config 层补 purpose）
> - 存疑-4 = 已运行时验证 ✅（方法真实存在，⑤-X 骨架作废跳过）
> - 存疑-5 = A-5（IntentContext.confidence 填意图聚合置信度）【已落地】
> - 存疑-6 = A-6（保留 REGISTERED ∪ request_snapshot 并集不删；经核对 app/core/tools/builtin 共 13 个真实注册工具入口，REGISTERED_ENABLED_TOOL_NAMES = 9 个主工具 + PIPELINE_INFRASTRUCTURE_TOOL_SET = 3 个 file_*，白名单交集完全覆盖用户的主要工具）
> - 存疑-7 = A-7（保守策略：Reflection 质量门只打 degraded=true + 日志告警，不做执行层 1 次重跑，留未来扩展点注释）【已落地】
> - 存疑-8 = A-8（代码改键名对齐链路图 pipeline_complexity_meta）【已落地】

| 顺序 | 类型 | 编号 | 目标文件 · 方法/类 | 对标文件/方法/类 | 内容说明 | 进度 |
|---|---|---|---|---|---|---|
| **1** | 【待重写】 | ⑦-A | [orchestration/agent_query_intent_pipeline.py](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/orchestration/agent_query_intent_pipeline.py) —— `build_orchestrator_input()` | 现有同方法 L91-L145 | force_override_mode 生效后同步清理不匹配的 hint：若覆盖成 react → `merged_slots.pop("initial_plan_hint", None)`；若覆盖成 plan_execute → `merged_slots.pop("first_tool_hint", None)`。避免 mode 和 slots hint 前后矛盾。 | ✅【已落地-20260902】L134-L143 |
| **2** | 【待补充】 | ⑧-A | [app/core/agent/orchestrator.py](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/app/core/agent/orchestrator.py) —— `run()` 内 Reflection 之后分支 + `_run_reflection()` 返回改三元组 | 对标 [reflection.py L158-167 should_retry_or_warn](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/app/core/agent/reflection.py#L158-L167) | 1) `_run_reflection` 内部算好 `should_retry_or_warn()` decision，返回 (report, driver, decision) 三元组；2) run() 内 warn/suggest_retry → 置 `is_degraded_execution = True` degraded=true；3) 按 A-7 保守策略不做执行层重跑，仅日志告警 + 注释留 B-7 扩展点。 | ✅【已落地-20260902】L269-L306 / L536-L571 |
| **3** | 【待重写】 | ⑧-B-1 | **（位置纠正）** ~~app/models/schemas.py~~ → [app/api/routes/chat.py L78 AgentChatResponse](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/app/api/routes/chat.py#L78) | 原 chat.py 8 字段已有 Pydantic 类定义 | 真实 AgentChatResponse 就在 chat.py 里（不在 schemas.py），追加 `degraded: bool = Field(False, ...)` + `reflection_quality: Optional[int] = Field(None, ge=0, le=100, ...)` 两字段。 | ✅【已落地-20260902】L86-L104 |
| **4** | 【待重写】 | ⑧-B-2 | [app/api/routes/chat.py](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/app/api/routes/chat.py) —— handle_agent_chat 返回段 | 同 L222-L245 | 返回 AgentChatResponse 中把 `orchestrator_result.degraded` 填入 degraded；把 `(orchestrator_result.reflection.quality_score if orchestrator_result.reflection else None)` 填入 reflection_quality，额外加 0~100 clamp + None 防御。 | ✅【已落地-20260902】L222-L245 |
| **5** | 【待补充】 | ⑥-A | orchestration/agent_query_intent_pipeline.py —— `build_orchestrator_input()` intent_payload_dict 组装块 | 同 L160-L167 | intent_payload_dict 加键 `confidence = float(intents_result.aggregated_confidence) or 1.0` + TypeError/ValueError 兜底 + 0.0~1.0 clamp。让 IntentContext.confidence 不再硬默认 1.0。 | ✅【已落地-20260902】L146-L164 |
| **6** | 【待重写】 | ⑤-A | orchestration/agent_query_intent_pipeline.py —— `_merge_slots()` L430 | 同 L430-L457 | 键名 `pipeline_complexity` → `pipeline_complexity_meta`（与链路图一致，便于前端 debug 面板读取）。**字段名纠正**：内部 `requires_external_data` → `has_external_data_dependency`（TaskComplexityAnalysis DTO 真实字段名），并补 `need_creative_output`。 | ✅【已落地-20260902】L430-L457 |
| **7** | 【待重写】 | 【可选加固】 | orchestration/agent_query_intent_pipeline.py —— `_merge_slots()` L434-L456 | `rewrite_result.complexity_analysis.xxx` | 在上述 ⑤-A 实现中同步完成：`ca_obj = getattr(rewrite_result, "complexity_analysis", None)` + 所有字段 `getattr(ca_obj, "xxx", default)` 兜底，避免上游 None 赋值路径爆 AttributeError。 | ✅【已落地-20260902】与 ⑤-A 合并完成 |
| **作废跳过** | 【待补充】 | ⑤-X | intent/intent_classify.py AgentIntentAggregator.collect_mcp_tool_ids_from_intents | 运行时验证 True | 2026-09-02 单行 python 已验证方法真实存在（L492-L505）。骨架原 80 行会覆盖真实实现（真实取 node.mcp_tool_id，不是 node_id/metadata），作废不落地。 | ❌【作废跳过】 |
| **8** | 【待移除·只改骨架文档】 | R-① | 原 repo_map.txt 【疑问-A】推荐方案对应的说明文字（不要删代码，只从骨架移除） | 无代码改动 | "要不要新增 ORCHESTRATION kind"—— 已拍板不新增（上一轮决策）。原 repo_map.txt 这两段描述从新版骨架移除。 | ⚪【可选清理，不影响运行】 |
| **9-11** | 【待移除·只改骨架文档】 | R-②~R-④ | 原 repo_map.txt 3 段过时描述（旧 DTO 三字段名 / Optional 描述 / 疑问-B classify_targets 重写） | 真实代码已全部纠正 | 仅骨架文档层面需要清理；本轮未改 repo_map.txt，不影响运行正确性。 | ⚪【可选清理，不影响运行】 |

---

## 第三部分 · 【待补充/重写】真实代码骨架（Python）

> 所有骨架都写了【顺序 N】对应上方清单。您按顺序改即可。
>
> 不改动 ReAct/Planner/Reflection 三大执行引擎内部实现（与原 repo_map 约定一致）。
>
> 只动 Pipeline / Orchestrator 路由层 / Pydantic Model / chat.py 返回结构 / Aggregator 缺方法。

---

### 顺序 1 ·【待重写-⑦-A】G-1 模式强覆盖后 hint 清理
**文件**：`orchestration/agent_query_intent_pipeline.py`
**对标方法**：`build_orchestrator_input()` L91-L145

```python
    # 在 L121 merged_slots 之后；L134 intent_payload_dict 之前插入：
    # ---- 【待重写-⑦-A · 顺序-1】G-1 模式强覆盖后：清理与新模式不匹配的 hint 字段 ----
    if force_override_mode:
        # mode 被前端 strategy 强覆盖后，原 Pipeline 产出的 hint 可能与新模式语义冲突。
        # 例：原本 plan_execute → slots 有 initial_plan_hint；现在强制 react，
        #     再保留 initial_plan_hint 会让 ReAct 拿到无用/误导信息。
        if force_override_mode == "react":
            # ReAct 不需要 initial_plan_hint（那是 Planner 冷启动 hint）
            merged_slots.pop("initial_plan_hint", None)
            # first_tool_hint 本来就是 react 用的 → 保留
        elif force_override_mode == "plan_execute":
            # PlanExecute 不需要 first_tool_hint（那是 ReAct 第一步提示）
            merged_slots.pop("first_tool_hint", None)
            # initial_plan_hint 本来就是 plan_execute 用的 → 保留
```

---

### 顺序 2 ·【待补充-⑧-A】Reflection 不通过 → degraded + 可选重跑
**文件**：`app/core/agent/orchestrator.py`
**对标类/方法**：`AgentOrchestrator.run()` 内 L269-281（反射阶段）
**对标方法**：`ReflectionAgent.should_retry_or_warn()` reflection.py L158-167

```python
            # 5. 触发可选的独立审视与反思层（原 L269-281 替换为如下片段）
            reflection_report: Optional[ReflectionReport] = None
            # --- 【待补充-⑧-A · 顺序-2】新增：质量判定 → degraded + 可选重跑 ---
            reflection_decision: Dict[str, Any] = {}
            if self._enable_reflection and final_answer_text:
                try:
                    reflection_report = await self._run_reflection(
                        user_query=user_input,
                        answer=final_answer_text,
                        trace_id=current_trace_id,
                        steps=executed_steps,
                    )
                    # 8 层框架对齐：Reflection 有独立 should_retry_or_warn 语义判定门
                    reflection_driver_for_decision = ReflectionAgent(
                        llm=_LLMAdapter(self._model_router.get_llm("reflection")),
                        # 注意：不要重复跑 reflect()，这里只是为了拿到 should_retry_or_warn；
                        # 如果能从 _run_reflection 内部直接把 agent 引出来更好，这里兜底构造。
                        min_quality_to_pass=int(self._config.get("reflection_min_quality", 60)),
                    )
                    reflection_decision = reflection_driver_for_decision.should_retry_or_warn(
                        reflection_report
                    )
                    # ——— PASS / FAIL 路由：标记 degraded ———
                    if reflection_decision.get("warn_low_quality") or reflection_decision.get(
                        "suggest_retry"
                    ):
                        is_degraded_execution = True
                        logger.warning(
                            "Reflection 判定质量未通过：decision=%r，quality=%r，自动标记 degraded=true",
                            reflection_decision,
                            reflection_report.quality_score,
                        )
                    # ——— FAIL → Retry：如果配置启用重试（默认先 0 次，保守）——
                    max_reflection_retries: int = int(
                        self._config.get("reflection_max_retries", 0)
                    )
                    if (
                        reflection_decision.get("suggest_retry")
                        and max_reflection_retries > 0
                        and not self.__dict__.get("_reflection_retried_already")
                    ):
                        # 保守实现：仅标记 1 次重试，并重新跑分支（react / plan_execute），
                        # 这里先只做 degraded 标记，真正的 rerun 逻辑需要您确认重试预算
                        # （是否要重新从 Pipeline 决策阶段跑？还是只跑执行层？）
                        # → 对应下方【存疑-7】，等您拍板后再补真实 rerun 代码。
                        object.__setattr__(self, "_reflection_retried_already", True)
                except Exception as reflection_error:
                    logger.warning("自我反思审计阶段执行异常，已自动跳过兜底: {}", reflection_error)
            # -----------------------------------------------------------------
```

---

### 顺序 3 ·【待重写-⑧-B-1】`AgentChatResponse` Pydantic Model 补字段
**文件**：`app/models/schemas.py`
**对标**：chat.py L202-L208 引用的 AgentChatResponse。

```python
# 【待重写-⑧-B-1 · 顺序-3】在 AgentChatResponse 已有的 5 个字段下追加 2 个：
class AgentChatResponse(BaseModel):
    # ... 已有 5 字段（status, session_id, trace_id, final_answer, steps_executed）保持不变 ...

    # ===== 新增：degradation 标记（链路图边界 ⑧ 要求）=====
    degraded: bool = Field(
        default=False,
        description="（Agent 接口）编排是否发生了降级。True 的场景："
                    "PlanExecute 失败 → 改用 React 兜底；或 Reflection 质量判定低质量/建议重试。"
                    "前端可根据该字段给用户'本次回答为降级结果'的轻提示。",
    )
    reflection_quality: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="（Agent 接口）Reflection 阶段给出的回答质量评分（0~100）。"
                    "启用 reflection 时有值；未启用或 Reflection 调用失败为 None。",
    )
```

---

### 顺序 4 ·【待重写-⑧-B-2】chat.py 把 degraded / reflection_quality 填进响应
**文件**：`app/api/routes/chat.py`
**位置**：handle_agent_chat L202-L208

```python
    # 【待重写-⑧-B-2 · 顺序-4】在原返回构造里补 degraded + reflection_quality：
    return AgentChatResponse(
        status="success",
        session_id=active_session_id,
        trace_id=orchestrator_result.trace_id,
        final_answer=orchestrator_result.answer,
        steps_executed=len(orchestrator_result.steps),
        # ===== 新增：边界 ⑧ Reflection Degradation =====
        degraded=bool(getattr(orchestrator_result, "degraded", False)),
        reflection_quality=(
            int(orchestrator_result.reflection.quality_score)
            if orchestrator_result.reflection is not None
            else None
        ),
    )
```

---

### 顺序 5 ·【待补充-⑥-A】IntentContext.confidence 显式填值
**文件**：`orchestration/agent_query_intent_pipeline.py`
**对标**：build_orchestrator_input() L134-L140

```python
        # 【待补充-⑥-A · 顺序-5】补充 confidence：
        intent_context_payload: Dict[str, Any] = {
            "preferred_mode": final_mode,
            "intent_classify_resolver": pipeline_output.intents_result.primary_intent_text
            or "general",
            # ===== 新增：从意图聚合结果填 confidence（默认 1.0 兜底）=====
            "confidence": float(
                getattr(pipeline_output.intents_result, "aggregated_confidence", 1.0)
            ) or 1.0,
            "allowed_tools": list(pipeline_output.allowed_tools_final or []),
            "slots": merged_slots,
        }
```

---

### 顺序 6 ·【待重写-⑤-A】`pipeline_complexity` → `pipeline_complexity_meta` 键名统一
**文件**：`orchestration/agent_query_intent_pipeline.py`
**位置**：`_merge_slots()` L402

```python
        # 【待重写-⑤-A · 顺序-6】键名统一为链路图的 pipeline_complexity_meta：
        # 原：merged["pipeline_complexity"] = {
        # 改：
        merged["pipeline_complexity_meta"] = {
            ...  # 内部 5 个字段保持不变
        }
```

---

### 顺序 7 ·【待补充-⑤-X】Aggregator 缺失的 collect_mcp_tool_ids_from_intents **→ 已验证真实存在 · 骨架作废（不要落地）**
**文件**：`intent/intent_classify.py`
**对标类**：`AgentIntentAggregator`
**对标调用点**：Pipeline L226-230 `self.intent_aggregator.collect_mcp_tool_ids_from_intents(intents_result)`

> **2026-09-02 验证结论**：运行 `python -c "from query_intent.intent.intent_classify import AgentIntentAggregator; print('collect_mcp_tool_ids_from_intents' in dir(AgentIntentAggregator))"` → **True**。真实方法定义在 [intent_classify.py L492-L505](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/intent/intent_classify.py#L492-L505)，实现逻辑：遍历 `agent_intents.mcp_node_scores` → 取 `node_score.node.mcp_tool_id` 属性 → 去重保序返回 List[str]。✅ **本条目【待补充-⑤-X】作废，跳过不写。**

> 原下方预留的 80 行 Python 骨架不再复制，避免与真实实现冲突覆盖（真实实现使用 `node_score.node.mcp_tool_id`，不是 `NodeScore.node_id / metadata`，两者数据来源不同，若落地下方旧骨架会引入错误交集逻辑）。

---

### 顺序 7（原 8）·【待重写·可选加固】rewrite.complexity_analysis None 防御加固
**文件**：`orchestration/agent_query_intent_pipeline.py`
**对标方法**：`_merge_slots()` L403-411
**对标类**：[AgentRewriteResult.complexity_analysis default_factory](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/intent_dto.py#L108)

```python
        # 【待重写·可选加固 · 顺序-7（原 8）】complexity_analysis 访问 None 防御
        # 原：直接访问 rewrite_result.complexity_analysis.estimated_steps
        # 改：加一层 getattr 兜底（default_factory 不会 None，但上游被手工赋 None 时防御）
        ca_obj = getattr(rewrite_result, "complexity_analysis", None)
        merged["pipeline_complexity_meta"] = {  # 配合【待重写-⑤-A】键名
            "estimated_steps": int(getattr(ca_obj, "estimated_steps", 1) or 1),
            "estimated_tool_calls": int(getattr(ca_obj, "estimated_tool_calls", 0) or 0),
            "has_multi_step_dependency": bool(
                getattr(ca_obj, "has_multi_step_dependency", False)
            ),
            "requires_external_data": bool(getattr(ca_obj, "requires_external_data", False)),
            "reasoning_notes": str(getattr(ca_obj, "reasoning_notes", "") or ""),
        }
```

---

## 第四部分 · 存疑点（需要您拍板）

下面 7 条是静态追踪中**无法自行决策、决定会影响全局行为**的点。请逐项回复策略（或默认按推荐 A 走）：

| 编号 | 问题 | 选项 A（推荐） | 选项 B（其他） | 影响的骨架条目 |
|---|---|---|---|---|
| **存疑-1** | 边界① chat.py → IntentChatMessage 转换时，目前把 assistant 消息也传进改写阶段。AgentChatContext 文档 L215 写的是「只读 USER 消息用于续问还原」，链路图也是「历史 USER 消息」含义。**实际会不会引发改写阶段 LLM 从历史答案里抄实体（信息泄露）？** | **A-1：改写阶段输入只保留 USER 消息**：在 chat.py L127-133 循环里把 `elif assistant` 分支删掉，只 append IntentChatMessage.user(m.content)，符合文档语义。 | B-2：保持现状，assistant 消息也传入——允许 LLM 根据历史答案做指代消解（风险=偶尔抄答案实体，但指代消解更准）。 | 不进骨架（是 chat.py 代码小改动），但直接影响 Stage1 改写质量 |
| **存疑-3** | 边界③ dependencies.py `_AppModelRouterIntentLLMAdapter.chat()` 中，purpose 首选 "intent_analysis"，失败才回退 "chat"。您部署的 ModelRouter 配置中**是否真的声明了 "intent_analysis" Purpose？**（如果线上没声明，每次都先走一次异常 catch 再回退，性能多 1 次 for_purpose round-trip + 1 次 Exception） | **A-3-1：保持现状容错**，您在真实环境的 `config.yaml / toml` 里显式补一个 "intent_analysis" Purpose（走专用便宜/快模型，省钱又快）。**推荐**。 | B-3-2：把默认 purpose 改成 "chat"，取消 "intent_analysis" 首次尝试——不会有 Exception，但所有意图/改写/模式决策都走大模型 chat purpose，成本略高、延迟略高。 | 无骨架条目，是 config 层 vs adapter 默认值的二选一 |
| **存疑-4 · ✅ 已验证·结论 PASS** | ~~边界⑤ `AgentIntentAggregator.collect_mcp_tool_ids_from_intents()` 真实实现里有没有？~~ → **结论：已实现**。2026-09-02 运行单行：`python -c "from query_intent.intent.intent_classify import AgentIntentAggregator; print('collect_mcp_tool_ids_from_intents' in dir(AgentIntentAggregator))"` → 返回 **True**；位置 [intent_classify.py L492-L505](file:///d:/pycharm/PyCharm%202026.1.1/PythonProject_deepagents/project-python/query_intent/intent/intent_classify.py#L492-L505)：遍历 `agent_intents.mcp_node_scores[*].node.mcp_tool_id` 去重保序。**本条目关闭，不需要您再跑命令。** | ——（无需选） | —— | 【待补充-⑤-X】已作废跳过。 |
| **存疑-5** | 边界⑥ IntentContext.confidence。链路图没写要填 confidence，但真实 IntentContext 有字段 `confidence: float = 1.0`（默认 1.0）。**填不填？**推荐 A：骨架里已按「填 AgentIntents.aggregated_confidence（意图聚合的置信度）」写了。您觉得对吗？ | **A-5：填意图聚合置信度**（跟骨架【待补充-⑥-A】一致）。 | B-5：填 ModeDecision.confidence（模式决策的置信度）。 | 【待补充-⑥-A】 |
| **存疑-6** | 边界⑤ MCP 白名单交集算法：当前允许 `并集 ∩ (REGISTERED ∪ request_snapshot_tools)`。**request_snapshot_tools = tool_registry.list_tool_names()**，内容其实和 REGISTERED 是完全相同的 9 个工具。`REGISTERED ∪ request_snapshot` 其实就等于 REGISTERED。**那多此一举办不办？** | **A-6：保持当前多此一举**——因为未来 request_snapshot 可能会是「会话级白名单（比如某用户禁用了某个工具）」，链路图里的 `request_snapshot_tools` 概念本身就预留了这种扩展；现在虽然相等，不影响正确性。✅ 不用改。 | B-6：删掉 `request_snapshot_tools` 这一支，直接只对 REGISTERED 做交集。 | 【待补充/重写】都不影响 |
| **存疑-7（边界⑧ 最大策略问题）** | 【待补充-⑧-A】中 Reflection.suggest_retry=True 时，**重试预算怎么执行**？当前骨架只做了 degraded 标记，没做真实 rerun。选项： | **A-7：推荐保守**：不做执行层重跑，只做 degraded=true 标记 + 前端告警。下次迭代再做真实重跑。**风险最低**。 | B-7：跑执行层重跑（最多 1 次）——重跑时保留 `mode`、把 `degraded 原因 + suggest_retry 报告` 注入 `intent.slots["reflection_retry_feedback"]` 给 Planner/ReAct 当 hint。性能影响最大（1 次完整 Agent 循环）。 | 【待补充-⑧-A】 |
| **存疑-8（最后确认链路图矛盾）** | 边界⑤链路图写 slots key 是 `pipeline_complexity_meta`，真实代码是 `pipeline_complexity`（少 `_meta`）。【待重写-⑤-A】就是改真实代码对齐链路图。**有没有前端/其他调用方已经在用这个少后缀的 key？** | **A-8：真实代码改键名对齐链路图**（【待重写-⑤-A】实现）。 | B-8：链路图迁就代码，您在链路图自己的本地笔记里把 `pipeline_complexity_meta` 改成 `pipeline_complexity`。 | 【待重写-⑤-A】 |

---

## 第五部分 · 是否需要新增文件？（请您选）

| 候选文件 | 用途 | 我建议？ |
|---|---|---|
| `tests/test_orchestration_8_boundary.py`（pytest） | 针对 8 边界的离线链路测试：Mock 所有 LLM/工具/记忆，跑 chat.py → Pipeline → Orchestrator 全程，断言每条边界 FAIL/PASS 条件（包括 ⑦-A hint 冲突 / ⑧-A degraded / ⑤-X MCP 白名单 / ⑥-A confidence 填充）。不依赖真实 Provider，每次改完都能秒级回归。 | ✅ **强烈建议新增**。当前所有链路全是静态断言、真实端到端一旦字段名漂移 0 防护。文件可写在 `query_intent/tests/` 目录下，全部 mock 不碰外部资源。 |
| `orchestration/checklist.py`（lint 用）| 把这份 8 边界审计表写成一份 50~100 行的 assert 清单，启动应用时做一次自检，发现字段名不对 / collect_mcp 没方法 / IntentContext 四键缺失就显式 WARN。不用任何依赖。 | 可选。优先级低于测试文件。 |

---

## 附录：边界 ⑦ 的一处隐形矛盾（附带修复方案已写进 顺序 1 骨架）

> 仅给您作为结论备忘：
>  **原链路图 G-1："前端 strategy 覆盖 Pipeline 决策 mode"** ✅
>  **原链路图 F-1："mode=plan_execute 才写 initial_plan_hint，mode=react 才写 first_tool_hint 进 slots"** ✅
>
> 但**两者连在一起就有 bug**：Pipeline 原本决策 plan_execute → slots 写了 initial_plan_hint；现在前端 strategy=react 强覆盖，ReAct 拿到的 slots 里还留着 initial_plan_hint。这不是错误，但会给调试日志/后续 hint 消费造成混乱。【顺序 1】骨架已修复：强覆盖后清理掉不匹配的 hint 字段。
