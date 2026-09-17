## Why

plan_execute 链路当前有两个独立缺陷，共同把单轮成本推到 **21.5 次 LLM 调用 / 67,819 tokens**（红线 6000 的 11.3 倍）：

1. **执行过程是"蒙眼"的**。每个子任务执行时，模型只拿到「当前任务描述 + 历史结论文本（`ctx_str`，只有已完成步骤的 conclusion）」，既看不到计划全貌（还剩几个 task、每个 task 解决哪个问题、用什么工具），也看不到各步成败状态。它因此**没有任何依据**去判断"答案已经够了"。
2. **计划一旦生成就无法收缩**。没有提前结束、没有跳过任务：`route_after_execute` 只看 `cursor < len(plan)`，必须把所有 task 跑完。即使第 2 步已取得充分证据，剩余 task 仍会逐个执行（每个 = 1 次工具调用 + 1 次提炼 LLM）。

叠加前一天引入的第三个问题：将「多个子问题可由同一次工具调用覆盖时必须合并」作为规划约束后，合并一旦未能同时覆盖全部子问题，就会有子问题无答案，而链路没有任何机制能发现这种遗漏。**子问题的覆盖可靠性优先于合并省下的 token**。

## What Changes

1. **回退子问题合并约束**：改回「每个子问题必须派发自己的子任务」（`plan_node` 的规划约束提示词），消除合并导致的漏答风险。
2. **新增计划台账视图**：每步执行前，把 plan 全貌渲染给模型——每行包含 `id / 该 task 解决哪个问题 / 调用什么工具（或无工具）/ 当前状态`。台账**从 `plan` + `subtask_results` 推导**（纯渲染函数），不维护第二份状态，避免双份状态漂移。
3. **新增控制协议 `SubTaskOutcomeSchema`**：子任务提炼那次 LLM 调用改为结构化输出，在结论之外附带 `next_action`（continue / finish）与 `skip_task_ids`。**复用已经必然发生的那次调用，不新增 LLM 调用**（对比"注册 end 工具"需要模型额外发起一轮）。
4. **state 新增 2 个字段**：`skipped_task_ids`（模型决定跳过的 task id）、`early_finish`（是否"答案够了"提前收尾，留痕审计用）。这是唯一真正需要新增的状态——其余信息（状态、工具）都可从现有数据推导。
5. **`route_after_execute` 支持跳转**：计算"下一个未被跳过的下标"推进游标；`finish` 归一为"跳过剩余全部"，因此不需要单独的路由分支。

**明确不做（范围约束）**：不提供新增 task 的能力，不提供修改已有 task 内容的能力——只允许「跳过」与「结束」。

## Capabilities

### New Capabilities

- `agent/plan-subtask-allocation`: 子问题与子任务的分配规则——每个被拆分出的子问题必须拥有独立对应的子任务，禁止为节省调用而合并。
- `agent/plan-execute-control`: 计划执行期的模型可控性——计划台账可见、可跳过指定子任务、可在证据充分时提前收尾。

### Modified Capabilities

（无既有 spec 需要修改）

## Impact

- `app/core/agent/graph/state.py`：新增 `skipped_task_ids` / `early_finish` 两个字段及初始值。
- `app/core/agent/graph/nodes/execute_node.py`：台账渲染 + 提炼调用改为结构化输出 + 写入跳过/结束状态。
- `app/core/agent/graph/builder.py`：`route_after_execute` 计算下一有效下标，支持跳过与提前收尾。
- `app/core/agent/graph/nodes/plan_node.py`：回退合并约束的提示词。
- `app/query_intent/llm_schemas.py`：新增 `SubTaskOutcomeSchema`（沿用 `SummaryVerdictSchema` 的范式与三档 response_format 降级链）。
- **评测口径**：跳过/提前结束会减少 `steps` 条数，需确认 `tool_success_rate` 与 `key_arg_recall` 的统计语义不受影响（只统计实际执行的步骤）。
- **自愈依赖**：提前收尾若导致证据不足，由既有 L3 闸门（`summarize_node` 的 `sufficient` 判定）与 `replan` 兜底，无需新增硬规则。
