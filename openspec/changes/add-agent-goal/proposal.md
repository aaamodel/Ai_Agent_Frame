## Why

上一个变更给执行期的模型补上了「计划台账」（每个子任务的标识 / 要解决的问题 / 工具 / 执行状态 / 是否解决）。但台账只回答了"**每个子任务各自怎么样了**"，没有回答"**这一切是为了达成什么**"。

后果是模型缺少判断依据：它在被要求决定「跳过某个子任务」或「提前收尾」时，只能看到一堆局部状态，无法判断某一步是否与最终目标无关、也无法判断上下文是否已经收集齐全。**缺的是一个目标锚点，不是更多状态。**

`agent_goal` 就是这个锚点：一句话说明本轮要交付什么。它必须由改写阶段产出（那里是唯一同时看到用户原问题、多轮历史与全部上下文的地方），并在每一轮调用中随上下文注入。

## What Changes

1. **改写阶段产出 `agent_goal`**：在既有改写结构化输出中新增一个字段——**一句话**描述本轮要达成的最终目标（交付物/结论形态）。复用既有的「改写+意图」组合调用，**不新增任何模型调用**。
2. **承载进状态**：`agent_goal` 随改写结果进入 Pipeline 的 `slots`，进而进入图状态的 `intent.slots`，可用于所有节点。
3. **全链路注入**：在三个决策点注入同一份目标——ReAct 系统附加段（两种协议）、plan 规划提示词、**子任务提炼（计划台账块的首行）**。
4. **台账承载方式**：目标作为台账块的**首行**呈现，而**不是表格中的一列**——它不是某个子任务的属性，而是所有子任务共同的约束；放进表格会让每一行重复一遍同样的文字。
5. **兜底**：改写未产出该字段时，回退为「改写后的问题」截断，保证注入永不为空。

**明确不做**：不引入第二条模型调用；不让 `agent_goal` 变成第二个「改写后问题」（它描述的是最终交付物，不是问题的复述）。

## Capabilities

### New Capabilities

- `agent/agent-goal`: 本轮 agent 目标的产生、承载与全链路注入——确保每个决策点都持有同一个目标锚点，使「跳过子任务」与「提前收尾」有可依据的判断基础。

### Modified Capabilities

（无既有 spec 需要修改）

## Impact

- `app/query_intent/llm_schemas.py`：只需在**基类** `AgentRewriteSchema` 新增 `agent_goal` 字段——主链路使用的 `AgentRewriteIntentCombinedSchema` **继承**它，会自动获得该字段（切勿在两处重复添加）。⚠️ 该 schema 的所有 Field description 会**逐字进入 response_format**（实测整份 ≈3.8KB / 1.1k token），故描述必须极简。
- `app/query_intent/prompts/agent-rewrite-intent-combined.st`：**主链路**（「改写+意图」组合调用）的输出格式段与零容忍校验段同步（要求简短、禁止复述原问题）。
- `app/query_intent/prompts/agent-question-rewrite.st`：**降级链路**（组合调用失败/返回空时，回退父类两段链路）的同一段落同步。两个提示词文件都必须改——只改主链路的话，一旦降级该字段就消失，而下游注入点拿到的只会是兜底值。
- `app/query_intent/intent_dto.py`：`AgentRewriteResult` 新增同名字段。
- `app/query_intent/rewrite/multi_question_rewrite_service.py`：解析时回填该字段（含兜底）。
- `app/query_intent/intent_3stage_pipeline/agent_query_intent_pipeline.py`：`_merge_slots` 写入 `slots["agent_goal"]`。
- `app/core/agent/graph/nodes/_common.py`：`build_extra_system`（ReAct 两种协议共用）注入目标；`render_plan_ledger` 支持目标作为首行。
- `app/core/agent/graph/nodes/plan_node.py` / `execute_node.py`：规划提示词与子任务提炼提示词注入目标。
- **评测影响**：改写调用的输入 schema 略微增大（每次调用 +约 120 字节）；工具评测的端到端 token 会相应微增，需在基线上记录。
