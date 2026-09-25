## Why

ReAct 多轮工具调用中，原始观测（尤其 RAG 每轮 5 条长片段）逐字累积进消息历史、每轮重发，实测 3 轮检索的最终答案轮 input 已达约 5000 token，且随轮次线性膨胀。现有压缩 `compact_tool_observations` 是纯**位置型硬截断**（最近 3 条全文、更早的压成 200 字首段），不看内容：关键证据若不在最近 3 条或不在块首部就会丢失，模型被迫带着残缺证据作答或重复检索，成本与错误率同时上升。

## What Changes

- 新增请求级**证据归一化层**：每次工具观测产生后，按工具类型确定性切分为统一的证据单元（Unit），只切分不改写、零 LLM 调用。
- 新增确定性证据管道：jieba 词项 + 自实现 BM25/实体/结构先验/新鲜度的固定权重打分 → 精确 hash + SimHash + 短文本 Jaccard 跨轮去重 → 按轮收紧的字符预算与 MMR 选择，每轮从全量 Unit 重算**证据板**。
- 错误/状态块不参与打分但**恒保留一行**；低分块保留"一行索引"，保证失败与被省略内容可感知。
- 新增**短观测豁免**：整条工具观测 ≤300 字符时不切分、不打分，整体作为豁免单元原文进入 ReAct 发送视图并保留 3 个轮次（不占证据板字符/条数预算，3 轮内总量有界）；超期后收敛为一行桩（工具名 + 原始字符数 + 证据编号 + 首句预览），模型仍可按编号回取；豁免单元照常参与 sha1 精确去重，重复观测（如反复列同一目录）不得占用原文保留名额。
- 新增 `fetch_evidence` **按需回取**能力：模型可用证据编号取回被省略块原文或相邻块，回取只读本请求已保存的观测、不重新调用外部工具；仅在 ReAct 路径注入。
- 新增 plan_execute **延迟一步回取**：子任务提炼输出协议新增独立字段 `requested_evidence_uids`（与选择候选资产/工具方向的 `selected_alternative_id` 语义分离），模型可索取本步被省略证据的全文；节点据此插入一个不调用外部工具的内部恢复步，从本请求已保存观测回填后再提炼，复用既有纠偏配额/留痕/链式闸门，每个子任务最多恢复一次。
- 替换三处 LLM 发送视图的观测组装方式：ReAct FC、文本协议 ReAct、plan_execute 子任务提炼（后者从 6000 字头部盲截断改为按相关性选择，≤300 字短观测原文直出，被省略单元支持延迟一步回取）。原始观测仍全文保存在现有 state 中，不引入新存储基础设施。
- 证据层任何环节异常时自动降级回现有压缩策略，主链路不中断。
- 新增每轮 trace 埋点（上板数/字符数/判重/低相关/无新增/回取/豁免/延迟恢复），用于真实会话验证压缩率与召回率。

## Capabilities

### New Capabilities

- `agent/evidence-board`: 请求内工具观测的确定性归一化、相关性打分、跨轮去重、按轮预算选择与证据板渲染；短观测的限期原文豁免与桩化；证据编号化的按需回取（ReAct 即时回取 / plan_execute 延迟一步回取）；以及 ReAct/plan_execute 三条 LLM 发送视图的证据呈现与降级约束。

### Modified Capabilities

<!-- 无。plan-execute-control 只约束计划台账/跳过/提前收尾，未约束子任务提炼时观测的截断与呈现方式；本次对提炼视图的改变由新能力 evidence-board 承载。 -->

## Impact

- 新增包 `app/core/agent/evidence/`（models / chunkers / pipeline / view / fetch，均为纯函数或节点层拦截）。
- 修改 `app/core/agent/graph/state.py`（新增 `evidence_units`、`evidence_meta` 两个请求级字段）。
- 修改 `app/core/agent/graph/nodes/execute_node.py`（FC 循环、文本协议循环、plan 提炼段三处观测入管与视图组装；fetch_evidence 定义注入与节点层拦截）。
- 修改 `app/core/agent/graph/nodes/_common.py`（`compact_tool_observations` 保留为降级 fallback；`compact_history_lines` 文本协议视图切换）。
- 不新增第三方依赖（BM25/SimHash 纯 Python 自实现，jieba 已在用）；不新增存储；不变更对外 HTTP/SSE API；不改变工具注册表（fetch_evidence 不注册为业务工具）。
- 测试：新增证据管道单元测试与 3 轮 RAG 夹具的端到端 token/召回验收；现有全量测试（基线 383 passed）不得回归。
