# LLM 生成过程逐字流式输出（设计规格）

日期：2026-09-20
状态：待审阅
涉及端点：`POST /api/v1/chat/with_agent`

---

## 1. 目标

把 Agent 工作任务的**模型生成过程**逐字推给前端，覆盖两处：

1. **问题改写 + 意图识别**（Pipeline Stage 1 的 LLM 调用）——逐字显示「改写后的问题」
2. **最终答案**（ReAct 节点的 LLM 调用）——逐字显示答案正文

### 为什么做

现状是**伪流式**。实测一次真实请求（打时间戳）：

```
step 事件    3 个     首个 48.67s → 末个 66.86s     ← 真流式，跨度 18 秒
content 分片 94 片    68.61s → 68.63s              ← 全部在 0.016 秒内到齐
```

工具过程（`step` 事件）已经是真流式；但**最终答案是整段生成完再按 8 字符切片**
（`chat.py:301-319` 的 `_stream_final_answer`，chunk 之间只有 `asyncio.sleep(0)`，
无真实时间间隔），94 片 / 752 字在 16 毫秒内一次性到达，用户看到的是"啪一下整段出现"。

Pipeline 更是**完全没有过程输出**——它整段跑完才返回，期间界面无任何反馈。

### 成功标准

**唯一的标准是"用户更早看到有意义的输出"**（用户已明确：目标就是耗时）。

- ✅ 通过：一次真实请求中，`rewrite` 阶段的第一个 delta 在 Pipeline 的 LLM 开始
  产出后即可见；`answer` 阶段的第一个 delta 在模型开始生成答案时即可见，
  **而不是等整段生成完**
- ❌ 不通过：仍然"整段生成完再一次性出现"；或端到端总耗时显著增加
- 本设计**不追求**"减少总耗时"——总耗时（模型生成时间 + 工具时间）不变；
  改变的是**感知等待**（把"黑屏"变成"边生成边看"）

---

## 2. 约束（按用户澄清后的真实意图重述）

用户原话：

> 不要那么死板，我指的不能改变原来 LLM 调用次数跟调用策略，是指**不会大幅增加整个 agent 的耗时**，
> 目标是**耗时**。像这种探测包式的调用可以不算重试。
> 如果有厂商不支持流式输出那就**出结果后一起渲染**。
> 为什么说不要改变原来 LLM 调用策略呢，因为**我怕改变 LLM 调用策略要大幅改动后面的
> rewrite question 的解析或 intent 的解析**，只要不大幅改动这些就可以。

归纳为三条真正的约束：

1. **不显著增加端到端耗时**——这才是根本目的。判断任何设计取舍都要以"用户能否更早
   看到输出"为准，而不是数调用次数。
2. **不破坏 `rewrite question` 与 `intent` 的解析路径**。这是用户最担心的点：
   改写结果的 schema 校验、容错解析、失败回退链路（`AgentRewriteIntentCombinedSchema`
   的 `model_validate_json` / `coerce_llm_json_to_schema` /
   `validate_tolerating_agent_goal`）**必须零改动**。
   本设计把全部提取逻辑放在显示层，业务解析路径完全不碰——满足该约束。
3. **不改变模型路由与候选顺序**：tier 解析、`PURPOSE_TIER_MAP`、熔断、重试策略保持原样。

**明确允许的事**：

- **探测包式的额外请求不算"增加调用次数"**（用户明确认可）——见 10.1 的方案 A
- 厂商不支持流式时，**退回非流式、结果出来一次性渲染**（用户明确要求的行为）
- 可以改的：**输出方式**（怎么把结果送出去）与**观测/追踪**（能看到什么）

**本设计不引入** `chat.py:722-737` 注释里建议的 `AsyncStreamRoutingExecutor`——
该建议的前提是"流式要自走候选降级"，而本设计不改路由，流式只是旁路观测。
（用户已否决改动路由层，故该注释描述的问题对本设计不适用。）

---

## 3. 后端现状（已核实的事实）

### 3.1 模型调用链（4 层）

```
execute_node.py:792        deps.model_router.chat_with_tools(...)
  → model_router.py:276      chat_with_tools：注入 tools/tool_choice 后转调 chat
    → model_router.py:208      chat：tier 解析 → select_chat_candidates → executor
      → async_model_executor.py:120   execute_with_candidate_fallback
                                       for target: allow_call → _call_with_attempts → mark_success/failure
        → async_model_executor.py:232   _call_with_attempts：预算 = tier.timeout_ms，重试 = tier.retries
          → async_model_executor.py:55    run_with_attempt_budget：attempts = 1 + retries
            → async_openai_caller.py:356    async_openai_chat_caller：组装 params
              → async_openai_caller.py:177    _chat_completion_with_langfuse
                                              → await client.chat.completions.create(**params)
```

补充事实：

- `client_resolver` = `ModelRouter._resolve_client`（`model_router.py:323-324`），
  按 `model_id` 查表，多候选即多客户端（`model_router.py:176-187` 预构建）
- 熔断时机：`allow_call`（`async_model_executor.py:168`）→ 调用 → `mark_success`（`:213`）
  或 `mark_failure`（`:185`）
- 返回类型 `AsyncOpenAICallResult`（`async_openai_caller.py:48-70`）字段：
  `content` / `model_id` / `usage` / `raw` / `tool_calls` / `reasoning_content`

### 3.2 调用器不支持流式

```python
# async_openai_caller.py:384-385
if stream:  # 兼容上层偶发 stream=True，但本调用器是非流式的
    stream = False
```

全仓 `stream=True` 只有两处：`chat.py:757`（`/chat/stream` 的真实流式）
与上面这处被强制关闭的地方。

`/chat/stream` 用的是**独立 client、完全绕过 ModelRouter**（`chat.py:744-758`），
本设计**不复用该做法**（它会引入第二条调用链）。

### 3.3 四个 LLM 调用点的输出形态

| 调用点 | 位置 | `response_format` | 输出形态 |
|---|---|---|---|
| 改写 + 意图 | `combined_rewrite_intent_service.py:179-181` | `AgentRewriteIntentCombinedSchema`（`:274-276`） | 结构化 JSON |
| 最终答案 · FC 协议 | `execute_node.py:792-799`，答案取 `:805-806` 的 `resp.content` | 无 | 纯文本 |
| 最终答案 · text 协议 | `execute_node.py:978-981`，解析见 `react_agent.py:235-240` | 无 | ReAct 草稿（`Thought:`/`Action:`/`Final Answer:`） |
| 汇总 · plan 路径 | `summarize_node.py:204-211` | `SummaryVerdictSchema`（`:46-72`） | 结构化 JSON |

**关键事实：改写与意图识别是同一次 LLM 调用**（`combined_rewrite_intent_service.py:60`
类文档："「改写 + 意图识别」单次 LLM 调用的组合服务"；`agent_query_intent_pipeline.py:233`
注释："组合调用已在 Stage1 单次 LLM 中产出逐问题意图打分，此处零 LLM 直接聚合"）。
因此这两者**只能同时出现**，做不到"先改写完、再意图完"。

### 3.4 线程模型（本设计最实的技术难点）

| 位置 | 执行上下文 |
|---|---|
| Pipeline | **同步**，被 `chat.py:415-422` 的 `asyncio.to_thread(pipeline.run, ...)` 丢到工作线程；其内部的意图适配器还自己 `asyncio.run(...)`（`dependencies.py:190-196`） |
| 图节点 | async，跑在主事件循环 |

因此把 token 推回 SSE **必须跨线程**。

### 3.5 LangGraph 能力

- 声明版本 `langgraph>=1.0.0`（`pyproject.toml:20`）；实测安装 **1.2.11**
- 当前只用 `stream_mode="updates"`（`runner.py:248`、`runner.py:327`）
- **全仓无 `get_stream_writer` 用法，无 `stream_mode="custom"`**

本设计**不使用** `get_stream_writer`：它只能覆盖 async 的图节点，覆盖不了
同步的 Pipeline（需跨线程），会导致两套机制并存。改用统一的上下文变量（见 4.1）。

### 3.6 既有约定（不影响本设计，但需说明）

`chat.py:722-737` 注释要求"流式接入熔断需新增 `AsyncStreamRoutingExecutor`"。
该约束的前提是"流式要自己走候选降级"。本设计不改路由，流式只是旁路观测，
因此**不需要也不引入**该执行器。

---

## 4. 设计

### 4.1 旁观通道（`app/core/agent/stream_sink.py`，新增）

一个模块级 `ContextVar`，保存"当前是否有人在听模型输出"：

```
ContextVar[StreamSink | None]
Sink 协议：push(event: dict) -> None   # 实现方负责线程安全
current_sink() -> StreamSink | None
use_sink(sink) -> 上下文管理器（进入时 set，退出时 reset）
```

**为什么用 `ContextVar` 而不是显式参数**（用户已选）：

- `asyncio.to_thread` 会**复制当前上下文**到工作线程，因此 Pipeline 那条同步链路
  也能读到同一个通道——一套机制覆盖两种情况
- 不需要改动 `ModelRouter` / `AsyncModelRoutingExecutor` 的函数签名
  （这两层有 `测试/test_llm_attempt_budget.py` 覆盖）

### 4.2 调用器的改动（唯一被改的**模型层**函数）

`async_openai_chat_caller`（`async_openai_caller.py:356`）内部：

```
sink = current_sink()
if sink 为 None:
    走今天完全相同的非流式路径（一行不变）
else:
    params["stream"] = True
    async for chunk in create(**params):
        累积 content / tool_calls 增量 / usage
        sink.push({"kind": "delta", "text": <本片文本>})
    用累积结果构造与今天**完全相同**的 AsyncOpenAICallResult 返回
```

**核心性质：对外契约不变**——调用器仍然"一次调用、一个完整结果"。
因此上游的 `run_with_attempt_budget`（重试）、`execute_with_candidate_fallback`
（候选降级）、熔断回写、以及节点里的结构化解析与全部闸门，
**一行都不需要改**。LLM 调用次数不变。

需要一并处理的细节：

- `async_openai_caller.py:384-385` 的强制 `stream = False` 改为"仅当无通道时强制"
- `tool_calls` 的流式增量需按 `index` 聚合还原成今天的结构化列表
  （`async_openai_caller.py:461-472` 的形态）
- **langfuse 包装**：`_chat_completion_with_langfuse`（`:177-223`）不感知流，
  流式分支直接调 `client.chat.completions.create`。
  **这是本次唯一的观测能力损失**——流式调用的 langfuse 记录会缺。
  取舍理由：不改动该包装函数即不影响非流式路径的观测。

### 4.3 SSE 层接线（`chat.py`）

```
生成器开始
  ├─ 装通道("rewrite") → await asyncio.to_thread(pipeline.run, ...) → 卸通道
  ├─ 装通道("answer")  → async for ev in agent_orchestrator.run_stream(...) → 卸通道
  └─ 结束
```

通道实现（SSE 侧）：内部持有 `asyncio.Queue` 与主循环引用，
`push` 走 `loop.call_soon_threadsafe(queue.put_nowait, ev)`。

**Pipeline 阶段的并发排空**：`await asyncio.to_thread(...)` 期间主循环是空闲的，
因此把 pipeline 包成 `asyncio.Task`，主循环侧 `while not task.done(): 排空队列并 yield`，
结束后再补排空一次，避免末尾片段丢失。

### 4.4 事件协议（新增 2 种 SSE 事件）

```json
{"delta": {"phase": "rewrite", "text": "我们优先"}}
{"delta": {"phase": "answer",  "text": "政企"}}
{"delta": {"phase": "answer",  "attempt_reset": true}}
```

第三种服务于用户选定的**「保留并标注」**策略：换候选/重试时不清空已显示内容，
而是让前端插入一条"上段因模型切换已废弃"的分隔提示。

**谁发 `attempt_reset`**：不改 executor——调用器**每次被调用即一次新尝试**，
所以它在开流前先发一条。

语义需明确（否则实现会歧义）：

- **第 1 条** `attempt_reset` 对应首次尝试，此时前端还没有任何内容，
  **不插分隔**——它只是"新一轮生成开始"的信号
- **从第 2 条起**（即发生过重试或候选降级）且前端已有内容时，才插入"上段已废弃"的分隔
- 判据是"**本次请求内收到的第几条**"，不是全局计数

### 4.5 四种输出的显示处理（本设计最需要评审的一节）

原始输出形态差异极大，必须分别处理。**提取逻辑全部放在 SSE 显示层**，
不进入业务层（符合"只改输出"）：

| 来源 | 处理 |
|---|---|
| 改写 + 意图 | 增量抽出 JSON 字段 `rewritten_question`，逐字显示；意图/子问题等 JSON 完整后仍用现有 `step` 事件显示 |
| 最终答案 · FC | 直接逐字显示（纯文本） |
| 最终答案 · text 协议 | **抑制**，直到缓冲区出现 `Final Answer:` 才显示其后内容（避免把 ReAct 草稿漏给用户） |
| 汇总 · plan | 增量抽出 JSON 字段 `answer` |

**判定方式（内容特征驱动）**，**必须按此顺序**逐条判定，命中即停：

1. **JSON 模式**：
   - `phase == "rewrite"`：缓冲区（去围栏后）以 `{` 开头即进入
     （该阶段已知是 `AgentRewriteIntentCombinedSchema`），抽 `rewritten_question`
   - `phase == "answer"`：**必须同时满足**"以 `{` 开头"**且**前 200 字符内含
     `"sufficient"`"才进入（抽 `answer`）
2. 含行首 `Thought:` / `Action:` / `Action Input:` → **抑制**（ReAct 草稿轮）
3. 含 `Final Answer:` → 只显示其后内容
4. 其余 → 原样显示

**为什么 `answer` 阶段的 JSON 判定要多一个 `"sufficient"` 条件**：
FC 协议的答案是纯文本，而**用户完全可能让 Agent 输出一段 JSON**
（例如"把结果以 JSON 给我"）。若只看"以 `{` 开头"，这种正常答案会被误判成
汇总的结构化输出，然后因为找不到 `answer` 字段而**整段不显示**。
`"sufficient"` 是 `SummaryVerdictSchema` 的首个字段（`summarize_node.py:46-72`），
用它做门禁可把误判压到极低。

**兜底方向是"不显示"，不是"错显示"**：字段/标记还没出现就什么都不推，
宁可晚出现，也绝不把 JSON 结构或草稿文本漏给用户。

### 4.6 增量 JSON 字段抽取器

输入：**每次追加一个片段的累积文本**；输出：该字段**当前已确定可见**的字符串值。

要求：

- 定位 `"<字段名>"` 键后进入字符串值区域
- 正确处理转义：`\"` / `\\` / `\n` / `\t` / `\uXXXX`
- 跨 chunk 边界安全（一次 push 的文本可能在任何位置断开）
- 字段未出现 / 值未闭合 → 返回"尚无内容"，不返回半截错误内容

---

## 5. 边界与失败处理

| 情况 | 处理 |
|---|---|
| `sink.push` 抛异常 | 内部 try/except **静默吞掉**——绝不能因为推送失败影响主链路 |
| 前端断连 / `abort` | 现有 `abort` 语义不变；通道随生成器回收，无资源泄漏 |
| 流式中途模型失败 | 现有重试/降级照常工作（调用器抛异常 → 老逻辑接管）；前端已显示内容保留并标注 |
| `tool_calls` 流式增量解析失败 | 退回"当作无 tool_calls"是危险的 → 改为**整轮判失败**，交由现有降级逻辑处理 |
| 无通道（例如其它调用方） | 走今天完全相同的非流式路径 |
| 超时 | `tier.timeout_ms` 语义不变（整段生成原本也要在该预算内完成） |

---

## 6. 测试策略

**新增**

- 通道模块单测：装/卸/嵌套/无通道时 `current_sink()` 为 `None`；`push` 抛异常不影响调用方
- 增量 JSON 抽取器单测：正常、转义（`\"` `\n` `\uXXXX`）、跨 chunk 边界、字段未出现、
  值未闭合、字段名出现在值内部（不应误判）
- 调用器流式分支单测：有通道时推送的片段拼接结果 == 返回的 `content`；
  无通道时行为与今天完全一致；`tool_calls` 增量聚合正确
- 前端：`streamEvents` 的 `delta` 判别；`useChatStream` 按 `phase` 分流
  （`rewrite` 进改写缓冲、`answer` 进正文）；`attempt_reset` 插分隔提示

**回归（必须原样通过）**

- `测试/test_llm_attempt_budget.py`（覆盖 executor 与 caller）
- `测试/test_graph_state_machine.py`（`FakeModelRouter` 跑全图）
- 前端现有 95 个用例

---

## 7. 明确排除

- 不改 `ModelRouter.chat` / `chat_with_tools` 的路由逻辑与签名
- 不改 `AsyncModelRoutingExecutor` 的候选循环、重试、熔断
- 不新增 `AsyncStreamRoutingExecutor`
- 不改 `PURPOSE_TIER_MAP` 与任何 prompt
- 不增加 LLM 调用次数
- 不改 `/chat`（非流式闲聊）与 `/chat/stream`（独立流式端点）
- 不改 `get_stream_writer` 机制、不改 `stream_mode`
- 复用现有 `step` 事件与「执行过程」前端组件，不新增阶段进度协议

---

## 8. 风险

1. **增量 JSON 抽取器**是本改动最易出错的部分（转义、分片边界、字段顺序）。
   缓解：兜底一律"不显示"；专门的边界单测；字段名误判用例。
2. **FC 的"前言"**：模型先输出"我先查一下…"再调工具时，那段文字会先显示、
   随后被标为"已废弃"。这是用户选定取舍的直接后果，会偶尔在界面上留下作废文字。
3. **langfuse 观测缺口**：流式调用不经过现有包装函数，这些调用在 langfuse 里不可见。
   缓解：在代码注释中显式标注；后续如需补齐，应给包装函数加流式支持而非另起通道。
4. **跨线程推送的时序**：`call_soon_threadsafe` 投递与生成器结束之间存在竞态，
   可能丢末尾片段。缓解：结束前补排空一次（见 4.3）。
5. **`stream=True` 与 `response_format` 的厂商兼容性（最需要你知情的一条）**：
   改写（`AgentRewriteIntentCombinedSchema`）与汇总（`SummaryVerdictSchema`）
   都是**结构化输出**调用。若某个候选厂商不支持"流式 + 结构化输出"，
   会直接 400——而**每个候选都会以同样方式失败**，最终整轮请求失败
   （现有降级机制救不了，因为不是模型不可用，是参数组合不被接受）。

   现有代码里已有一套能力降级机制 `_resolve_response_format`
   （`async_openai_caller.py:379-380` 注释："按候选能力自动降级
   （json_schema → json_object → 剥离）"），**但它不覆盖 stream 维度**。

   处理方式**已定**（用户确认）：开流前失败且异常特征像"参数不被接受"时，
   **同一次尝试内改用非流式重发**，结果一次性渲染；探测请求不计入"调用次数"。
   完整行为规格见 10.1。

---

## 9. 自审记录

（写完本规格后按四步自审：占位符扫描 / 内部一致性 / 范围 / 歧义）

| # | 问题 | 性质 | 修正 |
|---|---|---|---|
| 1 | 3.6 节引用了 `chat.py:722-737` 的既有约定，但未说明本设计是否受其约束 | 歧义 | 明确：该约定的前提是"流式自走候选降级"，本设计不改路由，因此不适用、也不引入该执行器 |
| 2 | 4.5 节四种输出的判定顺序未写清，存在"同时命中 JSON 与草稿标记"的解释空间 | 歧义 | 明确为**有序**判定（JSON → 草稿标记 → Final Answer → 原样），并写明兜底方向是"不显示" |
| 3 | 4.3 节未说明 Pipeline 阶段主循环空闲，可能出现队列积压 | 遗漏 | 补充"包成 Task + 循环排空 + 结束补排空" |
| 4 | 5 节未规定 `tool_calls` 增量解析失败的处理 | 遗漏 | 明确改为整轮判失败，交由现有降级逻辑处理（不可当作"无 tool_calls"） |
| 5 | 2 节与 7 节都能推出"不改路由"，但 2 节未点明它导致 `AsyncStreamRoutingExecutor` 方案作废 | 内部一致性 | 在 2 节补一句说明，避免实现者照旧注释去建执行器 |
| 6 | 4.2 标题声称调用器是"唯一被改的生产函数"，但 4.3 明确要改 `chat.py` | **内部矛盾** | 改为"唯一被改的**模型层**函数" |
| 7 | 4.4 只说"第 2 条 `attempt_reset` 才插分隔"，但没定义"第几条"是本次请求内计数还是全局 | 歧义 | 明确为**本次请求内**的序号；第 1 条只是"新一轮开始"信号，不插分隔 |
| 8 | 4.5 的 JSON 判定只看"以 `{` 开头"，**用户要求输出 JSON 时正常答案会被误判并整段不显示** | **逻辑漏洞** | `answer` 阶段额外要求前 200 字符含 `"sufficient"` 才进入 JSON 模式 |
| 9 | 未识别"`stream=True` + `response_format` 厂商兼容性"风险 | 遗漏 | 新增风险 5 与第 10 节（后经用户确认为方案 A） |
| 10 | 2 节把约束写成"不许增加调用次数"的硬规则，**偏离用户真实意图**（目的是耗时；怕的是改坏 rewrite/intent 解析） | **意图偏差** | 按用户澄清重写 2 节；1 节补"成功标准"（感知等待而非总耗时）；10.1 定为方案 A 并给出逐条行为规格 |

**审阅后的修订（第二轮之后）**：用户澄清了约束的真实意图，触发第 10 条修订。
本次修订只改"为什么"与"边界定义"，**未改动任何技术方案**（旁观通道、事件协议、
显示层提取、四个调用点的处理均不变）。

**范围结论**：本规格聚焦单一目标（把既有 LLM 调用的输出改为逐字送出），
不重构路由、不新增调用链，可作为单个实现计划。

---

## 10. 兼容性策略与待办

### 10.1 结构化输出调用遇到"不支持流式"的厂商时怎么办 —— **已定：方案 A**

用户已确认：探测包式的额外请求**不算增加调用次数**；厂商不支持流式时
**退回非流式、结果出来一次性渲染**。

**具体行为规格**：

1. 有通道时，该次调用先尝试 `stream=True`
2. **在收到首个 chunk 之前**失败，且异常为 `APIStatusError` 且状态码 ∈ {400, 422}
   （特征：参数组合不被厂商接受）→ **同一次尝试内改用非流式重发一次**
3. 非流式重发**不推送任何 delta**，结果照常返回——前端表现为"这一段没有逐字，
   直接出结果"，正是用户要求的行为
4. 其它异常（超时 / 连接错误 / 5xx / 未收到 chunk 前的流中断）→ **不重发**，
   原样抛出，交由现有 `run_with_attempt_budget` 重试与候选降级处理
5. 收到首 chunk **之后**才失败 → 不重发（已有内容已推给前端，重发会重复）；
   原样抛出，交由现有降级处理，前端按 `attempt_reset` 标注

**边界**：若某候选连续对多个请求都触发第 2 条，会形成"每个请求多一次失败探测"的
固定开销（一次失败请求通常几百毫秒内返回，不影响用户可感知耗时）。
本设计**不做**跨请求的负缓存（那需要改路由层，违反约束 3）。

### 10.2 待办（本次不做，但需记录）

- langfuse 对流式调用的观测缺口（风险 3）需后续补齐
- 若实测所有候选都支持"流式 + 结构化输出"，10.1 的第 2 条成为死代码，
  可在后续清理（保留不影响正确性）
