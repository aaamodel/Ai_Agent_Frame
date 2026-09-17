## Why

上一轮冒烟评测（`python -m evals.report --run-all --limit 2 --order random`）暴露了三处**互不相干**的缺陷，每处都有实测证据。

**① 超时预算被"一次调用 + SDK 自动重试"共用。** FAST tier 配了 50000ms，但日志出现
`Retrying request to /chat/completions in 0.47s` 紧接 `tier-timeout (50000ms(tier=fast))`。
根因：`async_build_openai_client` 从不传 `timeout`，客户端走 SDK 默认（读超时 600s、`max_retries=2`），
而 tier 超时是用 `asyncio.wait_for` 包在**外面**的——于是两次尝试共用一个 50s 预算。
后果不止报错：「改写+意图」的单次合成调用被打回旧的两段链路（白等 50s + 多一次 LLM 调用）。
同时 `light_rag.py` 的 LLM 调用**完全绕开 model_router**（硬编码百炼 `api_key`/`base_url` 直连，
只借用 tier 候选池里的模型名），超时用的是 LightRAG 自身全局 `llm_timeout`（实测日志 240s），
与 tier 配置、熔断、Langfuse 富化全部脱节。结果是**同一档模型在不同链路上行为不同**，
且超时口径无法从配置推导。

**② 表格工具不披露"能怎么聚合"。** `local_excel_read_tool` 的概览只列出列名，不说哪些列可作
`group_by`、哪些是数值列。实测「三季度业绩主要是哪条产品线贡献的」用了 **3 次**调用：
概览 → 再看一眼该 sheet 的具体数据 → 才敢写聚合参数。第 2 次纯属"工具没说，所以模型只能去看"。

**③ RAG 的 MRR 没有最小样本门。** `--limit 2` 只抽到 1 条可判分用例，该条排第 3 →
`mrr=0.333 < 0.55` 判红；而**全量 32 条实测 `mrr=0.699`（达标）**、`recall@5=0.963`。
报告本身已确立"样本量不足的指标不判定"的原则（P95 需 ≥20 条、边界准确率需 ≥5 条），
但 MRR 漏了这道门，导致**单条用例的排名直接等于整体水位**。

三件都是"预算 / 披露 / 读数"层面的小修，但分别决定：链路会不会白烧 LLM 调用、工具会不会多打一次往返、冒烟结果可不可信。

## What Changes

1. **单次模型调用独立计预算**：tier 超时语义从"整个候选一次预算"改为"**每次尝试**独立预算"——
   HTTP 层每尝试一个 tier 超时、SDK 隐式重试归零、重试由执行器显式控制并逐次记录耗时与结果。
2. **LightRAG 的 LLM 调用与 tier 对齐**：超时、重试、思考开关取自路由/tier 配置，而不是各写一套硬编码。
3. **表格工具披露可聚合能力**：概览按 dtype 给出每张表的「可用作分组列 / 可聚合数值列 / 可直接复用的参数示例」，
   使"想聚合"不必先"看一眼数据"。
4. **RAG 指标补样本量门**：为 MRR 增加最小样本数要求，未达下限时不参与质量门判定（与 P95 / 边界准确率同口径）。

**明确不做**（记录以免丢失）：

- 不新增或更换模型、不调整 tier 候选池（已排除"靠备选模型兜底"这条路）。
- 不改检索算法、不改分块参数。实测分块正常：`市场活动效果.md` 959 字 → 3 块，
  含关键事实那一段（chunk 1）正是被召回的那一块；全库按同参数重切得 60 块，与线上
  `sales_kb=60片` 一致。R24 排第 3 的真因是 top-5 分数密集并列（0.0308~0.0328）+
  该语料为交叉引用式叙述（同一事件被多篇文档提及），属数据/排序问题，不在本次范围。
- 不修 `_EXPLICIT_STEP_HINT_REGEX` 把「**优先**」里的"先"误判为步骤提示的缺陷
  （它把本次 T13 强制推成 `plan_execute`、`decision_source=explicit_hint`、置信度 0.92，
  是那次跑 134s 的起点）。它与本变更三项不同源，宜单独处理。

## Capabilities

### New Capabilities

- `platform/llm-call-budget`: 模型调用的超时与重试预算口径——单次调用独立计时、重试显式可控可观测，
  并确保所有 LLM 入口（含 LightRAG）对同一档模型使用同一套参数。
- `agent/excel-query`: 表格检索工具的能力披露——概览阶段即告知可用的筛选 / 分组 / 聚合能力，
  使聚合请求无需先做一次"探测性读取"。
- `evals/rag-quality-gate`: RAG 检索指标的质量门——含最小样本量门，避免小样本把指标判红或判绿。

### Modified Capabilities

（无。`openspec/specs/` 当前为空——`openspec-cn list --specs` 返回「未找到规范」，
本次三项均为首次登记的能力。）

## Impact

- `app/llm_model_router/async_model_executor.py`：超时语义改为"每次尝试独立预算"；重试显式化、逐次可观测。
- `app/llm_model_router/async_openai_caller.py`：客户端显式设置 `timeout` 与 `max_retries`，不再依赖 SDK 默认。
- `app/llm_model_router/model_router_config.py`：tier 配置需要承载"单次超时"与"重试次数"两个语义（若现有字段不足）。
- `app/infrastructure/knowledgebase/light_rag.py`：LLM 调用的超时 / 重试 / 思考开关与 tier 对齐。
- `app/core/tools/builtin/localexcel.py`：概览分支增加可聚合能力披露。
- `evals/metrics.py`、`evals/thresholds.yaml`、`evals/report.py`：MRR 最小样本门。
- **成本影响**：单次预算独立后，最坏总耗时由 `1 × tier_timeout` 变为 `(1 + 重试次数) × tier_timeout`，
  需要在 design 中给出重试次数取值与安全阀，避免"修好超时语义却把 P95 拉爆"。
- **评测影响**：MRR 加门后，`--limit 2` 这类小样本冒烟不再因该指标判红；全量结论（`mrr=0.699`）不变。
- **可观测性影响**：超时日志需要能区分"一次尝试超时"与"重试后仍超时"，否则同类问题仍难定位。
