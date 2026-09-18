## Context

动机见 `proposal.md - Why`。此处只列影响**方案选择**的现状与约束（均为本次实地核实的结果）：

1. **工具参数目前无法表达枚举**。`app/core/tools/base.py` 的 `ToolParameter` 只有
   `name / type / description / required` 四个字段；`BaseTool.schema_parameters()`
   对每个参数只输出 `{"type": ..., "description": ...}`——**数组类型连 `items` 都没有**。
   因此"给 `collection_names` 加 enum"不是改一处字面量，而是要先让参数模型能承载约束。

2. **`schema_parameters()` 是每请求调用的**（`prepare_node` 在构造 Function Calling
   定义与提示词 schema 文本时各调一次）。这意味着"运行时动态枚举"可以不引入任何刷新
   机制——每次请求天然重算。

3. **集合清单已有权威来源**。`KbCollectionRegistry` 是进程内只读快照，由
   Postgres `vector_collections` 表加载（上传/删除后事务提交时 `replace`）。
   `DefaultIntentClassifier` 已经从它合成动态意图节点，`prepare_node` 从
   `slots.top_kb_node.collection_names` 取集合定向。也就是说注册表已经是
   **意图层与编排层的共同上游**，工具 schema 再接它不会形成第二条加载路径。

4. **项目无迁移框架**。`app/main.py` 在启动期 `Base.metadata.create_all` 之后执行
   `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 的"轻量补列"。**没有任何 DROP 先例**。

5. **集合参数描述里目前写着一个不存在的集合名**作为示例（`如 ['hr_docs']`）。
   它本身在教模型"名字长这样"，属于必须一并去掉的诱导源。

6. **`retrieval_hint` 的落点横跨六处**：`agent_schemas`（字段）、`models`（DB 列）、
   `collection_repo`（读写）、`kb_collection_registry`（描述符 + `intent_text()` 拼接）、
   `document.py` / `kownledgebase.py`（上传 Form 与响应）、集合资产 JSON。

## Goals / Non-Goals

**Goals:**

- 让"集合名"从一个自由文本参数变成受约束的枚举选择，且**不依赖**意图层是否命中集合节点。
- 让集合枚举与集合描述**同处下发**——模型在同一处看到"有哪些""各是什么""只能选这些"。
- 把 `ToolParameter` 的约束表达能力补到能承载 `enum` 与数组 `items`，使这套做法可复用于
  其他运行时资产（技能、图谱集合、工具名）而不只是给一个工具打补丁。
- `retrieval_hint` 在存储、接口、注册表、提示词渲染四处**彻底消失**，且数据层不可逆移除。

**Non-Goals:**

- **不做硬拒绝**。越界取值静默过滤 + 回落全库（用户决策）。因此本设计**不追求
  "幻觉率为零"**，只追求"幻觉不再导致整轮失败"；代价是幻觉被掩盖、无法直接度量。
- **不做问题实体抽取**，也不做"意图 ↔ 实体"匹配度校验（proposal 已声明暂缓）。
- **不引入 Alembic 或任何迁移框架**。沿用启动期幂等 DDL。
- **不给 `_untagged` 补描述**。它保持无描述、因此不进枚举（用户决策）。
- **不改检索算法本身**（召回、重排、top_k 语义均不动）。

## Decisions

### D1：动态枚举落在"覆写 `schema_parameters()`"，而不是刷新 `self.parameters`

`RagKnowledgeSearchTool` 目前把 `self.parameters` 当静态列表在 `__init__` 里建好。
改为**保持 `parameters` 作为参数签名（名称/类型/是否必填/基础说明），在
`schema_parameters()` 覆写里对 `collection_names` 注入运行时枚举与描述**。

- 选它的理由：`schema_parameters()` 已被每请求调用（Context 第 2 条），枚举天然新鲜；
  上传新集合后下一次请求就能选到，不需要任何人去"通知工具实例刷新"。
- 备选一：注册表 `replace` 时回调刷新 `self.parameters`。**否**——注册表要反过来认识
  工具实例，且工具实例的生命周期与注册表刷新时机不一致，容易漂移出"陈旧枚举"。
- 备选二：在 `prepare_node` 里对导出的 schema 做后处理补 enum。**否**——把
  "工具该暴露什么参数"的知识泄漏到调用点，其它调用点（如 evals 直连工具）会漏掉。

### D2：扩展 `ToolParameter` 增加 `enum` 与 `items`，而不是让工具直接返回整份 dict schema

- `ToolParameter` 增加可选 `enum: list | None` 与 `items: dict | None`；
  `schema_parameters()` 在有值时把它们合并进 property 定义。
- 选它的理由：这是**参数模型的表达能力缺口**，不是单个工具的特例。补齐后
  `knowledge_graph_search` 的 `collection`、以及任何"运行时资产"参数都能复用。
- 备选：让 `RagKnowledgeSearchTool` 直接覆写返回一份裸 dict。**否**——会绕过
  `ToolParameter` 这条统一出口，形成"有的工具走声明式、有的走手写 schema"的分裂。

### D3：枚举口径 = 注册表中**描述非空**的集合名（用户决策）

- 依据：无描述的集合无法被判定适用场景，放进候选只会诱导误选；且本变更要求
  "每个枚举值都带一句描述"，无描述项会让该要求自相矛盾。
- 代价：`_untagged`（3 篇未分类文档）不可被显式指定。**接受**——它是系统兜底桶，
  不是业务语义上可判定的检索目标。
- 附加要求：枚举**不得**包含注册表之外的任何名字，包括作为示例出现在参数说明里
  （Context 第 5 条的 `hr_docs` 必须删掉）。

### D4：越界取值静默过滤并回落全库（用户决策）

- 行为：过滤出合法集合名 → 非空则限定检索；为空则不限定集合检索（全库）。
  两种情况都留可观测记录（被丢弃的取值）。
- 理由：枚举的强制力取决于模型与网关是否支持严格 schema，"还能检索到东西"优先于
  "严厉报错"。
- **明确代价（需在实现中如实承认）**：幻觉被静默吸收，`rag` 侧不会因此报错，事后只能
  靠留痕统计。因此留痕是这条决策的**必要配套**，不是可选项。

### D5：过滤点放在 `rag_service.retrieve_contexts` 的集合白名单入参处

- 该入参已经是"意图路由硬约束"的统一入口（`collection_names` 白名单），
  `rag_search.py` 只做参数形态容错（数组/字符串/None），不做语义校验。
- 选它的理由：无论调用来自 Planner、ReAct 还是 evals 直连，都经过这一个口，
  一处过滤即全覆盖。
- 备选：在 `rag_search.py` 工具层过滤。**否**——绕过工具直连 service 的调用点会漏。

### D6：`retrieval_hint` 彻底移除，含 `DROP COLUMN`（用户决策）

- 从六个落点全部删除（Context 第 6 条）。路由语义统一由 `description` 承担。
- 备选：保留 DB 列但全链路停用。**已被用户否决**——理由是"以后也不会使用"，
  留一个永不写入的列会成为下一次误用的入口。
- **不可逆**：DROP COLUMN 无法通过代码回滚恢复数据（见 Risks）。

### D7：DDL 落点在 `app/main.py` 启动期，幂等执行

- 沿用现有"轻量 DDL"惯例，在既有 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS engine`
  同一事务块内追加 `ALTER TABLE vector_collections DROP COLUMN IF EXISTS retrieval_hint`。
- 选它的理由：`IF EXISTS` 保证重复启动不报错；与现有 DDL 同处便于审阅。
- 备选：引入 Alembic。**否**——超出本变更范围，且与 Non-Goals 冲突。

### D8：`description` 作为唯一权威，但**不**做"描述长度/格式"约束

- 只要求"有且仅有一份"，不规定长短与写法。实测中 `sales_kb.description` 写得足够具体
  （覆盖产品与定价、折扣权限与异议话术），问题不在描述质量而在模型根本没拿到真实清单。
- 不做长度约束的理由：新增约束会引入"描述写多长才算合格"的不可判定标准，且本次目标
  是消除双份来源，不是提质。

## Risks / Trade-offs

- **`DROP COLUMN` 不可逆** → 部署前对 `vector_collections.retrieval_hint` 做一次
  `SELECT name, retrieval_hint` 导出留档（任务里落一条）；若需回滚，只能重建列并回灌，
  **且回灌数据来自该导出**。代码回滚不回数据。
- **静默回落掩盖幻觉** → 留痕必做（D4）；回归测试必须覆盖"全部越界 → 仍返回全库结果"
  与"部分越界 → 只保留合法值"，避免实现写成"遇非法直接空召回"。
- **枚举只对有描述集合生效，可能缩小可选范围** → 若某业务集合描述被清空，它会**静默
  退出枚举**，表现为"这个集合突然选不到了"。缓解：把"枚举集合数"打进可观测记录，
  与注册表总数对比，出现差值即可发现。
- **`retrieval_hint` 删除会改变意图匹配输入** → `intent_text()` 少一段文本，
  动态集合节点的 embedding/打分输入随之变化。缓解：回归必须跑意图匹配用例
  （动态集合命中），确认命中率不下降。
- **参数描述变长挤占提示词预算** → 8 个集合 × 一句描述会进入每次 Function Calling 定义。
  缓解：描述只取集合元数据原文、不额外加模板话术；并复用既有 `platform/llm-call-budget`
  的预算观测确认单请求增量可控。
- **`enum` 在严格模式下的兼容性** → 部分 OpenAI 兼容网关对 `enum` 支持不完整。
  缓解：D4 的静默回落正是为此设计的兜底，不依赖网关强制。

## Migration Plan

1. **部署前（人工，不可自动化的一部分）**：导出并留档现有映射——
   `SELECT name, retrieval_hint FROM vector_collections;`
2. 发布新版本：启动期 DDL 自动 `DROP COLUMN IF EXISTS retrieval_hint`；
   代码路径不再读写该列，因此删除与新代码无先后依赖（新代码不会去碰它）。
3. **回滚**：代码回滚即可恢复旧行为；`retrieval_hint` 列需手工重建并用第 1 步的导出回灌。
   **无法回滚的部分**：列的物理删除本身已发生。
4. 存量上传调用方：多传 `retrieval_hint` 会被 FastAPI 忽略，不报错；依赖响应字段的
   调用方需同步调整（proposal 已标注 **BREAKING**）。

## Open Questions

- 集合数量增长到几十个后，把全部描述注入单个参数说明是否会挤占过多预算——留待集合规模
  真的上来后用 `platform/llm-call-budget` 的观测数据再评估，**当前 8 个集合不构成问题**。
- 未来若要做"统一资产注册表"（proposal 已列为后续方向），本次的
  `ToolParameter.enum` 能否直接复用于技能名与工具名——留待那时验证，不影响本次任务拆分。
