## Why

一次真实 agent 调用（2026-09-18 10:45）里，Planner 给 `rag_knowledge_search` 传了**并不存在的集合名**：

```
[ t1 ] collection_names: ['product_docs', 'sales_policies']
[ t2 ] collection_names: ['pricing_guide', 'sales_policies']
```

而向量库真实集合只有 8 个：`Cross_Entity_Relation` / `Finance_Invoice_Manual` /
`Human_Resources_Employee` / `IT_Network_Support` / `Internal_System_Manual` /
`Org_System_Relation` / `sales_kb` / `_untagged`。两个子任务因此全部空召回，
`rag_knowledge_search` 连续 2 次无效后被硬熔断，整轮降级收尾——**一次完整的工具调用、
一次子任务提炼、一次台账更新全部白烧**。

根因不是"模型选错了集合"，而是**参数空间没有约束**，且系统在无意中教它编名字：

1. `collection_names` 的 `ToolParameter` **没有 enum**，`BaseTool.schema_parameters()`
   只导出 `{type, description}`——数组连 `items` 都没有；
2. 它的参数描述里写着 `如 ['hr_docs']`，而 **`hr_docs` 本身就是一个不存在的集合**。
   模型照抄的是"名字长这样"，而不是"名字只能是这些"；
3. 设计上原本有一条约束链（`KbCollectionRegistry` 动态意图节点 →
   `slots.top_kb_node.collection_names` → `prepare_node` 注入"必须将
   `collection_names` 设置为 X，严禁查询其他集合"的硬约束），但该次调用日志里**没有**
   `ReAct 已注入 KB 集合定向检索硬约束` 这条 INFO —— 意图层没有解析出 KB 节点，
   硬约束从未生效，Planner 是在**无约束状态**下凭空生成集合名。

另外，集合资产同时维护 `description`（这个集合是什么）与 `retrieval_hint`
（用户问什么时选它）两份文本，两者语义重叠、会被拼接成同一段注入文本
（`KbCollectionDescriptor.intent_text()`）、并各自持久化一列。实测表明
`sales_kb.description` 已经写了"产品与定价""折扣权限与异议话术"，模型仍然编了
`pricing_guide` —— **问题不在描述写得好不好，而在真实集合名从未作为受约束选项给过它**。
既然 `description` 已足够承担路由语义，双份文本只增加维护面与 token，决定退役。

## What Changes

- **新增**：`rag_knowledge_search` 的 `collection_names` 参数改为**运行时动态枚举**，
  取值恒等于当前注册表中**具备描述**的集合名全集；集合描述随参数 schema 一并下发，
  让"有哪些集合、各是什么"与"只能选这些"在同处呈现。
- **新增**：枚举越界时的**静默回落**——过滤掉不在注册表内的名字；过滤后为空则不限定
  集合检索（回落到全库），并在工具返回中留下可观测记录。不改用硬拒绝：容错优先，
  一次幻觉不应让整轮请求失败。
- **新增**：`ToolParameter` / `schema_parameters()` 支持 `enum` 与数组 `items`，
  使"运行时资产清单 → 受约束参数"成为可复用的机制，而不是只为这一个工具打的补丁。
- **移除（BREAKING）**：`retrieval_hint` 全链路退役——`agent_schemas` 字段、
  `vector_collections.retrieval_hint` 列（**DROP COLUMN**）、上传接口的两个
  `retrieval_hint` Form 字段、`KbCollectionDescriptor.retrieval_hint`、
  `intent_text()` 的拼接分支、集合资产 JSON 中的该字段。路由语义统一由
  `description` 承担。
- **移除（BREAKING）**：删除上传接口的 `retrieval_hint` 表单参数后，存量调用方若继续
  传该字段，FastAPI 会忽略未声明的多余 Form 字段（不会 422）；但**依赖响应中该字段**
  的调用方会取到 `None`/缺失。
- **数据侧**：**不为此类集合补写描述**。当前 `description` 为 `null` 的集合
  （`_untagged`，3 篇未分类文档）因此**不进入 enum**——其文档只能靠不限定集合的检索
  命中。这是本变更有意接受的取舍：它是系统兜底桶，不是业务语义上可判定的检索目标，
  给它编一句描述只会让模型在一个"其他"选项上做无意义的选择。

## Capabilities

### New Capabilities

- `agent/kb-collection-routing`: 知识库集合参数的**受约束取值**——集合枚举的来源、
  构成规则（仅含具备描述的集合）、随参数下发的集合描述、越界取值的静默回落与留痕。
  覆盖"模型只能在真实集合里选"这一行为要求。
- `platform/kb-collection-metadata`: 集合元数据的**单一权威字段**——`description` 作为
  唯一的路由语义来源；`retrieval_hint` 在存储、接口、注册表、提示词渲染四处全部退役，
  且不得以任何形式重新引入。覆盖元数据契约与接口兼容性要求。

### Modified Capabilities

<!-- 无：既有 6 个能力（agent/agent-goal、agent/excel-query、agent/plan-execute-control、
     agent/plan-subtask-allocation、evals/rag-quality-gate、platform/llm-call-budget）
     均不覆盖集合路由与集合元数据契约。检索质量门（evals/rag-quality-gate）的指标定义
     不受影响——它度量的是召回质量，而本次改的是"请求打给哪个集合"。 -->

## Impact

**代码**

- `app/core/tools/base.py` —— `ToolParameter` 增加 `enum` / `items`；`schema_parameters()` 导出它们
- `app/core/tools/builtin/rag_search.py` —— `collection_names` 改为动态 enum，去掉教模型编名字的示例 `['hr_docs']`
- `app/core/rag/rag_service.py` —— `retrieve_contexts` 的集合白名单入参增加越界过滤与回落
- `app/query_intent/kb_collection_registry.py` —— 移除 `retrieval_hint` 字段与 `intent_text()` 拼接分支；提供"具备描述的集合名"查询
- `app/models/agent_schemas.py` —— 移除 `retrieval_hint` 字段
- `app/infrastructure/database/models.py`、`collection_repo.py` —— 移除 `retrieval_hint` 列与读写
- `app/api/routes/document.py`、`app/api/routes/kownledgebase.py` —— 移除 `retrieval_hint` Form 与响应字段
- `app/main.py` —— 增加 `DROP COLUMN IF EXISTS retrieval_hint` 的轻量 DDL（项目无 Alembic，沿用现有启动期 DDL 惯例）

**数据 / 资产**

- `向量化数据库的集合.json` —— 移除 `retrieval_hint` 字段
- Postgres `vector_collections` —— 列被删除（**不可逆**）；`_untagged.description` 需补齐

**兼容性**

- 上传接口不再接受 `retrieval_hint`；响应中不再返回该字段。**BREAKING**
- 声明了 `retrieval_hint` 的集合描述文本长度会下降（少一份重复文本），意图匹配输入随之变化，
  需要回归验证动态集合节点的匹配是否仍准确

**不在本次范围**

- **问题实体抽取与"意图 ↔ 实体"匹配度校验**：项目当前没有 `QuestionEntity` 之类的结构，
  先暂缓。本次只解决"工具参数无约束"这一层的可直接观测故障。
- **统一资产注册表**（把集合/工具/技能/实体别名收敛到一处）：留作后续方向，
  本变更只把集合枚举这一条链打通。
