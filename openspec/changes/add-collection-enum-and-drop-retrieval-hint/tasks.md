## 1. 参数模型的约束表达能力（打底）

> 依据 design.md D2。当前 `ToolParameter` 无法表达 enum，数组连 `items` 都没有——
> 不先补这层，"给集合参数加枚举"无从下手。

- [ ] 1.1 为 `ToolParameter` 增加可选 `enum` 与 `items` 字段，并让 `schema_parameters()` 在有值时合并进 property 定义；验证：单测断言 `enum` / `items` 出现在导出结果中，且**未设置这两个字段的参数导出结果与改动前逐字一致**
- [ ] 1.2 确认既有工具（`rag_knowledge_search` / `knowledge_graph_search` / `local_excel_read_tool` 等）的 schema 导出无回归；验证：单测断言各工具导出结构不变，`pytest 测试 -q` 全绿

## 2. 集合枚举的来源与口径

> 依据 design.md D1 / D3 与 `agent/kb-collection-routing` 的前两条需求。
> 枚举必须**运行时**取自注册表，且只含有描述的集合。

- [ ] 2.1 在 `KbCollectionRegistry` 上提供"具备非空描述的集合（名 + 描述）"查询；验证：单测断言描述为空的集合被排除、有描述的按注册顺序返回
- [ ] 2.2 覆写 `rag_knowledge_search` 的 `schema_parameters()`，为 `collection_names` 注入动态枚举与逐条描述，并去掉参数说明里不存在的示例 `如 ['hr_docs']`；验证：单测断言 ① 枚举恰好等于"有描述集合"全集；② 每个枚举值都在说明里逐条出现；③ 说明文本中**不含任何注册表之外的名字**
- [ ] 2.3 验证枚举随注册表刷新：`replace` 新增/删除集合后，下一次 `schema_parameters()` 立即反映变化；验证：单测断言无需重建工具实例即可看到变化

## 3. 越界取值的静默过滤与回落

> 依据 design.md D4 / D5 与 `agent/kb-collection-routing` 的第三条需求。
> 过滤点必须在 `rag_service` 的白名单入参处——那是所有调用路径的唯一收口。

- [ ] 3.1 在 `rag_service.retrieve_contexts` 的集合白名单入参处按注册表过滤非法名；验证：单测断言越界名被丢弃、合法名保留
- [ ] 3.2 实现两条回落路径：过滤后非空 → 仅在合法集合内检索；过滤后为空 → **不限定集合**检索（全库），不得失败；验证：单测分别覆盖两种输入
- [ ] 3.3 为过滤与回落留下可观测记录（至少含被丢弃的取值）；验证：单测断言记录中能查到被丢弃的取值

## 4. retrieval_hint 全链路移除

> 依据 design.md D6 / D7 与 `platform/kb-collection-metadata`。这是 **BREAKING** 且
> `DROP COLUMN` **不可逆**，因此 4.1 必须先于 4.3 完成。

- [ ] 4.1 **（部署前人工步骤）** 导出并留档现存映射：`SELECT name, retrieval_hint FROM vector_collections;`。这是回滚时唯一的数据来源；验证：导出文件已保存且行数与集合数一致
- [ ] 4.2 移除 `agent_schemas` 的该字段、`models.py` 的列定义、`collection_repo.py` 的读写分支；验证：单测断言从存储构造描述符时不再读取该列
- [ ] 4.3 移除 `document.py` / `kownledgebase.py` 中该字段的 Form 入参与响应字段；验证：单测断言 ① 接口签名不含该参数；② 响应不含该字段；③ 调用方多传该字段时**请求仍成功**（不 422）
- [ ] 4.4 移除 `KbCollectionDescriptor.retrieval_hint` 字段与 `intent_text()` 中的拼接分支，路由语义只由 `description` 承担；验证：单测断言渲染文本只含一份语义文本、且不再出现"适用检索时机"字样
- [ ] 4.5 在 `app/main.py` 既有轻量 DDL 块内追加幂等的 `DROP COLUMN IF EXISTS`；验证：① 在已有该列的库上启动一次，列消失；② **再次启动不报错**（`IF EXISTS` 生效）
- [ ] 4.6 更新集合资产 `向量化数据库的集合.json`，移除该字段；验证：文件中不再出现该键，且其余字段与真实集合一一对应
- [ ] 4.7 全仓扫描确认无残留引用（含 `evals/tools/reembed_from_db.py` 等文档字符串）；验证：`retrieval_hint` 仅出现在本变更的 openspec 制品与历史记录中，应用代码与资产内为零

## 5. 回归与端到端

- [ ] 5.1 回归全部现有测试；验证：`pytest evals 测试 benchmark -q` 通过，且与变更前基线一致（当前存在 1 个既有的 cost gate 失败项，非本变更引入，不计入）
- [ ] 5.2 **意图匹配回归**：`intent_text()` 少一段文本会改变动态集合节点的匹配输入；验证：动态集合的意图命中用例通过，命中率不低于变更前（需要记录变更前基线数字）
- [ ] 5.3 用本次 trace 的原始问题（"私有化部署 上线周期/实施费 收费标准"类）端到端实跑；验证：① 工具调用参数中的集合名**全部属于真实注册表**；② 不再因空召回触发硬熔断；③ 与变更前（2 个子任务全空召回 + 工具熔断 + 降级收尾）对比
- [ ] 5.4 构造越界场景：桩掉模型输出一个不存在的集合名；验证：① 检索仍返回全库结果、请求不失败；② 记录中能查到被丢弃的取值
- [ ] 5.5 观测单请求提示词预算增量：8 个集合的描述进入 Function Calling 定义后，确认增量与 `platform/llm-call-budget` 的既有约束相容；验证：记录变更前后单请求输入字符数并对比
